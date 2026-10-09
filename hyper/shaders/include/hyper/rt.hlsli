#ifndef HYPER_RT_HLSLI
#define HYPER_RT_HLSLI

// Inline ray tracing against a hyper::RtScene. Bind with RtScene::AddLayoutItems /
// AddSetItems, plus the scene's bindless layout as descriptor set 1.
//
//     RtHit hit = TraceClosest(origin, dir, tMax);
//     if (hit.hit) { HitSurface s = GetHitSurface(hit, dir, lod); ... }
//     bool lit = TraceVisible(origin, sunDir, 1e6);
//
// Alpha-tested materials are tested against their base colour texture; alpha-
// blended and transmissive ones are ignored by rays, and so are instances whose
// materials the glTF hides from rays (hyper::Scene::SeenByRays).
//
// A ray tests at most `cutouts` alpha-tested candidates; the next one stops it. A ray
// that has threaded two dozen leaves is deep in foliage, and light through them is
// nothing; testing on is what made rays through a canopy cost a hundred times the
// rest. Rays that place things (where rain lands) pass HYPER_RT_EXACT.

#include <hyper/common.hlsli>
#include <donut/shaders/bindless.h>
#include <donut/shaders/packing.hlsli>

RaytracingAccelerationStructure t_SceneTlas : register(t40);
ByteAddressBuffer t_SceneInstances : register(t41);
ByteAddressBuffer t_SceneGeometries : register(t42);
ByteAddressBuffer t_SceneMaterials : register(t43);
StructuredBuffer<uint> t_SceneMaterialData : register(t44);   // the app's data on each material
SamplerState s_SceneMaterial : register(s40);
VK_BINDING(0, 1) ByteAddressBuffer t_BindlessBuffers[] : register(t0, space1);
VK_BINDING(1, 1) Texture2D t_BindlessTextures[] : register(t0, space2);

static const uint HYPER_RT_CUTOUTS = 24u;
static const uint HYPER_RT_EXACT = ~0u;

// The app's own data on a material (glTF extras "hyper": { "data": N }; 0 without one).
uint MaterialData(uint materialIndex)
{
    return materialIndex == ~0u ? 0u : t_SceneMaterialData[materialIndex];
}

struct RtHit
{
    bool hit;
    float t;
    uint instance;      // Donut instance index
    uint geometry;      // geometry index within the instance's mesh
    uint primitive;
    float2 barycentrics;
    float3x4 objectToWorld;
};

struct HitSurface
{
    float3 position;
    float3 normal;          // shading normal, facing the ray for double-sided materials
    float3 geometricNormal; // same orientation as normal
    bool frontFace;         // false: the ray hit the back of a single-sided surface
    float3 baseColor;
    float opacity;
    float metalness;
    float roughness;
    float3 emissive;
    float3 diffuseAlbedo;
    float3 specularF0;
};

GeometryData LoadGeometry(uint instance, uint geometry)
{
    const InstanceData inst = LoadInstanceData(t_SceneInstances, instance * c_SizeOfInstanceData);
    return LoadGeometryData(t_SceneGeometries, (inst.firstGeometryIndex + geometry) * c_SizeOfGeometryData);
}

MaterialConstants LoadMaterial(uint index)
{
    return LoadMaterialConstants(t_SceneMaterials, index * c_SizeOfMaterialConstants);
}

uint3 LoadTriangle(GeometryData g, uint primitive)
{
    return t_BindlessBuffers[NonUniformResourceIndex(g.indexBufferIndex)].Load3(g.indexOffset + primitive * c_SizeOfTriangleIndices);
}

float2 InterpolateTexcoord(GeometryData g, uint3 tri, float3 w)
{
    if (g.texCoord1Offset == ~0u)
        return 0;
    ByteAddressBuffer vb = t_BindlessBuffers[NonUniformResourceIndex(g.vertexBufferIndex)];
    const float2 t0 = asfloat(vb.Load2(g.texCoord1Offset + tri.x * c_SizeOfTexcoord));
    const float2 t1 = asfloat(vb.Load2(g.texCoord1Offset + tri.y * c_SizeOfTexcoord));
    const float2 t2 = asfloat(vb.Load2(g.texCoord1Offset + tri.z * c_SizeOfTexcoord));
    return t0 * w.x + t1 * w.y + t2 * w.z;
}

float4 SampleMaterialTexture(int index, float2 uv, float lod)
{
    return t_BindlessTextures[NonUniformResourceIndex(index)].SampleLevel(s_SceneMaterial, uv, lod);
}

// Whether a candidate hit on non-opaque geometry counts.
bool CandidateBlocks(uint instance, uint geometry, uint primitive, float2 bary)
{
    const GeometryData g = LoadGeometry(instance, geometry);
    if (g.materialIndex == ~0u)
        return true;
    const MaterialConstants m = LoadMaterial(g.materialIndex);
    if (m.domain != MaterialDomain_AlphaTested && m.domain != MaterialDomain_TransmissiveAlphaTested)
        return m.domain == MaterialDomain_Opaque;
    float opacity = m.opacity;
    if ((m.flags & MaterialFlags_UseBaseOrDiffuseTexture) != 0)
    {
        const uint3 tri = LoadTriangle(g, primitive);
        const float3 w = float3(1.0 - bary.x - bary.y, bary.x, bary.y);
        opacity *= SampleMaterialTexture(m.baseOrDiffuseTextureIndex, InterpolateTexcoord(g, tri, w), 0).a;
    }
    return opacity >= m.alphaCutoff;
}

// True if nothing blocks the segment.
bool TraceVisible(float3 origin, float3 dir, float tMax, uint cutouts = HYPER_RT_CUTOUTS)
{
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = dir;
    ray.TMin = 0.0;
    ray.TMax = tMax;
    RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES> q;
    q.TraceRayInline(t_SceneTlas, RAY_FLAG_NONE, 0xff, ray);
    uint tested = 0;
    while (q.Proceed())
    {
        if (q.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE
            && (++tested > cutouts
                || CandidateBlocks(q.CandidateInstanceID(), q.CandidateGeometryIndex(), q.CandidatePrimitiveIndex(), q.CandidateTriangleBarycentrics())))
            q.CommitNonOpaqueTriangleHit();
    }
    return q.CommittedStatus() == COMMITTED_NOTHING;
}

RtHit TraceClosest(float3 origin, float3 dir, float tMax, uint cutouts = HYPER_RT_CUTOUTS)
{
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = dir;
    ray.TMin = 0.0;
    ray.TMax = tMax;
    RayQuery<RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES> q;
    q.TraceRayInline(t_SceneTlas, RAY_FLAG_NONE, 0xff, ray);
    uint tested = 0;
    while (q.Proceed())
    {
        if (q.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE
            && (++tested > cutouts
                || CandidateBlocks(q.CandidateInstanceID(), q.CandidateGeometryIndex(), q.CandidatePrimitiveIndex(), q.CandidateTriangleBarycentrics())))
            q.CommitNonOpaqueTriangleHit();
    }

    RtHit h = (RtHit)0;
    h.hit = q.CommittedStatus() == COMMITTED_TRIANGLE_HIT;
    if (h.hit)
    {
        h.t = q.CommittedRayT();
        h.instance = q.CommittedInstanceID();
        h.geometry = q.CommittedGeometryIndex();
        h.primitive = q.CommittedPrimitiveIndex();
        h.barycentrics = q.CommittedTriangleBarycentrics();
        h.objectToWorld = q.CommittedObjectToWorld3x4();
    }
    return h;
}

// Surface at a hit. lod: texture mip bias for the material textures (rays have no derivatives).
// normalMap: apply the material's normal map (as Donut's ApplyNormalMap does); off, the
// interpolated vertex normal is the shading normal, which is enough for blurry GI rays.
HitSurface GetHitSurface(RtHit h, float3 rayDir, float lod, bool normalMap = false)
{
    const GeometryData g = LoadGeometry(h.instance, h.geometry);
    const uint3 tri = LoadTriangle(g, h.primitive);
    const float3 w = float3(1.0 - h.barycentrics.x - h.barycentrics.y, h.barycentrics.x, h.barycentrics.y);
    ByteAddressBuffer vb = t_BindlessBuffers[NonUniformResourceIndex(g.vertexBufferIndex)];

    const float3 p0 = asfloat(vb.Load3(g.positionOffset + tri.x * c_SizeOfPosition));
    const float3 p1 = asfloat(vb.Load3(g.positionOffset + tri.y * c_SizeOfPosition));
    const float3 p2 = asfloat(vb.Load3(g.positionOffset + tri.z * c_SizeOfPosition));

    HitSurface s = (HitSurface)0;
    s.position = mul(h.objectToWorld, float4(p0 * w.x + p1 * w.y + p2 * w.z, 1.0));
    float3 ng = normalize(mul(h.objectToWorld, float4(cross(p1 - p0, p2 - p0), 0.0)));
    float3 n = ng;
    if (g.normalOffset != ~0u)
    {
        const float3 n0 = Unpack_RGB8_SNORM(vb.Load(g.normalOffset + tri.x * c_SizeOfNormal));
        const float3 n1 = Unpack_RGB8_SNORM(vb.Load(g.normalOffset + tri.y * c_SizeOfNormal));
        const float3 n2 = Unpack_RGB8_SNORM(vb.Load(g.normalOffset + tri.z * c_SizeOfNormal));
        n = normalize(mul(h.objectToWorld, float4(n0 * w.x + n1 * w.y + n2 * w.z, 0.0)));
        // Winding is only a hint (mirrored transforms flip it); the vertex normals decide.
        if (dot(ng, n) < 0.0)
            ng = -ng;
    }

    MaterialConstants m = (MaterialConstants)0;
    m.baseOrDiffuseColor = 0.5;
    m.roughness = 1.0;
    m.opacity = 1.0;
    if (g.materialIndex != ~0u)
        m = LoadMaterial(g.materialIndex);

    const float2 uv = InterpolateTexcoord(g, tri, w);
    if (normalMap && (m.flags & MaterialFlags_UseNormalTexture) != 0 && g.tangentOffset != ~0u)
    {
        const float4 t0 = Unpack_RGBA8_SNORM(vb.Load(g.tangentOffset + tri.x * c_SizeOfNormal));
        const float4 t1 = Unpack_RGBA8_SNORM(vb.Load(g.tangentOffset + tri.y * c_SizeOfNormal));
        const float4 t2 = Unpack_RGBA8_SNORM(vb.Load(g.tangentOffset + tri.z * c_SizeOfNormal));
        const float3 tangent = mul(h.objectToWorld, float4(t0.xyz * w.x + t1.xyz * w.y + t2.xyz * w.z, 0.0));
        float3 texel = SampleMaterialTexture(m.normalTextureIndex, uv, lod).xyz;
        texel.xy = (texel.xy * 2.0 - 1.0) * m.normalTextureScale;
        texel.z = texel.z <= 0.0 ? sqrt(saturate(1.0 - dot(texel.xy, texel.xy))) : abs(texel.z * 2.0 - 1.0);
        if (dot(tangent, tangent) > 0.0 && t0.w != 0.0 && dot(texel, texel) > 0.0)
        {
            const float3 tn = normalize(tangent);
            const float3 bitangent = cross(n, tn) * t0.w;
            n = normalize(tn * texel.x + bitangent * texel.y + n * texel.z);
        }
    }

    s.frontFace = dot(ng, rayDir) < 0.0;
    if (!s.frontFace && (m.flags & MaterialFlags_DoubleSided) != 0)
        s.frontFace = true;
    if (dot(ng, rayDir) > 0.0)
    {
        ng = -ng;
        n = -n;
    }
    s.normal = n;
    s.geometricNormal = ng;

    s.baseColor = m.baseOrDiffuseColor;
    s.opacity = m.opacity;
    if ((m.flags & MaterialFlags_UseBaseOrDiffuseTexture) != 0)
    {
        const float4 c = SampleMaterialTexture(m.baseOrDiffuseTextureIndex, uv, lod);
        s.baseColor *= c.rgb;
        s.opacity *= c.a;
    }
    s.metalness = m.metalness;
    s.roughness = m.roughness;
    if ((m.flags & MaterialFlags_UseSpecularGlossModel) != 0)
    {
        s.metalness = 0.0;
    }
    else if ((m.flags & MaterialFlags_UseMetalRoughOrSpecularTexture) != 0)
    {
        const float4 mr = SampleMaterialTexture(m.metalRoughOrSpecularTextureIndex, uv, lod);
        s.roughness *= mr.g;
        s.metalness *= (m.flags & MaterialFlags_MetalnessInRedChannel) != 0 ? mr.r : mr.b;
    }
    s.emissive = m.emissiveColor;
    if ((m.flags & MaterialFlags_UseEmissiveTexture) != 0)
        s.emissive *= SampleMaterialTexture(m.emissiveTextureIndex, uv, lod).rgb;

    s.diffuseAlbedo = s.baseColor * (1.0 - s.metalness);
    s.specularF0 = lerp(0.04, s.baseColor, s.metalness);
    return s;
}

// Offsets a ray origin off a surface (Wächter & Binder, Ray Tracing Gems ch. 6).
float3 OffsetRayOrigin(float3 p, float3 n)
{
    const float origin = 1.0 / 32.0;
    const float floatScale = 1.0 / 65536.0;
    const float intScale = 256.0;
    const int3 of = int3(intScale * n);
    const float3 pi = float3(
        asfloat(asint(p.x) + (p.x < 0 ? -of.x : of.x)),
        asfloat(asint(p.y) + (p.y < 0 ? -of.y : of.y)),
        asfloat(asint(p.z) + (p.z < 0 ? -of.z : of.z)));
    return float3(
        abs(p.x) < origin ? p.x + floatScale * n.x : pi.x,
        abs(p.y) < origin ? p.y + floatScale * n.y : pi.y,
        abs(p.z) < origin ? p.z + floatScale * n.z : pi.z);
}

#endif
