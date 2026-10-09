// GPU-driven draws (hyper::GpuDrivenRenderer): vertex pulling from GpuScene,
// materials from Donut's material buffer and the bindless texture table.
//
//   vs / ps              the G-buffer, written exactly like Donut's GBufferFillPass
//   vs_depth / ps_depth  depth only (shadow cascades); the pixel shader only for alpha testing
//   task / mesh          the G-buffer by meshlets: the task shader culls 32 meshlets (frustum,
//                        normal cone, Hi-Z in phase 2, the LOD cut), the mesh shader emits the survivors
//   mesh_depth           the same for shadow cascades, depth only; the LOD cut is the camera's,
//                        so casters and receivers are the same surface
//
// ALPHA_TESTED: clip at the material's cutoff.

#include <hyper/gpu_scene.hlsli>
#include <donut/shaders/view_cb.h>
#include <donut/shaders/scene_material.hlsli>
#include <donut/shaders/motion_vectors.hlsli>

#ifndef ALPHA_TESTED
#define ALPHA_TESTED 0
#endif

struct DrawConstants
{
    PlanarViewConstants view;
    PlanarViewConstants viewPrev;
};
ConstantBuffer<DrawConstants> g_Draw : register(b0);
SamplerState s_Material : register(s0);
VK_BINDING(1, 1) Texture2D t_BindlessTextures[] : register(t0, space2);

struct VertexOut
{
    float4 position : SV_Position;
    float3 prevPos : PREV_POS;
    float2 texCoord : TEXCOORD;
    centroid float3 normal : NORMAL;
    centroid float4 tangent : TANGENT;
    nointerpolation uint draw : DRAW;
};

VertexOut ProcessVertex(uint vertex, uint drawIndex, GpuGeometry g, InstanceData inst)
{
    VertexOut o;
    const float3 pos = LoadGpuPosition(vertex);
    float3 prevPos = pos;
    if (g.prevBaseVertex != ~0u)
        prevPos = asfloat(t_GpuPrevPositions.Load3((vertex - g.baseVertex + g.prevBaseVertex) * 12u));
    const float3 normal = Unpack_RGB8_SNORM(t_GpuNormals.Load(vertex * 4u));
    const float4 tangent = Unpack_RGBA8_SNORM(t_GpuTangents.Load(vertex * 4u));

    const float3 world = mul(inst.transform, float4(pos, 1.0));
    o.prevPos = mul(inst.prevTransform, float4(prevPos, 1.0));
    o.texCoord = asfloat(t_GpuTexcoords.Load2(vertex * 8u));
    o.normal = mul(inst.transform, float4(normal, 0.0));
    o.tangent = float4(mul(inst.transform, float4(tangent.xyz, 0.0)), tangent.w);
    o.draw = drawIndex;
    o.position = mul(mul(float4(world, 1.0), g_Draw.view.matWorldToView), g_Draw.view.matViewToClip);
    return o;
}

void vs(uint vertex : SV_VertexID, uint drawIndex : SV_InstanceID, out VertexOut o)
{
    const GpuDraw d = t_GpuDraws[drawIndex];
    o = ProcessVertex(vertex, drawIndex, t_GpuGeometries[d.geometry], LoadGpuInstance(d.instance));
}

float4 SampleTexture(int index, float2 uv)
{
    return t_BindlessTextures[NonUniformResourceIndex(index)].Sample(s_Material, uv);
}

MaterialTextureSample SampleMaterial(MaterialConstants m, float2 uv, bool hasTangents)
{
    MaterialTextureSample t = DefaultMaterialTextures();
    if ((m.flags & MaterialFlags_UseBaseOrDiffuseTexture) != 0)
        t.baseOrDiffuse = SampleTexture(m.baseOrDiffuseTextureIndex, uv);
    if ((m.flags & MaterialFlags_UseMetalRoughOrSpecularTexture) != 0)
        t.metalRoughOrSpecular = SampleTexture(m.metalRoughOrSpecularTextureIndex, uv);
    if ((m.flags & MaterialFlags_UseEmissiveTexture) != 0)
        t.emissive = SampleTexture(m.emissiveTextureIndex, uv);
    if ((m.flags & MaterialFlags_UseNormalTexture) != 0 && hasTangents)
        t.normal = SampleTexture(m.normalTextureIndex, uv * m.normalTextureTransformScale);
    if ((m.flags & MaterialFlags_UseOcclusionTexture) != 0)
        t.occlusion = SampleTexture(m.occlusionTextureIndex, uv);
    if ((m.flags & MaterialFlags_UseTransmissionTexture) != 0)
        t.transmission = SampleTexture(m.transmissionTextureIndex, uv);
    if ((m.flags & MaterialFlags_UseOpacityTexture) != 0)
        t.opacity = SampleTexture(m.opacityTextureIndex, uv).r;
    return t;
}

MaterialConstants LoadDrawMaterial(GpuGeometry g)
{
    if (g.materialIndex != ~0u)
        return LoadGpuMaterial(g.materialIndex);
    MaterialConstants m = (MaterialConstants)0;
    m.baseOrDiffuseColor = 0.5;
    m.opacity = 1.0;
    m.roughness = 0.8;
    m.occlusionStrength = 1.0;
    return m;
}

struct GBufferOut
{
    float4 channel0 : SV_Target0;   // diffuse albedo, opacity
    float4 channel1 : SV_Target1;   // specular F0, occlusion
    float4 channel2 : SV_Target2;   // shading normal, roughness
    float4 channel3 : SV_Target3;   // emissive
    float3 motion : SV_Target4;
};

// Donut's GBufferFillPass output for a surface point (world-space normal and tangent).
GBufferOut MakeGBuffer(GpuGeometry g, MaterialConstants m, float3 normal, float4 tangent, float2 uv, bool frontFace, float3 motion)
{
    const bool hasTangents = (g.flags & GpuGeometry_HasTangents) != 0;
    // Without tangents the normal map is skipped (above) and the tangent is only a placeholder.
    MaterialTextureSample textures = SampleMaterial(m, uv, hasTangents);
    MaterialSample surface = EvaluateSceneMaterial(normal, hasTangents ? tangent : float4(0.0, 0.0, 0.0, 1.0), m, textures);
    if (!frontFace)
        surface.shadingNormal = -surface.shadingNormal;
    GBufferOut o;
    o.channel0 = float4(surface.diffuseAlbedo, surface.opacity);
    o.channel1 = float4(surface.specularF0, surface.occlusion);
    o.channel2 = float4(surface.shadingNormal, surface.roughness);
    // Emissive's spare channel: the material ID + 1, so passes after the G-buffer can
    // look up per-material data (0: no material; Donut's own fill pass writes 0).
    o.channel3 = float4(surface.emissiveColor, float(m.materialID + 1));
    o.motion = motion;
    return o;
}

GBufferOut ps(in VertexOut i, in bool isFrontFace : SV_IsFrontFace)
{
    const GpuGeometry g = t_GpuGeometries[t_GpuDraws[i.draw].geometry];
    const MaterialConstants m = LoadDrawMaterial(g);
    const GBufferOut o = MakeGBuffer(g, m, i.normal, i.tangent, i.texCoord, isFrontFace,
        GetMotionVector(i.position.xyz, i.prevPos, g_Draw.view, g_Draw.viewPrev));
#if ALPHA_TESTED
    if (m.domain != MaterialDomain_Opaque)
        clip(o.channel0.a - m.alphaCutoff);
#endif
    return o;
}

// ------------------------------------------------------------------ depth only

struct DepthOut
{
    float4 position : SV_Position;
    float2 texCoord : TEXCOORD;
    nointerpolation uint draw : DRAW;
};

void vs_depth(uint vertex : SV_VertexID, uint drawIndex : SV_InstanceID, out DepthOut o)
{
    const GpuDraw d = t_GpuDraws[drawIndex];
    const InstanceData inst = LoadGpuInstance(d.instance);
    const float3 world = mul(inst.transform, float4(LoadGpuPosition(vertex), 1.0));
    o.position = mul(mul(float4(world, 1.0), g_Draw.view.matWorldToView), g_Draw.view.matViewToClip);
    o.texCoord = asfloat(t_GpuTexcoords.Load2(vertex * 8u));
    o.draw = drawIndex;
}

// Alpha test for depth-only passes (alpha-tested geometry always has a material).
void AlphaTest(uint draw, float2 uv)
{
    const GpuGeometry g = t_GpuGeometries[t_GpuDraws[draw].geometry];
    const MaterialConstants m = LoadGpuMaterial(g.materialIndex);
    float opacity = m.opacity;
    if ((m.flags & MaterialFlags_UseBaseOrDiffuseTexture) != 0)
        opacity *= SampleTexture(m.baseOrDiffuseTextureIndex, uv).a;
    if ((m.flags & MaterialFlags_UseOpacityTexture) != 0)
        opacity *= SampleTexture(m.opacityTextureIndex, uv).r;
    clip(opacity - m.alphaCutoff);
}

void ps_depth(in DepthOut i)
{
#if ALPHA_TESTED
    AlphaTest(i.draw, i.texCoord);
#endif
}

// ------------------------------------------------------------------ meshlets

struct MeshCullConstants
{
    float4x4 viewProj;
    float4 planes[6];
    float3 cameraPos;
    uint phase;
    uint2 hizSize;
    uint hizMips;
    uint useOcclusion;
    float lodScale;          // camera projection[1][1] * viewport height / 2: object error -> pixels at distance 1
    float lodThreshold;      // pixels
    float lodNear;           // distances below this count as this
    uint useCone;            // 0 for shadow cascades (the cone is about the camera)
    float3 lodOrigin;        // the camera, also for shadow cascades
    uint countStats;         // the camera's passes only
};
ConstantBuffer<MeshCullConstants> g_MeshCull : register(b2);

struct TaskConstants
{
    uint taskBase;       // this bucket's first task entry
    uint3 pad;
};
DECLARE_PUSH_CONSTANTS(TaskConstants, g_Task, 1, 0);

ByteAddressBuffer t_Tasks : register(t72);
Texture2D<float2> t_HiZ : register(t73);
RWByteAddressBuffer u_Stats : register(u0);

static const uint c_StatTrianglesDrawn = 57;
static const uint c_StatMeshletsTested = 62;
static const uint c_StatMeshletsDrawn = 63;

struct Payload
{
    uint draw;
    uint meshlets[32];
};
groupshared Payload s_Payload;

bool SphereOccluded(float3 centre, float radius)
{
    float2 lo = 1e9, hi = -1e9;
    float nearest = 0.0;
    [unroll]
    for (uint k = 0; k < 8; ++k)
    {
        const float3 corner = centre + radius * float3((k & 1) ? 1.0 : -1.0, (k & 2) ? 1.0 : -1.0, (k & 4) ? 1.0 : -1.0);
        const float4 clip = mul(float4(corner, 1.0), g_MeshCull.viewProj);
        if (clip.w <= 1e-4)
            return false;
        const float3 ndc = clip.xyz / clip.w;
        const float2 uv = float2(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
        lo = min(lo, uv);
        hi = max(hi, uv);
        nearest = max(nearest, ndc.z);
    }
    lo = saturate(lo);
    hi = saturate(hi);
    const float2 sizePx = (hi - lo) * float2(g_MeshCull.hizSize);
    const uint mip = min(uint(ceil(log2(max(max(sizePx.x, sizePx.y), 1.0)))), g_MeshCull.hizMips - 1);
    const uint2 mipSize = max(g_MeshCull.hizSize >> mip, uint2(1, 1));
    const uint2 a = min(uint2(lo * float2(mipSize)), mipSize - 1);
    const uint2 b = min(uint2(hi * float2(mipSize)), mipSize - 1);
    const float farthest = min(min(t_HiZ.Load(int3(a, mip)).y, t_HiZ.Load(int3(b.x, a.y, mip)).y),
                               min(t_HiZ.Load(int3(a.x, b.y, mip)).y, t_HiZ.Load(int3(b, mip)).y));
    return nearest < farthest;
}

// Screen-space error in pixels of a LOD sphere (Karis 2021; meshoptimizer's clusterlod):
// error over distance, ignoring perspective distortion so that parents always project
// at least as large as their children and the cut stays consistent.
float ProjectedError(float3 centre, float radius, float error, float3x4 transform, float scale)
{
    const float3 world = mul(transform, float4(centre, 1.0));
    const float d = max(length(world - g_MeshCull.lodOrigin) - radius * scale, g_MeshCull.lodNear);
    return error * scale / d * g_MeshCull.lodScale;
}

// Whether this cluster is in the cut through the LOD DAG for this view. Flat meshlets
// (lodError 0, parentError FLT_MAX) always are.
bool ClusterInLodCut(GpuMeshlet m, float3x4 transform, float scale)
{
    return ProjectedError(m.parentCentre, m.parentRadius, m.parentError, transform, scale) > g_MeshCull.lodThreshold
        && ProjectedError(m.lodCentre, m.lodRadius, m.lodError, transform, scale) <= g_MeshCull.lodThreshold;
}

[numthreads(32, 1, 1)]
void task(uint gtid : SV_GroupThreadID, uint gid : SV_GroupID)
{
    const uint2 entry = t_Tasks.Load2((g_Task.taskBase + gid) * 8u);
    const uint drawIndex = entry.x;
    const GpuDraw d = t_GpuDraws[drawIndex];
    const GpuGeometry g = t_GpuGeometries[d.geometry];
    const InstanceData inst = LoadGpuInstance(d.instance);

    const uint meshlet = entry.y + gtid;
    bool visible = meshlet < g.meshletFirst + g.meshletCount;
    uint triangles = 0;
    if (visible)
    {
        const GpuMeshlet m = t_GpuMeshlets[meshlet];
        const float3 centre = mul(inst.transform, float4(m.centre, 1.0));
        const float3x3 r = (float3x3)inst.transform;
        const float scale = sqrt(max(max(dot(r[0], r[0]), dot(r[1], r[1])), dot(r[2], r[2])));
        const float radius = m.radius * scale;
        triangles = m.triangleCount;

        visible = ClusterInLodCut(m, inst.transform, scale);

        [unroll]
        for (uint i = 0; i < 6; ++i)
        {
            const float4 p = g_MeshCull.planes[i];
            if (dot(p.xyz, p.xyz) > 1e-12 && dot(p.xyz, centre) - radius > p.w)
                visible = false;
        }
        // Normal cone: every triangle faces away from the camera (meshoptimizer's test).
        if (visible && g_MeshCull.useCone != 0 && (g.flags & GpuGeometry_DoubleSided) == 0 && (g.flags & GpuGeometry_AlphaTested) == 0 && m.coneCutoff < 1.0)
        {
            const float3 apex = mul(inst.transform, float4(m.coneApex, 1.0));
            const float3 axis = normalize(mul(inst.transform, float4(m.coneAxis, 0.0)));
            if (dot(normalize(apex - g_MeshCull.cameraPos), axis) >= m.coneCutoff)
                visible = false;
        }
        if (visible && g_MeshCull.phase == 2 && g_MeshCull.useOcclusion != 0)
            visible = !SphereOccluded(centre, radius);
    }

    const uint index = WavePrefixCountBits(visible);
    if (visible)
        s_Payload.meshlets[index] = meshlet;
    const uint count = WaveActiveCountBits(visible);
    const uint drawnTriangles = WaveActiveSum(visible ? triangles : 0u);
    if (gtid == 0)
        s_Payload.draw = drawIndex;
    if (gtid == 0 && g_MeshCull.countStats != 0)
    {
        u_Stats.InterlockedAdd(c_StatMeshletsTested * 4u, min(g.meshletFirst + g.meshletCount - entry.y, 32u));
        u_Stats.InterlockedAdd(c_StatMeshletsDrawn * 4u, count);
        u_Stats.InterlockedAdd(c_StatTrianglesDrawn * 4u, drawnTriangles);
    }
    DispatchMesh(count, 1, 1, s_Payload);
}

[outputtopology("triangle")]
[numthreads(64, 1, 1)]
void mesh(uint gtid : SV_GroupThreadID, uint gid : SV_GroupID, in payload Payload p,
          out vertices VertexOut verts[64], out indices uint3 tris[124])
{
    const GpuMeshlet m = t_GpuMeshlets[p.meshlets[gid]];
    const GpuDraw d = t_GpuDraws[p.draw];
    const GpuGeometry g = t_GpuGeometries[d.geometry];
    const InstanceData inst = LoadGpuInstance(d.instance);
    SetMeshOutputCounts(m.vertexCount, m.triangleCount);
    if (gtid < m.vertexCount)
    {
        const uint vertex = g.baseVertex + t_GpuMeshletVertices.Load((m.vertexOffset + gtid) * 4u);
        verts[gtid] = ProcessVertex(vertex, p.draw, g, inst);
    }
    for (uint t = gtid; t < m.triangleCount; t += 64u)
    {
        const uint packed = t_GpuMeshletTriangles.Load((m.triangleOffset + t) * 4u);
        tris[t] = uint3(packed & 0xffu, (packed >> 8) & 0xffu, (packed >> 16) & 0xffu);
    }
}

[outputtopology("triangle")]
[numthreads(64, 1, 1)]
void mesh_depth(uint gtid : SV_GroupThreadID, uint gid : SV_GroupID, in payload Payload p,
                out vertices DepthOut verts[64], out indices uint3 tris[124])
{
    const GpuMeshlet m = t_GpuMeshlets[p.meshlets[gid]];
    const GpuDraw d = t_GpuDraws[p.draw];
    const GpuGeometry g = t_GpuGeometries[d.geometry];
    const InstanceData inst = LoadGpuInstance(d.instance);
    SetMeshOutputCounts(m.vertexCount, m.triangleCount);
    if (gtid < m.vertexCount)
    {
        const uint vertex = g.baseVertex + t_GpuMeshletVertices.Load((m.vertexOffset + gtid) * 4u);
        const float3 world = mul(inst.transform, float4(LoadGpuPosition(vertex), 1.0));
        DepthOut o;
        o.position = mul(mul(float4(world, 1.0), g_Draw.view.matWorldToView), g_Draw.view.matViewToClip);
        o.texCoord = asfloat(t_GpuTexcoords.Load2(vertex * 8u));
        o.draw = p.draw;
        verts[gtid] = o;
    }
    for (uint t = gtid; t < m.triangleCount; t += 64u)
    {
        const uint packed = t_GpuMeshletTriangles.Load((m.triangleOffset + t) * 4u);
        tris[t] = uint3(packed & 0xffu, (packed >> 8) & 0xffu, (packed >> 16) & 0xffu);
    }
}
