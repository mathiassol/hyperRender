// DDGI probe volume update and gather (Majercik et al. 2019; RTXGI-DDGI as reference).
//
//   trace               rays x probes: radiance + hit distance (negative: back face)
//   blend_irradiance    one group per probe: cosine-weighted radiance, hysteresis blend
//   blend_distance      one group per probe: mean / mean-squared distance
//   border_irradiance   octahedral border texels, so bilinear filtering wraps correctly
//   border_distance
//   relocate            moves probes out of geometry, deactivates ones stuck inside
//   gather              per pixel: all indirect light -> Donut's indirect-diffuse input
//                       (DDGI diffuse + the IBL probe's specular, occluded by the GI)

#include <hyper/rt_lighting.hlsli>
#include <donut/shaders/view_cb.h>

// trace
VK_IMAGE_FORMAT("rgba16f") RWTexture2D<float4> u_Rays : register(u0);

// blend / border / relocate
Texture2D<float4> t_Rays : register(t0);
Texture2D<float4> t_ProbeState : register(t1);
VK_IMAGE_FORMAT("rgba16f") RWTexture2D<float4> u_Atlas : register(u1);
VK_IMAGE_FORMAT("rgba16f") RWTexture2D<float4> u_ProbeData : register(u2);

// ------------------------------------------------------------------ trace

[numthreads(32, 1, 1)]
void trace(uint2 id : SV_DispatchThreadID)
{
    const uint ray = id.x;
    const uint probe = id.y;
    const uint probeCount = uint(g_Gi.counts.x * g_Gi.counts.y * g_Gi.counts.z);
    if (ray >= g_Gi.raysPerProbe || probe >= probeCount)
        return;

    const float3 origin = ProbeWorldPosition(probe);
    const float3 dir = DdgiRayDirection(ray);
    const RtHit hit = TraceClosest(origin, dir, g_Gi.maxRayDistance);
    if (!hit.hit)
    {
        u_Rays[id] = float4(MissRadiance(dir), 1e27);
        return;
    }
    const HitSurface s = GetHitSurface(hit, dir, 3.0);
    if (!s.frontFace)
    {
        // Shortened, so relocation prefers moving past the nearest back face.
        u_Rays[id] = float4(0.0, 0.0, 0.0, -hit.t * 0.2);
        return;
    }
    u_Rays[id] = float4(min(ShadeHitDiffuse(s, -dir), 65000.0), hit.t);
}

// ------------------------------------------------------------------ blend

static const uint c_MaxRays = 512;
groupshared float4 gs_Radiance[c_MaxRays];   // rgb radiance, a distance
groupshared float3 gs_Direction[c_MaxRays];

void LoadRays(uint probe, uint thread, uint threads)
{
    for (uint r = thread; r < g_Gi.raysPerProbe; r += threads)
    {
        gs_Radiance[r] = t_Rays[uint2(r, probe)];
        gs_Direction[r] = DdgiRayDirection(r);
    }
    GroupMemoryBarrierWithGroupSync();
}

float3 TexelDirection(uint2 texel, uint texels)
{
    return OctDecode(((float2(texel) + 0.5) / float(texels)) * 2.0 - 1.0);
}

[numthreads(6, 6, 1)]
void blend_irradiance(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID, uint index : SV_GroupIndex)
{
    const uint probe = group.x + group.y * g_Gi.tilesX;
    LoadRays(probe, index, c_IrradianceTexels * c_IrradianceTexels);
    const uint2 atlasTexel = group.xy * (c_IrradianceTexels + 2) + 1 + local.xy;
    if (t_ProbeState[group.xy].w <= 0.0 && g_Gi.reset == 0)
        return;

    const float3 dir = TexelDirection(local.xy, c_IrradianceTexels);
    float3 sum = 0.0;
    float weightSum = 0.0;
    for (uint r = 0; r < g_Gi.raysPerProbe; ++r)
    {
        if (gs_Radiance[r].a < 0.0)
            continue;   // back faces: the probe is inside geometry, relocation deals with it
        const float w = max(0.0, dot(dir, gs_Direction[r]));
        sum += w * gs_Radiance[r].rgb;
        weightSum += w;
    }
    float3 result = weightSum > 1e-4 ? sum / weightSum : 0.0;
    result = pow(max(result, 0.0), 1.0 / g_Gi.irradianceGamma);

    const float3 previous = u_Atlas[atlasTexel].rgb;
    float hysteresis = g_Gi.reset != 0 ? 0.0 : g_Gi.hysteresis;
    // Adapt faster to big changes (lights switched, doors opened).
    const float3 delta = abs(result - previous);
    if (max(delta.x, max(delta.y, delta.z)) > 0.25)
        hysteresis = max(0.0, hysteresis - 0.15);
    u_Atlas[atlasTexel] = float4(lerp(result, previous, hysteresis), 1.0);
}

[numthreads(14, 14, 1)]
void blend_distance(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID, uint index : SV_GroupIndex)
{
    const uint probe = group.x + group.y * g_Gi.tilesX;
    LoadRays(probe, index, c_DistanceTexels * c_DistanceTexels);
    const uint2 atlasTexel = group.xy * (c_DistanceTexels + 2) + 1 + local.xy;

    const float3 dir = TexelDirection(local.xy, c_DistanceTexels);
    const float maxDistance = length(g_Gi.spacing) * 1.5;
    float2 sum = 0.0;
    float weightSum = 0.0;
    for (uint r = 0; r < g_Gi.raysPerProbe; ++r)
    {
        const float w = pow(max(0.0, dot(dir, gs_Direction[r])), 50.0);
        const float d = min(abs(gs_Radiance[r].a), maxDistance);
        sum += w * float2(d, d * d);
        weightSum += w;
    }
    const float2 result = weightSum > 1e-6 ? sum / weightSum : float2(maxDistance, maxDistance * maxDistance);
    const float2 previous = u_Atlas[atlasTexel].xy;
    const float hysteresis = g_Gi.reset != 0 ? 0.0 : g_Gi.hysteresis;
    u_Atlas[atlasTexel] = float4(lerp(result, previous, hysteresis), 0.0, 1.0);
}

// Border texels copy the interior texel they wrap to (octahedral mirroring).
void CopyBorder(uint2 group, uint2 local, uint texels)
{
    const uint size = texels + 2;
    const uint last = size - 1;
    const bool edgeX = local.x == 0 || local.x == last;
    const bool edgeY = local.y == 0 || local.y == last;
    if (!edgeX && !edgeY)
        return;
    uint2 src;
    if (edgeX && edgeY)
        src = uint2(local.x == 0 ? texels : 1, local.y == 0 ? texels : 1);
    else if (edgeY)
        src = uint2(last - local.x, local.y == 0 ? 1 : texels);
    else
        src = uint2(local.x == 0 ? 1 : texels, last - local.y);
    const uint2 origin = group * size;
    u_Atlas[origin + local] = u_Atlas[origin + src];
}

[numthreads(8, 8, 1)]
void border_irradiance(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID)
{
    CopyBorder(group.xy, local.xy, c_IrradianceTexels);
}

[numthreads(16, 16, 1)]
void border_distance(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID)
{
    CopyBorder(group.xy, local.xy, c_DistanceTexels);
}

// ------------------------------------------------------------------ relocate

[numthreads(32, 1, 1)]
void relocate(uint id : SV_DispatchThreadID)
{
    const uint probeCount = uint(g_Gi.counts.x * g_Gi.counts.y * g_Gi.counts.z);
    if (id >= probeCount)
        return;
    const uint2 tile = ProbeTile(id);
    float4 data = u_ProbeData[tile];
    if (g_Gi.reset != 0)
        data = float4(0.0, 0.0, 0.0, 1.0);

    uint backfaces = 0;
    float closestBack = 1e27, closestFront = 1e27, farthestFront = 0.0;
    uint closestBackRay = 0, closestFrontRay = 0, farthestFrontRay = 0;
    for (uint r = 0; r < g_Gi.raysPerProbe; ++r)
    {
        float d = t_Rays[uint2(r, id)].a;
        if (d < 0.0)
        {
            ++backfaces;
            d *= -5.0;
            if (d < closestBack) { closestBack = d; closestBackRay = r; }
        }
        else
        {
            if (d < closestFront) { closestFront = d; closestFrontRay = r; }
            if (d > farthestFront) { farthestFront = d; farthestFrontRay = r; }
        }
    }

    const float backfaceRatio = float(backfaces) / float(g_Gi.raysPerProbe);
    float3 offset = data.xyz;
    if (backfaces > 0 && backfaceRatio > 0.25)
    {
        // Inside geometry: step through the nearest back face.
        offset += DdgiRayDirection(closestBackRay) * (closestBack + g_Gi.minFrontfaceDistance * 0.5);
    }
    else if (closestFront < g_Gi.minFrontfaceDistance)
    {
        // Too close to a surface: move towards open space, not through the far side.
        const float3 away = DdgiRayDirection(farthestFrontRay);
        if (dot(away, DdgiRayDirection(closestFrontRay)) <= 0.0)
            offset += away * min(farthestFront, g_Gi.minFrontfaceDistance);
    }
    if (all(abs(offset) <= g_Gi.spacing * 0.45))
        data.xyz = offset;

    // Still mostly seeing back faces: inside a wall, so it would only leak.
    data.w = backfaceRatio > 0.5 ? 0.0 : 1.0;
    u_ProbeData[tile] = data;
}

// ------------------------------------------------------------------ gather

struct GatherConstants
{
    PlanarViewConstants view;
    uint2 size;
    uint2 pad;
};
ConstantBuffer<GatherConstants> g_Gather : register(b0);
Texture2D<float> t_Depth : register(t2);
Texture2D<float4> t_Normals : register(t3);
Texture2D<float4> t_Diffuse : register(t4);
Texture2D<float4> t_Specular : register(t5);
VK_IMAGE_FORMAT("rgba16f") RWTexture2D<float4> u_Indirect : register(u3);

[numthreads(8, 8, 1)]
void gather(uint2 id : SV_DispatchThreadID)
{
    if (any(id >= g_Gather.size))
        return;
    const float depth = t_Depth[id];
    if (depth <= 0.0)
    {
        u_Indirect[id] = 0.0;
        return;
    }
    const float2 clip = (float2(id) + 0.5) * g_Gather.view.windowToClipScale + g_Gather.view.windowToClipBias;
    const float4 wp = mul(float4(clip, depth, 1.0), g_Gather.view.matClipToWorld);
    const float3 position = wp.xyz / wp.w;
    const float3 n = normalize(t_Normals[id].xyz);
    const float3 toViewer = normalize(g_Gather.view.cameraDirectionOrPosition.xyz - position);
    const float3 irradiance = DdgiIrradiance(position, n, toViewer);
    const float4 specular = t_Specular[id];   // F0, material occlusion

    // The probe sees open sky everywhere; the volume knows what is blocked. Their
    // ratio occludes the probe's specular (sky reflections indoors, mostly).
    float3 probeSpecular = 0.0;
    float specularOcclusion = 1.0;
    if (g_Gi.hasProbe != 0)
    {
        const float3 sky = t_GiProbeDiffuse.SampleLevel(s_Gi, float4(n, 0.0), 0).rgb * g_Gi.probeDiffuseScale;
        const float3 luma = float3(0.2126, 0.7152, 0.0722);
        specularOcclusion = saturate(dot(irradiance, luma) / max(dot(sky, luma), 1e-4));

        const float roughness = t_Normals[id].w;
        const float3 r = reflect(-toViewer, n);
        const float3 env = t_GiProbeSpecular.SampleLevel(s_Gi, float4(r, 0.0),
            sqrt(saturate(roughness)) * (g_Gi.probeSpecularMips - 1.0)).rgb * g_Gi.probeSpecularScale;
        // Karis' fit of the split-sum environment BRDF, as the reflection composite uses.
        const float nDotV = saturate(dot(n, toViewer));
        const float4 c0 = float4(-1.0, -0.0275, -0.572, 0.022);
        const float4 c1 = float4(1.0, 0.0425, 1.04, -0.04);
        const float4 k = roughness * c0 + c1;
        const float a004 = min(k.x * k.x, exp2(-9.28 * nDotV)) * k.x + k.y;
        const float2 ab = float2(-1.04, 1.04) * a004 + k.zw;
        probeSpecular = env * (specular.rgb * ab.x + ab.y) * specularOcclusion;
    }
    // Donut adds this as-is (times AO): albedo and the material's occlusion go in here.
    // .a: the specular occlusion, for the reflection composite.
    u_Indirect[id] = float4((irradiance * t_Diffuse[id].rgb + probeSpecular) * specular.a, specularOcclusion);
}
