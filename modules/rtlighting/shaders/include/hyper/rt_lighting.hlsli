#ifndef HYPER_RT_LIGHTING_HLSLI
#define HYPER_RT_LIGHTING_HLSLI

// Lighting for ray hits and the DDGI volume (after Majercik et al. 2019,
// "Dynamic Diffuse Global Illumination with Ray-Traced Irradiance Fields").
// Bound by hyper::Ddgi::AddLayoutItems / AddSetItems (b50, t50..t54, s50),
// whether or not DDGI is enabled: without it, indirect light at hits comes
// from the IBL probe.
//
// Probe atlases: probe p = x + z * countX + y * countX * countZ sits in tile
// (p % tilesX, p / tilesX); a tile is the octahedral interior plus a 1-texel
// border. Irradiance holds cosine-weighted mean radiance (E / pi), gamma-encoded;
// diffuse light is albedo * DdgiIrradiance(...), as in Donut's probe convention.

#include <hyper/rt.hlsli>

struct GiConstants
{
    float3 volumeOrigin;    // world position of probe (0, 0, 0)
    uint enabled;
    float3 spacing;
    uint raysPerProbe;
    int3 counts;
    uint tilesX;
    float2 irradianceInvSize;
    float2 distanceInvSize;
    float normalBias;
    float viewBias;
    float intensity;
    float maxRayDistance;
    float3 sunDir;          // towards the sun
    uint hasProbe;          // IBL diffuse available (fallback without DDGI)
    float3 sunRadiance;     // colour * irradiance
    float envMip;           // environment mip for ray misses
    float4 rayRotation[3];  // rows, re-randomised every frame
    float hysteresis;
    float minFrontfaceDistance;
    uint frameIndex;
    uint reset;             // first update: no history
    float probeDiffuseScale;
    float irradianceGamma;
    float probeSpecularScale;
    float probeSpecularMips;
};

ConstantBuffer<GiConstants> g_Gi : register(b50);
Texture2D<float4> t_DdgiIrradiance : register(t50);
Texture2D<float4> t_DdgiDistance : register(t51);
Texture2D<float4> t_DdgiProbeData : register(t52);   // xyz relocation offset, w: 1 active / 0 inactive
TextureCube<float4> t_GiEnvironment : register(t53);  // radiance for rays that miss
TextureCubeArray<float4> t_GiProbeDiffuse : register(t54);
TextureCubeArray<float4> t_GiProbeSpecular : register(t55);
SamplerState s_Gi : register(s50);

static const uint c_IrradianceTexels = 6;
static const uint c_DistanceTexels = 14;

float2 OctEncode(float3 d)
{
    const float l1 = abs(d.x) + abs(d.y) + abs(d.z);
    float2 uv = d.xy / l1;
    if (d.z < 0.0)
        uv = (1.0 - abs(uv.yx)) * select(uv >= 0.0, 1.0, -1.0);
    return uv;
}

float3 OctDecode(float2 uv)
{
    float3 d = float3(uv, 1.0 - abs(uv.x) - abs(uv.y));
    if (d.z < 0.0)
        d.xy = (1.0 - abs(d.yx)) * select(d.xy >= 0.0, 1.0, -1.0);
    return normalize(d);
}

uint ProbeIndex(int3 c)
{
    return uint(c.x + c.z * g_Gi.counts.x + c.y * g_Gi.counts.x * g_Gi.counts.z);
}

int3 ProbeCoords(uint p)
{
    const uint slice = uint(g_Gi.counts.x * g_Gi.counts.z);
    const uint y = p / slice;
    const uint r = p - y * slice;
    return int3(r % uint(g_Gi.counts.x), y, r / uint(g_Gi.counts.x));
}

uint2 ProbeTile(uint p)
{
    return uint2(p % g_Gi.tilesX, p / g_Gi.tilesX);
}

float3 ProbeWorldPosition(uint p)
{
    return g_Gi.volumeOrigin + float3(ProbeCoords(p)) * g_Gi.spacing + t_DdgiProbeData[ProbeTile(p)].xyz;
}

float2 ProbeAtlasUv(uint p, float3 dir, uint texels, float2 invSize)
{
    const float2 tile = float2(ProbeTile(p)) * float(texels + 2);
    return (tile + 1.0 + (OctEncode(dir) * 0.5 + 0.5) * float(texels)) * invSize;
}

float3 SphericalFibonacci(float i, float n)
{
    const float phi = HYPER_TWO_PI * frac(i * 0.61803398875);
    const float cosTheta = 1.0 - (2.0 * i + 1.0) / n;
    const float sinTheta = sqrt(saturate(1.0 - cosTheta * cosTheta));
    return float3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);
}

float3 DdgiRayDirection(uint ray)
{
    const float3 d = SphericalFibonacci(float(ray), float(g_Gi.raysPerProbe));
    return normalize(float3(dot(g_Gi.rayRotation[0].xyz, d), dot(g_Gi.rayRotation[1].xyz, d), dot(g_Gi.rayRotation[2].xyz, d)));
}

// Irradiance / pi at a surface: the 8 surrounding probes, weighted by trilinear
// position, direction (wrap) and Chebyshev visibility.
float3 DdgiIrradiance(float3 position, float3 normal, float3 toViewer)
{
    const float minSpacing = min(g_Gi.spacing.x, min(g_Gi.spacing.y, g_Gi.spacing.z));
    const float3 biased = position + (normal * g_Gi.normalBias + toViewer * g_Gi.viewBias) * minSpacing;
    const int3 base = clamp(int3(floor((biased - g_Gi.volumeOrigin) / g_Gi.spacing)), 0, g_Gi.counts - 1);
    const float3 alpha = saturate((biased - (g_Gi.volumeOrigin + float3(base) * g_Gi.spacing)) / g_Gi.spacing);

    float3 sum = 0.0;
    float weightSum = 0.0;
    [unroll]
    for (uint i = 0; i < 8; ++i)
    {
        const int3 offset = int3(i, i >> 1, i >> 2) & 1;
        const int3 coords = clamp(base + offset, 0, g_Gi.counts - 1);
        const uint p = ProbeIndex(coords);
        const float4 data = t_DdgiProbeData[ProbeTile(p)];
        if (data.w <= 0.0)
            continue;
        const float3 probePos = g_Gi.volumeOrigin + float3(coords) * g_Gi.spacing + data.xyz;

        const float3 trilinear = lerp(1.0 - alpha, alpha, float3(offset));
        float weight = 1.0;

        // Wrap shading: probes behind the surface count less.
        const float3 toProbe = normalize(probePos - position);
        const float wrap = (dot(toProbe, normal) + 1.0) * 0.5;
        weight *= wrap * wrap + 0.2;

        // Chebyshev visibility from the probe's mean / mean-squared distance.
        const float3 probeToPoint = biased - probePos;
        const float distance = length(probeToPoint);
        const float2 moments = t_DdgiDistance.SampleLevel(s_Gi,
            ProbeAtlasUv(p, probeToPoint / max(distance, 1e-4), c_DistanceTexels, g_Gi.distanceInvSize), 0).xy;
        if (distance > moments.x)
        {
            const float variance = abs(moments.x * moments.x - moments.y);
            const float d = distance - moments.x;
            const float chebyshev = variance / (variance + d * d);
            weight *= max(chebyshev * chebyshev * chebyshev, 0.0);
        }
        weight = max(weight, 1e-6);
        // Crush tiny weights, which otherwise leak light through thin walls.
        if (weight < 0.2)
            weight *= weight * weight / (0.2 * 0.2);
        weight *= trilinear.x * trilinear.y * trilinear.z;

        const float3 texel = t_DdgiIrradiance.SampleLevel(s_Gi, ProbeAtlasUv(p, normal, c_IrradianceTexels, g_Gi.irradianceInvSize), 0).rgb;
        // Stored with gamma g; blend with gamma 2 (perceptual), square at the end.
        sum += weight * pow(max(texel, 0.0), 0.5 * g_Gi.irradianceGamma);
        weightSum += weight;
    }
    if (weightSum <= 0.0)
        return 0.0;
    const float3 e = sum / weightSum;
    return e * e * g_Gi.intensity;
}

// Irradiance / pi from whatever indirect source is bound.
float3 IndirectIrradiance(float3 position, float3 normal, float3 toViewer)
{
    if (g_Gi.enabled != 0)
        return DdgiIrradiance(position, normal, toViewer);
    if (g_Gi.hasProbe != 0)
        return t_GiProbeDiffuse.SampleLevel(s_Gi, float4(normal, 0.0), 0).rgb * g_Gi.probeDiffuseScale;
    return 0.0;
}

float3 MissRadiance(float3 dir, float lod)
{
    return t_GiEnvironment.SampleLevel(s_Gi, dir, lod).rgb;
}

// At the blur that suits diffuse rays (sharp environments would add noise).
float3 MissRadiance(float3 dir)
{
    return MissRadiance(dir, g_Gi.envMip);
}

// Diffuse radiance leaving a hit from emission and the sun (with a shadow ray).
// Back faces of single-sided surfaces are black.
float3 ShadeHitDirect(HitSurface s)
{
    if (!s.frontFace)
        return 0.0;
    float3 radiance = s.emissive;
    const float nl = dot(s.normal, g_Gi.sunDir);
    if (nl > 0.0 && dot(s.geometricNormal, g_Gi.sunDir) > 0.0 && any(g_Gi.sunRadiance > 0.0))
    {
        if (TraceVisible(OffsetRayOrigin(s.position, s.geometricNormal), g_Gi.sunDir, 1e6))
            radiance += s.diffuseAlbedo * (nl / HYPER_PI) * g_Gi.sunRadiance;
    }
    return radiance;
}

// ... plus indirect light towards toViewer.
float3 ShadeHitDiffuse(HitSurface s, float3 toViewer)
{
    if (!s.frontFace)
        return 0.0;
    return ShadeHitDirect(s) + s.diffuseAlbedo * IndirectIrradiance(s.position, s.normal, toViewer);
}

#endif
