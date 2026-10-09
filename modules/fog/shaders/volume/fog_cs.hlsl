// Froxel volumetric fog (after Wronski 2014 and Hillaire 2015): a view-aligned
// grid with exponential depth slices.
//
//   inject      per froxel: media (height fog, noise, local boxes/spheres) and light
//               (sun with shadows, point/spot lights, ambient), blended with the
//               reprojected previous frame
//   integrate   per column, front to back: in-scattering and transmittance
//   apply       per pixel: colour * transmittance + in-scattering at the pixel's depth
//
// Variants: FOG_RT (sun shadows by ray query, ambient from DDGI / IBL through
// hyper/rt_lighting.hlsli) or not (Donut's cascaded shadow map, IBL ambient);

#include <hyper/common.hlsli>
#include <hyper/noise.hlsli>
#include <hyper/volume.hlsli>
#include <donut/shaders/view_cb.h>
#include <donut/shaders/light_cb.h>
#include <donut/shaders/light_types.h>

#if FOG_RT
#include <hyper/rt_lighting.hlsli>
#else
#include <donut/shaders/utils.hlsli>
#include <donut/shaders/shadows.hlsli>
#endif


static const uint c_MaxVolumes = 8;
static const uint c_MaxLights = 16;

struct FogConstants
{
    PlanarViewConstants view;
    PlanarViewConstants viewPrev;
    uint3 grid;
    uint frameIndex;
    float nearZ;
    float farZ;
    float temporalWeight;
    uint historyValid;
    float density;
    float heightFalloff;
    float baseHeight;
    float anisotropy;
    float3 albedo;
    float noiseStrength;
    float3 windOffset;
    float noiseScale;
    float3 sunDir;              // towards the sun
    uint shadowMode;            // 0 none, 1 cascades, 2 ray query
    float3 sunRadiance;
    uint numLights;
    float ambientIntensity;
    float localIntensity;
    uint numVolumes;
    uint numCascades;
    float3 cameraPos;
    uint hasProbe;
    uint2 size;                 // apply: target size
    float2 pad;
    float4 volumeCentre[c_MaxVolumes];   // xyz, w density
    float4 volumeExtent[c_MaxVolumes];   // xyz half extent, w: 0 box, 1 sphere
    ShadowConstants cascades[4];
    LightConstants lights[c_MaxLights];
};

ConstantBuffer<FogConstants> g_Fog : register(b0);

Texture3D<float> t_Noise : register(t0);
Texture3D<float4> t_History : register(t1);
Texture3D<float4> t_Froxels : register(t2);
Texture3D<float4> t_Integrated : register(t3);
Texture2D<float> t_Depth : register(t4);
TextureCubeArray<float4> t_ProbeDiffuse : register(t5);
#if !FOG_RT
Texture2DArray t_ShadowMap : register(t6);
SamplerComparisonState s_ShadowCompare : register(s2);
#endif
SamplerState s_LinearWrap : register(s0);
SamplerState s_LinearClamp : register(s1);

VK_IMAGE_FORMAT("rgba16f") RWTexture3D<float4> u_Froxels : register(u0);
VK_IMAGE_FORMAT("rgba16f") RWTexture3D<float4> u_Integrated : register(u1);
VK_IMAGE_FORMAT("rgba16f") RWTexture2D<float4> u_Color : register(u2);

// ------------------------------------------------------------------ slices

// View depth of slice coordinate w in [0, 1]: exponential from nearZ to farZ.
float SliceDepth(float w)
{
    return g_Fog.nearZ * pow(g_Fog.farZ / g_Fog.nearZ, w);
}

float SliceCoord(float viewZ)
{
    return log(max(viewZ, 1e-4) / g_Fog.nearZ) / log(g_Fog.farZ / g_Fog.nearZ);
}

float3 CameraForward(PlanarViewConstants v)
{
    // Row 2 of view-to-world: +Z of Donut's (left-handed) view space.
    return normalize(v.matViewToWorld[2].xyz);
}

float3 RayDirection(PlanarViewConstants v, float2 uv)
{
    const float2 clip = float2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    const float4 p = mul(float4(clip, 0.5, 1.0), v.matClipToWorld);
    return normalize(p.xyz / p.w - v.cameraDirectionOrPosition.xyz);
}

// ------------------------------------------------------------------ media

float Density(float3 p)
{
    const float height = max(p.y - g_Fog.baseHeight, 0.0);
    float d = g_Fog.density * exp(-height * g_Fog.heightFalloff);
    if (g_Fog.noiseStrength > 0.0)
    {
        const float3 q = (p + g_Fog.windOffset) * g_Fog.noiseScale;
        const float n = t_Noise.SampleLevel(s_LinearWrap, q * 0.125, 0) * 0.6
                      + t_Noise.SampleLevel(s_LinearWrap, q * 0.37, 0) * 0.4;
        d *= saturate(1.0 + g_Fog.noiseStrength * 2.0 * n);
    }
    for (uint i = 0; i < g_Fog.numVolumes; ++i)
    {
        const float3 local = (p - g_Fog.volumeCentre[i].xyz) / max(g_Fog.volumeExtent[i].xyz, 1e-3);
        // Soft edges over the outer 20%.
        const float r = g_Fog.volumeExtent[i].w > 0.5 ? length(local) : max(abs(local.x), max(abs(local.y), abs(local.z)));
        d += g_Fog.volumeCentre[i].w * saturate((1.0 - r) * 5.0);
    }
    return d;
}

// ------------------------------------------------------------------ light

float SunVisibility(float3 p)
{
#if FOG_RT
    if (g_Fog.shadowMode == 2)
        return TraceVisible(p, g_Fog.sunDir, 1e6) ? 1.0 : 0.0;
#else
    if (g_Fog.shadowMode == 1)
    {
        for (uint c = 0; c < g_Fog.numCascades; ++c)
        {
            const float2 s = EvaluateShadowPCF(t_ShadowMap, s_ShadowCompare, g_Fog.cascades[c], p);
            if (s.y > 0.5)
                return s.x;
        }
    }
#endif
    return 1.0;
}

// Radiance / pi of the surroundings, averaged over up and down (isotropic media).
float3 Ambient(float3 p)
{
#if FOG_RT
    return 0.5 * (IndirectIrradiance(p, float3(0, 1, 0), 0.0) + IndirectIrradiance(p, float3(0, -1, 0), 0.0));
#else
    if (g_Fog.hasProbe == 0)
        return 0.0;
    return 0.5 * (t_ProbeDiffuse.SampleLevel(s_LinearClamp, float4(0, 1, 0, 0), 0).rgb
                + t_ProbeDiffuse.SampleLevel(s_LinearClamp, float4(0, -1, 0, 0), 0).rgb);
#endif
}

float3 LocalLights(float3 p, float3 rayDir)
{
    float3 sum = 0.0;
    for (uint i = 0; i < g_Fog.numLights; ++i)
    {
        const LightConstants light = g_Fog.lights[i];
        const float3 toPoint = p - light.position;
        const float d = length(toPoint);
        const float3 l = toPoint / max(d, 1e-4);
        float attenuation = 1.0;
        if (light.angularSizeOrInvRange > 0.0)
        {
            const float x = d * light.angularSizeOrInvRange;
            attenuation = saturate(1.0 - x * x * x * x);
            attenuation *= attenuation;
        }
        if (light.lightType == LightType_Spot)
            attenuation *= 1.0 - smoothstep(light.innerAngle, light.outerAngle, acos(clamp(dot(l, light.direction), -1.0, 1.0)));
        const float irradiance = light.intensity / max(d * d, 1e-2) * attenuation;
        // Forward scattering peaks where the light travels towards the camera (l = -rayDir).
        sum += light.color * irradiance * HenyeyGreenstein(dot(-l, rayDir), g_Fog.anisotropy);
    }
    return sum;
}

// ------------------------------------------------------------------ inject

[numthreads(4, 4, 4)]
void inject(uint3 id : SV_DispatchThreadID)
{
    if (any(id >= g_Fog.grid))
        return;

    const float2 uv = (float2(id.xy) + 0.5) / float2(g_Fog.grid.xy);
    // Jitter within the slice; the temporal blend integrates it.
    const float jitter = frac(InterleavedGradientNoise(id.xy, g_Fog.frameIndex) + float(id.z) * 0.618034);
    const float w = (float(id.z) + jitter) / float(g_Fog.grid.z);
    const float3 rayDir = RayDirection(g_Fog.view, uv);
    const float z = SliceDepth(w);
    const float3 p = g_Fog.cameraPos + rayDir * (z / max(dot(rayDir, CameraForward(g_Fog.view)), 1e-3));

    const float sigmaT = Density(p);
    float3 inScatter = 0.0;
    if (sigmaT > 1e-6)
    {
        const float3 sigmaS = sigmaT * g_Fog.albedo;
        float3 light = 0.0;
        if (any(g_Fog.sunRadiance > 0.0))
            light += g_Fog.sunRadiance * HenyeyGreenstein(dot(g_Fog.sunDir, rayDir), g_Fog.anisotropy) * SunVisibility(p);
        light += Ambient(p) * g_Fog.ambientIntensity;
        light += LocalLights(p, rayDir) * g_Fog.localIntensity;
        inScatter = sigmaS * light;
    }
    float4 current = float4(inScatter, sigmaT);

    // Reproject the froxel centre (unjittered) into last frame's grid.
    if (g_Fog.historyValid != 0)
    {
        const float zc = SliceDepth((float(id.z) + 0.5) / float(g_Fog.grid.z));
        const float3 pc = g_Fog.cameraPos + rayDir * (zc / max(dot(rayDir, CameraForward(g_Fog.view)), 1e-3));
        const float4 clip = mul(float4(pc, 1.0), g_Fog.viewPrev.matWorldToClip);
        if (clip.w > 0.0)
        {
            const float2 prevUv = float2(clip.x / clip.w * 0.5 + 0.5, 0.5 - clip.y / clip.w * 0.5);
            const float prevZ = dot(pc - g_Fog.viewPrev.cameraDirectionOrPosition.xyz, CameraForward(g_Fog.viewPrev));
            const float prevW = SliceCoord(prevZ);
            if (all(prevUv >= 0.0) && all(prevUv <= 1.0) && prevW >= 0.0 && prevW <= 1.0)
            {
                const float4 history = t_History.SampleLevel(s_LinearClamp, float3(prevUv, prevW), 0);
                current = lerp(current, history, g_Fog.temporalWeight);
            }
        }
    }
    u_Froxels[id] = current;
}

// ------------------------------------------------------------------ integrate

[numthreads(8, 8, 1)]
void integrate(uint2 id : SV_DispatchThreadID)
{
    if (any(id >= g_Fog.grid.xy))
        return;
    const float2 uv = (float2(id) + 0.5) / float2(g_Fog.grid.xy);
    const float3 rayDir = RayDirection(g_Fog.view, uv);
    const float stretch = 1.0 / max(dot(rayDir, CameraForward(g_Fog.view)), 1e-3);   // view z -> ray length

    float3 scattering = 0.0;
    float transmittance = 1.0;
    float zPrev = 0.0;
    for (uint k = 0; k < g_Fog.grid.z; ++k)
    {
        const float zEnd = SliceDepth(float(k + 1) / float(g_Fog.grid.z));
        const float dt = (zEnd - zPrev) * stretch;
        zPrev = zEnd;
        const float4 f = t_Froxels[uint3(id, k)];
        scattering += transmittance * IntegrateScattering(f.rgb, f.a, dt);
        transmittance *= exp(-f.a * dt);
        u_Integrated[uint3(id, k)] = float4(scattering, transmittance);
    }
}

// ------------------------------------------------------------------ apply

[numthreads(8, 8, 1)]
void apply(uint2 id : SV_DispatchThreadID)
{
    if (any(id >= g_Fog.size))
        return;
    const float depth = t_Depth[id];
    const float2 uv = (float2(id) + 0.5) / float2(g_Fog.size);
    // Reverse-Z, infinite far plane: view z = zNear / depth; the sky gets the whole range.
    const float viewZ = depth > 0.0 ? g_Fog.view.matViewToClip[3][2] / depth : g_Fog.farZ;
    const float w = SliceCoord(min(viewZ, g_Fog.farZ));
    // Slice k holds the integral up to its far end.
    const float coord = w - 0.5 / float(g_Fog.grid.z);
    float4 fog = t_Integrated.SampleLevel(s_LinearClamp, float3(uv, max(coord, 0.0)), 0);
    if (coord < 0.0)
    {
        // Closer than the first slice's end: fade in from nothing.
        const float t = saturate(w * float(g_Fog.grid.z));
        fog = lerp(float4(0.0, 0.0, 0.0, 1.0), fog, t);
    }
    const float4 colour = u_Color[id];
    u_Color[id] = float4(colour.rgb * fog.a + fog.rgb, colour.a);
}
