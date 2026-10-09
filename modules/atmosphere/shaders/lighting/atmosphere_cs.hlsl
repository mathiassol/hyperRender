// Physically based sky and aerial perspective after Hillaire 2020, "A Scalable
// and Production Ready Sky and Atmosphere Rendering Technique" (reference
// implementation: sebh/UnrealEngineSkyAtmosphere, MIT).
//
//   transmittance   256x64 LUT: transmittance to the top of the atmosphere by (height, cos zenith)
//   multiscatter    32x32 LUT: isotropic multiple-scattering transfer by (height, cos sun zenith)
//   skyview         192x108 LUT: sky luminance around the camera, latitude warped towards the horizon
//   aerial          32^3 froxels: in-scattering + transmittance along view rays, up to aerialMaxKm
//   sky (ps)        sky + sun disk, a fullscreen quad depth-tested against empty depth
//   apply           aerial perspective over geometry
//   capture         the sky into a cubemap (no sun disk) for image-based lighting
//
// Luminance is for a sun of unit irradiance; apply/capture scale by the sun's.
// Distances are km, in a planet frame whose up is world +Y.

#include <hyper/common.hlsli>
#include <hyper/cubemap.hlsli>
#include <donut/shaders/view_cb.h>


struct AtmosphereConstants
{
    PlanarViewConstants view;

    float3 sunDir;          // towards the sun
    float sunIrradiance;
    float3 rayleighScattering;
    float mieScattering;
    float3 ozoneAbsorption;
    float mieExtinction;
    float3 groundAlbedo;
    float mieG;
    float3 cameraKm;        // relative to the planet centre
    float groundRadius;
    float3 cameraWorld;
    float topRadius;
    float unitKm;           // km per world unit, times the aerial-perspective scale
    float aerialMaxKm;
    float sunCosRadius;
    float sunDiskRadiance;
    uint2 size;             // apply: target size; capture: cube size in x
    uint2 pad;
};

ConstantBuffer<AtmosphereConstants> g_Atmo : register(b0);

Texture2D<float4> t_Transmittance : register(t0);
Texture2D<float4> t_MultiScatter : register(t1);
Texture2D<float4> t_SkyView : register(t2);
Texture3D<float4> t_Aerial : register(t3);
Texture2D<float> t_Depth : register(t4);
SamplerState s_Linear : register(s0);

VK_IMAGE_FORMAT("rgba16f") RWTexture2D<float4> u_Out : register(u0);
VK_IMAGE_FORMAT("rgba16f") RWTexture3D<float4> u_Volume : register(u1);
VK_IMAGE_FORMAT("rgba16f") RWTexture2DArray<float4> u_Cube : register(u2);

static const uint2 c_TransmittanceSize = uint2(256, 64);
static const uint c_MultiScatterSize = 32;
static const uint2 c_SkyViewSize = uint2(192, 108);
static const uint c_AerialSize = 32;

// ------------------------------------------------------------------ medium

struct Medium
{
    float3 rayleigh;     // scattering
    float3 mie;          // scattering
    float3 scattering;
    float3 extinction;
};

Medium SampleMedium(float3 posKm)
{
    const float h = max(length(posKm) - g_Atmo.groundRadius, 0.0);
    const float rayleighDensity = exp(-h / 8.0);
    const float mieDensity = exp(-h / 1.2);
    const float ozoneDensity = max(0.0, 1.0 - abs(h - 25.0) / 15.0);
    Medium m;
    m.rayleigh = g_Atmo.rayleighScattering * rayleighDensity;
    m.mie = g_Atmo.mieScattering * mieDensity;
    m.scattering = m.rayleigh + m.mie;
    m.extinction = m.rayleigh + g_Atmo.mieExtinction * mieDensity + g_Atmo.ozoneAbsorption * ozoneDensity;
    return m;
}

float RayleighPhase(float c)
{
    return 3.0 / (16.0 * HYPER_PI) * (1.0 + c * c);
}

// Cornette-Shanks.
float MiePhase(float c)
{
    const float g = g_Atmo.mieG;
    const float g2 = g * g;
    return 3.0 / (8.0 * HYPER_PI) * ((1.0 - g2) * (1.0 + c * c)) / ((2.0 + g2) * pow(max(1.0 + g2 - 2.0 * g * c, 1e-4), 1.5));
}

// Distance along a unit ray to a sphere at the origin: the entry point from
// outside, the exit from inside; -1 if it misses or lies behind.
float RaySphere(float3 ro, float3 rd, float radius)
{
    const float b = dot(ro, rd);
    const float c = dot(ro, ro) - radius * radius;
    if (c > 0.0 && b > 0.0)
        return -1.0;
    const float disc = b * b - c;
    if (disc < 0.0)
        return -1.0;
    const float s = sqrt(disc);
    return -b - s > 0.0 ? -b - s : -b + s;
}

float2 HeightCosUv(float3 posKm, float3 dir)
{
    const float r = length(posKm);
    const float c = dot(posKm / r, dir);
    return float2(c * 0.5 + 0.5, saturate((r - g_Atmo.groundRadius) / (g_Atmo.topRadius - g_Atmo.groundRadius)));
}

// To the top of the atmosphere; 0 where the ray hits the ground (the planet's shadow).
float3 Transmittance(float3 posKm, float3 dir)
{
    return t_Transmittance.SampleLevel(s_Linear, HeightCosUv(posKm, dir), 0).rgb;
}

float3 MultiScatter(float3 posKm, float3 sun)
{
    return t_MultiScatter.SampleLevel(s_Linear, HeightCosUv(posKm, sun), 0).rgb;
}

// Single scattering of the sun plus the multiple-scattering LUT, marched with
// samples bunched towards the camera. Stops at tLimit (or the ground / top).
float3 IntegrateScattering(float3 p, float3 dir, float3 sun, float tLimit, uint steps, out float3 transmittance)
{
    transmittance = 1.0;
    const float tGround = RaySphere(p, dir, g_Atmo.groundRadius);
    const float tTop = RaySphere(p, dir, g_Atmo.topRadius);
    float tMax = tGround > 0.0 ? tGround : tTop;
    if (tMax <= 0.0)
        return 0.0;
    tMax = min(tMax, tLimit);

    const float cosTheta = dot(dir, sun);
    const float phaseR = RayleighPhase(cosTheta);
    const float phaseM = MiePhase(cosTheta);

    float3 luminance = 0.0;
    float tPrev = 0.0;
    for (uint i = 0; i < steps; ++i)
    {
        const float x = float(i + 1) / float(steps);
        const float t1 = tMax * x * x;
        const float dt = t1 - tPrev;
        const float3 pos = p + dir * (0.5 * (tPrev + t1));
        tPrev = t1;

        const Medium m = SampleMedium(pos);
        const float3 sampleT = exp(-m.extinction * dt);
        const float3 sunT = Transmittance(pos, sun);
        const float3 s = (m.rayleigh * phaseR + m.mie * phaseM) * sunT + m.scattering * MultiScatter(pos, sun);
        luminance += transmittance * (s - s * sampleT) / max(m.extinction, 1e-7);
        transmittance *= sampleT;
    }
    return luminance;
}

// ------------------------------------------------------------------ LUTs

[numthreads(8, 8, 1)]
void transmittance(uint2 id : SV_DispatchThreadID)
{
    if (any(id >= c_TransmittanceSize))
        return;
    const float2 uv = (float2(id) + 0.5) / float2(c_TransmittanceSize);
    const float c = uv.x * 2.0 - 1.0;
    const float3 p = float3(0.0, g_Atmo.groundRadius + uv.y * (g_Atmo.topRadius - g_Atmo.groundRadius), 0.0);
    const float3 d = float3(sqrt(saturate(1.0 - c * c)), c, 0.0);

    if (RaySphere(p, d, g_Atmo.groundRadius) > 0.0)
    {
        u_Out[id] = 0.0;
        return;
    }
    const float tMax = RaySphere(p, d, g_Atmo.topRadius);
    const uint steps = 40;
    float3 depth = 0.0;
    for (uint i = 0; i < steps; ++i)
        depth += SampleMedium(p + d * ((float(i) + 0.5) / float(steps) * tMax)).extinction;
    u_Out[id] = float4(exp(-depth * (tMax / float(steps))), 1.0);
}

// Hillaire 2020 section 5.5: second-order scattering and the transfer f_ms
// integrated over the sphere, psi = L2 / (1 - f_ms).
[numthreads(8, 8, 1)]
void multiscatter(uint2 id : SV_DispatchThreadID)
{
    if (any(id >= c_MultiScatterSize))
        return;
    const float2 uv = (float2(id) + 0.5) / float(c_MultiScatterSize);
    const float sunCos = uv.x * 2.0 - 1.0;
    const float3 p = float3(0.0, g_Atmo.groundRadius + max(uv.y * (g_Atmo.topRadius - g_Atmo.groundRadius), 1e-3), 0.0);
    const float3 sun = float3(sqrt(saturate(1.0 - sunCos * sunCos)), sunCos, 0.0);
    const float isotropic = 1.0 / (4.0 * HYPER_PI);

    const uint n = 8, steps = 20;
    float3 l2 = 0.0, fms = 0.0;
    for (uint a = 0; a < n; ++a)
    for (uint b = 0; b < n; ++b)
    {
        const float cosT = 1.0 - 2.0 * (float(b) + 0.5) / float(n);
        const float sinT = sqrt(saturate(1.0 - cosT * cosT));
        const float phi = HYPER_TWO_PI * (float(a) + 0.5) / float(n);
        const float3 dir = float3(sinT * cos(phi), cosT, sinT * sin(phi));

        const float tGround = RaySphere(p, dir, g_Atmo.groundRadius);
        const float tMax = tGround > 0.0 ? tGround : RaySphere(p, dir, g_Atmo.topRadius);
        const float dt = tMax / float(steps);
        float3 t = 1.0, lum = 0.0, f = 0.0;
        for (uint i = 0; i < steps; ++i)
        {
            const float3 pos = p + dir * ((float(i) + 0.5) * dt);
            const Medium m = SampleMedium(pos);
            const float3 sampleT = exp(-m.extinction * dt);
            const float3 invExt = 1.0 / max(m.extinction, 1e-7);
            const float3 s = m.scattering * Transmittance(pos, sun) * isotropic;
            lum += t * (s - s * sampleT) * invExt;
            f += t * (m.scattering - m.scattering * sampleT) * invExt;
            t *= sampleT;
        }
        if (tGround > 0.0)
        {
            const float3 pg = p + dir * tGround;
            lum += t * Transmittance(pg, sun) * saturate(dot(normalize(pg), sun)) * g_Atmo.groundAlbedo / HYPER_PI;
        }
        l2 += lum;
        fms += f;
    }
    l2 /= float(n * n);
    fms /= float(n * n);
    u_Out[id] = float4(l2 / max(1.0 - fms, 1e-3), 1.0);
}

// Sky-view parameterisation around the camera: u = azimuth from the sun
// (0..pi, the sky is symmetric about the sun's plane), v = elevation above the
// true horizon, square-root warped so the horizon gets the most texels.
float HorizonDip(float r)
{
    return acos(clamp(g_Atmo.groundRadius / r, -1.0, 1.0));
}

float2 SkyViewUv(float3 dir)
{
    const float3 up = normalize(g_Atmo.cameraKm);
    const float beta = HorizonDip(length(g_Atmo.cameraKm));
    const float a = asin(clamp(dot(dir, up), -1.0, 1.0)) + beta;
    const float v = a >= 0.0 ? 0.5 + 0.5 * sqrt(saturate(a / (0.5 * HYPER_PI + beta)))
                             : 0.5 - 0.5 * sqrt(saturate(-a / (0.5 * HYPER_PI - beta)));
    const float3 h = dir - up * dot(dir, up);
    const float3 sunH = g_Atmo.sunDir - up * dot(g_Atmo.sunDir, up);
    float phi = 0.0;
    if (dot(h, h) > 1e-10 && dot(sunH, sunH) > 1e-10)
        phi = acos(clamp(dot(normalize(h), normalize(sunH)), -1.0, 1.0));
    return float2(phi / HYPER_PI, v);
}

[numthreads(8, 8, 1)]
void skyview(uint2 id : SV_DispatchThreadID)
{
    if (any(id >= c_SkyViewSize))
        return;
    const float2 uv = (float2(id) + 0.5) / float2(c_SkyViewSize);
    const float r = length(g_Atmo.cameraKm);
    const float beta = HorizonDip(r);
    const float a = uv.y >= 0.5 ? (2.0 * uv.y - 1.0) * (2.0 * uv.y - 1.0) * (0.5 * HYPER_PI + beta)
                                : -(1.0 - 2.0 * uv.y) * (1.0 - 2.0 * uv.y) * (0.5 * HYPER_PI - beta);
    const float elevation = a - beta;
    const float phi = uv.x * HYPER_PI;

    // Local frame: up = +Y, the sun in the XY plane.
    const float sunCos = dot(normalize(g_Atmo.cameraKm), g_Atmo.sunDir);
    const float3 sun = float3(sqrt(saturate(1.0 - sunCos * sunCos)), sunCos, 0.0);
    const float3 dir = float3(cos(elevation) * cos(phi), sin(elevation), cos(elevation) * sin(phi));

    float3 t;
    const float3 lum = IntegrateScattering(float3(0.0, r, 0.0), dir, sun, 1e9, 32, t);
    u_Out[id] = float4(lum, 1.0);
}

float3 WorldFromClip(float2 clip, float depth)
{
    const float4 p = mul(float4(clip, depth, 1.0), g_Atmo.view.matClipToWorld);
    return p.xyz / p.w;
}

// One thread per froxel column; slice k holds the result at (k + 1) / 32 of aerialMaxKm.
[numthreads(8, 8, 1)]
void aerial(uint2 id : SV_DispatchThreadID)
{
    if (any(id >= c_AerialSize))
        return;
    const float2 uv = (float2(id) + 0.5) / float(c_AerialSize);
    const float2 clip = float2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    const float3 dir = normalize(WorldFromClip(clip, 0.5) - g_Atmo.cameraWorld);
    const float3 p = g_Atmo.cameraKm;
    const float3 sun = g_Atmo.sunDir;
    const float phaseR = RayleighPhase(dot(dir, sun));
    const float phaseM = MiePhase(dot(dir, sun));

    const float tGround = RaySphere(p, dir, g_Atmo.groundRadius);
    const float sliceKm = g_Atmo.aerialMaxKm / float(c_AerialSize);
    float3 lum = 0.0, t = 1.0;
    float tPrev = 0.0;
    for (uint k = 0; k < c_AerialSize; ++k)
    {
        // Two samples per slice, none past the ground.
        for (uint j = 0; j < 2; ++j)
        {
            float t1 = (float(k) + (float(j) + 1.0) * 0.5) * sliceKm;
            if (tGround > 0.0)
                t1 = min(t1, tGround);
            const float dt = t1 - tPrev;
            if (dt <= 0.0)
                continue;
            const float3 pos = p + dir * (0.5 * (tPrev + t1));
            tPrev = t1;
            const Medium m = SampleMedium(pos);
            const float3 sampleT = exp(-m.extinction * dt);
            const float3 s = (m.rayleigh * phaseR + m.mie * phaseM) * Transmittance(pos, sun) + m.scattering * MultiScatter(pos, sun);
            lum += t * (s - s * sampleT) / max(m.extinction, 1e-7);
            t *= sampleT;
        }
        u_Volume[uint3(id, k)] = float4(lum, dot(t, 1.0 / 3.0));
    }
}

// ------------------------------------------------------------------ apply / capture

// Sky luminance for a unit sun, plus the lit ground below the horizon.
float3 SkyLuminance(float3 dir, bool sunDisk)
{
    const float3 p = g_Atmo.cameraKm;
    float3 lum = t_SkyView.SampleLevel(s_Linear, SkyViewUv(dir), 0).rgb;

    const float tGround = RaySphere(p, dir, g_Atmo.groundRadius);
    if (tGround > 0.0)
    {
        const float3 pg = p + dir * tGround;
        const float3 viewT = Transmittance(pg, -dir) / max(Transmittance(p, -dir), 1e-6);
        lum += viewT * Transmittance(pg, g_Atmo.sunDir) * saturate(dot(normalize(pg), g_Atmo.sunDir)) * g_Atmo.groundAlbedo / HYPER_PI;
    }
    else if (sunDisk)
    {
        const float c = dot(dir, g_Atmo.sunDir);
        const float edge = smoothstep(g_Atmo.sunCosRadius - 2e-6, g_Atmo.sunCosRadius + 2e-6, c);
        lum += edge * Transmittance(p, dir) * g_Atmo.sunDiskRadiance;
    }
    return lum;
}

[numthreads(8, 8, 1)]
void apply(uint2 id : SV_DispatchThreadID)
{
    if (any(id >= g_Atmo.size))
        return;
    const float depth = t_Depth[id];
    const float2 pixel = float2(id) + 0.5;
    const float2 clip = pixel * g_Atmo.view.windowToClipScale + g_Atmo.view.windowToClipBias;

    if (depth <= 0.0)
        return;

    const float distanceKm = length(WorldFromClip(clip, depth) - g_Atmo.cameraWorld) * g_Atmo.unitKm;
    const float w = distanceKm / g_Atmo.aerialMaxKm;
    float4 ap = t_Aerial.SampleLevel(s_Linear, float3(pixel / float2(g_Atmo.size), max(w - 0.5 / float(c_AerialSize), 0.0)), 0);
    // Fade in over the first slice (its value is for the slice's far end).
    const float fade = saturate(w * float(c_AerialSize));
    ap = lerp(float4(0.0, 0.0, 0.0, 1.0), ap, fade);

    const float4 colour = u_Out[id];
    u_Out[id] = float4(colour.rgb * ap.a + ap.rgb * g_Atmo.sunIrradiance, colour.a);
}

void sky(in float4 pos : SV_Position, in float2 uv : UV, out float4 o_Color : SV_Target)
{
    const float2 clip = pos.xy * g_Atmo.view.windowToClipScale + g_Atmo.view.windowToClipBias;
    const float3 dir = normalize(WorldFromClip(clip, 0.5) - g_Atmo.cameraWorld);
    o_Color = float4(min(SkyLuminance(dir, true) * g_Atmo.sunIrradiance, 60000.0), 1.0);
}

[numthreads(8, 8, 1)]
void capture(uint3 id : SV_DispatchThreadID)
{
    const uint size = g_Atmo.size.x;
    if (id.x >= size || id.y >= size)
        return;
    const float3 dir = CubeDirection(id.z, (float2(id.xy) + 0.5) / float(size));
    u_Cube[id] = float4(min(SkyLuminance(dir, false) * g_Atmo.sunIrradiance, 60000.0), 1.0);
}
