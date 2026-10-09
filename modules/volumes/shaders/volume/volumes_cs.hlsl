// Ray-marched participating media (hyper::RaymarchedVolumes): local boxes and
// spheres of noisy, optionally emissive media, and a cloud layer (after
// Schneider, "Nubis" 2015-2023: weather coverage, Perlin-Worley shapes eroded by
// Worley detail, Beer-powder, two-lobe phase).
//
//   march       half resolution, jittered: in-scattering + transmittance, and the
//               distances the upsample and reprojection need
//   temporal    reprojected accumulation, clamped to the current neighbourhood
//   composite   full resolution, depth-aware upsample: colour * T + S

#include <hyper/common.hlsli>
#include <hyper/noise.hlsli>
#include <hyper/volume.hlsli>
#include <donut/shaders/view_cb.h>

static const uint c_MaxBoxes = 8;

struct VolumeConstants
{
    PlanarViewConstants view;
    PlanarViewConstants viewPrev;
    uint2 fullSize;
    uint2 halfSize;
    float3 cameraPos;
    uint frameIndex;
    float3 sunDir;              // towards the sun
    uint steps;
    float3 sunRadiance;
    uint lightSteps;
    float maxDistance;
    float temporalWeight;
    uint historyValid;
    uint numBoxes;
    uint cloudsEnabled;
    float cloudBottom;
    float cloudTop;
    float cloudCoverage;
    float cloudDensity;
    float cloudShapeScale;
    float cloudDetailScale;
    float cloudWeatherScale;
    float3 cloudWind;           // offset, world units
    float cloudPowder;
    float gForward;
    float gBack;
    float lobeBlend;
    float ambientIntensity;
    float3 cloudAlbedo;
    uint hasProbe;
    uint hasAtmosphere;
    float aerialMaxKm;
    float unitKm;               // km per world unit (aerial perspective scale included)
    float time;
    float airIrradiance;        // the atmosphere LUTs are for a unit sun
    float3 pad;
    float4 boxCentre[c_MaxBoxes];     // xyz, w density
    float4 boxExtent[c_MaxBoxes];     // xyz half extent, w: 0 box, 1 sphere
    float4 boxAlbedo[c_MaxBoxes];     // rgb, w noise strength
    float4 boxEmission[c_MaxBoxes];   // rgb (radiance per unit density), w noise cycles per unit
};

ConstantBuffer<VolumeConstants> g_Vol : register(b0);

Texture3D<float4> t_Shape : register(t0);       // NoiseType::PerlinWorley
Texture3D<float4> t_Detail : register(t1);      // NoiseType::Worley
Texture2D<float> t_Depth : register(t2);        // full-resolution scene depth (reverse-Z)
TextureCubeArray<float4> t_ProbeDiffuse : register(t3);
Texture3D<float4> t_Aerial : register(t4);      // Atmosphere's aerial perspective (optional)
Texture2D<float4> t_Current : register(t5);
Texture2D<float4> t_History : register(t6);
Texture2D<float2> t_Distances : register(t7);   // x: representative distance, y: scene view z
SamplerState s_Wrap : register(s0);
SamplerState s_Clamp : register(s1);

VK_IMAGE_FORMAT("rgba16f") RWTexture2D<float4> u_March : register(u0);
VK_IMAGE_FORMAT("rg32f") RWTexture2D<float2> u_Distances : register(u1);
VK_IMAGE_FORMAT("rgba16f") RWTexture2D<float4> u_Resolved : register(u2);
VK_IMAGE_FORMAT("rgba16f") RWTexture2D<float4> u_Color : register(u3);

float3 RayDirection(float2 uv)
{
    const float2 clip = float2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    const float4 p = mul(float4(clip, 0.5, 1.0), g_Vol.view.matClipToWorld);
    return normalize(p.xyz / p.w - g_Vol.cameraPos);
}

float3 CameraForward()
{
    return normalize(g_Vol.view.matViewToWorld[2].xyz);
}

float ViewZ(float depth)
{
    return depth > 0.0 ? g_Vol.view.matViewToClip[3][2] / depth : 1e9;
}

// ------------------------------------------------------------------ media

struct Medium
{
    float density;      // extinction per unit
    float3 albedo;      // single-scattering albedo
    float3 emission;
    float cloud;        // share of the density that is cloud (phase, ambient)
    float cloudHeight;  // 0 at the cloud base, 1 at the top
};

float CloudDensity(float3 p, bool detail, out float height)
{
    height = (p.y - g_Vol.cloudBottom) / max(g_Vol.cloudTop - g_Vol.cloudBottom, 1e-3);
    if (height <= 0.0 || height >= 1.0)
        return 0.0;

    // Weather: large-scale coverage from a slice of the shape noise.
    const float2 wp = (p.xz + g_Vol.cloudWind.xz * 0.5) * g_Vol.cloudWeatherScale;
    const float weather = t_Shape.SampleLevel(s_Wrap, float3(wp, 0.37), 0).r;
    const float coverage = saturate(Remap(weather, 1.0 - g_Vol.cloudCoverage, 1.0, 0.0, 1.0));
    if (coverage <= 0.0)
        return 0.0;

    // Cumulus profile: rounded base, tapering top.
    const float profile = saturate(Remap(height, 0.0, 0.08, 0.0, 1.0)) * saturate(Remap(height, 0.35, 1.0, 1.0, 0.0));

    const float4 s = t_Shape.SampleLevel(s_Wrap, (p + g_Vol.cloudWind) * g_Vol.cloudShapeScale, 0);
    const float lowFbm = s.g * 0.625 + s.b * 0.25 + s.a * 0.125;
    float base = saturate(Remap(s.r, -(1.0 - lowFbm), 1.0, 0.0, 1.0));
    base = saturate(Remap(base * profile, 1.0 - coverage, 1.0, 0.0, 1.0)) * coverage;
    if (base <= 0.0 || !detail)
        return base * g_Vol.cloudDensity;

    // Erode the edges with high-frequency Worley: wispy at the base, billowy above.
    const float3 d = t_Detail.SampleLevel(s_Wrap, (p + g_Vol.cloudWind * 2.0) * g_Vol.cloudDetailScale, 0).rgb;
    const float detailFbm = d.r * 0.625 + d.g * 0.25 + d.b * 0.125;
    const float erosion = lerp(detailFbm, 1.0 - detailFbm, saturate(height * 8.0));
    return saturate(Remap(base, erosion * 0.45, 1.0, 0.0, 1.0)) * g_Vol.cloudDensity;
}

Medium SampleMedium(float3 p, bool detail)
{
    Medium m = (Medium)0;
    for (uint i = 0; i < g_Vol.numBoxes; ++i)
    {
        const float3 local = (p - g_Vol.boxCentre[i].xyz) / max(g_Vol.boxExtent[i].xyz, 1e-3);
        const float r = g_Vol.boxExtent[i].w > 0.5 ? length(local) : max(abs(local.x), max(abs(local.y), abs(local.z)));
        float d = g_Vol.boxCentre[i].w * saturate((1.0 - r) * 4.0);
        if (d <= 0.0)
            continue;
        const float strength = g_Vol.boxAlbedo[i].w;
        if (strength > 0.0 && detail)
        {
            // Billows carved by thresholded Perlin-Worley, edges eroded by Worley detail; rising slowly.
            const float3 q = p * g_Vol.boxEmission[i].w + float3(0.0, -g_Vol.time * 0.15, 0.0);
            const float4 s = t_Shape.SampleLevel(s_Wrap, q * 0.25, 0);
            const float n = s.r * 0.4 + s.g * 0.4 + s.b * 0.2;
            const float detailN = t_Detail.SampleLevel(s_Wrap, q, 0).r;
            // Baked ranges: R 0.35..1, GBA 0.1..0.9, so n is about 0.3..0.85.
            d *= lerp(1.0, saturate(Remap(n, 0.45, 0.7, 0.0, 1.0)) * lerp(0.5, 1.0, detailN), strength);
        }
        m.albedo += g_Vol.boxAlbedo[i].rgb * d;
        m.emission += g_Vol.boxEmission[i].rgb * d;
        m.density += d;
    }
    if (g_Vol.cloudsEnabled != 0)
    {
        float h;
        const float c = CloudDensity(p, detail, h);
        if (c > 0.0)
        {
            m.albedo += g_Vol.cloudAlbedo * c;
            m.density += c;
            m.cloud = c;
            m.cloudHeight = h;
        }
    }
    if (m.density > 0.0)
    {
        m.cloud /= m.density;
        m.albedo /= m.density;
    }
    return m;
}

// Optical depth towards the sun, with steps growing away from the sample.
float SunOpticalDepth(float3 p)
{
    float depth = 0.0, t = 0.0;
    float step = g_Vol.cloudsEnabled != 0 ? (g_Vol.cloudTop - g_Vol.cloudBottom) / float(g_Vol.lightSteps * 2u) : 0.5;
    for (uint i = 0; i < g_Vol.lightSteps; ++i)
    {
        const float3 q = p + g_Vol.sunDir * (t + 0.5 * step);
        depth += SampleMedium(q, i < 2u).density * step;
        t += step;
        step *= 1.6;
    }
    return depth;
}

// ------------------------------------------------------------------ march

// Where the ray can meet media: boxes and the cloud slab.
float2 MediaRange(float3 o, float3 d, float tMax)
{
    float2 range = float2(1e9, 0.0);
    for (uint i = 0; i < g_Vol.numBoxes; ++i)
    {
        const float3 e = g_Vol.boxExtent[i].xyz;
        const float2 hit = RayBox(o, d, g_Vol.boxCentre[i].xyz - e, g_Vol.boxCentre[i].xyz + e);
        if (hit.x < hit.y)
            range = float2(min(range.x, hit.x), max(range.y, hit.y));
    }
    if (g_Vol.cloudsEnabled != 0)
    {
        // Horizontal slab (no planet curvature within the march range).
        float t0 = 0.0, t1 = 1e9;
        if (abs(d.y) > 1e-5)
        {
            const float a = (g_Vol.cloudBottom - o.y) / d.y;
            const float b = (g_Vol.cloudTop - o.y) / d.y;
            t0 = max(min(a, b), 0.0);
            t1 = max(a, b);
        }
        else if (o.y < g_Vol.cloudBottom || o.y > g_Vol.cloudTop)
        {
            t1 = -1.0;
        }
        if (t0 < t1)
            range = float2(min(range.x, t0), max(range.y, t1));
    }
    range.y = min(range.y, min(tMax, g_Vol.maxDistance));
    return range;
}

[numthreads(8, 8, 1)]
void march(uint2 id : SV_DispatchThreadID)
{
    if (any(id >= g_Vol.halfSize))
        return;
    // The nearest of the 2x2 full-resolution depths: media never shows through foreground edges.
    const uint2 full = min(id * 2u, g_Vol.fullSize - 1u);
    float depth = 0.0;
    [unroll]
    for (uint k = 0; k < 4; ++k)
        depth = max(depth, t_Depth[min(full + uint2(k & 1u, k >> 1), g_Vol.fullSize - 1u)]);

    const float2 uv = (float2(id) + 0.5) / float2(g_Vol.halfSize);
    const float3 dir = RayDirection(uv);
    const float sceneZ = ViewZ(depth);
    const float tScene = sceneZ / max(dot(dir, CameraForward()), 1e-3);
    const float2 range = MediaRange(g_Vol.cameraPos, dir, depth > 0.0 ? tScene : 1e9);

    float3 scattering = 0.0;
    float transmittance = 1.0;
    float representative = range.x < range.y ? range.y : min(tScene, g_Vol.maxDistance);
    bool first = true;

    if (range.x < range.y)
    {
        const uint steps = max(g_Vol.steps, 8u);
        const float dt = (range.y - range.x) / float(steps);
        const float jitter = InterleavedGradientNoise(id, g_Vol.frameIndex);
        const float cosTheta = dot(g_Vol.sunDir, dir);
        // Ambient from above (the sky) and below (the lit ground), radiance / pi.
        float3 skyAmbient = 0.0, groundAmbient = 0.0;
        if (g_Vol.hasProbe != 0)
        {
            skyAmbient = t_ProbeDiffuse.SampleLevel(s_Clamp, float4(0.0, 1.0, 0.0, 0.0), 0).rgb * g_Vol.ambientIntensity;
            groundAmbient = t_ProbeDiffuse.SampleLevel(s_Clamp, float4(0.0, -1.0, 0.0, 0.0), 0).rgb * g_Vol.ambientIntensity;
        }

        for (uint i = 0; i < steps; ++i)
        {
            const float t = range.x + (float(i) + jitter) * dt;
            const float3 p = g_Vol.cameraPos + dir * t;
            const Medium m = SampleMedium(p, true);
            if (m.density <= 1e-6)
                continue;
            if (first)
            {
                representative = t;
                first = false;
            }
            // Multiple scattering, approximated by octaves of weaker extinction, softer
            // phase and less energy (Wrenninge 2013; Hillaire 2016).
            const float opticalDepth = SunOpticalDepth(p);
            float sunLight = 0.0;
            float a = 1.0, e = 1.0, g = 1.0;
            [unroll]
            for (uint octave = 0; octave < 3; ++octave)
            {
                const float phaseO = lerp(HenyeyGreenstein(cosTheta, 0.3 * g),
                    DualLobeHenyeyGreenstein(cosTheta, g_Vol.gForward * g, g_Vol.gBack * g, g_Vol.lobeBlend), m.cloud);
                sunLight += a * phaseO * BeerPowder(opticalDepth * e, g_Vol.cloudPowder * m.cloud);
                a *= 0.5;
                e *= 0.4;
                g *= 0.5;
            }
            // Cloud bases see the ground, tops the sky; other media an even mix.
            const float up = lerp(0.5, m.cloudHeight, m.cloud);
            const float3 light = g_Vol.sunRadiance * sunLight + lerp(groundAmbient, skyAmbient, up);
            const float3 source = m.density * m.albedo * light + m.emission;
            scattering += transmittance * IntegrateScattering(source, m.density, dt);
            transmittance *= exp(-m.density * dt);
            if (transmittance < 0.01)
                break;
        }
    }

    // Aerial perspective between the camera and the media (clouds are kilometres away).
    if (g_Vol.hasAtmosphere != 0 && transmittance < 1.0)
    {
        const float w = representative * g_Vol.unitKm / g_Vol.aerialMaxKm;
        const float4 ap = t_Aerial.SampleLevel(s_Clamp, float3(uv, saturate(w - 1.0 / 64.0)), 0);
        const float fade = saturate(w * 32.0);
        const float airT = lerp(1.0, ap.a, fade);
        const float3 airS = ap.rgb * fade * g_Vol.airIrradiance;
        scattering = scattering * airT + airS * (1.0 - transmittance);
    }

    u_March[id] = float4(scattering, transmittance);
    u_Distances[id] = float2(representative, sceneZ);
}

// ------------------------------------------------------------------ temporal

[numthreads(8, 8, 1)]
void temporal(uint2 id : SV_DispatchThreadID)
{
    if (any(id >= g_Vol.halfSize))
        return;
    const float4 current = t_Current[id];
    float4 lo = current, hi = current;
    [unroll]
    for (int y = -1; y <= 1; ++y)
    [unroll]
    for (int x = -1; x <= 1; ++x)
    {
        const float4 s = t_Current[clamp(int2(id) + int2(x, y), int2(0, 0), int2(g_Vol.halfSize) - 1)];
        lo = min(lo, s);
        hi = max(hi, s);
    }

    float4 result = current;
    if (g_Vol.historyValid != 0)
    {
        const float2 uv = (float2(id) + 0.5) / float2(g_Vol.halfSize);
        const float3 p = g_Vol.cameraPos + RayDirection(uv) * t_Distances[id].x;
        const float4 clip = mul(float4(p, 1.0), g_Vol.viewPrev.matWorldToClip);
        if (clip.w > 0.0)
        {
            const float2 prevUv = float2(clip.x / clip.w * 0.5 + 0.5, 0.5 - clip.y / clip.w * 0.5);
            if (all(prevUv >= 0.0) && all(prevUv <= 1.0))
            {
                const float4 history = clamp(t_History.SampleLevel(s_Clamp, prevUv, 0), lo, hi);
                result = lerp(current, history, g_Vol.temporalWeight);
            }
        }
    }
    u_Resolved[id] = result;
}

// ------------------------------------------------------------------ composite

[numthreads(8, 8, 1)]
void composite(uint2 id : SV_DispatchThreadID)
{
    if (any(id >= g_Vol.fullSize))
        return;
    const float z = ViewZ(t_Depth[id]);
    // Bilinear over the 4 nearest half-resolution texels, weighted by depth similarity.
    const float2 halfPos = (float2(id) + 0.5) * 0.5 - 0.5;
    const int2 base = int2(floor(halfPos));
    const float2 f = halfPos - float2(base);
    float4 sum = 0.0;
    float weightSum = 0.0;
    [unroll]
    for (uint k = 0; k < 4; ++k)
    {
        const int2 o = int2(k & 1u, k >> 1);
        const int2 c = clamp(base + o, int2(0, 0), int2(g_Vol.halfSize) - 1);
        const float bilinear = (o.x ? f.x : 1.0 - f.x) * (o.y ? f.y : 1.0 - f.y);
        const float zc = t_Distances[c].y;
        const float similarity = 1.0 / (1e-3 + abs(min(z, 1e6) - min(zc, 1e6)) / max(min(z, 1e6), 1e-2));
        const float w = bilinear * similarity;
        sum += t_Current[c] * w;
        weightSum += w;
    }
    const float4 media = weightSum > 0.0 ? sum / weightSum : float4(0.0, 0.0, 0.0, 1.0);
    const float4 colour = u_Color[id];
    u_Color[id] = float4(colour.rgb * media.a + media.rgb, colour.a);
}
