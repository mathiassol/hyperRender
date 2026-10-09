#ifndef HYPER_TRACED_LIGHTING_HLSLI
#define HYPER_TRACED_LIGHTING_HLSLI

// Traced bounce light and reflections for hyper::TracedLighting. This file is the
// mechanism; what a ray sees is the app's. An app compiles it in its own shader set (so
// its hit shading can use its own resources): include this, then define
//
//   // Radiance arriving at a surface along a bounce ray (the sky's on a miss).
//   float3 TraceDiffuse(float3 origin, float3 dir, inout Rng rng);
//   // Whether the surface at a full-res pixel reflects, and where, facing which way, how rough.
//   bool ReflectingSurface(uint2 pixel, out float3 position, out float3 normal, out float roughness);
//   // Radiance arriving along a reflection ray, and how far it went.
//   float3 TraceReflection(float3 origin, float3 dir, float roughness, inout Rng rng, out float dist);
//
// and list the entry points in its shaders.cfg: gi_trace, gi_temporal, gi_filter,
// reflect_trace, reflect_temporal, reflect_filter.
//
// Bindings (hyper/pass.hlsli's slots): t0 depth and t1 normals (xyz, roughness in w) at
// full resolution, t3 the pass's input, t4 its history, u0 its output; t2 and t5..t9 are
// the app's.
//
//   gi          one bounce, at half resolution: cosine rays from the surface each half-res
//               pixel stands for; history reprojected through the surface, dropped where the
//               camera's distance to it changed; three edge-avoiding a-trous passes.
//   reflections a GGX visible-normal ray per pixel, but where it's rough (> 0.35) one pixel
//               in each 2 x 2 traces, a different one each frame (the reflection is soft
//               there, and rough rays are the ones that wander into foliage); history
//               reprojected through the reflected point for smooth surfaces (a mirror's image
//               moves with what it shows), clipped to the neighbourhood; a blur that grows
//               with roughness.

#include <hyper/pass.hlsli>
#include <hyper/sampling.hlsli>
#include <hyper/rt.hlsli>

struct TracedConstants
{
    uint2 size;             // full resolution
    uint frameIndex;
    uint step;              // a-trous step; for the temporal passes, 0: no history yet
    uint rays;              // bounce rays per half-res pixel
    float giClamp;          // luminance one pixel's bounce light may carry (fireflies)
    float reflectionClamp;
    float tolerance;        // m: a depth difference history and filters accept, on top of a few % of the distance
};
VK_PUSH_CONSTANT ConstantBuffer<TracedConstants> g_Traced : register(b3);

Texture2D<float> t_Depth : register(t0);
Texture2D<float4> t_Normal : register(t1);
Texture2D<float4> t_In : register(t3);
Texture2D<float4> t_History : register(t4);
VK_IMAGE_FORMAT("rgba16f") RWTexture2D<float4> u_Out : register(u0);

float3 TraceDiffuse(float3 origin, float3 dir, inout Rng rng);
bool ReflectingSurface(uint2 pixel, out float3 position, out float3 normal, out float roughness);
float3 TraceReflection(float3 origin, float3 dir, float roughness, inout Rng rng, out float dist);

// ------------------------------------------------------------------ bounce light

uint2 HalfSize()
{
    return (g_Traced.size + 1u) / 2u;
}

// The full-res pixel a half-res texel stands for, and its surface.
bool GiSurface(uint2 id, out float3 p, out float3 n, out float dist)
{
    const uint2 full = min(id * 2u, g_Traced.size - 1u);
    const float depth = t_Depth[full];
    n = normalize(t_Normal[full].xyz);
    p = 0.0;
    dist = 0.0;
    if (depth <= 0.0)
        return false;
    p = WorldFromDepth(float2(full), depth);
    dist = length(p - CameraPosition());
    return true;
}

[numthreads(8, 8, 1)]
void gi_trace(uint2 id : SV_DispatchThreadID)
{
    if (any(id >= HalfSize()))
        return;
    float3 p, n;
    float dist;
    if (!GiSurface(id, p, n, dist))
    {
        u_Out[id] = 0.0;
        return;
    }
    Rng rng = MakeRngHashed(id, g_Traced.frameIndex * 5u + 2u);
    float3 radiance = 0.0;
    for (uint i = 0; i < g_Traced.rays; ++i)
        radiance += TraceDiffuse(OffsetRayOrigin(p, n), SampleCosine(n, rng), rng);
    radiance /= float(max(g_Traced.rays, 1u));
    radiance *= min(1.0, g_Traced.giClamp / max(Luminance(radiance), 1e-4));
    u_Out[id] = float4(radiance, dist);
}

[numthreads(8, 8, 1)]
void gi_temporal(uint2 id : SV_DispatchThreadID)
{
    if (any(id >= HalfSize()))
        return;
    float3 p, n;
    float dist;
    const float4 current = t_In[id];
    if (!GiSurface(id, p, n, dist))
    {
        u_Out[id] = 0.0;
        return;
    }
    const float2 prevPixel = PixelFromWorld(p, g_ViewPrev) * 0.5;
    const float prevDist = length(p - g_ViewPrev.matViewToWorld[3].xyz);
    const float2 hs = float2(HalfSize());
    float3 result = current.rgb;
    if (g_Traced.step != 0u && all(prevPixel >= 0.5) && all(prevPixel < hs - 0.5))
    {
        const float4 h = t_History.SampleLevel(s_Linear, prevPixel / hs, 0);
        if (abs(h.a - prevDist) < 0.03 * prevDist + g_Traced.tolerance)
            result = lerp(h.rgb, current.rgb, 0.1);
    }
    u_Out[id] = float4(result, dist);
}

static const float c_TracedKernel[3] = { 3.0 / 8.0, 1.0 / 4.0, 1.0 / 16.0 };

[numthreads(8, 8, 1)]
void gi_filter(uint2 id : SV_DispatchThreadID)
{
    const uint2 hs = HalfSize();
    if (any(id >= hs))
        return;
    float3 p, n;
    float dist;
    const float4 centre = t_In[id];
    if (!GiSurface(id, p, n, dist))
    {
        u_Out[id] = centre;
        return;
    }
    const int step = int(g_Traced.step);
    float3 sum = centre.rgb * c_TracedKernel[0] * c_TracedKernel[0];
    float weights = c_TracedKernel[0] * c_TracedKernel[0];
    [unroll]
    for (int y = -2; y <= 2; ++y)
        [unroll]
        for (int x = -2; x <= 2; ++x)
        {
            if (x == 0 && y == 0)
                continue;
            const int2 q = int2(id) + int2(x, y) * step;
            if (any(q < 0) || any(q >= int2(hs)))
                continue;
            const float4 s = t_In[q];
            const float3 nq = normalize(t_Normal[min(uint2(q) * 2u, g_Traced.size - 1u)].xyz);
            const float w = c_TracedKernel[abs(x)] * c_TracedKernel[abs(y)]
                * pow(saturate(dot(n, nq)), 16.0)
                * exp(-abs(s.a - dist) / (0.02 * dist * float(step) + g_Traced.tolerance));
            sum += s.rgb * w;
            weights += w;
        }
    u_Out[id] = float4(sum / weights, dist);
}

// ------------------------------------------------------------------ reflections

[numthreads(8, 8, 1)]
void reflect_trace(uint2 id : SV_DispatchThreadID)
{
    if (any(id >= g_Traced.size))
        return;
    float3 p, n;
    float roughness;
    if (!ReflectingSurface(id, p, n, roughness))
    {
        u_Out[id] = 0.0;
        return;
    }
    if (roughness > 0.35 && ((id.x & 1u) + 2u * (id.y & 1u)) != (g_Traced.frameIndex & 3u))
    {
        u_Out[id] = float4(0.0, 0.0, 0.0, -1.0);     // not traced this frame
        return;
    }
    Rng rng = MakeRngHashed(id, g_Traced.frameIndex * 3u + 1u);
    const float3 dir = SampleSpecular(n, normalize(CameraPosition() - p), roughness, rng);
    float dist;
    float3 radiance = TraceReflection(OffsetRayOrigin(p, n), dir, roughness, rng, dist);
    radiance *= min(1.0, g_Traced.reflectionClamp / max(Luminance(radiance), 1e-4));
    u_Out[id] = float4(radiance, min(dist, 1e4));
}

float3 RgbToYCoCg(float3 c)
{
    return float3(0.25 * c.r + 0.5 * c.g + 0.25 * c.b, 0.5 * c.r - 0.5 * c.b, -0.25 * c.r + 0.5 * c.g - 0.25 * c.b);
}

float3 YCoCgToRgb(float3 c)
{
    return float3(c.x + c.y - c.z, c.x + c.z, c.x - c.y - c.z);
}

[numthreads(8, 8, 1)]
void reflect_temporal(uint2 id : SV_DispatchThreadID)
{
    if (any(id >= g_Traced.size))
        return;
    float3 p, n;
    float roughness;
    if (!ReflectingSurface(id, p, n, roughness))
    {
        u_Out[id] = 0.0;
        return;
    }
    float4 current = t_In[id];
    // The neighbourhood's statistics from the pixels traced this frame (all of them where
    // it's glossy, a quarter where it's rough); an untraced pixel takes their mean.
    float3 m1 = 0.0, m2 = 0.0, fill = 0.0;
    float count = 0.0, fillDist = 0.0;
    [unroll]
    for (int y = -1; y <= 1; ++y)
        [unroll]
        for (int x = -1; x <= 1; ++x)
        {
            const float4 s = t_In[clamp(int2(id) + int2(x, y), 0, int2(g_Traced.size) - 1)];
            if (s.a < 0.0)
                continue;
            const float3 c = RgbToYCoCg(s.rgb);
            const float3 cc = c / (1.0 + c.x);
            m1 += cc;
            m2 += cc * cc;
            fill += s.rgb;
            fillDist += s.a;
            count += 1.0;
        }
    m1 /= max(count, 1.0);
    m2 /= max(count, 1.0);
    const bool traced = current.a >= 0.0;
    if (!traced)
        current = count > 0.0 ? float4(fill / count, fillDist / count) : float4(0.0, 0.0, 0.0, 1.0);
    const float3 sigma = sqrt(max(m2 - m1 * m1, 0.0));
    const float3 v = normalize(CameraPosition() - p);
    const float mirror = saturate(1.0 - roughness / 0.3);
    const float2 prev = lerp(PixelFromWorld(p, g_ViewPrev), PixelFromWorld(p - v * current.a, g_ViewPrev), mirror);
    // A quarter of the samples where it's rough: a quarter of the weight for a filled-in one.
    float alpha = lerp(0.4, 0.06, saturate(roughness / 0.35)) * (traced ? 1.0 : 0.25);
    float3 history = current.rgb;
    if (g_Traced.step != 0u && all(prev >= 0.5) && all(prev < float2(g_Traced.size) - 0.5))
    {
        history = t_History.SampleLevel(s_Linear, prev / float2(g_Traced.size), 0).rgb;
        // Clip in compressed YCoCg to the box of mean +- 1.25 sigma.
        float3 h = RgbToYCoCg(history);
        float3 hc = h / (1.0 + h.x);
        hc = clamp(hc, m1 - 1.25 * sigma, m1 + 1.25 * sigma);
        h = hc / max(1.0 - hc.x, 1e-3);
        history = max(YCoCgToRgb(h), 0.0);
    }
    else
        alpha = 1.0;
    u_Out[id] = float4(lerp(history, current.rgb, alpha), current.a);
}

[numthreads(8, 8, 1)]
void reflect_filter(uint2 id : SV_DispatchThreadID)
{
    if (any(id >= g_Traced.size))
        return;
    float3 p, n;
    float roughness;
    if (!ReflectingSurface(id, p, n, roughness))
    {
        u_Out[id] = 0.0;
        return;
    }
    const float4 centre = t_In[id];
    if (roughness < 0.1)
    {
        u_Out[id] = centre;
        return;
    }
    const float depthC = t_Depth[id];
    const float step = 1.0 + roughness * 5.0;
    float3 sum = centre.rgb;
    float weights = 1.0;
    [unroll]
    for (int y = -2; y <= 2; ++y)
        [unroll]
        for (int x = -2; x <= 2; ++x)
        {
            if (x == 0 && y == 0)
                continue;
            const int2 q = int2(id) + int2(round(float2(x, y) * step));
            if (any(q < 0) || any(q >= int2(g_Traced.size)))
                continue;
            const float4 wn = t_Normal[q];
            const float dq = t_Depth[q];
            const float w = exp(-0.5 * float(x * x + y * y) / 2.0)
                * pow(saturate(dot(n, normalize(wn.xyz))), 32.0)
                * exp(-abs(dq - depthC) / max(depthC * 0.02, 1e-6))
                * exp(-abs(wn.w - roughness) * 8.0);
            sum += t_In[q].rgb * w;
            weights += w;
        }
    u_Out[id] = float4(sum / weights, centre.a);
}

#endif
