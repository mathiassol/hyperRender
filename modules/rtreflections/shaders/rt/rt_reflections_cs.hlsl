// Ray-traced reflections: one GGX-sampled ray per pixel from the G-buffer, the
// hit shaded with hyper/rt_lighting.hlsli (sun + shadow ray, emissive, DDGI or
// IBL indirect). Output matches the SSR trace (rgb radiance, a confidence), so
// ScreenSpaceReflections' temporal filter and composite finish it. Misses get
// confidence 0: the IBL probe's reflection, already in the lit image, stays.

#include <hyper/rt_lighting.hlsli>
#include <donut/shaders/view_cb.h>

struct RtReflectionConstants
{
    PlanarViewConstants view;
    uint2 size;
    uint frameIndex;
    float maxRoughness;
    float normalBias;
    float3 pad;
};

ConstantBuffer<RtReflectionConstants> g_Refl : register(b0);
Texture2D<float> t_Depth : register(t0);
Texture2D<float4> t_Normals : register(t1);
VK_IMAGE_FORMAT("rgba16f") RWTexture2D<float4> u_Traced : register(u0);

float Hash(uint3 v)
{
    v = v * 1664525u + 1013904223u;
    v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
    v ^= v >> 16u;
    v.x += v.y * v.z;
    return float(v.x) / 4294967295.0;
}

// Heitz 2018, visible normals. v in tangent space (z = normal).
float3 SampleGgxVndf(float3 v, float alpha, float2 u)
{
    const float3 vh = normalize(float3(alpha * v.x, alpha * v.y, v.z));
    const float lensq = vh.x * vh.x + vh.y * vh.y;
    const float3 t1 = lensq > 0.0 ? float3(-vh.y, vh.x, 0.0) * rsqrt(lensq) : float3(1.0, 0.0, 0.0);
    const float3 t2 = cross(vh, t1);
    const float r = sqrt(u.x);
    const float phi = HYPER_TWO_PI * u.y;
    const float p1 = r * cos(phi);
    float p2 = r * sin(phi);
    const float s = 0.5 * (1.0 + vh.z);
    p2 = (1.0 - s) * sqrt(1.0 - p1 * p1) + s * p2;
    const float3 nh = p1 * t1 + p2 * t2 + sqrt(max(0.0, 1.0 - p1 * p1 - p2 * p2)) * vh;
    return normalize(float3(alpha * nh.x, alpha * nh.y, max(0.0, nh.z)));
}

[numthreads(8, 8, 1)]
void main(uint2 id : SV_DispatchThreadID)
{
    if (any(id >= g_Refl.size))
        return;
    const float depth = t_Depth[id];
    const float4 nr = t_Normals[id];
    const float roughness = nr.w;
    if (depth <= 0.0 || roughness > g_Refl.maxRoughness)
    {
        u_Traced[id] = 0.0;
        return;
    }

    const float2 clip = (float2(id) + 0.5) * g_Refl.view.windowToClipScale + g_Refl.view.windowToClipBias;
    const float4 wp = mul(float4(clip, depth, 1.0), g_Refl.view.matClipToWorld);
    const float3 position = wp.xyz / wp.w;
    const float3 n = normalize(nr.xyz);
    const float3 incident = normalize(position - g_Refl.view.cameraDirectionOrPosition.xyz);

    const float3 up = abs(n.y) < 0.999 ? float3(0, 1, 0) : float3(1, 0, 0);
    const float3 tx = normalize(cross(up, n));
    const float3 ty = cross(n, tx);
    const float3 toEye = -incident;
    const float3 vLocal = float3(dot(toEye, tx), dot(toEye, ty), dot(toEye, n));
    const float2 xi = float2(Hash(uint3(id, g_Refl.frameIndex)), Hash(uint3(id.yx, g_Refl.frameIndex + 7919u)));
    const float alpha = max(roughness * roughness, 1e-3);
    const float3 hLocal = SampleGgxVndf(vLocal, alpha, xi);
    const float3 h = normalize(hLocal.x * tx + hLocal.y * ty + hLocal.z * n);
    float3 r = reflect(incident, h);
    if (dot(r, n) <= 0.0)
        r = reflect(incident, n);

    const float viewDistance = length(position - g_Refl.view.cameraDirectionOrPosition.xyz);
    const float3 origin = position + n * g_Refl.normalBias * (1.0 + 0.05 * viewDistance);
    const RtHit hit = TraceClosest(origin, r, 1e4);
    if (!hit.hit)
    {
        u_Traced[id] = 0.0;
        return;
    }

    // Texture detail the reflection can resolve: coarser with distance and roughness.
    const float lod = clamp(log2(max(hit.t * (alpha + 0.002) * 512.0, 1.0)), 0.0, 10.0);
    const HitSurface s = GetHitSurface(hit, r, lod);
    const float3 radiance = ShadeHitDiffuse(s, -r);
    const float confidence = 1.0 - smoothstep(g_Refl.maxRoughness * 0.6, g_Refl.maxRoughness, roughness);
    u_Traced[id] = float4(min(radiance, 1000.0), confidence);
}
