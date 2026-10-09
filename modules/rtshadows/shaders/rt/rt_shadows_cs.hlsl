// Ray-traced sun shadows: one ray per pixel from the G-buffer towards a point on
// the sun's disk (jittered per frame, so the penumbra resolves under TAA).
// Output: visibility in .r of a shadow-channel texture for Donut's deferred pass.

#include <hyper/rt.hlsli>
#include <donut/shaders/view_cb.h>

struct RtShadowConstants
{
    PlanarViewConstants view;
    float3 sunDir;          // towards the sun
    float tanRadius;        // tangent of the sampled disk's angular radius
    uint2 size;
    uint frameIndex;
    float normalBias;
};

ConstantBuffer<RtShadowConstants> g_Shadow : register(b0);
Texture2D<float> t_Depth : register(t0);
Texture2D<float4> t_Normals : register(t1);
VK_IMAGE_FORMAT("rgba8") RWTexture2D<float4> u_Shadow : register(u0);

float Hash(uint3 v)
{
    v = v * 1664525u + 1013904223u;
    v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
    v ^= v >> 16u;
    v.x += v.y * v.z;
    return float(v.x) / 4294967295.0;
}

[numthreads(8, 8, 1)]
void main(uint2 id : SV_DispatchThreadID)
{
    if (any(id >= g_Shadow.size))
        return;
    const float depth = t_Depth[id];
    if (depth <= 0.0)
    {
        u_Shadow[id] = 1.0;
        return;
    }

    const float2 clip = (float2(id) + 0.5) * g_Shadow.view.windowToClipScale + g_Shadow.view.windowToClipBias;
    const float4 wp = mul(float4(clip, depth, 1.0), g_Shadow.view.matClipToWorld);
    const float3 position = wp.xyz / wp.w;
    const float3 n = normalize(t_Normals[id].xyz);
    const float3 l = normalize(g_Shadow.sunDir);
    if (dot(n, l) <= 0.0)
    {
        u_Shadow[id] = 0.0;
        return;
    }

    // A point on the sun's disk, uniformly.
    const float3 t = normalize(cross(abs(l.y) < 0.99 ? float3(0, 1, 0) : float3(1, 0, 0), l));
    const float3 b = cross(l, t);
    const float r = sqrt(Hash(uint3(id, g_Shadow.frameIndex))) * g_Shadow.tanRadius;
    const float phi = HYPER_TWO_PI * Hash(uint3(id.yx, g_Shadow.frameIndex * 7u + 3u));
    const float3 dir = normalize(l + (t * cos(phi) + b * sin(phi)) * r);

    // Depth-reconstructed positions drift with distance; bias with it.
    const float viewDistance = length(position - g_Shadow.view.cameraDirectionOrPosition.xyz);
    const float3 origin = position + n * g_Shadow.normalBias * (1.0 + 0.05 * viewDistance);
    u_Shadow[id] = TraceVisible(origin, dir, 1e6) ? 1.0 : 0.0;
}
