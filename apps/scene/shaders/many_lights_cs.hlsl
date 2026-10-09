// The scene app's point lights past the 16 Donut's deferred pass takes (a scene can have
// thousands): tiled, added to the lit image. Each 16 x 16 tile finds its depth range,
// tests every light's sphere against it and against its rectangle (the sphere projected,
// a circle that covers it), marks the ones that reach it in a bitmask, and shades its
// pixels with them in index order (the same sum every run), by Donut's own ShadeSurface,
// as its deferred pass would. At most MAX_LIGHTS lights.
//
//   t0..t4  depth, the G-buffer's four channels (Donut's order)
//   u0      the lit image (added to)
//   t20     the lights (Donut's LightConstants), t21 each one's sphere (world xyz, radius)

#pragma pack_matrix(row_major)

#include <hyper/pass.hlsli>
#include <donut/shaders/gbuffer.hlsli>
#include <donut/shaders/lighting.hlsli>

Texture2D<float> t_Depth : register(t0);
Texture2D t_GBuffer0 : register(t1);
Texture2D t_GBuffer1 : register(t2);
Texture2D t_GBuffer2 : register(t3);
Texture2D t_GBuffer3 : register(t4);
VK_IMAGE_FORMAT("rgba16f") RWTexture2D<float4> u_Output : register(u0);
StructuredBuffer<LightConstants> t_Lights : register(t20);
StructuredBuffer<float4> t_Spheres : register(t21);

struct ManyLightsPush
{
    uint count;
    uint3 pad;
};
DECLARE_PUSH_CONSTANTS(ManyLightsPush, g_Push, 3, 0);

#define TILE 16
#define MAX_LIGHTS 16384   // the lights a tile can hold, a bit each: lit in index order, so the sum is the same every run

groupshared uint s_MinZ, s_MaxZ;
groupshared uint s_Mask[MAX_LIGHTS / 32];

float ViewDepth(float depth)
{
    // Reverse-Z depth to view-space z (forward), through the inverse projection.
    const float4 v = mul(float4(0.0, 0.0, depth, 1.0), g_View.matClipToView);
    return v.z / v.w;
}


[numthreads(TILE, TILE, 1)]
void main(uint2 id : SV_DispatchThreadID, uint2 group : SV_GroupID, uint index : SV_GroupIndex)
{
    if (index == 0)
    {
        s_MinZ = 0x7f7fffff;
        s_MaxZ = 0;
    }
    for (uint w = index; w < MAX_LIGHTS / 32; w += TILE * TILE)
        s_Mask[w] = 0;
    GroupMemoryBarrierWithGroupSync();

    const bool inside = all(id < uint2(g_View.viewportSize));
    const float depth = inside ? t_Depth[id] : 0.0;
    const bool surface = depth > 0.0;   // reverse-Z: 0 is the sky
    float z = 0.0;
    if (surface)
    {
        z = ViewDepth(depth);
        InterlockedMin(s_MinZ, asuint(max(z, 0.0)));
        InterlockedMax(s_MaxZ, asuint(max(z, 0.0)));
    }
    GroupMemoryBarrierWithGroupSync();
    if (s_MaxZ == 0)
        return;   // all sky

    // The tile's rectangle in pixels and depth range; each light's sphere, projected, a
    // circle that covers it (its radius at the sphere's nearest depth), against both.
    const float2 lo = float2(group * TILE), hi = lo + TILE;
    const float minZ = asfloat(s_MinZ), maxZ = asfloat(s_MaxZ);
    const float4x4 proj = g_View.matViewToClip;
    const float2 scale = 0.5 * g_View.viewportSize * float2(proj[0][0], proj[1][1]);
    const uint count = min(g_Push.count, uint(MAX_LIGHTS));
    for (uint i = index; i < count; i += TILE * TILE)
    {
        const float4 s = t_Spheres[i];
        const float3 v = mul(float4(s.xyz, 1.0), g_View.matWorldToView).xyz;
        bool hit = v.z + s.w >= minZ && v.z - s.w <= maxZ;
        if (hit && v.z - s.w > 0.05)
        {
            const float4 clip = mul(float4(v, 1.0), proj);
            const float2 centre = float2(clip.x / clip.w * 0.5 + 0.5, 0.5 - clip.y / clip.w * 0.5) * g_View.viewportSize;
            const float radius = s.w * max(scale.x, scale.y) / (v.z - s.w);
            const float2 nearest = clamp(centre, lo, hi);
            hit = dot(nearest - centre, nearest - centre) <= radius * radius;
        }
        if (hit)
            InterlockedOr(s_Mask[i / 32], 1u << (i % 32));
    }
    GroupMemoryBarrierWithGroupSync();
    if (!surface)
        return;

    float4 channels[4];
    channels[0] = t_GBuffer0[id];
    channels[1] = t_GBuffer1[id];
    channels[2] = t_GBuffer2[id];
    channels[3] = t_GBuffer3[id];
    const MaterialSample material = DecodeGBuffer(channels);
    const float3 position = WorldFromDepth(float2(id), depth);
    const float3 incident = normalize(position - CameraPosition());
    float3 light = 0.0;
    bool any = false;
    for (uint word = 0; word < (count + 31) / 32; ++word)
    {
        uint bits = s_Mask[word];
        while (bits != 0)
        {
            const uint j = word * 32 + firstbitlow(bits);
            bits &= bits - 1;
            // (ShadeSurface leaves the colour to the caller, as Donut's deferred pass multiplies it.)
            const LightConstants l = t_Lights[j];
            float3 diffuse, specular;
            ShadeSurface(l, material, position, incident, diffuse, specular);
            light += (diffuse + specular) * l.color;
            any = true;
        }
    }
    if (any)
        u_Output[id] = float4(u_Output[id].rgb + light, u_Output[id].a);
}
