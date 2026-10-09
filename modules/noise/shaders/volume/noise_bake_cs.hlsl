// Bakes tileable 3D noise volumes (hyper::NoiseVolume). Sampled with trilinear
// filtering and wrap addressing, one fetch replaces a hashed lattice lookup.
//
//   gradient        R16F  gradient noise, [-1, 1], `period` cells across the volume
//   perlin_worley   RGBA8 R: Perlin-Worley (cloud shapes), GBA: Worley fbm at 2x, 4x, 8x the period
//   worley          RGBA8 RGB: Worley fbm at 1x, 2x, 4x the period (cloud erosion detail)

#include <hyper/common.hlsli>
#include <hyper/noise.hlsli>

struct NoiseConstants
{
    uint size;          // texels per side
    uint period;        // lattice cells per side at the base frequency
    float2 pad;
};

DECLARE_PUSH_CONSTANTS(NoiseConstants, g_Noise, 0, 0);
VK_IMAGE_FORMAT("r16f") RWTexture3D<float> u_Scalar : register(u0);
VK_IMAGE_FORMAT("rgba8") RWTexture3D<float4> u_Color : register(u1);

float3 Position(uint3 id)
{
    // Texel centres, so trilinear filtering reproduces the noise between them.
    return (float3(id) + 0.5) / float(g_Noise.size) * float(g_Noise.period);
}

[numthreads(4, 4, 4)]
void gradient(uint3 id : SV_DispatchThreadID)
{
    if (any(id >= g_Noise.size))
        return;
    u_Scalar[id] = PeriodicGradientNoise(Position(id), g_Noise.period);
}

[numthreads(4, 4, 4)]
void perlin_worley(uint3 id : SV_DispatchThreadID)
{
    if (any(id >= g_Noise.size))
        return;
    const float3 p = Position(id);
    const uint period = g_Noise.period;
    // Gradient fbm spans about +-0.3; stretch it to use the whole [0, 1].
    const float perlin = saturate(PeriodicGradientFbm(p, period, 4) * 1.4 + 0.5);
    const float worley = PeriodicWorleyFbm(p, period);
    // Perlin-Worley (Schneider 2015): Perlin's shapes, eroded into billows by Worley.
    const float perlinWorley = saturate(Remap(perlin, worley - 1.0, 1.0, 0.0, 1.0));
    u_Color[id] = float4(perlinWorley,
                         PeriodicWorleyFbm(p * 2.0, period * 2u),
                         PeriodicWorleyFbm(p * 4.0, period * 4u),
                         PeriodicWorleyFbm(p * 8.0, period * 8u));
}

[numthreads(4, 4, 4)]
void worley(uint3 id : SV_DispatchThreadID)
{
    if (any(id >= g_Noise.size))
        return;
    const float3 p = Position(id);
    const uint period = g_Noise.period;
    u_Color[id] = float4(PeriodicWorleyFbm(p, period), PeriodicWorleyFbm(p * 2.0, period * 2u),
                         PeriodicWorleyFbm(p * 4.0, period * 4u), 1.0);
}
