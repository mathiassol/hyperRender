// Energy-conserving mip-chain bloom (Jimenez, "Next Generation Post Processing
// in Call of Duty: Advanced Warfare", SIGGRAPH 2014).
//   downsample: 13-tap filter, Karis average on the first level to kill fireflies
//   upsample:   3x3 tent, accumulated into the next larger level

#include <hyper/common.hlsli>

struct BloomConstants
{
    float2 srcTexelSize;
    float2 dstSize;
    float filterRadius;
    float3 pad;
};

DECLARE_PUSH_CONSTANTS(BloomConstants, g_Bloom, 0, 0);
Texture2D<float4> t_Source : register(t0);
VK_IMAGE_FORMAT("rgba16f") RWTexture2D<float4> u_Dest : register(u0);
SamplerState s_Linear : register(s0);

float3 Tap(float2 uv, float2 offset)
{
    return t_Source.SampleLevel(s_Linear, uv + offset * g_Bloom.srcTexelSize, 0).rgb;
}

float KarisWeight(float3 c)
{
    return 1.0 / (1.0 + dot(c, float3(0.2126, 0.7152, 0.0722)));
}

[numthreads(8, 8, 1)]
void downsample(uint2 id : SV_DispatchThreadID)
{
    if (any(float2(id) >= g_Bloom.dstSize))
        return;

    float2 uv = (float2(id) + 0.5) / g_Bloom.dstSize;

    float3 a = Tap(uv, float2(-2,  2)), b = Tap(uv, float2(0,  2)), c = Tap(uv, float2(2,  2));
    float3 d = Tap(uv, float2(-1,  1)), e = Tap(uv, float2(1,  1));
    float3 f = Tap(uv, float2(-2,  0)), g = Tap(uv, float2(0,  0)), h = Tap(uv, float2(2,  0));
    float3 i = Tap(uv, float2(-1, -1)), j = Tap(uv, float2(1, -1));
    float3 k = Tap(uv, float2(-2, -2)), l = Tap(uv, float2(0, -2)), m = Tap(uv, float2(2, -2));

#if KARIS
    // Five overlapping 2x2 boxes, each weighted by 1/(1+luma) so single bright pixels can't dominate.
    float3 g0 = (d + e + i + j) * 0.25;
    float3 g1 = (a + b + f + g) * 0.25;
    float3 g2 = (b + c + g + h) * 0.25;
    float3 g3 = (f + g + k + l) * 0.25;
    float3 g4 = (g + h + l + m) * 0.25;
    float w0 = KarisWeight(g0) * 0.5;
    float w1 = KarisWeight(g1) * 0.125;
    float w2 = KarisWeight(g2) * 0.125;
    float w3 = KarisWeight(g3) * 0.125;
    float w4 = KarisWeight(g4) * 0.125;
    float3 result = (g0 * w0 + g1 * w1 + g2 * w2 + g3 * w3 + g4 * w4) / (w0 + w1 + w2 + w3 + w4);
    // NaN/Inf from the scene would spread across the whole bloom chain.
    if (any(isnan(result)) || any(isinf(result)))
        result = 0;
#else
    float3 result = (d + e + i + j) * 0.125
                  + (b + f + h + l) * 0.0625
                  + (a + c + k + m) * 0.03125
                  + g * 0.125;
#endif

    u_Dest[id] = float4(max(result, 0.0), 1.0);
}

[numthreads(8, 8, 1)]
void upsample(uint2 id : SV_DispatchThreadID)
{
    if (any(float2(id) >= g_Bloom.dstSize))
        return;

    float2 uv = (float2(id) + 0.5) / g_Bloom.dstSize;
    float r = g_Bloom.filterRadius;

    float3 up = Tap(uv, float2(0, 0)) * 4.0
              + (Tap(uv, float2(0, r)) + Tap(uv, float2(-r, 0)) + Tap(uv, float2(r, 0)) + Tap(uv, float2(0, -r))) * 2.0
              + (Tap(uv, float2(-r, r)) + Tap(uv, float2(r, r)) + Tap(uv, float2(-r, -r)) + Tap(uv, float2(r, -r)));

    u_Dest[id] = float4(u_Dest[id].rgb + up / 16.0, 1.0);
}
