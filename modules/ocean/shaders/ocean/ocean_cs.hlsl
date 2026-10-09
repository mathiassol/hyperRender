// The ocean's simulation, per cascade (the z of every dispatch):
//   evolve    the spectrum at this time, as the eight fields the surface needs, packed two
//             real fields to a complex one
//   fft       an inverse FFT of OCEAN_N along rows, then along columns, in groupshared memory
//   assemble  displacement, slopes and foam from the transformed fields
//   mip       one mip level of displacement and slopes from the one above

#include <hyper/common.hlsli>
#include <hyper/ocean_cb.h>

cbuffer c_Sim : register(b0) { OceanSimConstants c_Sim; }

struct OceanPush
{
    uint columns;     // fft: 0 rows, 1 columns
    uint size;        // mip: the destination's size
};
DECLARE_PUSH_CONSTANTS(OceanPush, g_Push, 1, 0);

Texture2DArray<float4> t_Spectrum : register(t0);
Texture2DArray<float4> t_PrevDisplacement : register(t1);
Texture2DArray<float4> t_MipDisplacement : register(t2);
Texture2DArray<float4> t_MipSlopes : register(t3);
VK_IMAGE_FORMAT("rgba32f") RWTexture2DArray<float4> u_FftA : register(u0);
VK_IMAGE_FORMAT("rgba32f") RWTexture2DArray<float4> u_FftB : register(u1);
VK_IMAGE_FORMAT("rgba16f") RWTexture2DArray<float4> u_Displacement : register(u2);
VK_IMAGE_FORMAT("rgba16f") RWTexture2DArray<float4> u_Slopes : register(u3);

static const float c_Gravity = 9.81;

float2 CMul(float2 a, float2 b) { return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x); }
float2 TimesI(float2 a) { return float2(-a.y, a.x); }

// h(k, t) = h0(k) e^(i w t) + conj(h0(-k)) e^(-i w t), then (with x' = x + choppiness D):
//   Dy = h               Dx = i kx/k h         Dz = i kz/k h
//   dDy/dx = i kx h      dDy/dz = i kz h
//   dDx/dx = -kx^2/k h   dDz/dz = -kz^2/k h    dDx/dz = -kx kz/k h
// Each pair of real fields goes into one complex one, A + iB, whose transform is a + ib.
[numthreads(8, 8, 1)]
void evolve(uint3 id : SV_DispatchThreadID)
{
    const float2 n = float2(id.xy) - OCEAN_N / 2;
    const float2 k = n * (HYPER_TWO_PI / c_Sim.patchSize[id.z]);
    const float kl = length(k);
    const float4 h0 = t_Spectrum[id];

    float s, c;
    sincos(sqrt(c_Gravity * kl) * c_Sim.time, s, c);
    const float2 h = CMul(h0.xy, float2(c, s)) + CMul(h0.zw, float2(c, -s));
    const float2 ih = TimesI(h);
    const float2 kn = kl > 1e-6 ? k / kl : 0.0;
    const float invK = kl > 1e-6 ? 1.0 / kl : 0.0;

    const float2 dx = ih * kn.x, dy = h, dz = ih * kn.y;
    const float2 sx = ih * k.x, sz = ih * k.y;
    const float2 jxx = -h * (k.x * k.x * invK), jzz = -h * (k.y * k.y * invK), jxz = -h * (k.x * k.y * invK);

    u_FftA[id] = float4(dx + TimesI(dy), dz + TimesI(sx));
    u_FftB[id] = float4(sz + TimesI(jxx), jzz + TimesI(jxz));
}

// Stockham radix-2 (Bainville's formulation), both textures' two complex values at once.
groupshared float4 g_A[2][OCEAN_N];
groupshared float4 g_B[2][OCEAN_N];

[numthreads(OCEAN_N / 2, 1, 1)]
void fft(uint3 group : SV_GroupID, uint t : SV_GroupIndex)
{
    const uint half = OCEAN_N / 2;
    const uint3 p0 = g_Push.columns ? uint3(group.x, t, group.z) : uint3(t, group.x, group.z);
    const uint3 p1 = g_Push.columns ? uint3(group.x, t + half, group.z) : uint3(t + half, group.x, group.z);
    g_A[0][t] = u_FftA[p0];
    g_A[0][t + half] = u_FftA[p1];
    g_B[0][t] = u_FftB[p0];
    g_B[0][t + half] = u_FftB[p1];
    GroupMemoryBarrierWithGroupSync();

    uint src = 0;
    [unroll]
    for (uint stage = 0; stage < OCEAN_LOG_N; ++stage)
    {
        const uint p = 1u << stage;
        const uint k = t & (p - 1);
        float s, c;
        sincos(HYPER_PI * float(k) / float(p), s, c);   // +: the inverse transform
        const float2 w = float2(c, s);
        const float4 a0 = g_A[src][t], b0 = g_B[src][t];
        float4 a1 = g_A[src][t + half], b1 = g_B[src][t + half];
        a1 = float4(CMul(a1.xy, w), CMul(a1.zw, w));
        b1 = float4(CMul(b1.xy, w), CMul(b1.zw, w));
        const uint j = (t << 1) - k;
        g_A[src ^ 1][j] = a0 + a1;
        g_A[src ^ 1][j + p] = a0 - a1;
        g_B[src ^ 1][j] = b0 + b1;
        g_B[src ^ 1][j + p] = b0 - b1;
        src ^= 1;
        GroupMemoryBarrierWithGroupSync();
    }

    u_FftA[p0] = g_A[src][t];
    u_FftA[p1] = g_A[src][t + half];
    u_FftB[p0] = g_B[src][t];
    u_FftB[p1] = g_B[src][t + half];
}

[numthreads(8, 8, 1)]
void assemble(uint3 id : SV_DispatchThreadID)
{
    // The spectrum is stored centred (k = 0 in the middle), which multiplies the
    // transform by (-1)^(x + y).
    const float sign = ((id.x + id.y) & 1) ? -1.0 : 1.0;
    const float4 a = u_FftA[id] * sign;
    const float4 b = u_FftB[id] * sign;
    const float lambda = c_Sim.choppiness;
    const float3 displacement = float3(a.x * lambda, a.y, a.z * lambda);
    const float2 slope = float2(a.w, b.x);
    const float jxx = b.y * lambda, jzz = b.z * lambda, jxz = b.w * lambda;

    // Where the Jacobian of x -> x + D drops towards 0 the surface folds over: foam,
    // which then fades.
    const float jacobian = (1.0 + jxx) * (1.0 + jzz) - jxz * jxz;
    const float made = saturate((c_Sim.foamThreshold - jacobian) * c_Sim.foamGain);
    const float foam = max(t_PrevDisplacement[id].w * c_Sim.foamKeep, made);

    u_Displacement[id] = float4(displacement, foam);
    u_Slopes[id] = float4(slope, slope * slope);
}

[numthreads(8, 8, 1)]
void mip(uint3 id : SV_DispatchThreadID)
{
    if (any(id.xy >= g_Push.size))
        return;
    const uint3 s = uint3(id.xy * 2, id.z);
    u_Displacement[id] = 0.25 * (t_MipDisplacement[s] + t_MipDisplacement[s + uint3(1, 0, 0)]
                               + t_MipDisplacement[s + uint3(0, 1, 0)] + t_MipDisplacement[s + uint3(1, 1, 0)]);
    u_Slopes[id] = 0.25 * (t_MipSlopes[s] + t_MipSlopes[s + uint3(1, 0, 0)]
                         + t_MipSlopes[s + uint3(0, 1, 0)] + t_MipSlopes[s + uint3(1, 1, 0)]);
}
