#ifndef HYPER_NOISE_HLSLI
#define HYPER_NOISE_HLSLI

// Hashes and noise, shared by hyper's volumetrics, the noise baker
// (hyper::NoiseVolume) and apps (the black hole's disk).
//
//   Pcg3d / Hash1 / Hash3        integer hashes (Jarzynski & Olano 2020)
//   GradientNoise                Perlin-style gradient noise, about [-1, 1]
//   PeriodicGradientNoise        the same, tiling every `period` lattice cells
//   PeriodicWorley               1 - nearest-feature distance (cellular), tiling
//   InterleavedGradientNoise     per-pixel dither for jittered marches (Jimenez 2014)

uint3 Pcg3d(uint3 v)
{
    v = v * 1664525u + 1013904223u;
    v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
    v ^= v >> 16u;
    v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
    return v;
}

float Hash1(uint3 v)
{
    return float(Pcg3d(v).x) / 4294967295.0;
}

float3 Hash3(uint3 v)
{
    return float3(Pcg3d(v)) / 4294967295.0;
}

float3 Gradient(int3 c)
{
    return float3(Pcg3d(uint3(c))) * (2.0 / 4294967295.0) - 1.0;
}

float GradientNoiseLattice(float3 p, uint period, bool periodic)
{
    const int3 i = int3(floor(p));
    const float3 f = p - floor(p);
    const float3 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    float n[8];
    [unroll]
    for (uint k = 0; k < 8; ++k)
    {
        const int3 o = int3(k, k >> 1, k >> 2) & 1;
        const int3 c = periodic ? int3(uint3(i + o) % period) : i + o;
        n[k] = dot(Gradient(c), f - float3(o));
    }
    const float nx00 = lerp(n[0], n[1], u.x);
    const float nx10 = lerp(n[2], n[3], u.x);
    const float nx01 = lerp(n[4], n[5], u.x);
    const float nx11 = lerp(n[6], n[7], u.x);
    return lerp(lerp(nx00, nx10, u.y), lerp(nx01, nx11, u.y), u.z);
}

float GradientNoise(float3 p)
{
    return GradientNoiseLattice(p, 1u, false);
}

// p in lattice cells; the noise repeats every `period` cells on each axis.
float PeriodicGradientNoise(float3 p, uint period)
{
    return GradientNoiseLattice(p, period, true);
}

// Cellular noise, one feature point per cell: 1 at a feature, falling to 0 about
// a cell away. p in cells; repeats every `period` cells.
float PeriodicWorley(float3 p, uint period)
{
    const int3 cell = int3(floor(p));
    const float3 f = p - floor(p);
    float nearest = 1e9;
    [unroll]
    for (int z = -1; z <= 1; ++z)
    [unroll]
    for (int y = -1; y <= 1; ++y)
    [unroll]
    for (int x = -1; x <= 1; ++x)
    {
        const int3 o = int3(x, y, z);
        const uint3 c = uint3(cell + o + int3(period, period, period)) % period;
        const float3 feature = float3(o) + Hash3(c * 1u + 0x2545F491u);
        const float3 d = feature - f;
        nearest = min(nearest, dot(d, d));
    }
    return 1.0 - saturate(sqrt(nearest));
}

// Fractal sums with doubling periods, so they tile like their first octave.
float PeriodicGradientFbm(float3 p, uint period, uint octaves)
{
    float sum = 0.0, amplitude = 0.5, norm = 0.0;
    for (uint o = 0; o < octaves; ++o)
    {
        sum += amplitude * PeriodicGradientNoise(p, period);
        norm += amplitude;
        p *= 2.0;
        period *= 2u;
        amplitude *= 0.5;
    }
    return sum / norm;
}

float PeriodicWorleyFbm(float3 p, uint period)
{
    return PeriodicWorley(p, period) * 0.625 + PeriodicWorley(p * 2.0, period * 2u) * 0.25
         + PeriodicWorley(p * 4.0, period * 4u) * 0.125;
}

float Remap(float v, float lo, float hi, float newLo, float newHi)
{
    return newLo + (v - lo) / max(hi - lo, 1e-6) * (newHi - newLo);
}

// [0, 1) dither that varies per pixel and frame; use as a march's start offset.
float InterleavedGradientNoise(uint2 pixel, uint frame)
{
    const float2 p = float2(pixel) + 5.588238 * float(frame % 64u);
    return frac(52.9829189 * frac(dot(p, float2(0.06711056, 0.00583715))));
}

#endif
