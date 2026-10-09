#ifndef HYPER_SAMPLING_HLSLI
#define HYPER_SAMPLING_HLSLI

// Random numbers and direction sampling for Monte Carlo passes (traced lighting,
// the reference path tracer, apps' own).

#include <hyper/common.hlsli>

// PCG-style integer hash (Jarzynski & Olano 2020).
uint PcgHash(uint v)
{
    const uint state = v * 747796405u + 2891336453u;
    const uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

struct Rng
{
    uint state;
    float Next()
    {
        state = state * 747796405u + 2891336453u;
        uint w = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
        return float((w >> 22u) ^ w) / 4294967296.0;
    }
};

Rng MakeRng(uint2 pixel, uint frame, uint stream)
{
    Rng r;
    r.state = (pixel.x * 1973u + pixel.y * 9277u + frame * 26699u + stream * 104729u) | 1u;
    r.Next();
    return r;
}

// Decorrelated across pixels and samples, for accumulating thousands of them.
Rng MakeRngHashed(uint2 pixel, uint sampleIndex)
{
    Rng r;
    r.state = PcgHash(PcgHash(pixel.x + (pixel.y << 16)) ^ PcgHash(sampleIndex + 0x9E3779B9u));
    return r;
}

// Orthonormal tangent frame around n (Duff et al. 2017).
void Basis(float3 n, out float3 t, out float3 b)
{
    const float s = n.z >= 0.0 ? 1.0 : -1.0;
    const float a = -1.0 / (s + n.z);
    const float c = n.x * n.y * a;
    t = float3(1.0 + s * n.x * n.x * a, s * c, -s * n.x);
    b = float3(c, s + n.y * n.y * a, -n.y);
}

float3 SampleCosine(float3 n, inout Rng rng)
{
    const float u = rng.Next(), v = rng.Next();
    const float r = sqrt(u), phi = HYPER_TWO_PI * v;
    float3 t, b;
    Basis(n, t, b);
    return normalize(t * (r * cos(phi)) + b * (r * sin(phi)) + n * sqrt(max(0.0, 1.0 - u)));
}

// Uniform over the cone of directions within acos(cosMax) of axis.
float3 SampleCone(float3 axis, float cosMax, float2 u)
{
    const float cosTheta = 1.0 - u.x * (1.0 - cosMax);
    const float sinTheta = sqrt(saturate(1.0 - cosTheta * cosTheta));
    const float phi = HYPER_TWO_PI * u.y;
    float3 t, b;
    Basis(axis, t, b);
    return normalize(t * (sinTheta * cos(phi)) + b * (sinTheta * sin(phi)) + axis * cosTheta);
}

// Heitz 2018, visible normals; v in tangent space (z = normal).
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

// A GGX half vector around n, from its visible normals, in world space.
float3 SampleGgxHalfVector(float3 n, float3 toViewer, float alpha, float2 u)
{
    float3 t, b;
    Basis(n, t, b);
    const float3 vLocal = float3(dot(toViewer, t), dot(toViewer, b), dot(toViewer, n));
    const float3 hLocal = SampleGgxVndf(vLocal, alpha, u);
    return normalize(hLocal.x * t + hLocal.y * b + hLocal.z * n);
}

float3 SampleSpecular(float3 n, float3 toViewer, float roughness, inout Rng rng)
{
    const float alpha = max(roughness * roughness, 1e-3);
    const float3 h = SampleGgxHalfVector(n, toViewer, alpha, float2(rng.Next(), rng.Next()));
    float3 r = reflect(-toViewer, h);
    if (dot(r, n) <= 0.0)
        r = reflect(-toViewer, n);
    return r;
}

// Karis' fit of the split-sum environment BRDF.
float3 EnvBrdf(float3 f0, float roughness, float nDotV)
{
    const float4 c0 = float4(-1.0, -0.0275, -0.572, 0.022);
    const float4 c1 = float4(1.0, 0.0425, 1.04, -0.04);
    const float4 k = roughness * c0 + c1;
    const float a004 = min(k.x * k.x, exp2(-9.28 * nDotV)) * k.x + k.y;
    const float2 ab = float2(-1.04, 1.04) * a004 + k.zw;
    return f0 * ab.x + ab.y;
}

#endif
