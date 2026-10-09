#ifndef HYPER_VOLUME_HLSLI
#define HYPER_VOLUME_HLSLI

// Participating-media helpers for hyper's fog, volumes and clouds.

#include <hyper/common.hlsli>

// Henyey-Greenstein phase function; g in (-1, 1), cosTheta between the light's
// travel direction reversed (towards the light) and the view ray reversed.
float HenyeyGreenstein(float cosTheta, float g)
{
    const float g2 = g * g;
    return (1.0 - g2) / (4.0 * HYPER_PI * pow(max(1.0 + g2 - 2.0 * g * cosTheta, 1e-4), 1.5));
}

// Two lobes: a strong forward one and a weaker backward one (clouds: silver lining + glow).
float DualLobeHenyeyGreenstein(float cosTheta, float gForward, float gBack, float blend)
{
    return lerp(HenyeyGreenstein(cosTheta, gForward), HenyeyGreenstein(cosTheta, -gBack), blend);
}

// Beer-Lambert with Schneider's "powder" darkening of thin edges facing the light.
float BeerPowder(float opticalDepth, float powder)
{
    return exp(-opticalDepth) * lerp(1.0, 1.0 - exp(-2.0 * opticalDepth), powder);
}

// Energy-conserving integration of in-scattering S over a segment of length dt
// with extinction sigmaT (Hillaire 2015): what the segment adds, before the
// transmittance up to it is applied.
float3 IntegrateScattering(float3 scattering, float sigmaT, float dt)
{
    const float t = exp(-sigmaT * dt);
    return sigmaT > 1e-7 ? scattering * (1.0 - t) / sigmaT : scattering * dt;
}

// Distance along a ray to an axis-aligned box: .x entry (clamped to 0), .y exit; x > y means no hit.
float2 RayBox(float3 origin, float3 dir, float3 boxMin, float3 boxMax)
{
    const float3 inv = 1.0 / select(abs(dir) > 1e-8, dir, 1e-8);
    const float3 t0 = (boxMin - origin) * inv;
    const float3 t1 = (boxMax - origin) * inv;
    const float3 lo = min(t0, t1), hi = max(t0, t1);
    return float2(max(max(lo.x, max(lo.y, lo.z)), 0.0), min(hi.x, min(hi.y, hi.z)));
}

#endif
