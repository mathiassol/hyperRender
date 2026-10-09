#ifndef HYPER_PLANET_ATMOSPHERE_HLSLI
#define HYPER_PLANET_ATMOSPHERE_HLSLI

// A planet's atmosphere in shaders (the atmospheres module): its media, its lookup tables,
// and the in-scattered light along a ray (Hillaire 2020, "A Scalable and Production Ready
// Sky and Atmosphere Rendering Technique": single scattering marched, multiple scattering
// from a table). The tables are built on the CPU (PlanetAtmospheres.cpp) with the same
// parameterisation as below.
//
// Bind the tables as Texture2DArray (one slice per atmosphere) and a clamp sampler, then:
//     float3 transmittance;
//     float3 light = AtmosphereAlongRay(a, origin, dir, tMax, transmittance, ...textures);

#include <hyper/planet_atmosphere_cb.h>

static const float c_AtmoPi = 3.14159265;

// What's in the air at a height: Rayleigh and aerosol scattering, and total extinction.
struct AtmoMedium
{
    float3 rayleigh;
    float3 mieScattering;
    float3 extinction;
};

AtmoMedium AtmosphereMedium(PlanetAtmosphereConstants a, float height)
{
    AtmoMedium m;
    const float h = max(height, 0.0);
    m.rayleigh = a.rayleigh * exp(-h / a.rayleighH);
    m.mieScattering = a.mieScattering * exp(-h / a.mieH);
    const float3 mieExt = a.mieExtinction * exp(-h / a.mieH);
    const float ozone = a.ozoneWidth > 0.0 ? max(0.0, 1.0 - abs(h - a.ozonePeak) / (0.5 * a.ozoneWidth)) : 0.0;
    m.extinction = m.rayleigh + mieExt + a.absorption * ozone;
    return m;
}

// The transmittance table's coordinates (Bruneton 2017): the distance to the top along mu,
// between its least and greatest, and the height as rho / H.
float2 TransmittanceUv(PlanetAtmosphereConstants a, float r, float mu)
{
    const float H = sqrt(max(a.top * a.top - a.bottom * a.bottom, 0.0));
    const float rho = sqrt(max(r * r - a.bottom * a.bottom, 0.0));
    const float disc = r * r * (mu * mu - 1.0) + a.top * a.top;
    const float d = max(-r * mu + sqrt(max(disc, 0.0)), 0.0);
    const float dMin = a.top - r, dMax = rho + H;
    const float xMu = (d - dMin) / max(dMax - dMin, 1e-3);
    const float xR = rho / max(H, 1e-3);
    // Texel centres at the table's edges.
    return float2(0.5 / ATMO_TRANSMITTANCE_W + xMu * (1.0 - 1.0 / ATMO_TRANSMITTANCE_W),
                  0.5 / ATMO_TRANSMITTANCE_H + xR * (1.0 - 1.0 / ATMO_TRANSMITTANCE_H));
}

float2 MultiScatterUv(PlanetAtmosphereConstants a, float r, float muS)
{
    const float x = saturate(muS * 0.5 + 0.5), y = saturate((r - a.bottom) / (a.top - a.bottom));
    return float2(0.5 / ATMO_MULTISCATTER + x * (1.0 - 1.0 / ATMO_MULTISCATTER), 0.5 / ATMO_MULTISCATTER + y * (1.0 - 1.0 / ATMO_MULTISCATTER));
}

float2 IrradianceUv(PlanetAtmosphereConstants a, float r, float muS)
{
    const float x = saturate(muS * 0.5 + 0.5), y = saturate((r - a.bottom) / (a.top - a.bottom));
    return float2(0.5 / ATMO_IRRADIANCE_W + x * (1.0 - 1.0 / ATMO_IRRADIANCE_W), 0.5 / ATMO_IRRADIANCE_H + y * (1.0 - 1.0 / ATMO_IRRADIANCE_H));
}

// Whether a ray from radius r at cosine mu (from the zenith) meets the ground.
bool RayHitsGround(PlanetAtmosphereConstants a, float r, float mu)
{
    return mu < 0.0 && r * r * (mu * mu - 1.0) + a.bottom * a.bottom >= 0.0;
}

// Sunlight's transmittance to a point (r, cosine of the Sun's zenith angle there): 0 behind the ground.
float3 SunTransmittance(PlanetAtmosphereConstants a, float r, float muS, Texture2DArray<float4> transmittance, SamplerState s)
{
    if (RayHitsGround(a, r, muS))
        return 0.0;
    return transmittance.SampleLevel(s, float3(TransmittanceUv(a, r, muS), a.slice), 0).rgb;
}

float3 SkyIrradiance(PlanetAtmosphereConstants a, float r, float muS, Texture2DArray<float4> irradiance, SamplerState s)
{
    return irradiance.SampleLevel(s, float3(IrradianceUv(a, r, muS), a.slice), 0).rgb;
}

// How much of the star's disk a point in the air sees past the occluder (an eclipse): the
// overlap of two disks, angular radii s and o, centres theta apart.
float AtmoUncovered(float s, float o, float theta)
{
    if (theta >= s + o)
        return 1.0;
    if (theta <= o - s)
        return 0.0;
    if (theta <= s - o)
        return 1.0 - (o * o) / (s * s);
    const float a = acos(clamp((theta * theta + s * s - o * o) / (2.0 * theta * s), -1.0, 1.0));
    const float b = acos(clamp((theta * theta + o * o - s * s) / (2.0 * theta * o), -1.0, 1.0));
    const float overlap = s * s * (a - 0.5 * sin(2.0 * a)) + o * o * (b - 0.5 * sin(2.0 * b));
    return saturate(1.0 - overlap / (c_AtmoPi * s * s));
}

float StarVisible(PlanetAtmosphereConstants a, float3 p)
{
    if (a.occluderRadius <= 0.0 || a.sunRadius <= 0.0)
        return 1.0;
    const float3 toSun = a.sunCenter - p, toOcc = a.occluderCenter - p;
    const float ds = length(toSun), dO = length(toOcc);
    if (dO >= ds || dot(toSun, toOcc) <= 0.0)
        return 1.0;
    const float s = asin(saturate(a.sunRadius / ds)), o = asin(saturate(a.occluderRadius / dO));
    // The angle between them from the chord: acos of a dot product near 1 is too coarse in floats.
    return AtmoUncovered(max(s, 1e-6), o, 2.0 * asin(saturate(0.5 * length(toSun / ds - toOcc / dO))));
}

float RayleighPhase(float c) { return 3.0 / (16.0 * c_AtmoPi) * (1.0 + c * c); }

// Cornette-Shanks: Henyey-Greenstein made to match Mie better at back angles.
float3 MiePhase(float3 g, float c)
{
    const float3 g2 = g * g;
    return 3.0 / (8.0 * c_AtmoPi) * (1.0 - g2) * (1.0 + c * c) / ((2.0 + g2) * pow(max(1.0 + g2 - 2.0 * g * c, 1e-4), 1.5));
}

// The ray's stretch inside the shell [tEnter, tExit] (false: it misses).
bool AtmosphereSpan(PlanetAtmosphereConstants a, float3 origin, float3 dir, out float tEnter, out float tExit)
{
    const float3 oc = origin - a.center;
    const float b = dot(oc, dir);
    const float3 q = oc - b * dir;
    const float h2 = a.top * a.top - dot(q, q);
    tEnter = 0.0;
    tExit = 0.0;
    if (h2 < 0.0)
        return false;
    const float h = sqrt(h2);
    tEnter = max(-b - h, 0.0);
    tExit = -b + h;
    return tExit > 0.0;
}

// Where the ray first meets the ground, or a big number.
float GroundDistance(PlanetAtmosphereConstants a, float3 origin, float3 dir)
{
    const float3 oc = origin - a.center;
    const float b = dot(oc, dir);
    const float3 q = oc - b * dir;
    const float h2 = a.bottom * a.bottom - dot(q, q);
    if (h2 < 0.0)
        return 3.0e38;
    const float t = -b - sqrt(h2);
    return t > 0.0 ? t : 3.0e38;
}

// The light the air scatters towards the origin along dir, up to tMax (a surface, or
// infinity), lit by the planet's star; and the ray's transmittance. Exposed (the star's
// illuminance is).
// The aerosol's phase function at a cosine: its Mie table (theta = pi u^2), else Cornette-Shanks.
float3 AerosolPhase(PlanetAtmosphereConstants a, float c, Texture2DArray<float4> phase, SamplerState s)
{
    if (a.phaseTable < 0.5)
        return MiePhase(a.mieG, c);
    // The angle from the chord to the forward direction: acos is coarse near 1.
    const float theta = 2.0 * asin(saturate(0.5 * sqrt(max(2.0 - 2.0 * c, 0.0))));
    const float u = sqrt(theta / c_AtmoPi);
    return phase.SampleLevel(s, float3((0.5 + u * (ATMO_PHASE - 1)) / ATMO_PHASE, 0.5, a.slice), 0).rgb;
}

float3 AtmosphereAlongRay(PlanetAtmosphereConstants a, float3 origin, float3 dir, float tMax, uint steps,
                          out float3 transmittance, Texture2DArray<float4> tTransmittance,
                          Texture2DArray<float4> tMultiScatter, Texture2DArray<float4> tPhase, SamplerState s)
{
    transmittance = 1.0;
    float tEnter, tExit;
    if (!AtmosphereSpan(a, origin, dir, tEnter, tExit))
        return 0.0;
    tExit = min(tExit, min(tMax, GroundDistance(a, origin, dir)));
    if (tExit <= tEnter)
        return 0.0;
    float3 light = 0.0;
    const float dt = (tExit - tEnter) / float(steps);
    for (uint i = 0; i < steps; ++i)
    {
        const float t = tEnter + (float(i) + 0.5) * dt;
        const float3 p = origin + dir * t - a.center;
        const float r = length(p);
        const float3 up = p / r;
        const float3 towardsSun = normalize(a.sunCenter - (origin + dir * t));
        const float muS = dot(up, towardsSun);
        const float c = dot(dir, towardsSun);
        const AtmoMedium m = AtmosphereMedium(a, r - a.bottom);
        // In an eclipse's shadow the air is lit by what's left of the star's disk (and its
        // multiple scattering, nearly: the shadow is narrow beside the air it's lit from).
        const float eclipse = StarVisible(a, origin + dir * t);
        const float3 sun = SunTransmittance(a, r, muS, tTransmittance, s) * eclipse;
        const float3 multi = tMultiScatter.SampleLevel(s, float3(MultiScatterUv(a, r, muS), a.slice), 0).rgb * eclipse;
        const float3 scattered = (m.rayleigh * RayleighPhase(c) + m.mieScattering * AerosolPhase(a, c, tPhase, s)) * sun
                               + (m.rayleigh + m.mieScattering) * multi;
        // Integrated analytically over the step (Hillaire's energy-conserving form).
        const float3 stepT = exp(-m.extinction * dt);
        light += transmittance * scattered * (1.0 - stepT) / max(m.extinction, 1e-12);
        transmittance *= stepT;
    }
    return light * a.sunIlluminance;
}

#endif
