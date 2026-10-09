#ifndef HYPER_PLANET_ATMOSPHERE_CB_H
#define HYPER_PLANET_ATMOSPHERE_CB_H

// One planet's atmosphere, for shaders (hyper/planet_atmosphere.hlsli): its shell, its
// scattering and absorption, and its lookup tables' slice. Shared by PlanetAtmospheres.cpp
// (after <hyper/shared.h>) and the shaders. Lengths in metres, coefficients per metre, per
// channel (612, 549, 465 nm), positions camera-relative.

#define PLANET_ATMOSPHERES 4            // at most this many at once (the tables' slices)
#define ATMO_TRANSMITTANCE_W 256        // transmittance: (mu, r)
#define ATMO_TRANSMITTANCE_H 64
#define ATMO_MULTISCATTER 32            // multiple scattering: (mu_s, r)
#define ATMO_IRRADIANCE_W 64            // sky irradiance on the ground: (mu_s, r)
#define ATMO_IRRADIANCE_H 16
#define ATMO_PHASE 128                  // the aerosol's phase function: angle theta = pi (u)^2, u = 0..1

struct PlanetAtmosphereConstants
{
    float3 center;           // the planet's centre, camera-relative
    float bottom;            // the ground's radius
    float3 rayleigh;         // Rayleigh scattering at the ground
    float top;               // the atmosphere's top radius
    float3 mieScattering;    // aerosol scattering at the ground
    float rayleighH;         // scale heights
    float3 mieExtinction;
    float mieH;
    float3 mieG;             // aerosol asymmetry, per channel
    float ozonePeak;         // altitude of the absorbing layer's peak
    float3 absorption;       // ozone (or another absorber) at its peak
    float ozoneWidth;        // the tent's full width
    float3 groundAlbedo;
    float slice;             // its lookup tables' slice
    float3 sunIlluminance;   // exposed lux x colour of its star, at the planet
    float sunRadius;         // the star's radius
    float3 sunCenter;        // its star, camera-relative
    float occluderRadius;    // > 0: a body that can eclipse the star (a moon), so the air in its shadow goes dark
    float3 occluderCenter;   // camera-relative
    float phaseTable;        // 1: the aerosol's phase function is its table (Mie theory's), else Cornette-Shanks from mieG
};

#endif
