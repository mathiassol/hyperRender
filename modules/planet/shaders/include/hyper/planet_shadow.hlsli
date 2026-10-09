#ifndef HYPER_PLANET_SHADOW_HLSLI
#define HYPER_PLANET_SHADOW_HLSLI

// The ground's own shadows (the planet module): how much of a star's disk a point on a
// planet sees past the terrain between them. A march towards the star through the shared
// height function, coarsened to wavelengths over a kilometre, at distances growing by 70% a
// step from 30 m to about 150 km; the lowest the line to the star passes over the ground, as
// an angle against the star's angular radius, gives the penumbra. The point's own coarse
// height is its zero, so the coarse ground never shadows the fine ground it approximates.
//
// The includer defines TERRAIN_MAP_LOAD (see planet_terrain.h) when the body has a map.

#include <hyper/planet_terrain.h>

// p: the point, toStar: the unit direction to the star, both in the planet's axes (metres
// from its centre); starRadius: the star's angular radius (radians).
float TerrainShadow(PlanetTerrainConstants c, float3 p, float3 toStar, float starRadius)
{
    PlanetTerrainConstants coarse = c;
    coarse.minWavelength = max(c.minWavelength, 1000.0);
    coarse.patchSpacing = 500.0;
    const float r0 = length(p);
    float own = TerrainHeight(coarse, p / r0);
    if (c.ocean > 0.0)
        own = max(own, 0.0);
    const float lift = (r0 - c.radius) - own;   // how far the fine ground is above the coarse here
    float lowest = 1.0;                          // the line's least clearance, as an angle
    float t = 30.0;
    for (int k = 0; k < 16; ++k, t *= 1.7)
    {
        const float3 q = p + toStar * t;
        const float rq = length(q);
        float h = TerrainHeight(coarse, q / rq);
        if (c.ocean > 0.0)
            h = max(h, 0.0);
        const float clearance = (rq - c.radius - h - lift) / t;
        lowest = min(lowest, clearance);
        if (rq - c.radius > h + 2.0 * (rq - r0) + 10000.0)
            break;   // far above anything that could be in the way
    }
    return smoothstep(-starRadius, starRadius, lowest);
}

#endif
