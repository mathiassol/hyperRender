#ifndef HYPER_PLANET_TILE_HLSLI
#define HYPER_PLANET_TILE_HLSLI

// A point of a tile, as an offset from the tile's centre that floats can hold exactly.
//
// The cube point is c0 + d (c0 the tile's centre on the face, d the step across it); the
// sphere point is (c0 + d) / |c0 + d|. Far from the centre of the planet both are ~1, and
// their difference from the centre's direction would cancel, so it's written in the small
// quantities alone:
//     dir - dir0 = d / L + c0 (L0 - L) / (L L0),  L0 - L = -(2 c0.d + d.d) / (L0 + L).
// The app places the tile's centre (radius x dir0) from doubles; a vertex is that plus
// (R + h) (dir - dir0) + h dir0, small numbers all of them.

#include <hyper/planet_tiles_cb.h>

struct TilePoint
{
    float3 dir0;     // the tile centre's direction
    float3 delta;    // this point's direction minus it
};

// ij: the vertex (0..TERRAIN_GRID-1, or a step beyond for neighbours).
TilePoint TileDirection(uint face, float2 centre, float halfSize, float2 ij)
{
    const float3 n = terrain_faceNormal(face), u = terrain_faceU(face), v = terrain_faceV(face);
    const float3 c0 = n + u * centre.x + v * centre.y;
    const float2 step = (ij * (2.0 / float(TERRAIN_GRID - 1)) - 1.0) * halfSize;
    const float3 d = u * step.x + v * step.y;
    const float l0 = length(c0), l = length(c0 + d);
    TilePoint t;
    t.dir0 = c0 / l0;
    const float shorter = -(2.0 * dot(c0, d) + dot(d, d)) / (l0 + l);   // L0 - L
    t.delta = d / l + c0 * (shorter / (l * l0));
    return t;
}

// The point on the ground, relative to radius x dir0, in the planet's axes.
float3 TileLocal(TilePoint t, float radius, float height)
{
    return t.delta * (radius + height) + t.dir0 * height;
}

uint TerrainPackNormal(float3 n)
{
    // Octahedral.
    float2 p = n.xy / (abs(n.x) + abs(n.y) + abs(n.z));
    if (n.z < 0.0)
        p = (1.0 - abs(p.yx)) * float2(p.x >= 0.0 ? 1.0 : -1.0, p.y >= 0.0 ? 1.0 : -1.0);
    const uint2 q = uint2(round(saturate(p * 0.5 + 0.5) * 65535.0));
    return q.x | (q.y << 16);
}

float3 TerrainUnpackNormal(uint packed)
{
    const float2 p = float2(packed & 0xffffu, packed >> 16) / 65535.0 * 2.0 - 1.0;
    float3 n = float3(p, 1.0 - abs(p.x) - abs(p.y));
    if (n.z < 0.0)
        n.xy = (1.0 - abs(n.yx)) * float2(n.x >= 0.0 ? 1.0 : -1.0, n.y >= 0.0 ? 1.0 : -1.0);
    return normalize(n);
}

// Albedo, square-root encoded (more steps in the dark), and the flags.
uint TerrainPackAlbedo(float3 albedo, uint flags)
{
    const uint3 q = uint3(round(sqrt(saturate(albedo)) * 255.0));
    return q.x | (q.y << 8) | (q.z << 16) | (flags << 24);
}

float3 TerrainUnpackAlbedo(uint packed)
{
    const float3 s = float3(packed & 0xffu, (packed >> 8) & 0xffu, (packed >> 16) & 0xffu) / 255.0;
    return s * s;
}

#endif
