#ifndef HYPER_PLANET_TILES_CB_H
#define HYPER_PLANET_TILES_CB_H

// The planet module's GPU records (C++ after <hyper/shared.h>, and HLSL). A tile is a
// square of a cube face (its centre in the face's coordinates, -1..1, and half its size),
// pushed out onto the sphere: TERRAIN_GRID x TERRAIN_GRID vertices, each a float4 in the
// pool: height (m, the sea's surface where it's water), the normal (octahedral, 2 x 16
// bits, planet axes), the albedo (sqrt-encoded 8 bits a channel) and flags, the water's
// depth (m, signed: negative on land).

#include <hyper/planet_terrain.h>

#define TERRAIN_MAX_DRAWS 4096

// The cube's faces: each one's outward normal and the two axes across it.
TERRAIN_FN float3 terrain_faceNormal(uint f)
{
    return f == 0u ? float3(1.0f, 0.0f, 0.0f) : f == 1u ? float3(-1.0f, 0.0f, 0.0f) : f == 2u ? float3(0.0f, 1.0f, 0.0f)
         : f == 3u ? float3(0.0f, -1.0f, 0.0f) : f == 4u ? float3(0.0f, 0.0f, 1.0f) : float3(0.0f, 0.0f, -1.0f);
}
TERRAIN_FN float3 terrain_faceU(uint f)
{
    return f == 0u ? float3(0.0f, 1.0f, 0.0f) : f == 1u ? float3(0.0f, 0.0f, 1.0f) : f == 2u ? float3(0.0f, 0.0f, 1.0f)
         : f == 3u ? float3(1.0f, 0.0f, 0.0f) : f == 4u ? float3(1.0f, 0.0f, 0.0f) : float3(0.0f, 1.0f, 0.0f);
}
TERRAIN_FN float3 terrain_faceV(uint f)
{
    return f == 0u ? float3(0.0f, 0.0f, 1.0f) : f == 1u ? float3(0.0f, 1.0f, 0.0f) : f == 2u ? float3(1.0f, 0.0f, 0.0f)
         : f == 3u ? float3(0.0f, 0.0f, 1.0f) : f == 4u ? float3(0.0f, 1.0f, 0.0f) : float3(1.0f, 0.0f, 0.0f);
}

struct TerrainGenerateConstants
{
    PlanetTerrainConstants terrain;
    float2 centre;           // the tile's centre on its face
    float halfSize;
    uint face;
    uint slot;               // where in the pool
    uint pad0, pad1, pad2;
};

// A body with terrain, this frame: its axes in the camera's world (camera at the origin),
// and last frame's, for motion vectors.
struct TerrainBody
{
    float4 axisX;            // xyz; w: its radius
    float4 axisY;            // w: written to the G-buffer's emissive.w (the app's id for the body)
    float4 axisZ;
    float4 prevAxisX;
    float4 prevAxisY;
    float4 prevAxisZ;
};

struct TerrainDraw
{
    float3 centre;           // radius x the tile centre's direction, camera-relative (from doubles)
    uint slot;
    float3 prevCentre;       // the same last frame, in this frame's camera space
    uint packed;             // face (3 bits), body (3 bits), level (5 bits)
    float2 uv;               // the tile's centre on its face
    float halfSize;
    float skirt;             // how far the edges hang down, metres
};

#endif
