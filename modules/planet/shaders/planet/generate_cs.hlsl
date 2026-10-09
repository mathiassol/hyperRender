// The planet module: tiles' vertices, from the shared height and material functions
// (hyper/planet_terrain.h), one tile a layer of groups (SV_GroupID.z: its job). Each vertex: its height (the sea's surface over water), its
// normal from its neighbours' heights, what the ground is, and the water's depth.

// The measured elevation maps, one slice a body (planet_terrain.h reads them through this).
Texture2DArray<float> t_Maps : register(t0);
#define TERRAIN_MAP_LOAD(slice, x, y) t_Maps.Load(int4(x, y, slice, 0))
// The finer maps of a few places (planet_terrain.h's TerrainPatch, three uint4s each), and their samples.
StructuredBuffer<uint4> t_Patches : register(t2);
StructuredBuffer<float> t_PatchSamples : register(t3);
#define TERRAIN_PATCH(i) terrain_patchFrom(t_Patches[(i) * 3u], t_Patches[(i) * 3u + 1u], t_Patches[(i) * 3u + 2u])
#define TERRAIN_PATCH_LOAD(i) t_PatchSamples[i]
#include <hyper/planet_tile.hlsli>

StructuredBuffer<TerrainGenerateConstants> t_Jobs : register(t1);
RWStructuredBuffer<float4> u_Vertices : register(u0);
static TerrainGenerateConstants g_C;

// The ground as drawn: over a sea, the sea's surface.
float DrawnHeight(float h)
{
    return g_C.terrain.ocean > 0.0 ? max(h, 0.0) : h;
}

float3 Neighbour(float2 ij)
{
    const TilePoint t = TileDirection(g_C.face, g_C.centre, g_C.halfSize, ij);
    const float h = DrawnHeight(TerrainHeightSplit(g_C.terrain, t.dir0, t.delta));
    return TileLocal(t, g_C.terrain.radius, h);
}

[numthreads(TERRAIN_GROUP, TERRAIN_GROUP, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= TERRAIN_GRID || id.y >= TERRAIN_GRID)
        return;
    g_C = t_Jobs[id.z];
    const float2 ij = float2(id.xy);
    const TilePoint t = TileDirection(g_C.face, g_C.centre, g_C.halfSize, ij);
    const float3 dir = t.dir0 + t.delta;
    const float h = TerrainHeightSplit(g_C.terrain, t.dir0, t.delta);
    const float drawn = DrawnHeight(h);

    // The normal from the neighbours a step away, beyond the tile at its edges.
    const float3 n = normalize(cross(Neighbour(ij + float2(1, 0)) - Neighbour(ij - float2(1, 0)),
                                     Neighbour(ij + float2(0, 1)) - Neighbour(ij - float2(0, 1))));
    const float3 up = normalize(dir);
    float3 normal = dot(n, up) < 0.0 ? -n : n;

    // (A face spans two units of the cube, about 1.6 radii across.)
    const float spacing = 1.6 * g_C.halfSize / float(TERRAIN_GRID - 1);
    const float4 material = TerrainMaterial(g_C.terrain, up, h, dot(normal, up), spacing);
    uint flags = 0u;
    if (material.w > 0.0)
    {
        flags |= TERRAIN_MATERIAL_WATER;
        normal = up;   // a calm sea's surface; light_cs adds the waves
    }
    if (g_C.terrain.meltTemperature > 0.0 && TerrainTemperature(g_C.terrain, up, h) > g_C.terrain.meltTemperature)
        flags |= TERRAIN_MATERIAL_LAVA;
    // w: the water's depth, signed (above the sea, minus the ground's height), so a coast
    // falls where it crosses 0 between vertices rather than along their grid.
    const float depth = flags & TERRAIN_MATERIAL_WATER ? material.w : (g_C.terrain.ocean > 0.0 ? min(-h, -0.01) : -1.0);
    u_Vertices[g_C.slot * TERRAIN_GRID * TERRAIN_GRID + id.y * TERRAIN_GRID + id.x] =
        float4(drawn, asfloat(TerrainPackNormal(normal)), asfloat(TerrainPackAlbedo(material.xyz, flags)), depth);
}
