// The planet module: tiles into the G-buffer (Donut's layout: albedo, specular, normal,
// emissive, motion). Instanced: one instance a tile, its vertices from the pool. Positions
// are camera-relative, each the tile's centre (from doubles) plus a small offset.
//
// What the lighting reads, besides albedo and normal (roughness in normal.w):
//   specular.r  1 over water (else 0.04), specular.g its depth as d / (d + 20 m)
//   emissive.w  the body's light index (the app's), as the app's other bodies write it

#include <hyper/common.hlsli>
#include <donut/shaders/view_cb.h>
#include <donut/shaders/motion_vectors.hlsli>
#include <hyper/planet_tile.hlsli>

ConstantBuffer<PlanarViewConstants> g_View : register(b0);
ConstantBuffer<PlanarViewConstants> g_ViewPrev : register(b1);
StructuredBuffer<TerrainDraw> t_Draws : register(t0);
StructuredBuffer<TerrainBody> t_Bodies : register(t1);
StructuredBuffer<float4> t_Vertices : register(t2);

struct VsOut
{
    float4 position : SV_Position;
    float3 world : WORLD;
    float3 prevWorld : PREV_WORLD;
    float3 normal : NORMAL;
    float3 albedo : ALBEDO;
    float water : WATER;
    nointerpolation float lightBody : LIGHT_BODY;
};

VsOut vs(uint vertex : SV_VertexID, uint instance : SV_InstanceID)
{
    const TerrainDraw d = t_Draws[instance];
    const uint face = d.packed & 7u, bodyIndex = (d.packed >> 3) & 7u;
    const TerrainBody b = t_Bodies[bodyIndex];
    // The grid with a ring of skirt vertices round it, hanging down to hide the cracks
    // between tiles of different detail.
    const uint row = TERRAIN_GRID + 2;
    const int gi = int(vertex % row), gj = int(vertex / row);
    const bool skirt = gi == 0 || gj == 0 || gi == int(row) - 1 || gj == int(row) - 1;
    const uint i = uint(clamp(gi - 1, 0, TERRAIN_GRID - 1)), j = uint(clamp(gj - 1, 0, TERRAIN_GRID - 1));
    const float4 v = t_Vertices[d.slot * TERRAIN_GRID * TERRAIN_GRID + j * TERRAIN_GRID + i];

    const TilePoint t = TileDirection(face, d.uv, d.halfSize, float2(i, j));
    const float h = v.x - (skirt ? d.skirt : 0.0);
    const float3 local = TileLocal(t, b.axisX.w, h);
    const float3 n = TerrainUnpackNormal(asuint(v.y));

    VsOut o;
    o.world = d.centre + b.axisX.xyz * local.x + b.axisY.xyz * local.y + b.axisZ.xyz * local.z;
    o.prevWorld = d.prevCentre + b.prevAxisX.xyz * local.x + b.prevAxisY.xyz * local.y + b.prevAxisZ.xyz * local.z;
    o.normal = b.axisX.xyz * n.x + b.axisY.xyz * n.y + b.axisZ.xyz * n.z;
    o.albedo = TerrainUnpackAlbedo(asuint(v.z));
    o.water = v.w;
    o.lightBody = b.axisY.w;
    o.position = mul(float4(o.world, 1.0), g_View.matWorldToClip);
    return o;
}

struct GBufferOut
{
    float4 albedo : SV_Target0;
    float4 specular : SV_Target1;
    float4 normal : SV_Target2;
    float4 emissive : SV_Target3;
    float3 motion : SV_Target4;
};

GBufferOut ps(VsOut i)
{
    GBufferOut o;
    const bool water = i.water > 0.0;
    o.albedo = float4(i.albedo, 1.0);
    o.specular = water ? float4(1.0, i.water / (i.water + 20.0), 0.0, 1.0) : float4(0.04, 0.0, 0.0, 1.0);
    o.normal = float4(normalize(i.normal), 0.8);
    o.emissive = float4(0.0, 0.0, 0.0, i.lightBody);
    o.motion = GetMotionVector(i.position.xyz, i.prevWorld, g_View, g_ViewPrev);
    return o;
}
