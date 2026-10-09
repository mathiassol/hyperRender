// The ocean surface into the G-buffer: a clipmap of OCEAN_GRID x OCEAN_GRID-cell levels
// around the camera, one instance per level, each twice the cell size of the one inside
// it. A level is centred on a multiple of twice its cell, so its edge lies on the next
// level's grid lines; near its edge its odd vertices slide onto the even ones, so the
// boundary is exactly the coarser level's (no cracks). The coarser level skips what the
// finer one covers.

#include <hyper/common.hlsli>
#include <hyper/ocean_cb.h>
#include <donut/shaders/motion_vectors.hlsli>

cbuffer c_Surface : register(b0) { OceanSurfaceConstants c; }
Texture2DArray<float4> t_Displacement : register(t0);
Texture2DArray<float4> t_PrevDisplacement : register(t1);
Texture2DArray<float4> t_Slopes : register(t2);
SamplerState s_Wrap : register(s0);

struct VsOut
{
    float4 position : SV_Position;
    float3 world : WORLD;           // displaced
    float3 prevWorld : PREV_WORLD;
    float3 grid : GRID;             // undisplaced xz, and the height of the waves there
    nointerpolation uint level : LEVEL;
};

// About the size of a mesh cell at this distance; what the displacement is filtered to.
float FootprintAt(float2 xz)
{
    const float2 d = xz - c.cameraPos.xz;
    const float h = c.cameraPos.y - c.seaLevel;
    const float distance = sqrt(dot(d, d) + h * h);
    return max(distance * (2.0 / OCEAN_GRID), 1e-3);
}

// Texel m of a cascade holds the surface at x = m L / N (the transform's samples), which
// is the texel's corner, not its centre.
float2 PatchUv(float2 xz, uint cascade)
{
    return xz * c.invPatchSize[cascade] + 0.5 / OCEAN_N;
}

float4 Displacement(Texture2DArray<float4> tex, float2 xz)
{
    const float footprint = FootprintAt(xz);
    float4 sum = 0.0;
    [unroll]
    for (uint i = 0; i < OCEAN_CASCADES; ++i)
    {
        const float mip = max(0.0, log2(footprint * OCEAN_N * c.invPatchSize[i]));
        sum += tex.SampleLevel(s_Wrap, float3(PatchUv(xz, i), i), mip);
    }
    return sum;
}

bool Inside(OceanLevel level, float2 xz, float margin)
{
    const float2 d = abs(xz - level.center);
    return max(d.x, d.y) < level.halfExtent - margin;
}

VsOut vs(uint vertex : SV_VertexID, uint instance : SV_InstanceID)
{
    const OceanLevel level = c.levels[instance];
    const uint2 g = uint2(vertex % (OCEAN_GRID + 1), vertex / (OCEAN_GRID + 1));
    float2 xz = level.center + (float2(g) - OCEAN_GRID / 2) * level.cellSize;

    // Odd vertices slide onto their even neighbours towards the edge.
    const float2 rel = abs(xz - level.center) / level.halfExtent;
    const float morph = saturate((max(rel.x, rel.y) - c.morphStart) / (0.95 - c.morphStart));
    xz -= float2(g & 1) * level.cellSize * morph;

    VsOut o;
    o.level = instance;
    // Inside the finer level: collapse (the pixel shader drops what's left at its edge).
    if (instance > 0 && Inside(c.levels[instance - 1], xz, 0.5 * level.cellSize))
    {
        o.position = asfloat(0x7fc00000);
        o.world = o.prevWorld = o.grid = 0.0;
        return o;
    }

    // The planet curves away under the camera.
    const float2 fromCamera = xz - c.cameraPos.xz;
    const float drop = dot(fromCamera, fromCamera) / (2.0 * c.earthRadius);
    const float3 base = float3(xz.x, c.seaLevel - drop, xz.y);
    const float4 d = Displacement(t_Displacement, xz);
    o.world = base + d.xyz;
    o.prevWorld = base + Displacement(t_PrevDisplacement, xz).xyz;
    o.grid = float3(xz, d.y);
    o.position = mul(float4(o.world, 1.0), c.view.matWorldToClip);
    return o;
}

void ps(VsOut i,
        out float4 o_albedo : SV_Target0, out float4 o_specular : SV_Target1, out float4 o_normal : SV_Target2,
        out float4 o_emissive : SV_Target3, out float3 o_motion : SV_Target4)
{
    if (i.level > 0 && Inside(c.levels[i.level - 1], i.grid.xy, 1e-3 * c.levels[i.level].cellSize))
        discard;

    // Slopes and foam, filtered over the pixel. The slopes' mips keep their squares too,
    // so what the filtering smooths away comes back as roughness (LEAN mapping).
    float2 slope = 0.0;
    float2 variance = 0.0;
    float foam = 0.0;
    [unroll]
    for (uint k = 0; k < OCEAN_CASCADES; ++k)
    {
        const float3 uv = float3(PatchUv(i.grid.xy, k), k);
        const float4 s = t_Slopes.Sample(s_Wrap, uv);
        slope += s.xy;
        variance += max(s.zw - s.xy * s.xy, 0.0);
        foam += t_Displacement.Sample(s_Wrap, uv).w;
    }
    float3 n = normalize(float3(-slope.x, 1.0, -slope.y));
    const float3 toCamera = c.cameraPos - i.world;
    const float3 v = normalize(toCamera);
    // The water faces up whichever way a triangle winds (the camera stays above it). A
    // normal facing away from the camera (a steep crest seen at a grazing angle) would
    // light as a back face; tilt it up to the silhouette.
    n = normalize(n + v * max(0.0, 0.05 - dot(n, v)));

    foam = saturate(foam * c.foamAmount);
    foam = smoothstep(0.15, 0.85, foam);

    const float r2 = c.roughness * c.roughness;
    const float alpha = sqrt(r2 * r2 + variance.x + variance.y);
    const float roughness = lerp(sqrt(alpha), 0.6, foam);

    // Light through thin crests, from the sun behind them, and the sunlit body of the
    // water towards the viewer: no surface reflection, so it goes in as emissive.
    const float3 l = c.sunDirection;
    const float height = saturate(i.grid.z / max(c.waveHeight, 1e-3) * 0.5 + 0.5);
    const float behind = pow(saturate(dot(l, -v)), 4.0) * pow(saturate(0.5 - 0.5 * dot(l, n)), 3.0);
    const float body = 0.25 * pow(saturate(dot(v, n)), 2.0) * saturate(l.y);
    const float3 scatter = c.scatterColor * c.scatterStrength * c.sunRadiance * (2.0 * height * height * behind + height * body);

    o_albedo = float4(lerp(c.waterAlbedo, c.foamAlbedo, foam), 1.0);
    o_specular = float4(0.02.xxx, 1.0);
    o_normal = float4(n, roughness);
    o_emissive = float4(scatter * (1.0 - foam), 0.0);
    o_motion = GetMotionVector(i.position.xyz, i.prevWorld, c.view, c.viewPrev);
}
