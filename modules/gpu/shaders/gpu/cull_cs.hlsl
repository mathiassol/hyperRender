// GPU culling for hyper::GpuDrivenRenderer: one thread per draw
// (instance x geometry) tests its world-space box against the view frustum
// and, in phase 2, a Hi-Z pyramid; survivors append DrawIndexedIndirect
// arguments to their bucket (compacted, counted for drawIndexedIndirectCount).
//
// Two-phase occlusion (Haar & Aaltonen 2015):
//   phase 1   frustum + visible last frame -> draw; then the pyramid is built from that depth
//   phase 2   frustum + Hi-Z against it -> draw what phase 1 missed; visibility for next frame
//   single    frustum only (shadow cascades, or occlusion off)
//
// Draws with meshlets (and useMeshlets) become task-shader work instead: one
// entry (draw, first meshlet) per 32 meshlets, counted into a DispatchMesh
// indirect record per bucket; the task shader culls the meshlets themselves.

#include <hyper/gpu_scene.hlsli>

struct CullConstants
{
    float4x4 viewProj;          // world -> clip (reverse-Z for the camera)
    float4 planes[6];           // world space, normals outwards; zero normal = unused
    uint drawCount;
    uint phase;                 // 0 single, 1, 2
    uint argsBase;              // first argument record of this pass (bucket 0)
    uint countBase;             // first counter of this pass (bucket 0)
    uint maxDraws;              // argument records per bucket
    uint useOcclusion;
    uint2 hizSize;
    uint hizMips;
    uint reverseZ;
    uint useMeshlets;
    uint taskCapacity;          // task entries per bucket
};

ConstantBuffer<CullConstants> g_Cull : register(b0);
Texture2D<float2> t_HiZ : register(t0);                 // x nearest (max), y farthest (min) reverse-Z depth
RWByteAddressBuffer u_Visibility : register(u0);        // uint per draw: visible last frame
RWByteAddressBuffer u_Args : register(u1);              // DrawIndexedIndirectArguments records
RWByteAddressBuffer u_Counts : register(u2);            // per pass x bucket counters, then statistics
RWByteAddressBuffer u_Tasks : register(u3);             // uint2 (draw, first meshlet) per task group
RWByteAddressBuffer u_MeshArgs : register(u4);          // DispatchMesh (x, 1, 1) per pass x bucket

static const uint c_StatDrawnPhase1 = 58;               // counter indices past the pass counters
static const uint c_StatDrawnPhase2 = 59;
static const uint c_StatFrustumCulled = 60;
static const uint c_StatOccluded = 61;

void WorldBox(GpuGeometry g, InstanceData inst, out float3 centre, out float3 extent)
{
    centre = mul(inst.transform, float4(g.boundsCentre, 1.0));
    // float3x4 rows are the matrix rows: each world axis sums the rotated extents.
    const float3x3 m = (float3x3)inst.transform;
    extent = float3(dot(abs(m[0]), g.boundsExtent), dot(abs(m[1]), g.boundsExtent), dot(abs(m[2]), g.boundsExtent));
}

bool InFrustum(float3 centre, float3 extent)
{
    [unroll]
    for (uint i = 0; i < 6; ++i)
    {
        const float4 p = g_Cull.planes[i];
        if (dot(p.xyz, p.xyz) < 1e-12)
            continue;
        if (dot(p.xyz, centre) - dot(abs(p.xyz), extent) > p.w)
            return false;
    }
    return true;
}

// True if the box is certainly hidden behind the depth in the pyramid.
bool Occluded(float3 centre, float3 extent)
{
    float2 lo = 1e9, hi = -1e9;
    float nearest = 0.0;   // reverse-Z: larger is nearer
    [unroll]
    for (uint k = 0; k < 8; ++k)
    {
        const float3 corner = centre + extent * float3((k & 1) ? 1.0 : -1.0, (k & 2) ? 1.0 : -1.0, (k & 4) ? 1.0 : -1.0);
        const float4 clip = mul(float4(corner, 1.0), g_Cull.viewProj);
        if (clip.w <= 1e-4)
            return false;   // straddles the camera
        const float3 ndc = clip.xyz / clip.w;
        const float2 uv = float2(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
        lo = min(lo, uv);
        hi = max(hi, uv);
        nearest = max(nearest, ndc.z);
    }
    lo = saturate(lo);
    hi = saturate(hi);
    const float2 sizePx = (hi - lo) * float2(g_Cull.hizSize);
    // The mip where the box covers at most 2x2 texels.
    const uint mip = min(uint(ceil(log2(max(max(sizePx.x, sizePx.y), 1.0)))), g_Cull.hizMips - 1);
    const uint2 mipSize = max(g_Cull.hizSize >> mip, uint2(1, 1));
    const uint2 a = min(uint2(lo * float2(mipSize)), mipSize - 1);
    const uint2 b = min(uint2(hi * float2(mipSize)), mipSize - 1);
    const float farthest = min(min(t_HiZ.Load(int3(a, mip)).y, t_HiZ.Load(int3(b.x, a.y, mip)).y),
                               min(t_HiZ.Load(int3(a.x, b.y, mip)).y, t_HiZ.Load(int3(b, mip)).y));
    return nearest < farthest;
}

void Emit(uint drawIndex, GpuDraw d, GpuGeometry g)
{
    if (g_Cull.countBase < 8u)   // the camera's two passes
        u_Counts.InterlockedAdd((g_Cull.countBase == 0 ? c_StatDrawnPhase1 : c_StatDrawnPhase2) * 4u, 1u);
    if (g_Cull.useMeshlets != 0 && g.meshletCount > 0)
    {
        const uint groups = (g.meshletCount + 31u) / 32u;
        uint first;
        u_MeshArgs.InterlockedAdd((g_Cull.countBase + d.bucket) * 12u, groups, first);
        if (first + groups > g_Cull.taskCapacity)
            return;
        const uint base = (g_Cull.countBase + d.bucket) * g_Cull.taskCapacity + first;
        for (uint i = 0; i < groups; ++i)
            u_Tasks.Store2((base + i) * 8u, uint2(drawIndex, g.meshletFirst + i * 32u));
        return;
    }
    uint slot;
    u_Counts.InterlockedAdd((g_Cull.countBase + d.bucket) * 4u, 1u, slot);
    if (slot >= g_Cull.maxDraws)
        return;
    const uint record = g_Cull.argsBase + d.bucket * g_Cull.maxDraws + slot;
    // indexCount, instanceCount, startIndex, baseVertex, startInstance (= the draw, for the vertex shader)
    u_Args.Store4(record * 20u, uint4(g.indexCount, 1u, g.firstIndex, g.baseVertex));
    u_Args.Store(record * 20u + 16u, drawIndex);
}

[numthreads(64, 1, 1)]
void main(uint id : SV_DispatchThreadID)
{
    if (id >= g_Cull.drawCount)
        return;
    const GpuDraw d = t_GpuDraws[id];
    const GpuGeometry g = t_GpuGeometries[d.geometry];
    const InstanceData inst = LoadGpuInstance(d.instance);

    float3 centre, extent;
    WorldBox(g, inst, centre, extent);
    const bool inFrustum = InFrustum(centre, extent);
    if (!inFrustum)
        u_Counts.InterlockedAdd(c_StatFrustumCulled * 4u, g_Cull.phase == 2 ? 1u : 0u);

    if (g_Cull.phase == 0)
    {
        if (inFrustum)
            Emit(id, d, g);
        return;
    }

    const bool wasVisible = u_Visibility.Load(id * 4u) != 0;
    if (g_Cull.phase == 1)
    {
        if (inFrustum && wasVisible)
            Emit(id, d, g);
        return;
    }

    // Phase 2.
    const bool occluded = inFrustum && g_Cull.useOcclusion != 0 && Occluded(centre, extent);
    const bool visible = inFrustum && !occluded;
    if (occluded)
        u_Counts.InterlockedAdd(c_StatOccluded * 4u, 1u);
    if (visible && !wasVisible)
        Emit(id, d, g);
    u_Visibility.Store(id * 4u, visible ? 1u : 0u);
}
