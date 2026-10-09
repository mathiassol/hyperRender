#ifndef HYPER_GPU_SCENE_HLSLI
#define HYPER_GPU_SCENE_HLSLI

// hyper::GpuScene on the GPU: the whole scene's geometry in shared buffers,
// a record per geometry and per draw (instance x geometry). Bind with
// GpuScene::AddLayoutItems / AddSetItems (t60..t71); materials' textures come
// from the scene's bindless table (set 1, binding 1).
//
// Vertices are pulled by global index: SV_VertexID of an indexed draw already
// includes the geometry's baseVertex.

#include <hyper/common.hlsli>
#include <donut/shaders/bindless.h>
#include <donut/shaders/packing.hlsli>

static const uint GpuGeometry_AlphaTested = 1u;
static const uint GpuGeometry_DoubleSided = 2u;
static const uint GpuGeometry_HasTangents = 4u;
static const uint GpuGeometry_HasTexcoords = 8u;

struct GpuGeometry
{
    uint firstIndex;
    uint indexCount;
    uint baseVertex;
    uint prevBaseVertex;     // ~0: no previous positions (static)
    uint materialIndex;
    uint flags;
    uint meshletFirst;
    uint meshletCount;
    float3 boundsCentre;     // object space
    float pad0;
    float3 boundsExtent;
    float pad1;
};

// meshoptimizer meshlet: up to 64 vertices, 124 triangles, with culling bounds (object space).
// With cluster LOD, a node of the LOD DAG: drawn when its own error projects to at most
// the threshold and its parent's to more (see ClusterInLodCut in draw.hlsl).
struct GpuMeshlet
{
    uint vertexOffset;       // into t_GpuMeshletVertices (geometry-local vertex indices)
    uint triangleOffset;     // into t_GpuMeshletTriangles (3 x 8-bit local indices per uint)
    uint vertexCount;
    uint triangleCount;
    float3 centre;           // culling sphere
    float radius;
    float3 coneApex;
    float coneCutoff;        // >= 1: no usable cone
    float3 coneAxis;
    float lodError;          // simplification error, object space; 0 at full detail
    float3 lodCentre;        // the sphere lodError is measured from
    float lodRadius;
    float3 parentCentre;     // the coarser clusters' sphere and error
    float parentRadius;
    float parentError;       // FLT_MAX: nothing coarser
    float3 pad;
};

struct GpuDraw
{
    uint instance;           // Donut instance index (transforms)
    uint geometry;           // GpuGeometry index
    uint bucket;             // 0 opaque, 1 opaque double-sided, 2 alpha-tested, 3 alpha-tested double-sided
    uint pad;
};

ByteAddressBuffer t_GpuPositions : register(t60);
ByteAddressBuffer t_GpuPrevPositions : register(t61);
ByteAddressBuffer t_GpuNormals : register(t62);
ByteAddressBuffer t_GpuTangents : register(t63);
ByteAddressBuffer t_GpuTexcoords : register(t64);
StructuredBuffer<GpuGeometry> t_GpuGeometries : register(t65);
StructuredBuffer<GpuDraw> t_GpuDraws : register(t66);
ByteAddressBuffer t_GpuInstances : register(t67);    // Donut's InstanceData
ByteAddressBuffer t_GpuMaterials : register(t68);    // Donut's MaterialConstants
StructuredBuffer<GpuMeshlet> t_GpuMeshlets : register(t69);
ByteAddressBuffer t_GpuMeshletVertices : register(t70);
ByteAddressBuffer t_GpuMeshletTriangles : register(t71);

InstanceData LoadGpuInstance(uint index)
{
    return LoadInstanceData(t_GpuInstances, index * c_SizeOfInstanceData);
}

MaterialConstants LoadGpuMaterial(uint index)
{
    return LoadMaterialConstants(t_GpuMaterials, index * c_SizeOfMaterialConstants);
}

float3 LoadGpuPosition(uint vertex)
{
    return asfloat(t_GpuPositions.Load3(vertex * 12u));
}

#endif
