#pragma once

#include <hyper/platform.h>

#include <donut/core/math/math.h>
#include <nvrhi/nvrhi.h>

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

namespace donut::engine { struct MeshInfo; struct MeshGeometry; }

namespace hyper
{
    struct Context;
    class Scene;

    // First-fit range allocator with coalescing free list, in elements. Backs
    // GpuScene's shared vertex and index buffers; usable for any sub-allocated
    // GPU buffer (voxel chunks, streamed meshes).
    class GpuArena
    {
    public:
        explicit GpuArena(uint32_t capacity = 0) { Reset(capacity); }

        void Reset(uint32_t capacity);
        // Adds [Capacity(), capacity) to the free ranges.
        void Grow(uint32_t capacity);
        [[nodiscard]] std::optional<uint32_t> Allocate(uint32_t count);
        void Free(uint32_t offset, uint32_t count);

        [[nodiscard]] uint32_t Capacity() const { return m_Capacity; }
        [[nodiscard]] uint32_t Used() const { return m_Used; }

    private:
        std::map<uint32_t, uint32_t> m_Free;   // offset -> count
        uint32_t m_Capacity = 0;
        uint32_t m_Used = 0;
    };

    // The scene's geometry for GPU-driven rendering: every mesh's
    // indices and vertex attributes copied (GPU to GPU) into one buffer per kind,
    // sub-allocated with GpuArena, so one indirect draw call can cover all of it.
    // Plus a record per geometry (ranges, material, bounds) and per draw
    // (instance x geometry, the unit the culling works on). Transforms stay in
    // Donut's instance buffer; skinned meshes are re-copied when Donut re-skins them.
    //
    // Static geometry is also split into meshlets (meshoptimizer, 64 vertices /
    // 124 triangles, with bounding spheres and normal cones) for mesh shading.
    // With cluster LOD, the meshlets are a DAG of simplified clusters
    // (meshoptimizer's clusterlod: groups of clusters merged and simplified to half,
    // level after level), each with its own and its parent's error, so the task
    // shader can pick a cut per view.
    //
    // It follows Scene::Revision() by difference: a mesh no instance draws any more
    // gives its ranges back, a new one is copied and split (the frame after Donut
    // uploads it), and only the draw list is rebuilt for all. Buffers grow, keeping
    // their contents, when an arena runs out.
    //
    // Shaders: hyper/gpu_scene.hlsli.
    class GpuScene
    {
    public:
        explicit GpuScene(Context& ctx);

        // After scene.RefreshBuffers(cmd). Follows the scene's changes, and clusterLod's
        // (building the LOD DAG takes a moment). False while the scene isn't ready or
        // isn't bindless, or has nothing to draw yet.
        bool Update(nvrhi::ICommandList* cmd, Scene& scene, bool clusterLod = false);

        [[nodiscard]] uint32_t DrawCount() const { return uint32_t(m_Draws.size()); }
        [[nodiscard]] uint32_t TriangleCount() const { return m_Triangles; }
        [[nodiscard]] nvrhi::IBuffer* IndexBuffer() const { return m_Indices; }

        static void AddLayoutItems(nvrhi::BindingLayoutDesc& desc);
        void AddSetItems(nvrhi::BindingSetDesc& desc, Scene& scene) const;

        // Mirrors hyper/gpu_scene.hlsli.
        struct Geometry
        {
            uint32_t firstIndex, indexCount, baseVertex, prevBaseVertex;
            uint32_t materialIndex, flags, meshletFirst, meshletCount;
            float boundsCentre[3]; float pad0;
            float boundsExtent[3]; float pad1;
        };
        struct Draw
        {
            uint32_t instance, geometry, bucket, pad;
        };
        struct Meshlet
        {
            uint32_t vertexOffset, triangleOffset, vertexCount, triangleCount;
            float centre[3]; float radius;                 // culling sphere
            float coneApex[3]; float coneCutoff;
            float coneAxis[3]; float lodError;             // simplification error, object space; 0 at full detail
            float lodCentre[3]; float lodRadius;           // the sphere lodError is measured from
            float parentCentre[3]; float parentRadius;     // the coarser clusters' sphere and error
            float parentError; float pad[3];               // FLT_MAX: nothing coarser
        };
        static constexpr uint32_t c_MeshletsPerTask = 32;
        static constexpr uint32_t c_Buckets = 4;

        // Increases whenever the draws change (scene edits, the LOD toggle), for buffers sized from them.
        [[nodiscard]] uint32_t Revision() const { return m_Revision; }
        [[nodiscard]] bool HasClusterLod() const { return m_BuiltLod; }
        [[nodiscard]] uint32_t LodLevels() const { return m_LodLevels; }
        [[nodiscard]] uint32_t MeshletCount() const { return m_Pools[MeshletPool].arena.Used(); }
        // Task-shader groups needed if every meshlet draw were visible (sizes the task lists).
        [[nodiscard]] uint32_t TaskCapacity() const { return m_TaskCapacity; }

    private:
        // Buffers that share one arena: an element's range is at the same offset in each.
        enum PoolId { VertexPool, IndexPool, PrevPool, MeshletPool, MeshletVertexPool, MeshletTrianglePool, PoolCount };
        struct Pool
        {
            GpuArena arena;
            struct Part { nvrhi::BufferHandle* buffer; nvrhi::BufferDesc desc; uint32_t stride; };
            std::vector<Part> parts;
        };
        // A geometry record's CPU side; one without a mesh is free.
        struct Slot
        {
            std::shared_ptr<donut::engine::MeshInfo> mesh;   // kept alive while it holds ranges
            const donut::engine::MeshGeometry* geometry = nullptr;
            uint32_t vertexCount = 0, prevCount = 0;
            uint32_t meshletVertexFirst = 0, meshletVertexCount = 0, meshletTriangleFirst = 0, meshletTriangleCount = 0;
        };
        struct Waiting
        {
            std::shared_ptr<donut::engine::MeshInfo> mesh;
            uint64_t frame;      // first seen; added on a later frame, once Donut has uploaded it
        };

        Context& m_Ctx;
        Pool m_Pools[PoolCount];
        nvrhi::BufferHandle m_Indices, m_Positions, m_PrevPositions, m_Normals, m_Tangents, m_Texcoords;
        nvrhi::BufferHandle m_MeshletBuffer, m_MeshletVertexBuffer, m_MeshletTriangleBuffer;
        nvrhi::BufferHandle m_GeometryBuffer, m_DrawBuffer;
        nvrhi::BindingLayoutHandle m_CopyLayout;
        nvrhi::ComputePipelineHandle m_Copy;
        std::vector<Geometry> m_Geometries;    // by slot
        std::vector<Slot> m_Slots;
        std::vector<uint32_t> m_FreeSlots;
        std::unordered_map<const donut::engine::MeshInfo*, std::vector<uint32_t>> m_MeshSlots;
        std::unordered_map<const donut::engine::MeshInfo*, Waiting> m_Waiting;
        std::vector<Draw> m_Draws;
        std::vector<uint32_t> m_Skinned;       // slots re-copied every frame
        uint32_t m_TaskCapacity = 0;
        bool m_BuiltLod = false;
        uint32_t m_SeenRevision = ~0u;
        uint32_t m_Revision = 0;
        uint32_t m_LodLevels = 0;
        uint32_t m_Triangles = 0;
        bool m_Used = false;   // the copy pipeline is built on first use

        uint32_t Allocate(nvrhi::ICommandList* cmd, PoolId pool, uint32_t count);
        void Release(uint32_t slot);
        void Drop(Scene& scene, bool all);
        void Add(nvrhi::ICommandList* cmd, uint64_t before);
        void WriteDraws(nvrhi::ICommandList* cmd, Scene& scene);
        // One geometry's meshlets (or, given attributes, 5 floats per vertex: normal and
        // texcoord, its LOD DAG), appended to the vectors with offsets into them.
        void BuildMeshlets(uint32_t indexCount, const uint32_t* indices, const float* positions, const float* attributes,
                           uint32_t vertexCount, std::vector<Meshlet>& meshlets, std::vector<uint32_t>& vertices,
                           std::vector<uint32_t>& triangles);
        void CopyVertices(nvrhi::ICommandList* cmd, uint32_t slot, bool prevToo);
        void CopyWords(nvrhi::ICommandList* cmd, nvrhi::IBuffer* dst, uint64_t dstOffset, nvrhi::IBuffer* src,
                       uint64_t srcOffset, uint64_t bytes);

    public:
        void CreatePipelines();
    };
}
