#include <hyper/gpu/GpuScene.h>

#include <hyper/App.h>
#include <hyper/scene/Scene.h>

#include <donut/core/log.h>
#include <donut/engine/Scene.h>
#include <donut/engine/SceneGraph.h>

#include <meshoptimizer.h>
#include <cassert>
#define CLUSTERLOD_IMPLEMENTATION
#pragma warning(push, 0)
#include <clusterlod.h>
#pragma warning(pop)

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <memory>
#include <string>

using namespace donut;

namespace hyper
{
    // ------------------------------------------------------------------ GpuArena

    void GpuArena::Reset(uint32_t capacity)
    {
        m_Free.clear();
        m_Capacity = capacity;
        m_Used = 0;
        if (capacity)
            m_Free[0] = capacity;
    }

    void GpuArena::Grow(uint32_t capacity)
    {
        if (capacity <= m_Capacity)
            return;
        auto last = m_Free.empty() ? m_Free.end() : std::prev(m_Free.end());
        if (last != m_Free.end() && last->first + last->second == m_Capacity)
            last->second += capacity - m_Capacity;
        else
            m_Free[m_Capacity] = capacity - m_Capacity;
        m_Capacity = capacity;
    }

    std::optional<uint32_t> GpuArena::Allocate(uint32_t count)
    {
        if (count == 0)
            return 0u;
        for (auto it = m_Free.begin(); it != m_Free.end(); ++it)
        {
            if (it->second < count)
                continue;
            const uint32_t offset = it->first;
            const uint32_t remaining = it->second - count;
            m_Free.erase(it);
            if (remaining)
                m_Free[offset + count] = remaining;
            m_Used += count;
            return offset;
        }
        return std::nullopt;
    }

    void GpuArena::Free(uint32_t offset, uint32_t count)
    {
        if (count == 0)
            return;
        m_Used -= count;
        auto next = m_Free.lower_bound(offset);
        // Merge with the following range...
        if (next != m_Free.end() && offset + count == next->first)
        {
            count += next->second;
            next = m_Free.erase(next);
        }
        // ...and the preceding one.
        if (next != m_Free.begin())
        {
            auto prev = std::prev(next);
            if (prev->first + prev->second == offset)
            {
                prev->second += count;
                return;
            }
        }
        m_Free[offset] = count;
    }

    // ------------------------------------------------------------------ GpuScene

    namespace
    {
        constexpr uint32_t c_AlphaTested = 1u, c_DoubleSided = 2u, c_HasTangents = 4u, c_HasTexcoords = 8u;

        // Skinning prototypes are never an instance's mesh (their skinned copies are), so the
        // instances alone decide. (Joint weights in the buffers don't mark a prototype: Donut
        // shares one vertex buffer between all meshes of a glTF file.)
        bool Drawable(const engine::MeshInfo& mesh)
        {
            return mesh.type == engine::MeshType::Triangles && mesh.buffers && mesh.buffers->indexBuffer
                && mesh.buffers->vertexBuffer && mesh.buffers->hasAttribute(engine::VertexAttribute::Position);
        }

        // Opaque and alpha-tested only: blended and transmissive stay on Donut's forward pass.
        bool Deferrable(const engine::Material* m)
        {
            return !m || m->domain == engine::MaterialDomain::Opaque || m->domain == engine::MaterialDomain::AlphaTested;
        }
    }

    GpuScene::GpuScene(Context& ctx) : m_Ctx(ctx)
    {
        m_CopyLayout = ctx.device->createBindingLayout(nvrhi::BindingLayoutDesc().setVisibility(nvrhi::ShaderType::Compute)
            .addItem(nvrhi::BindingLayoutItem::PushConstants(0, 16))
            .addItem(nvrhi::BindingLayoutItem::RawBuffer_SRV(0))
            .addItem(nvrhi::BindingLayoutItem::RawBuffer_UAV(0)));

        auto raw = [](const char* name)
        {
            return nvrhi::BufferDesc().setDebugName(name).setCanHaveRawViews(true).setCanHaveUAVs(true)
                .setInitialState(nvrhi::ResourceStates::ShaderResource).setKeepInitialState(true);
        };
        auto index = nvrhi::BufferDesc().setDebugName("hyper/gpu-indices").setIsIndexBuffer(true).setCanHaveRawViews(true)
            .setCanHaveUAVs(true).setInitialState(nvrhi::ResourceStates::IndexBuffer).setKeepInitialState(true);
        auto meshlets = nvrhi::BufferDesc().setDebugName("hyper/gpu-meshlets").setStructStride(sizeof(Meshlet))
            .setInitialState(nvrhi::ResourceStates::ShaderResource).setKeepInitialState(true);
        m_Pools[VertexPool].parts = { { std::addressof(m_Positions), raw("hyper/gpu-positions"), 12 }, { std::addressof(m_Normals), raw("hyper/gpu-normals"), 4 },
                                      { std::addressof(m_Tangents), raw("hyper/gpu-tangents"), 4 }, { std::addressof(m_Texcoords), raw("hyper/gpu-texcoords"), 8 } };
        m_Pools[IndexPool].parts = { { std::addressof(m_Indices), index, 4 } };
        m_Pools[PrevPool].parts = { { std::addressof(m_PrevPositions), raw("hyper/gpu-prev-positions"), 12 } };
        m_Pools[MeshletPool].parts = { { std::addressof(m_MeshletBuffer), meshlets, uint32_t(sizeof(Meshlet)) } };
        m_Pools[MeshletVertexPool].parts = { { std::addressof(m_MeshletVertexBuffer), raw("hyper/gpu-meshlet-vertices"), 4 } };
        m_Pools[MeshletTrianglePool].parts = { { std::addressof(m_MeshletTriangleBuffer), raw("hyper/gpu-meshlet-triangles"), 4 } };
        // Small to start with: they grow with what's added.
        for (Pool& pool : m_Pools)
        {
            pool.arena.Reset(1024);
            for (auto& part : pool.parts)
                *part.buffer = ctx.device->createBuffer(nvrhi::BufferDesc(part.desc).setByteSize(uint64_t(1024) * part.stride));
        }
    }

    void GpuScene::CreatePipelines()
    {
        if (!m_Used)
            return;   // built on first use
        auto shader = m_Ctx.shaders->CreateShader("hyper/gpu/copy_cs", "main", nullptr, nvrhi::ShaderType::Compute);
        if (shader)
            m_Copy = m_Ctx.device->createComputePipeline(nvrhi::ComputePipelineDesc().setComputeShader(shader).addBindingLayout(m_CopyLayout));
    }

    void GpuScene::CopyWords(nvrhi::ICommandList* cmd, nvrhi::IBuffer* dst, uint64_t dstOffset, nvrhi::IBuffer* src,
                             uint64_t srcOffset, uint64_t bytes)
    {
        if (!m_Copy || bytes == 0)
            return;
        const uint32_t c[4] = { uint32_t(srcOffset), uint32_t(dstOffset), uint32_t(bytes / 4), 0 };
        auto set = m_Ctx.bindingCache->GetOrCreateBindingSet(nvrhi::BindingSetDesc()
            .addItem(nvrhi::BindingSetItem::PushConstants(0, 16))
            .addItem(nvrhi::BindingSetItem::RawBuffer_SRV(0, src))
            .addItem(nvrhi::BindingSetItem::RawBuffer_UAV(0, dst)), m_CopyLayout);
        cmd->setComputeState(nvrhi::ComputeState().setPipeline(m_Copy).addBindingSet(set));
        cmd->setPushConstants(c, sizeof(c));
        cmd->dispatch(uint32_t((bytes / 4 + 63) / 64));
    }

    void GpuScene::CopyVertices(nvrhi::ICommandList* cmd, uint32_t slot, bool prevToo)
    {
        const Slot& s = m_Slots[slot];
        const Geometry& g = m_Geometries[slot];
        const auto& buffers = *s.mesh->buffers;
        const uint64_t first = uint64_t(s.mesh->vertexOffset + s.geometry->vertexOffsetInMesh);
        const uint64_t count = s.geometry->numVertices;
        auto copy = [&](nvrhi::IBuffer* dst, uint32_t dstVertex, engine::VertexAttribute attr, uint32_t stride)
        {
            if (!buffers.hasAttribute(attr))
                return;
            const auto& range = buffers.getVertexBufferRange(attr);
            CopyWords(cmd, dst, uint64_t(dstVertex) * stride, buffers.vertexBuffer, range.byteOffset + first * stride, count * stride);
        };
        copy(m_Positions, g.baseVertex, engine::VertexAttribute::Position, 12);
        copy(m_Normals, g.baseVertex, engine::VertexAttribute::Normal, 4);
        copy(m_Tangents, g.baseVertex, engine::VertexAttribute::Tangent, 4);
        copy(m_Texcoords, g.baseVertex, engine::VertexAttribute::TexCoord1, 8);
        if (prevToo && g.prevBaseVertex != ~0u)
            copy(m_PrevPositions, g.prevBaseVertex, engine::VertexAttribute::PrevPosition, 12);
    }

    // A range from a pool; when its arena has none that fits, its buffers grow by half
    // (at least by the request), contents copied on cmd.
    uint32_t GpuScene::Allocate(nvrhi::ICommandList* cmd, PoolId id, uint32_t count)
    {
        Pool& pool = m_Pools[id];
        if (auto offset = pool.arena.Allocate(count))
            return *offset;
        const uint32_t capacity = pool.arena.Capacity() + std::max(count, pool.arena.Capacity() / 2);
        for (auto& part : pool.parts)
        {
            nvrhi::BufferHandle grown = m_Ctx.device->createBuffer(nvrhi::BufferDesc(part.desc).setByteSize(uint64_t(capacity) * part.stride));
            cmd->copyBuffer(grown, 0, *part.buffer, 0, (*part.buffer)->getDesc().byteSize);
            *part.buffer = grown;
        }
        pool.arena.Grow(capacity);
        return *pool.arena.Allocate(count);
    }

    void GpuScene::Release(uint32_t slot)
    {
        Slot& s = m_Slots[slot];
        Geometry& g = m_Geometries[slot];
        m_Pools[VertexPool].arena.Free(g.baseVertex, s.vertexCount);
        m_Pools[IndexPool].arena.Free(g.firstIndex, g.indexCount);
        if (g.prevBaseVertex != ~0u)
            m_Pools[PrevPool].arena.Free(g.prevBaseVertex, s.prevCount);
        m_Pools[MeshletPool].arena.Free(g.meshletFirst, g.meshletCount);
        m_Pools[MeshletVertexPool].arena.Free(s.meshletVertexFirst, s.meshletVertexCount);
        m_Pools[MeshletTrianglePool].arena.Free(s.meshletTriangleFirst, s.meshletTriangleCount);
        m_Triangles -= g.indexCount / 3;
        s = Slot{};
        g = Geometry{};
        m_FreeSlots.push_back(slot);
    }

    // Gives back the ranges of meshes no instance draws any more (all: every mesh), and
    // notes the drawn ones that have none yet.
    void GpuScene::Drop(Scene& scene, bool all)
    {
        std::unordered_map<const engine::MeshInfo*, std::shared_ptr<engine::MeshInfo>> drawn;
        for (const auto& instance : scene.Graph()->GetMeshInstances())
        {
            const auto& mesh = instance->GetMesh();
            if (mesh && Drawable(*mesh))
                drawn.emplace(mesh.get(), mesh);
        }
        uint32_t dropped = 0;
        for (auto it = m_MeshSlots.begin(); it != m_MeshSlots.end();)
        {
            if (!all && drawn.count(it->first))
            {
                ++it;
                continue;
            }
            for (uint32_t slot : it->second)
                Release(slot);
            dropped += uint32_t(it->second.size());
            it = m_MeshSlots.erase(it);
        }
        std::erase_if(m_Waiting, [&](const auto& w) { return !drawn.count(w.first); });
        for (const auto& [key, mesh] : drawn)
            if (!m_MeshSlots.count(key) && !m_Waiting.count(key))
                m_Waiting[key] = { mesh, all ? uint64_t(0) : m_Ctx.frameIndex };
        if (dropped)
            log::info("hyper: GPU scene dropped %u geometries", dropped);
    }

    // Copies the waiting meshes first seen before frame `before` into the shared buffers
    // and splits them into meshlets.
    void GpuScene::Add(nvrhi::ICommandList* cmd, uint64_t before)
    {
        if (std::none_of(m_Waiting.begin(), m_Waiting.end(), [&](const auto& w) { return w.second.frame < before; }))
            return;
        const auto start = std::chrono::steady_clock::now();
        std::vector<uint32_t> added;
        // The copies run on their own command list: their results are read back for the
        // meshlets. Not immediate: the frame's command list is open (Vulkan allows one
        // immediate list at a time).
        nvrhi::CommandListHandle load = m_Ctx.device->createCommandList(
            nvrhi::CommandListParameters().setEnableImmediateExecution(false));
        load->open();
        for (auto it = m_Waiting.begin(); it != m_Waiting.end();)
        {
            if (it->second.frame >= before)
            {
                ++it;
                continue;
            }
            const auto& mesh = it->second.mesh;
            const bool skinned = mesh->skinPrototype != nullptr;
            const auto& buffers = *mesh->buffers;
            std::vector<uint32_t>& slots = m_MeshSlots[mesh.get()];
            for (const auto& g : mesh->geometries)
            {
                if (g->type != engine::MeshGeometryPrimitiveType::Triangles || !Deferrable(g->material.get()))
                    continue;
                uint32_t slot;
                if (!m_FreeSlots.empty())
                {
                    slot = m_FreeSlots.back();
                    m_FreeSlots.pop_back();
                }
                else
                {
                    slot = uint32_t(m_Slots.size());
                    m_Slots.emplace_back();
                    m_Geometries.emplace_back();
                }
                Slot& s = m_Slots[slot];
                s.mesh = mesh;
                s.geometry = g.get();
                s.vertexCount = g->numVertices;
                Geometry rec{};
                rec.baseVertex = Allocate(load, VertexPool, g->numVertices);
                rec.firstIndex = Allocate(load, IndexPool, g->numIndices);
                rec.indexCount = g->numIndices;
                rec.prevBaseVertex = ~0u;
                if (skinned && buffers.hasAttribute(engine::VertexAttribute::PrevPosition))
                {
                    rec.prevBaseVertex = Allocate(load, PrevPool, g->numVertices);
                    s.prevCount = g->numVertices;
                }
                const auto* material = g->material.get();
                rec.materialIndex = material ? uint32_t(material->materialID) : ~0u;
                if (material && material->domain == engine::MaterialDomain::AlphaTested)
                    rec.flags |= c_AlphaTested;
                if (material && material->doubleSided)
                    rec.flags |= c_DoubleSided;
                if (buffers.hasAttribute(engine::VertexAttribute::Tangent))
                    rec.flags |= c_HasTangents;
                if (buffers.hasAttribute(engine::VertexAttribute::TexCoord1))
                    rec.flags |= c_HasTexcoords;
                const dm::float3 c = g->objectSpaceBounds.center(), e = g->objectSpaceBounds.diagonal() * 0.5f;
                rec.boundsCentre[0] = c.x; rec.boundsCentre[1] = c.y; rec.boundsCentre[2] = c.z;
                rec.boundsExtent[0] = e.x; rec.boundsExtent[1] = e.y; rec.boundsExtent[2] = e.z;
                m_Geometries[slot] = rec;
                m_Triangles += g->numIndices / 3;
                slots.push_back(slot);
                added.push_back(slot);
            }
            it = m_Waiting.erase(it);
        }
        for (uint32_t slot : added)
        {
            const Slot& s = m_Slots[slot];
            const Geometry& rec = m_Geometries[slot];
            CopyWords(load, m_Indices, uint64_t(rec.firstIndex) * 4, s.mesh->buffers->indexBuffer,
                uint64_t(s.mesh->indexOffset + s.geometry->indexOffsetInMesh) * 4, uint64_t(rec.indexCount) * 4);
            CopyVertices(load, slot, true);
        }

        // Meshlets need the indices and positions on the CPU, and Donut frees its copies
        // after upload: read the new ranges back, packed one after another.
        std::vector<uint32_t> statics;
        uint64_t indexWords = 0, vertexWords = 0;
        for (uint32_t slot : added)
            if (!m_Slots[slot].mesh->skinPrototype)
            {
                statics.push_back(slot);
                indexWords += m_Geometries[slot].indexCount;
                vertexWords += m_Slots[slot].vertexCount;
            }
        auto staging = [&](const char* name, uint64_t bytes)
        {
            return m_Ctx.device->createBuffer(nvrhi::BufferDesc().setDebugName(name).setByteSize(std::max<uint64_t>(bytes, 16))
                .setCpuAccess(nvrhi::CpuAccessMode::Read).setInitialState(nvrhi::ResourceStates::CopyDest).setKeepInitialState(true));
        };
        auto indexStaging = staging("hyper/gpu-readback-indices", indexWords * 4);
        auto positionStaging = staging("hyper/gpu-readback-positions", vertexWords * 12);
        // Cluster LOD simplifies with normals and texcoords too.
        nvrhi::BufferHandle normalStaging, texcoordStaging;
        if (m_BuiltLod)
        {
            normalStaging = staging("hyper/gpu-readback-normals", vertexWords * 4);
            texcoordStaging = staging("hyper/gpu-readback-texcoords", vertexWords * 8);
        }
        uint64_t indexCursor = 0, vertexCursor = 0;
        for (uint32_t slot : statics)
        {
            const Geometry& rec = m_Geometries[slot];
            const uint64_t n = m_Slots[slot].vertexCount;
            load->copyBuffer(indexStaging, indexCursor * 4, m_Indices, uint64_t(rec.firstIndex) * 4, uint64_t(rec.indexCount) * 4);
            load->copyBuffer(positionStaging, vertexCursor * 12, m_Positions, uint64_t(rec.baseVertex) * 12, n * 12);
            if (m_BuiltLod)
            {
                load->copyBuffer(normalStaging, vertexCursor * 4, m_Normals, uint64_t(rec.baseVertex) * 4, n * 4);
                load->copyBuffer(texcoordStaging, vertexCursor * 8, m_Texcoords, uint64_t(rec.baseVertex) * 8, n * 8);
            }
            indexCursor += rec.indexCount;
            vertexCursor += n;
        }
        load->close();
        m_Ctx.device->executeCommandList(load);
        m_Ctx.device->waitForIdle();

        const auto* idx = static_cast<const uint32_t*>(m_Ctx.device->mapBuffer(indexStaging, nvrhi::CpuAccessMode::Read));
        const auto* pos = static_cast<const float*>(m_Ctx.device->mapBuffer(positionStaging, nvrhi::CpuAccessMode::Read));
        std::vector<float> attributes;
        if (m_BuiltLod)
        {
            const auto* nrm = static_cast<const uint32_t*>(m_Ctx.device->mapBuffer(normalStaging, nvrhi::CpuAccessMode::Read));
            const auto* uv = static_cast<const float*>(m_Ctx.device->mapBuffer(texcoordStaging, nvrhi::CpuAccessMode::Read));
            if (nrm && uv)
            {
                attributes.resize(size_t(vertexWords) * 5);
                for (size_t v = 0; v < size_t(vertexWords); ++v)
                {
                    for (int k = 0; k < 3; ++k)   // RGB8 SNORM, as Donut packs normals
                        attributes[v * 5 + k] = std::max(float(int8_t((nrm[v] >> (8 * k)) & 0xff)) / 127.f, -1.f);
                    attributes[v * 5 + 3] = uv[v * 2 + 0];
                    attributes[v * 5 + 4] = uv[v * 2 + 1];
                }
            }
            if (nrm)
                m_Ctx.device->unmapBuffer(normalStaging);
            if (uv)
                m_Ctx.device->unmapBuffer(texcoordStaging);
        }
        uint32_t meshletCount = 0;
        if (idx && pos)
        {
            std::vector<Meshlet> meshlets;
            std::vector<uint32_t> vertices, triangles;
            indexCursor = vertexCursor = 0;
            for (uint32_t slot : statics)
            {
                Geometry& rec = m_Geometries[slot];
                Slot& s = m_Slots[slot];
                meshlets.clear();
                vertices.clear();
                triangles.clear();
                BuildMeshlets(rec.indexCount, idx + indexCursor, pos + vertexCursor * 3,
                              attributes.empty() ? nullptr : attributes.data() + vertexCursor * 5, s.vertexCount,
                              meshlets, vertices, triangles);
                indexCursor += rec.indexCount;
                vertexCursor += s.vertexCount;
                rec.meshletCount = uint32_t(meshlets.size());
                rec.meshletFirst = Allocate(cmd, MeshletPool, rec.meshletCount);
                s.meshletVertexCount = uint32_t(vertices.size());
                s.meshletVertexFirst = Allocate(cmd, MeshletVertexPool, s.meshletVertexCount);
                s.meshletTriangleCount = uint32_t(triangles.size());
                s.meshletTriangleFirst = Allocate(cmd, MeshletTrianglePool, s.meshletTriangleCount);
                for (Meshlet& m : meshlets)
                {
                    m.vertexOffset += s.meshletVertexFirst;
                    m.triangleOffset += s.meshletTriangleFirst;
                }
                if (!meshlets.empty())
                {
                    cmd->writeBuffer(m_MeshletBuffer, meshlets.data(), meshlets.size() * sizeof(Meshlet), uint64_t(rec.meshletFirst) * sizeof(Meshlet));
                    cmd->writeBuffer(m_MeshletVertexBuffer, vertices.data(), vertices.size() * 4, uint64_t(s.meshletVertexFirst) * 4);
                    cmd->writeBuffer(m_MeshletTriangleBuffer, triangles.data(), triangles.size() * 4, uint64_t(s.meshletTriangleFirst) * 4);
                }
                meshletCount += rec.meshletCount;
            }
        }
        if (idx)
            m_Ctx.device->unmapBuffer(indexStaging);
        if (pos)
            m_Ctx.device->unmapBuffer(positionStaging);

        uint64_t triangles = 0;
        for (uint32_t slot : added)
            triangles += m_Geometries[slot].indexCount / 3;
        log::info("hyper: GPU scene added %zu geometries, %llu triangles, %u meshlets%s (%.0f ms)",
            added.size(), (unsigned long long)triangles, meshletCount,
            m_BuiltLod ? (" in a cluster LOD DAG of " + std::to_string(m_LodLevels) + " levels").c_str() : "",
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
    }

    // Draw records (instance x geometry) for every instance whose mesh is in, and the
    // geometry records; both rewritten whole (they're small), since Donut renumbers
    // instances when the graph changes.
    void GpuScene::WriteDraws(nvrhi::ICommandList* cmd, Scene& scene)
    {
        m_Draws.clear();
        m_TaskCapacity = 0;
        for (const auto& instance : scene.Graph()->GetMeshInstances())
        {
            auto it = m_MeshSlots.find(instance->GetMesh().get());
            if (it == m_MeshSlots.end())
                continue;
            for (uint32_t slot : it->second)
            {
                const Geometry& rec = m_Geometries[slot];
                Draw d{};
                d.instance = uint32_t(instance->GetInstanceIndex());
                d.geometry = slot;
                d.bucket = ((rec.flags & c_AlphaTested) ? 2u : 0u) + ((rec.flags & c_DoubleSided) ? 1u : 0u);
                m_Draws.push_back(d);
                m_TaskCapacity += (rec.meshletCount + c_MeshletsPerTask - 1) / c_MeshletsPerTask;
            }
        }
        // Material IDs can move when materials come and go.
        m_Skinned.clear();
        for (uint32_t slot = 0; slot < uint32_t(m_Slots.size()); ++slot)
        {
            const Slot& s = m_Slots[slot];
            if (!s.mesh)
                continue;
            m_Geometries[slot].materialIndex = s.geometry->material ? uint32_t(s.geometry->material->materialID) : ~0u;
            if (s.mesh->skinPrototype)
                m_Skinned.push_back(slot);
        }

        auto structured = [&](nvrhi::BufferHandle& buffer, const char* name, uint32_t stride, size_t count)
        {
            const uint64_t bytes = uint64_t(std::max<size_t>(count, 1)) * stride;
            if (!buffer || buffer->getDesc().byteSize < bytes)
                buffer = m_Ctx.device->createBuffer(nvrhi::BufferDesc().setDebugName(name).setByteSize(bytes + bytes / 2)
                    .setStructStride(stride).setInitialState(nvrhi::ResourceStates::ShaderResource).setKeepInitialState(true));
        };
        structured(m_GeometryBuffer, "hyper/gpu-geometries", sizeof(Geometry), m_Geometries.size());
        structured(m_DrawBuffer, "hyper/gpu-draws", sizeof(Draw), m_Draws.size());
        if (!m_Geometries.empty())
            cmd->writeBuffer(m_GeometryBuffer, m_Geometries.data(), m_Geometries.size() * sizeof(Geometry));
        if (!m_Draws.empty())
            cmd->writeBuffer(m_DrawBuffer, m_Draws.data(), m_Draws.size() * sizeof(Draw));
        m_Ctx.bindingCache->Clear();
        ++m_Revision;
    }

    void GpuScene::BuildMeshlets(uint32_t indexCount, const uint32_t* indices, const float* positions, const float* attributes,
                                 uint32_t vertexCount, std::vector<Meshlet>& out, std::vector<uint32_t>& outVertices,
                                 std::vector<uint32_t>& outTriangles)
    {
        // Appends one meshlet with its culling bounds; m carries the LOD fields.
        auto add = [&](const uint32_t* vertices, const uint8_t* triangles, uint32_t vCount, uint32_t tCount, Meshlet m)
        {
            const meshopt_Bounds b = meshopt_computeMeshletBounds(vertices, triangles, tCount, positions, vertexCount, sizeof(float) * 3);
            m.vertexOffset = uint32_t(outVertices.size());
            m.triangleOffset = uint32_t(outTriangles.size());
            m.vertexCount = vCount;
            m.triangleCount = tCount;
            std::copy_n(b.center, 3, m.centre);
            m.radius = b.radius;
            std::copy_n(b.cone_apex, 3, m.coneApex);
            std::copy_n(b.cone_axis, 3, m.coneAxis);
            m.coneCutoff = b.cone_cutoff;
            if (m.lodError == 0.f && m.lodRadius == 0.f)
            {
                std::copy_n(b.center, 3, m.lodCentre);
                m.lodRadius = b.radius;
            }
            out.push_back(m);
            outVertices.insert(outVertices.end(), vertices, vertices + vCount);
            for (uint32_t t = 0; t < tCount; ++t)
            {
                const uint8_t* tri = &triangles[t * 3];
                outTriangles.push_back(uint32_t(tri[0]) | (uint32_t(tri[1]) << 8) | (uint32_t(tri[2]) << 16));
            }
        };

        if (!attributes)
        {
            constexpr size_t maxVertices = 64, maxTriangles = 124;
            const size_t bound = meshopt_buildMeshletsBound(indexCount, maxVertices, maxTriangles);
            std::vector<meshopt_Meshlet> meshlets(bound);
            std::vector<unsigned int> vertices(indexCount);
            std::vector<unsigned char> triangles(indexCount);
            const size_t count = meshopt_buildMeshlets(meshlets.data(), vertices.data(), triangles.data(), indices, indexCount,
                positions, vertexCount, sizeof(float) * 3, maxVertices, maxTriangles, 0.25f);
            for (size_t i = 0; i < count; ++i)
            {
                const meshopt_Meshlet& m = meshlets[i];
                meshopt_optimizeMeshlet(&vertices[m.vertex_offset], &triangles[m.triangle_offset], m.triangle_count, m.vertex_count);
                Meshlet one{};
                one.parentError = FLT_MAX;   // no LOD: always the one to draw
                add(&vertices[m.vertex_offset], &triangles[m.triangle_offset], m.vertex_count, m.triangle_count, one);
            }
            return;
        }

        // Cluster LOD. As meshoptimizer's nanite demo: normals weigh in, UV seams are protected.
        clodConfig config = clodDefaultConfig(124);
        config.max_vertices = 64;   // the mesh shader's limit
        static const float normalWeights[3] = { 0.5f, 0.5f, 0.5f };
        clodMesh mesh{};
        mesh.indices = indices;
        mesh.index_count = indexCount;
        mesh.vertex_count = vertexCount;
        mesh.vertex_positions = positions;
        mesh.vertex_positions_stride = sizeof(float) * 3;
        mesh.vertex_attributes = attributes;
        mesh.vertex_attributes_stride = sizeof(float) * 5;
        mesh.attribute_weights = normalWeights;
        mesh.attribute_count = 3;
        mesh.attribute_protect_mask = (1u << 3) | (1u << 4);

        std::vector<clodBounds> groups;   // each group's simplified bounds, by the id clodBuild hands back
        std::vector<uint32_t> local(size_t(124) * 3);
        std::vector<uint8_t> triangles(size_t(124) * 3);
        clodBuild(config, mesh, [&](clodGroup group, const clodCluster* clusters, size_t count) -> int
        {
            m_LodLevels = std::max(m_LodLevels, uint32_t(group.depth + 1));
            for (size_t i = 0; i < count; ++i)
            {
                const clodCluster& c = clusters[i];
                const size_t unique = clodLocalIndices(local.data(), triangles.data(), c.indices, c.index_count);
                // Drawn while its own error is small enough and its parent's is not.
                Meshlet m{};
                if (c.refined >= 0)
                {
                    const clodBounds& self = groups[size_t(c.refined)];
                    std::copy_n(self.center, 3, m.lodCentre);
                    m.lodRadius = self.radius;
                    m.lodError = self.error;
                }
                std::copy_n(group.simplified.center, 3, m.parentCentre);
                m.parentRadius = group.simplified.radius;
                m.parentError = group.simplified.error;
                add(local.data(), triangles.data(), uint32_t(unique), uint32_t(c.index_count / 3), m);
            }
            groups.push_back(group.simplified);
            return int(groups.size() - 1);
        });
    }

    bool GpuScene::Update(nvrhi::ICommandList* cmd, Scene& scene, bool clusterLod)
    {
        if (!m_Used)
        {
            m_Used = true;
            CreatePipelines();
        }
        if (!scene.IsReady() || !scene.DescriptorTable())
            return false;
        bool changed = false;
        if (clusterLod != m_BuiltLod)
        {
            // Every mesh again, split the other way (they're on the GPU already).
            m_BuiltLod = clusterLod;
            m_LodLevels = 0;
            Drop(scene, true);
            m_SeenRevision = scene.Revision();
            changed = true;
        }
        else if (scene.Revision() != m_SeenRevision)
        {
            m_SeenRevision = scene.Revision();
            Drop(scene, false);
            changed = true;
        }
        // New meshes wait a frame: Donut uploads them in the frame's command list.
        const size_t waiting = m_Waiting.size();
        if (waiting)
            Add(cmd, m_Ctx.frameIndex);
        if (changed || m_Waiting.size() != waiting)
        {
            const auto start = std::chrono::steady_clock::now();
            WriteDraws(cmd, scene);
            log::info("hyper: GPU scene %zu draws (%.2f ms)", m_Draws.size(),
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
        }
        // Skinned meshes: Donut re-skins into their own buffers; copy the fresh vertices.
        for (uint32_t slot : m_Skinned)
            CopyVertices(cmd, slot, true);
        return !m_Draws.empty();
    }

    void GpuScene::AddLayoutItems(nvrhi::BindingLayoutDesc& desc)
    {
        for (uint32_t t = 60; t <= 64; ++t)
            desc.addItem(nvrhi::BindingLayoutItem::RawBuffer_SRV(t));
        desc.addItem(nvrhi::BindingLayoutItem::StructuredBuffer_SRV(65))
            .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_SRV(66))
            .addItem(nvrhi::BindingLayoutItem::RawBuffer_SRV(67))
            .addItem(nvrhi::BindingLayoutItem::RawBuffer_SRV(68))
            .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_SRV(69))
            .addItem(nvrhi::BindingLayoutItem::RawBuffer_SRV(70))
            .addItem(nvrhi::BindingLayoutItem::RawBuffer_SRV(71));
    }

    void GpuScene::AddSetItems(nvrhi::BindingSetDesc& desc, Scene& scene) const
    {
        const engine::Scene& s = *scene.Get();
        desc.addItem(nvrhi::BindingSetItem::RawBuffer_SRV(60, m_Positions))
            .addItem(nvrhi::BindingSetItem::RawBuffer_SRV(61, m_PrevPositions))
            .addItem(nvrhi::BindingSetItem::RawBuffer_SRV(62, m_Normals))
            .addItem(nvrhi::BindingSetItem::RawBuffer_SRV(63, m_Tangents))
            .addItem(nvrhi::BindingSetItem::RawBuffer_SRV(64, m_Texcoords))
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(65, m_GeometryBuffer))
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(66, m_DrawBuffer))
            .addItem(nvrhi::BindingSetItem::RawBuffer_SRV(67, s.GetInstanceBuffer()))
            .addItem(nvrhi::BindingSetItem::RawBuffer_SRV(68, s.GetMaterialBuffer()))
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(69, m_MeshletBuffer))
            .addItem(nvrhi::BindingSetItem::RawBuffer_SRV(70, m_MeshletVertexBuffer))
            .addItem(nvrhi::BindingSetItem::RawBuffer_SRV(71, m_MeshletTriangleBuffer));
    }
}
