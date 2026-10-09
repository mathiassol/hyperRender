#include <hyper/rt/RtScene.h>

#include <hyper/App.h>
#include <hyper/scene/Scene.h>

#include <donut/core/log.h>
#include <donut/engine/Scene.h>
#include <donut/engine/SceneGraph.h>
#include <nvrhi/utils.h>


using namespace donut;

namespace hyper
{
    namespace
    {
        bool IsTriangleMesh(const engine::MeshInfo& mesh)
        {
            return mesh.type == engine::MeshType::Triangles && mesh.buffers && mesh.buffers->indexBuffer
                && mesh.buffers->vertexBuffer && mesh.buffers->hasAttribute(engine::VertexAttribute::Position);
        }

        nvrhi::rt::AccelStructDesc BlasDesc(const engine::MeshInfo& mesh, bool updatable)
        {
            nvrhi::rt::AccelStructDesc desc;
            desc.isTopLevel = false;
            desc.debugName = mesh.name;
            const auto& positions = mesh.buffers->getVertexBufferRange(engine::VertexAttribute::Position);
            for (const auto& geometry : mesh.geometries)
            {
                if (geometry->type != engine::MeshGeometryPrimitiveType::Triangles || geometry->numIndices == 0)
                    continue;
                nvrhi::rt::GeometryDesc g;
                auto& t = g.geometryData.triangles;
                t.indexBuffer = mesh.buffers->indexBuffer;
                t.indexOffset = uint64_t(mesh.indexOffset + geometry->indexOffsetInMesh) * sizeof(uint32_t);
                t.indexFormat = nvrhi::Format::R32_UINT;
                t.indexCount = geometry->numIndices;
                t.vertexBuffer = mesh.buffers->vertexBuffer;
                t.vertexOffset = uint64_t(mesh.vertexOffset + geometry->vertexOffsetInMesh) * sizeof(dm::float3) + positions.byteOffset;
                t.vertexFormat = nvrhi::Format::RGB32_FLOAT;
                t.vertexStride = sizeof(dm::float3);
                t.vertexCount = geometry->numVertices;
                g.geometryType = nvrhi::rt::GeometryType::Triangles;
                // Non-opaque geometry goes through the shaders' alpha test.
                const bool opaque = !geometry->material || geometry->material->domain == engine::MaterialDomain::Opaque;
                g.flags = opaque ? nvrhi::rt::GeometryFlags::Opaque : nvrhi::rt::GeometryFlags::None;
                desc.bottomLevelGeometries.push_back(g);
            }
            desc.buildFlags = nvrhi::rt::AccelStructBuildFlags::PreferFastTrace;
            if (updatable)
                desc.buildFlags = nvrhi::rt::AccelStructBuildFlags::PreferFastBuild | nvrhi::rt::AccelStructBuildFlags::AllowUpdate;
            return desc;
        }
    }

    nvrhi::IBindingSet* RtBindingSets::Get(nvrhi::IDevice* device, const nvrhi::BindingSetDesc& desc, nvrhi::IBindingLayout* layout)
    {
        for (size_t i = 0; i < m_Entries.size(); ++i)
        {
            if (m_Entries[i].layout == layout && m_Entries[i].desc == desc)
                return m_Entries[i].set;
        }
        // Ping-pong passes alternate between a couple of sets; keep a handful.
        constexpr size_t c_Max = 8;
        if (m_Entries.size() >= c_Max)
            m_Entries.erase(m_Entries.begin());
        m_Entries.push_back({ desc, layout, device->createBindingSet(desc, layout) });
        return m_Entries.back().set;
    }

    RtScene::RtScene(Context& ctx) : m_Ctx(ctx) { }

    void RtScene::BuildBlases(nvrhi::ICommandList* cmd, Scene& scene)
    {
        auto graph = scene.Graph();
        // Skinning prototypes are never drawn; their skinned copies are. (Joint weights in
        // the buffers don't tell: Donut shares one vertex buffer between a file's meshes.)
        std::unordered_set<const engine::MeshInfo*> prototypes;
        for (const auto& skinned : graph->GetSkinnedMeshInstances())
            prototypes.insert(skinned->GetPrototypeMesh().get());
        size_t count = 0;
        for (const auto& mesh : graph->GetMeshes())
        {
            if (mesh->accelStruct || !IsTriangleMesh(*mesh) || prototypes.count(mesh.get()))
                continue;
            const bool skinned = mesh->skinPrototype != nullptr;
            auto desc = BlasDesc(*mesh, skinned);
            if (desc.bottomLevelGeometries.empty())
                continue;
            mesh->accelStruct = m_Ctx.device->createAccelStruct(desc);
            // Skinned copies are built after their first skinning, in Update.
            if (skinned)
                m_Unskinned.insert(mesh.get());
            else
                nvrhi::utils::BuildBottomLevelAccelStruct(cmd, mesh->accelStruct, desc);
            ++count;
        }
        if (count)
            log::info("hyper: built %zu BLAS", count);
    }

    bool RtScene::Update(nvrhi::ICommandList* cmd, Scene& scene)
    {
        if (!m_Ctx.rayQuery || !scene.IsReady() || !scene.Get()->GetDescriptorTable())
            return false;

        auto graph = scene.Graph();
        const uint32_t frame = uint32_t(m_Ctx.frameIndex);
        // Meshes that came in since (all of them, for a new scene) get theirs; a mesh
        // keeps its BLAS while it lives.
        if (scene.Revision() != m_Revision)
        {
            m_Revision = scene.Revision();
            BuildBlases(cmd, scene);
        }

        // Skinned meshes: full build the first time, refit after that.
        for (const auto& skinned : graph->GetSkinnedMeshInstances())
        {
            const auto& mesh = skinned->GetMesh();
            const bool first = m_Unskinned.erase(mesh.get()) > 0;
            if (!mesh->accelStruct || (!first && skinned->GetLastUpdateFrameIndex() < frame))
                continue;
            auto desc = BlasDesc(*mesh, true);
            if (!first)
                desc.buildFlags = desc.buildFlags | nvrhi::rt::AccelStructBuildFlags::PerformUpdate;
            nvrhi::utils::BuildBottomLevelAccelStruct(cmd, mesh->accelStruct, desc);
        }

        std::vector<nvrhi::rt::InstanceDesc> instances;
        instances.reserve(graph->GetMeshInstances().size());
        for (const auto& instance : graph->GetMeshInstances())
        {
            const auto& mesh = instance->GetMesh();
            if (!mesh->accelStruct || !instance->GetNode())
                continue;
            // Instances whose materials are all hidden from rays (Scene::SeenByRays) are
            // masked out: rays pass them by without testing them at all.
            bool seen = false;
            for (const auto& g : mesh->geometries)
                seen = seen || !g->material || scene.SeenByRays(uint32_t(g->material->materialID));
            nvrhi::rt::InstanceDesc d;
            d.bottomLevelAS = mesh->accelStruct;
            d.instanceMask = seen ? 0xff : 0x00;
            d.instanceID = uint32_t(instance->GetInstanceIndex());
            dm::affineToColumnMajor(instance->GetNode()->GetLocalToWorldTransformFloat(), d.transform);
            instances.push_back(d);
        }

        if (!m_Tlas || instances.size() > m_TlasCapacity)
        {
            m_TlasCapacity = uint32_t(std::max<size_t>(instances.size() + instances.size() / 2, 16));
            nvrhi::rt::AccelStructDesc desc;
            desc.isTopLevel = true;
            desc.topLevelMaxInstances = m_TlasCapacity;
            desc.debugName = "hyper/tlas";
            m_Tlas = m_Ctx.device->createAccelStruct(desc);
            m_Ctx.bindingCache->Clear();
        }

        auto scope = m_Ctx.profiler->Measure(cmd, "tlas");
        cmd->buildTopLevelAccelStruct(m_Tlas, instances.data(), instances.size());
        m_Built = true;
        return true;
    }

    void RtScene::AddLayoutItems(nvrhi::BindingLayoutDesc& desc)
    {
        desc.addItem(nvrhi::BindingLayoutItem::RayTracingAccelStruct(40))
            .addItem(nvrhi::BindingLayoutItem::RawBuffer_SRV(41))
            .addItem(nvrhi::BindingLayoutItem::RawBuffer_SRV(42))
            .addItem(nvrhi::BindingLayoutItem::RawBuffer_SRV(43))
            .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_SRV(44))
            .addItem(nvrhi::BindingLayoutItem::Sampler(40));
    }

    void RtScene::AddSetItems(nvrhi::BindingSetDesc& desc, Scene& scene) const
    {
        const engine::Scene& s = *scene.Get();
        desc.addItem(nvrhi::BindingSetItem::RayTracingAccelStruct(40, m_Tlas))
            .addItem(nvrhi::BindingSetItem::RawBuffer_SRV(41, s.GetInstanceBuffer()))
            .addItem(nvrhi::BindingSetItem::RawBuffer_SRV(42, s.GetGeometryBuffer()))
            .addItem(nvrhi::BindingSetItem::RawBuffer_SRV(43, s.GetMaterialBuffer()))
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(44, scene.MaterialDataBuffer()))
            .addItem(nvrhi::BindingSetItem::Sampler(40, m_Ctx.common->m_LinearWrapSampler));
    }
}
