#pragma once

#include <hyper/platform.h>

#include <nvrhi/nvrhi.h>

#include <unordered_set>
#include <vector>

namespace donut::engine { struct MeshInfo; }

namespace hyper
{
    struct Context;
    class Scene;

    // A few binding sets, reused while their description is unchanged. RT passes
    // use this instead of Context::bindingCache: Donut's cache checks every hit
    // against the set's stored description, and NVRHI's validation layer stores
    // acceleration structures unwrapped, so sets holding a TLAS never match.
    class RtBindingSets
    {
    public:
        nvrhi::IBindingSet* Get(nvrhi::IDevice* device, const nvrhi::BindingSetDesc& desc, nvrhi::IBindingLayout* layout);
        void Clear() { m_Entries.clear(); }

    private:
        struct Entry
        {
            nvrhi::BindingSetDesc desc;
            nvrhi::IBindingLayout* layout = nullptr;
            nvrhi::BindingSetHandle set;
        };
        std::vector<Entry> m_Entries;   // most recent last
    };

    // Acceleration structures for a hyper::Scene: one BLAS per mesh (built when the
    // mesh comes into the scene, Scene::Revision(); skinned meshes are refit whenever
    // Donut re-skins them) and a TLAS rebuilt every frame from the instances' transforms. The TLAS instance ID is Donut's
    // instance index, so shaders find InstanceData / GeometryData / materials
    // with hyper/rt.hlsli.
    //
    // RT passes bind the scene with AddLayoutItems / AddSetItems (t40..t44, s40)
    // plus the scene's bindless layout as descriptor set 1.
    class RtScene
    {
    public:
        explicit RtScene(Context& ctx);

        // After scene.RefreshBuffers(cmd) (which skins). Returns false while the
        // scene isn't ready or ray queries are unavailable.
        bool Update(nvrhi::ICommandList* cmd, Scene& scene);

        [[nodiscard]] bool Ready() const { return m_Tlas != nullptr && m_Built; }
        [[nodiscard]] nvrhi::rt::IAccelStruct* Tlas() const { return m_Tlas; }

        static void AddLayoutItems(nvrhi::BindingLayoutDesc& desc);
        void AddSetItems(nvrhi::BindingSetDesc& desc, Scene& scene) const;

    private:
        Context& m_Ctx;
        nvrhi::rt::AccelStructHandle m_Tlas;
        uint32_t m_TlasCapacity = 0;
        uint32_t m_Revision = 0;
        bool m_Built = false;
        std::unordered_set<const donut::engine::MeshInfo*> m_Unskinned;   // skinned meshes not built yet

        void BuildBlases(nvrhi::ICommandList* cmd, Scene& scene);
    };
}
