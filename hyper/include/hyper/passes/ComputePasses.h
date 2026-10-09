#pragma once

#include <hyper/platform.h>

#include <hyper/rt/RtScene.h>

#include <nvrhi/nvrhi.h>

#include <functional>
#include <initializer_list>
#include <vector>

namespace donut::engine { class PlanarView; }

namespace hyper
{
    struct Context;
    class Scene;

    // An app's compute passes over one binding layout (hyper/pass.hlsli has the slots): the
    // views at b0 / b1, 32 bytes of push constants at b3, up to 10 textures read (t0..t9)
    // and 4 written (u0..u3) per dispatch, the app's shared resources, and the scene for
    // rays (t40..t44 plus the bindless set) when it traces.
    //
    //     ComputePasses passes(ctx, { app items: b2, t20.., ... }, [&](nvrhi::BindingSetDesc& d) { ...the same, bound }, &rt);
    //     passes.Pipeline(m_Light, "app/light_cs", "main", scene.BindlessLayout());
    //     passes.BeginFrame(cmd, view, previous, &scene);   // the scene only when the passes trace
    //     passes.Dispatch(cmd, m_Light, { depth, normals }, { lit }, width, height);
    class ComputePasses
    {
    public:
        static constexpr uint32_t c_Srvs = 10, c_Uavs = 4, c_PushBytes = 32;

        // appItems: the app's shared bindings; appSet adds the matching set items (each
        // dispatch, so they can change; null when there are none). rt: the app's RtScene,
        // bound for hyper/rt.hlsli (null: the passes don't trace).
        ComputePasses(Context& ctx, std::vector<nvrhi::BindingLayoutItem> appItems,
                      std::function<void(nvrhi::BindingSetDesc&)> appSet, const RtScene* rt);

        // Builds the pipeline for a shader entry point into `pipeline`. If the shader didn't
        // compile (a typo during hot reload), it keeps the one it had and returns false.
        // bindless: the scene's layout (Scene::BindlessLayout) when the passes trace.
        bool Pipeline(nvrhi::ComputePipelineHandle& pipeline, const char* shader, const char* entry,
                      nvrhi::IBindingLayout* bindless) const;

        // Once a frame, before the dispatches: the views (b0 / b1), and the scene whose rays
        // the passes trace (needed only when built with an RtScene).
        void BeginFrame(nvrhi::ICommandList* cmd, const donut::engine::PlanarView& view,
                        const donut::engine::PlanarView& previous, Scene* scene = nullptr);

        // threads: the dispatch's extent in threads (groups of group x group). push: up
        // to c_PushBytes for the shader's b3. Unused slots get black / a dummy target.
        void Dispatch(nvrhi::ICommandList* cmd, nvrhi::IComputePipeline* pipeline,
                      std::initializer_list<nvrhi::ITexture*> srvs, std::initializer_list<nvrhi::ITexture*> uavs,
                      uint32_t width, uint32_t height, const void* push = nullptr, uint32_t pushBytes = 0, uint32_t group = 8);

        // Forget cached binding sets (targets were recreated).
        void ClearSets() { m_Sets.Clear(); }

        [[nodiscard]] nvrhi::IBindingLayout* Layout() const { return m_Layout; }
        // The views BeginFrame wrote, for the app's other pipelines (b0 / b1).
        [[nodiscard]] nvrhi::IBuffer* ViewBuffer() const { return m_View; }
        [[nodiscard]] nvrhi::IBuffer* PrevViewBuffer() const { return m_PrevView; }

    private:
        Context& m_Ctx;
        const RtScene* m_Rt;
        Scene* m_Scene = nullptr;
        RtBindingSets m_Sets;
        std::function<void(nvrhi::BindingSetDesc&)> m_AppSet;
        nvrhi::BindingLayoutHandle m_Layout;
        nvrhi::BufferHandle m_View, m_PrevView;
        nvrhi::TextureHandle m_Dummy;
    };
}
