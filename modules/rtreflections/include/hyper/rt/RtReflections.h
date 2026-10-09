#pragma once

#include <hyper/platform.h>

#include <hyper/rt/RtScene.h>

#include <nvrhi/nvrhi.h>

namespace donut::engine { class PlanarView; }

namespace hyper
{
    struct Context;
    class Scene;
    class RtLighting;

    struct RtReflectionSettings
    {
        bool enabled = false;        // opt in: replaces the SSR trace
        float maxRoughness = 0.7f;
        float normalBias = 0.01f;
    };

    // Ray-traced reflections (one bounce, diffuse shading at the hit), replacing
    // the SSR trace. The result goes through ScreenSpaceReflections' temporal
    // filter and composite; its settings (intensity, history) apply.
    class RtReflections
    {
    public:
        explicit RtReflections(Context& ctx);

        RtReflectionSettings settings;

        // Returns the traced texture (rgb radiance, a confidence).
        nvrhi::ITexture* Trace(nvrhi::ICommandList* cmd, const donut::engine::PlanarView& view, Scene& scene,
                               const RtScene& rt, const RtLighting& lighting, nvrhi::ITexture* depth, nvrhi::ITexture* normals);
        void CreatePipelines();
        void BuildUI();

    private:
        Context& m_Ctx;
        nvrhi::BindingLayoutHandle m_Layout;
        nvrhi::IBindingLayout* m_BindlessLayout = nullptr;
        nvrhi::ComputePipelineHandle m_Pipeline;
        nvrhi::BufferHandle m_Constants;
        nvrhi::TextureHandle m_Output;
        RtBindingSets m_Sets;
    };
}
