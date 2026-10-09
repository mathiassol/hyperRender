#pragma once

#include <hyper/platform.h>

#include <hyper/rt/RtScene.h>

#include <nvrhi/nvrhi.h>

namespace donut::engine { class PlanarView; }

namespace hyper
{
    struct Context;
    class Scene;
    class RtScene;

    struct RtShadowSettings
    {
        bool enabled = false;        // opt in: replaces the shadow cascades on the deferred path
        float softness = 1.f;        // multiplies the sun's angular size: wider penumbrae
        float normalBias = 0.01f;    // world units, grows with view distance
    };

    // Ray-traced sun shadows from the G-buffer, replacing the shadow map on the
    // deferred path. Writes Donut's "shadow channels" texture (.r = sun).
    class RtShadows
    {
    public:
        explicit RtShadows(Context& ctx);

        RtShadowSettings settings;

        // Returns the shadow-channel texture (RGBA8, .r = visibility).
        nvrhi::ITexture* Render(nvrhi::ICommandList* cmd, const donut::engine::PlanarView& view, Scene& scene,
                                const RtScene& rt, nvrhi::ITexture* depth, nvrhi::ITexture* normals);
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
