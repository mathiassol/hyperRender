#pragma once

#include <hyper/platform.h>

#include <memory>

namespace donut::engine { class PlanarView; class FramebufferFactory; }
namespace donut::render { class CascadedShadowMap; class DepthPass; class InstancedOpaqueDrawStrategy; }
namespace nvrhi { class ICommandList; }

namespace hyper
{
    struct Context;
    class Scene;
    class GpuDrivenRenderer;

    struct ShadowSettings
    {
        bool enabled = true;
        int resolution = 2048;          // per cascade
        int cascades = 4;
        float maxDistance = 60.f;       // world units covered by the cascades
        float exponent = 4.f;           // cascade split distribution: higher = more detail near the camera
        int depthBias = 100;
        float slopeBias = 4.f;
    };

    // Cascaded shadow maps for the scene's sun (Donut's CascadedShadowMap, stable
    // cascades so they don't shimmer as the camera moves). Render() draws them
    // and hooks them to the light, so Donut's deferred and forward passes pick
    // them up.
    class SunShadows
    {
    public:
        explicit SunShadows(Context& ctx);
        ~SunShadows();

        ShadowSettings settings;

        // gpu: draw the cascades GPU-driven (frustum culled per cascade) instead of with Donut's DepthPass.
        void Render(nvrhi::ICommandList* cmd, Scene& scene, const donut::engine::PlanarView& view,
                    GpuDrivenRenderer* gpu = nullptr);
        void CreatePipelines();
        void BuildUI();

    private:
        Context& m_Ctx;
        std::shared_ptr<donut::render::CascadedShadowMap> m_Csm;
        std::shared_ptr<donut::engine::FramebufferFactory> m_Framebuffer;
        std::unique_ptr<donut::render::DepthPass> m_DepthPass;
        std::unique_ptr<donut::render::InstancedOpaqueDrawStrategy> m_Strategy;
        int m_BuiltResolution = 0;
        int m_BuiltCascades = 0;
        int m_BuiltBias = 0;
        float m_BuiltSlopeBias = 0.f;
    };
}
