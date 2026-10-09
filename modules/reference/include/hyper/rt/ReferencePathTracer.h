#pragma once

#include <hyper/platform.h>

#include <hyper/rt/RtScene.h>

#include <nvrhi/nvrhi.h>

#include <cstdint>
#include <vector>

namespace donut::engine { class PlanarView; }

namespace hyper
{
    struct Context;
    class Scene;
    class RtLighting;

    struct ReferencePathTracerSettings
    {
        int maxBounces = 8;          // indirect bounces (Russian roulette after 3)
        int samplesPerFrame = 1;
        int maxSamples = 4096;       // stop accumulating here; 0 = never stop
        float clamp = 0.f;           // per-sample luminance clamp against fireflies; 0 = off (unbiased)
    };

    // Ground truth for the renderer's lighting model: a path tracer that
    // accumulates a running mean while nothing changes, to check the real-time passes
    // (CSM, SSR, DDGI, traced lighting) against. Same BSDF as Donut's lighting, the sun
    // with MIS, the environment cube, emission, normal maps.
    //
    // Not modelled: point and spot lights, alpha-blended and transmissive materials
    // (rays pass through them), fog, volumes and aerial perspective.
    //
    // It needs a ready RtScene and RtLighting (prepared this frame) for the sun and the
    // environment; apps/scene's SceneRenderer shows it with --set reference=1.
    class ReferencePathTracer
    {
    public:
        explicit ReferencePathTracer(Context& ctx);

        ReferencePathTracerSettings settings;

        // Adds samplesPerFrame samples and returns the image (HDR radiance, RGBA32F).
        // Starts over when the view, sun, scene, environment or settings change.
        // environmentRevision: bump to restart for environment changes (EnvironmentLighting::Revision()).
        nvrhi::ITexture* Render(nvrhi::ICommandList* cmd, const donut::engine::PlanarView& view, Scene& scene,
                                const RtScene& rt, const RtLighting& lighting, uint32_t width, uint32_t height,
                                uint32_t environmentRevision);
        void Reset() { m_Samples = 0; }
        [[nodiscard]] uint32_t Samples() const { return m_Samples; }

        void CreatePipelines();
        void BuildUI();

    private:
        Context& m_Ctx;
        nvrhi::BindingLayoutHandle m_Layout;
        nvrhi::IBindingLayout* m_BindlessLayout = nullptr;
        nvrhi::ComputePipelineHandle m_Pipeline;
        nvrhi::BufferHandle m_Constants;
        nvrhi::TextureHandle m_Accumulation;
        RtBindingSets m_Sets;
        uint32_t m_Samples = 0;
        std::vector<uint8_t> m_Key;   // what the accumulated image was rendered with
    };
}
