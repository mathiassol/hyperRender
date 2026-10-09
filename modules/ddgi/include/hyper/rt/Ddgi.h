#pragma once

#include <hyper/platform.h>

#include <hyper/rt/RtLighting.h>
#include <hyper/rt/RtScene.h>

#include <donut/core/math/math.h>
#include <nvrhi/nvrhi.h>

namespace donut::engine { class PlanarView; }

namespace hyper
{
    struct Context;
    class Scene;
    class GBuffer;

    struct DdgiSettings
    {
        bool enabled = false;            // opt in: replaces the probe's diffuse
        int probeBudget = 4096;          // probes across the scene's bounds
        int raysPerProbe = 128;          // per frame, multiple of 32, up to 512
        float hysteresis = 0.97f;        // weight of the history: higher = smoother, slower
        float normalBias = 0.1f;         // fractions of the probe spacing
        float viewBias = 0.3f;
        float intensity = 1.f;
        bool relocate = true;            // move probes out of geometry, turn off ones stuck inside
    };

    // Dynamic diffuse GI with ray-traced probes (DDGI): a probe grid over the
    // scene's bounds; each frame every probe traces raysPerProbe rays, shades the
    // hits with RtLighting (sun + shadow ray, emissive, and the volume itself for
    // further bounces) and blends them into octahedral irradiance / distance atlases.
    // Gather() turns that into Donut's indirect-diffuse input. The volume also lights
    // every other pass's ray hits, through RtLighting.
    //
    //     lighting.Prepare(cmd, scene, env, probe, scale, ddgi.Prepare(cmd, scene));
    //     ddgi.Update(cmd, scene, rt, lighting);
    class Ddgi
    {
    public:
        explicit Ddgi(Context& ctx);

        DdgiSettings settings;

        // Every frame RT runs: (re)allocates and returns the volume for RtLighting;
        // null while disabled.
        const RtProbeVolume* Prepare(nvrhi::ICommandList* cmd, Scene& scene);
        // Traces and blends the probes (no-op when disabled), after lighting.Prepare.
        void Update(nvrhi::ICommandList* cmd, Scene& scene, const RtScene& rt, const RtLighting& lighting);
        // Per pixel, all indirect light for DeferredLightingPass::Inputs::indirectDiffuse:
        // DDGI diffuse plus the IBL probe's specular occluded by the GI (so pass no
        // light probes to the deferred pass). .a is the specular occlusion.
        nvrhi::ITexture* Gather(nvrhi::ICommandList* cmd, const donut::engine::PlanarView& view, GBuffer& gbuffer,
                                const RtLighting& lighting);

        [[nodiscard]] bool Active() const { return settings.enabled && m_Irradiance && m_Trace; }
        [[nodiscard]] dm::int3 ProbeCounts() const { return m_Counts; }

        void CreatePipelines();
        void BuildUI();

    private:
        Context& m_Ctx;
        RtProbeVolume m_Volume;
        nvrhi::BufferHandle m_GatherConstants;
        nvrhi::BindingLayoutHandle m_TraceLayout, m_BlendLayout, m_RelocateLayout, m_GatherLayout;
        nvrhi::IBindingLayout* m_BindlessLayout = nullptr;
        nvrhi::ComputePipelineHandle m_Trace, m_BlendIrradiance, m_BlendDistance, m_BorderIrradiance, m_BorderDistance,
            m_Relocate, m_Gather;
        nvrhi::TextureHandle m_Rays, m_Irradiance, m_Distance, m_ProbeData, m_Indirect;
        RtBindingSets m_Sets;

        dm::int3 m_Counts = dm::int3(0);
        dm::float3 m_Origin = dm::float3(0.f);
        dm::float3 m_Spacing = dm::float3(1.f);
        uint32_t m_TilesX = 0;
        uint32_t m_Generation = ~0u;
        int m_BuiltBudget = 0, m_BuiltRays = 0;
        uint32_t m_FramesSinceReset = 0;
        bool m_Reset = true;

        void Allocate(Scene& scene);
    };
}
