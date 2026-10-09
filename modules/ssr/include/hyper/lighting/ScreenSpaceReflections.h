#pragma once

#include <hyper/platform.h>

#include <nvrhi/nvrhi.h>

namespace donut::engine { class PlanarView; struct LightProbe; }

namespace hyper
{
    struct Context;
    class GBuffer;
    class DepthPyramid;

    struct ReflectionSettings
    {
        bool enabled = true;
        float maxRoughness = 0.6f;   // rougher surfaces keep the probe's reflection only
        float thickness = 0.3f;      // world units a surface is assumed to extend behind its depth
        int maxSteps = 96;
        float intensity = 1.f;
        float historyWeight = 0.15f; // weight of the new frame in the temporal blend
    };

    // Stochastic screen-space reflections traced through the Hi-Z pyramid,
    // resolved temporally, and composited over the lit image. Where a ray
    // leaves the screen or misses, the IBL probe's reflection (already in the
    // lit image) stays.
    class ScreenSpaceReflections
    {
    public:
        explicit ScreenSpaceReflections(Context& ctx);

        ReflectionSettings settings;

        // Reads gbuffer.HdrColor (after lighting and sky) and writes the result back into it.
        // externalTrace: another tracer's result in the same format (rgb radiance,
        // a confidence), e.g. RtReflections; then only the temporal filter and
        // composite run, whatever settings.enabled says. specularOcclusion (.a):
        // how much of the probe's specular the lit image holds (Ddgi::Gather).
        void Render(nvrhi::ICommandList* cmd, const donut::engine::PlanarView& view, GBuffer& gbuffer,
                    DepthPyramid* hiz, const donut::engine::LightProbe* probe, float zNear,
                    nvrhi::ITexture* externalTrace = nullptr, nvrhi::ITexture* specularOcclusion = nullptr);
        void CreatePipelines();
        void BuildUI();

        // The resolved reflections (rgb) and confidence (a) of the last frame.
        [[nodiscard]] nvrhi::ITexture* Output() const { return m_Frame ? m_History[(m_Frame - 1) & 1u].Get() : nullptr; }

    private:
        bool m_Used = false;   // pipelines are built on first use
        Context& m_Ctx;
        nvrhi::BindingLayoutHandle m_Layout;
        nvrhi::ComputePipelineHandle m_Trace, m_Temporal, m_Composite;
        nvrhi::BufferHandle m_Constants;
        nvrhi::TextureHandle m_Traced;
        nvrhi::TextureHandle m_History[2];
        uint32_t m_Width = 0, m_Height = 0;
        uint32_t m_Frame = 0;
        bool m_HistoryValid = false;
    };
}
