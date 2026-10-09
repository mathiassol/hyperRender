#pragma once

#include <hyper/platform.h>

#include <hyper/RenderTarget.h>
#include <hyper/passes/Bloom.h>
#include <hyper/passes/Tonemap.h>

#include <donut/engine/View.h>
#include <donut/render/TemporalAntiAliasingPass.h>

#include <memory>

namespace hyper
{
    struct Context;

    enum class AntiAliasing
    {
        None,
        Taa,    // needs Inputs::depth; motion vectors come from depth if not given
    };

    struct PostSettings
    {
        AntiAliasing aa = AntiAliasing::Taa;
        donut::render::TemporalAntiAliasingParameters taa;

        bool bloom = true;
        float bloomIntensity = 0.04f;
        float bloomRadius = 1.f;

        TonemapSettings tonemap;
    };

    // The standard HDR back half of a frame:
    //   [TAA] -> bloom -> exposure + tonemap -> target
    //
    // Inputs may be smaller than the target: TAA then upscales (TAAU); without
    // TAA the tonemapper samples bilinearly.
    //
    // Conventions for the inputs (docs/architecture.md, section 5):
    //   color          linear scene-referred Rec.709, RGBA16_FLOAT, SRV
    //   depth          reverse-Z (1 near, 0 = far/infinity), any SRV-able format
    //   motionVectors  optional, RG16_FLOAT, pixels from current to previous frame
    //
    // Per frame:
    //   ctx.camera.GetView(view, w, h);
    //   post.PrepareView(view);          // jitters the view when TAA is on
    //   ... render the scene with `view` into color (+ depth) ...
    //   post.Render(cmd, { color, depth }, view, framebuffer);
    class PostProcess
    {
    public:
        explicit PostProcess(Context& ctx);
        ~PostProcess();

        PostSettings settings;

        struct Inputs
        {
            nvrhi::ITexture* color = nullptr;
            nvrhi::ITexture* depth = nullptr;
            nvrhi::ITexture* motionVectors = nullptr;
            // With motionVectors: stencil bit(s) in depth marking pixels that already have
            // motion vectors (e.g. the G-buffer's). The rest get camera motion from depth.
            uint32_t motionVectorStencilMask = 0;
            // Optional HDR layer added after TAA (resampled to fit): things that must not
            // be temporally filtered, e.g. analytic point lights that would smear.
            nvrhi::ITexture* overlay = nullptr;
        };

        void PrepareView(donut::engine::PlanarView& view);
        void Render(nvrhi::ICommandList* cmd, const Inputs& inputs, const donut::engine::PlanarView& view, nvrhi::IFramebuffer* target);

        // Call from App::ShadersReloaded().
        void ShadersReloaded();

        // Forget TAA history (camera cuts, teleports).
        void ResetHistory() { m_HistoryValid = false; }

        // ImGui controls for `settings`; call inside a window.
        void BuildUI();

        [[nodiscard]] Tonemap& GetTonemap() { return m_Tonemap; }
        [[nodiscard]] Bloom& GetBloom() { return m_Bloom; }

    private:
        Context& m_Ctx;
        Bloom m_Bloom;
        Tonemap m_Tonemap;

        // TAA
        std::unique_ptr<donut::render::TemporalAntiAliasingPass> m_Taa;
        RenderTarget m_TaaOutput;
        RenderTarget m_Feedback1;
        RenderTarget m_Feedback2;
        RenderTarget m_MotionVectors;
        nvrhi::ITexture* m_TaaColor = nullptr;
        nvrhi::ITexture* m_TaaDepth = nullptr;
        nvrhi::ITexture* m_TaaMotion = nullptr;
        uint32_t m_TaaStencilMask = 0;
        donut::engine::PlanarView m_PreviousView;
        donut::engine::PlanarView m_OutputView;   // input view resized to the output (TAA target)
        bool m_HistoryValid = false;
        bool m_WarnedNoDepth = false;

        // overlay
        RenderTarget m_Combined;
        nvrhi::BindingLayoutHandle m_AddLayout;
        nvrhi::ComputePipelineHandle m_AddPipeline;
        void CreateAddPipeline();
        nvrhi::ITexture* AddOverlay(nvrhi::ICommandList* cmd, nvrhi::ITexture* base, nvrhi::ITexture* overlay);

        nvrhi::ITexture* RunTaa(nvrhi::ICommandList* cmd, const Inputs& inputs, const donut::engine::PlanarView& view,
                                uint32_t outputWidth, uint32_t outputHeight);
    };
}
