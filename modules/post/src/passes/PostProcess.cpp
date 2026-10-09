#include <hyper/passes/PostProcess.h>

#include <hyper/App.h>

#include <donut/core/log.h>
#include <donut/engine/FramebufferFactory.h>

#include <imgui.h>

using namespace donut;

namespace hyper
{
    namespace
    {
        nvrhi::TextureDesc UavTarget(const char* name, nvrhi::Format format)
        {
            return nvrhi::TextureDesc().setDebugName(name).setFormat(format).setIsUAV(true);
        }
    }

    PostProcess::PostProcess(Context& ctx)
        : m_Ctx(ctx)
        , m_Bloom(ctx)
        , m_Tonemap(ctx)
        , m_TaaOutput(UavTarget("hyper/taa-output", nvrhi::Format::RGBA16_FLOAT))
        , m_Feedback1(UavTarget("hyper/taa-feedback1", nvrhi::Format::RGBA16_FLOAT))
        , m_Feedback2(UavTarget("hyper/taa-feedback2", nvrhi::Format::RGBA16_FLOAT))
        , m_MotionVectors(nvrhi::TextureDesc().setDebugName("hyper/motion-vectors").setFormat(nvrhi::Format::RG16_FLOAT).setIsRenderTarget(true))
        , m_Combined(UavTarget("hyper/combined", nvrhi::Format::RGBA16_FLOAT))
    {
        m_AddLayout = ctx.device->createBindingLayout(nvrhi::BindingLayoutDesc()
            .setVisibility(nvrhi::ShaderType::Compute)
            .addItem(nvrhi::BindingLayoutItem::PushConstants(0, 4 * sizeof(float)))
            .addItem(nvrhi::BindingLayoutItem::Texture_SRV(0))
            .addItem(nvrhi::BindingLayoutItem::Texture_SRV(1))
            .addItem(nvrhi::BindingLayoutItem::Sampler(0))
            .addItem(nvrhi::BindingLayoutItem::Texture_UAV(0)));
        CreateAddPipeline();
    }

    void PostProcess::CreateAddPipeline()
    {
        auto shader = m_Ctx.shaders->CreateShader("hyper/passes/add_cs", "main", nullptr, nvrhi::ShaderType::Compute);
        if (shader)
            m_AddPipeline = m_Ctx.device->createComputePipeline(
                nvrhi::ComputePipelineDesc().setComputeShader(shader).addBindingLayout(m_AddLayout));
    }

    nvrhi::ITexture* PostProcess::AddOverlay(nvrhi::ICommandList* cmd, nvrhi::ITexture* base, nvrhi::ITexture* overlay)
    {
        if (!m_AddPipeline)
            return base;
        const auto& desc = base->getDesc();
        m_Combined.Resize(m_Ctx.device, desc.width, desc.height);

        auto set = m_Ctx.bindingCache->GetOrCreateBindingSet(nvrhi::BindingSetDesc()
            .addItem(nvrhi::BindingSetItem::PushConstants(0, 4 * sizeof(float)))
            .addItem(nvrhi::BindingSetItem::Texture_SRV(0, base))
            .addItem(nvrhi::BindingSetItem::Texture_SRV(1, overlay))
            .addItem(nvrhi::BindingSetItem::Sampler(0, m_Ctx.common->m_LinearClampSampler))
            .addItem(nvrhi::BindingSetItem::Texture_UAV(0, m_Combined)), m_AddLayout);
        const float constants[4] = { float(desc.width), float(desc.height), 0.f, 0.f };
        cmd->setComputeState(nvrhi::ComputeState().setPipeline(m_AddPipeline).addBindingSet(set));
        cmd->setPushConstants(constants, sizeof(constants));
        cmd->dispatch((desc.width + 7) / 8, (desc.height + 7) / 8);
        return m_Combined;
    }

    // Out of line: Donut's TAA pass needs FramebufferFactory complete to be destroyed.
    PostProcess::~PostProcess() = default;

    void PostProcess::ShadersReloaded()
    {
        m_Bloom.CreatePipelines();
        m_Tonemap.CreatePipelines();
        CreateAddPipeline();
    }

    void PostProcess::PrepareView(engine::PlanarView& view)
    {
        const bool jitter = settings.aa == AntiAliasing::Taa && m_Taa;
        view.SetPixelOffset(jitter ? m_Taa->GetCurrentPixelOffset() : dm::float2::zero());
        view.UpdateCache();
    }

    nvrhi::ITexture* PostProcess::RunTaa(nvrhi::ICommandList* cmd, const Inputs& inputs, const engine::PlanarView& view,
                                         uint32_t outputWidth, uint32_t outputHeight)
    {
        // TAA resolves straight to the output size, so a lower-resolution input is upscaled (TAAU).
        const auto& colorDesc = inputs.color->getDesc();
        const bool outputChanged = m_TaaOutput.Width() != outputWidth || m_TaaOutput.Height() != outputHeight;
        m_TaaOutput.Resize(m_Ctx.device, outputWidth, outputHeight);
        m_Feedback1.Resize(m_Ctx.device, outputWidth, outputHeight);
        m_Feedback2.Resize(m_Ctx.device, outputWidth, outputHeight);
        if (!inputs.motionVectors)
            m_MotionVectors.Resize(m_Ctx.device, colorDesc.width, colorDesc.height);
        nvrhi::ITexture* motion = inputs.motionVectors ? inputs.motionVectors : m_MotionVectors.Get();

        m_OutputView = view;
        m_OutputView.SetViewport(nvrhi::Viewport(float(outputWidth), float(outputHeight)));
        m_OutputView.SetPixelOffset(dm::float2::zero());
        m_OutputView.UpdateCache();

        // Donut's TAA binds its textures at construction, so rebuild it when any of them change.
        if (!m_Taa || m_TaaColor != inputs.color || m_TaaDepth != inputs.depth || m_TaaMotion != motion
            || m_TaaStencilMask != inputs.motionVectorStencilMask || outputChanged)
        {
            render::TemporalAntiAliasingPass::CreateParameters p;
            p.sourceDepth = inputs.depth;
            p.motionVectors = motion;
            p.unresolvedColor = inputs.color;
            p.resolvedColor = m_TaaOutput;
            p.feedback1 = m_Feedback1;
            p.feedback2 = m_Feedback2;
            p.motionVectorStencilMask = inputs.motionVectors ? inputs.motionVectorStencilMask : 0;
            m_Taa = std::make_unique<render::TemporalAntiAliasingPass>(m_Ctx.device, m_Ctx.shaders, m_Ctx.common, view, p);
            m_Taa->SetJitter(render::TemporalAntiAliasingJitter::Halton);
            m_TaaColor = inputs.color;
            m_TaaDepth = inputs.depth;
            m_TaaMotion = motion;
            m_TaaStencilMask = inputs.motionVectorStencilMask;
            m_HistoryValid = false;
        }

        // Camera motion from depth: everywhere without given vectors, or only where the stencil says none were written.
        if (!inputs.motionVectors || inputs.motionVectorStencilMask)
            m_Taa->RenderMotionVectors(cmd, view, m_HistoryValid ? m_PreviousView : view);

        m_Taa->TemporalResolve(cmd, settings.taa, m_HistoryValid, view, m_OutputView);
        m_Taa->AdvanceFrame();
        m_HistoryValid = true;
        return m_TaaOutput;
    }

    void PostProcess::Render(nvrhi::ICommandList* cmd, const Inputs& inputs, const engine::PlanarView& view, nvrhi::IFramebuffer* target)
    {
        auto scope = m_Ctx.profiler->Measure(cmd, "post");

        nvrhi::ITexture* hdr = inputs.color;

        AntiAliasing aa = settings.aa;
        if (aa == AntiAliasing::Taa && !inputs.depth)
        {
            if (!m_WarnedNoDepth)
                log::warning("hyper: TAA needs PostProcess::Inputs::depth; running without anti-aliasing");
            m_WarnedNoDepth = true;
            aa = AntiAliasing::None;
        }

        // The view that matches `hdr`: the output view once TAA has resolved to output size.
        const engine::PlanarView* hdrView = &view;

        if (aa == AntiAliasing::Taa)
        {
            auto t = m_Ctx.profiler->Measure(cmd, "taa");
            const auto& fbInfo = target->getFramebufferInfo();
            hdr = RunTaa(cmd, inputs, view, fbInfo.width, fbInfo.height);
            hdrView = &m_OutputView;
        }
        else
        {
            m_HistoryValid = false;
        }
        m_PreviousView = view;

        if (inputs.overlay)
        {
            auto t = m_Ctx.profiler->Measure(cmd, "overlay");
            hdr = AddOverlay(cmd, hdr, inputs.overlay);
        }

        Tonemap::BloomInput bloom;
        if (settings.bloom && settings.bloomIntensity > 0.f)
        {
            auto t = m_Ctx.profiler->Measure(cmd, "bloom");
            m_Bloom.Render(cmd, hdr, settings.bloomRadius);
            bloom = { m_Bloom.Output(), settings.bloomIntensity, m_Bloom.Normalization() };
        }

        auto t = m_Ctx.profiler->Measure(cmd, "tonemap");
        m_Tonemap.Render(cmd, settings.tonemap, *hdrView, hdr, bloom, target);
    }

    void PostProcess::BuildUI()
    {
        bool taa = settings.aa == AntiAliasing::Taa;
        if (ImGui::Checkbox("TAA", &taa))
            settings.aa = taa ? AntiAliasing::Taa : AntiAliasing::None;
        if (settings.aa == AntiAliasing::Taa)
            ImGui::SliderFloat("TAA new frame weight", &settings.taa.newFrameWeight, 0.01f, 1.f);

        ImGui::Checkbox("bloom", &settings.bloom);
        if (settings.bloom)
        {
            ImGui::SliderFloat("bloom intensity", &settings.bloomIntensity, 0.f, 0.3f);
            ImGui::SliderFloat("bloom radius", &settings.bloomRadius, 0.5f, 3.f);
        }

        TonemapSettings& tm = settings.tonemap;
        const char* opNames[] = { "AgX", "ACES (fitted)", "Khronos Neutral", "Reinhard", "clamp" };
        int op = int(tm.op);
        if (ImGui::Combo("tonemapper", &op, opNames, IM_ARRAYSIZE(opNames)))
            tm.op = TonemapOperator(op);
        ImGui::SliderFloat("exposure EV", &tm.exposureEV, -8.f, 8.f);
        ImGui::Checkbox("auto exposure", &tm.autoExposure);
        ImGui::Checkbox("dither", &tm.dither);
        ImGui::SliderFloat("film grain", &tm.grain, 0.f, 1.f);
        ImGui::SliderFloat("look: contrast (power)", &tm.lookPower, 0.5f, 2.f);
        ImGui::SliderFloat("look: saturation", &tm.lookSaturation, 0.f, 2.f);
    }
}
