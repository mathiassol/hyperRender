#include <hyper/passes/Tonemap.h>

#include <hyper/App.h>

#include <donut/engine/FramebufferFactory.h>
#include <donut/engine/View.h>
#include <donut/render/ToneMappingPasses.h>

#include <cmath>

using namespace donut;

namespace hyper
{
    namespace
    {
        struct TonemapConstants
        {
            float exposureScale;
            float bloomIntensity;
            float bloomNorm;
            uint32_t op;
            uint32_t flags;
            uint32_t frameIndex;
            float grain;
            float power;
            float saturation;
            float pad0, pad1, pad2;
        };

        constexpr uint32_t c_FlagAutoExposure = 1;
        constexpr uint32_t c_FlagDither = 2;
        constexpr uint32_t c_FlagEncodeSrgb = 4;
    }

    Tonemap::Tonemap(Context& ctx) : m_Ctx(ctx)
    {
        m_Layout = ctx.device->createBindingLayout(nvrhi::BindingLayoutDesc()
            .setVisibility(nvrhi::ShaderType::Pixel)
            .addItem(nvrhi::BindingLayoutItem::PushConstants(0, sizeof(TonemapConstants)))
            .addItem(nvrhi::BindingLayoutItem::Texture_SRV(0))
            .addItem(nvrhi::BindingLayoutItem::Texture_SRV(1))
            .addItem(nvrhi::BindingLayoutItem::TypedBuffer_SRV(2))
            .addItem(nvrhi::BindingLayoutItem::Sampler(0)));

        m_ExposureBuffer = ctx.device->createBuffer(nvrhi::BufferDesc()
            .setDebugName("hyper/exposure")
            .setByteSize(sizeof(uint32_t))
            .setFormat(nvrhi::Format::R32_UINT)
            .setCanHaveUAVs(true)
            .setCanHaveTypedViews(true)
            .setInitialState(nvrhi::ResourceStates::UnorderedAccess)
            .setKeepInitialState(true));

        // Donut's ToneMappingPass wants a framebuffer to build its own (unused) draw pipeline.
        m_DummyFramebuffer = std::make_shared<engine::FramebufferFactory>(ctx.device);
        m_DummyFramebuffer->RenderTargets = { ctx.device->createTexture(nvrhi::TextureDesc()
            .setDebugName("hyper/tonemap-dummy")
            .setFormat(nvrhi::Format::RGBA8_UNORM)
            .setIsRenderTarget(true)
            .setInitialState(nvrhi::ResourceStates::RenderTarget)
            .setKeepInitialState(true)) };

        CreatePipelines();
    }

    Tonemap::~Tonemap() = default;

    void Tonemap::CreatePipelines()
    {
        auto ps = m_Ctx.shaders->CreateShader("hyper/passes/tonemap_ps", "main", nullptr, nvrhi::ShaderType::Pixel);
        if (!ps)
            return;
        m_PixelShader = ps;
        m_Pipeline = nullptr;   // rebuilt on the next Render for the target's format
    }

    void Tonemap::Render(nvrhi::ICommandList* cmd, const TonemapSettings& settings, const engine::PlanarView& view,
                         nvrhi::ITexture* hdr, const BloomInput& bloom, nvrhi::IFramebuffer* target)
    {
        if (!m_PixelShader)
            return;

        if (settings.autoExposure)
        {
            if (!m_Exposure || m_ExposureSource != hdr)
            {
                render::ToneMappingPass::CreateParameters params;
                params.exposureBufferOverride = m_ExposureBuffer;
                m_Exposure = std::make_unique<render::ToneMappingPass>(
                    m_Ctx.device, m_Ctx.shaders, m_Ctx.common, m_DummyFramebuffer, view, params);
                m_ExposureSource = hdr;
            }

            if (!m_ExposureReset)
            {
                m_Exposure->ResetExposure(cmd, 0.18f);
                m_ExposureReset = true;
            }

            render::ToneMappingParameters p;
            p.eyeAdaptationSpeedUp = settings.adaptSpeedUp;
            p.eyeAdaptationSpeedDown = settings.adaptSpeedDown;
            p.minAdaptedLuminance = settings.minLuminance;
            p.maxAdaptedLuminance = settings.maxLuminance;
            p.histogramLowPercentile = settings.histogramLow;
            p.histogramHighPercentile = settings.histogramHigh;

            m_Exposure->AdvanceFrame(m_Ctx.deltaTime);
            m_Exposure->ResetHistogram(cmd);
            m_Exposure->AddFrameToHistogram(cmd, view, hdr);
            m_Exposure->ComputeExposure(cmd, p);
        }

        const auto& fbInfo = target->getFramebufferInfo();
        // An sRGB target encodes on write, so it gets linear values; anything else gets encoded ones.
        const bool encodeSrgb = fbInfo.colorFormats.empty() || !nvrhi::getFormatInfo(fbInfo.colorFormats[0]).isSRGB;
        if (!m_Pipeline || !(m_PipelineFbInfo == fbInfo))
        {
            nvrhi::GraphicsPipelineDesc desc;
            desc.primType = nvrhi::PrimitiveType::TriangleStrip;
            desc.VS = m_Ctx.common->m_FullscreenVS;
            desc.PS = m_PixelShader;
            desc.bindingLayouts = { m_Layout };
            desc.renderState.rasterState.setCullNone();
            desc.renderState.depthStencilState.depthTestEnable = false;
            desc.renderState.depthStencilState.stencilEnable = false;
            m_Pipeline = m_Ctx.device->createGraphicsPipeline(desc, fbInfo);
            m_PipelineFbInfo = fbInfo;
        }

        nvrhi::ITexture* bloomTexture = bloom.texture ? bloom.texture : m_Ctx.common->m_BlackTexture.Get();
        auto bindings = m_Ctx.bindingCache->GetOrCreateBindingSet(nvrhi::BindingSetDesc()
            .addItem(nvrhi::BindingSetItem::PushConstants(0, sizeof(TonemapConstants)))
            .addItem(nvrhi::BindingSetItem::Texture_SRV(0, hdr))
            .addItem(nvrhi::BindingSetItem::Texture_SRV(1, bloomTexture, nvrhi::Format::UNKNOWN, nvrhi::TextureSubresourceSet(0, 1, 0, 1)))
            .addItem(nvrhi::BindingSetItem::TypedBuffer_SRV(2, m_ExposureBuffer))
            .addItem(nvrhi::BindingSetItem::Sampler(0, m_Ctx.common->m_LinearClampSampler)), m_Layout);

        TonemapConstants c{};
        c.exposureScale = std::exp2(settings.exposureEV);
        c.bloomIntensity = bloom.texture ? bloom.intensity : 0.f;
        c.bloomNorm = bloom.normalization;
        c.op = uint32_t(settings.op);
        c.flags = (settings.autoExposure ? c_FlagAutoExposure : 0)
                | (settings.dither ? c_FlagDither : 0)
                | (encodeSrgb ? c_FlagEncodeSrgb : 0);
        c.frameIndex = uint32_t(m_Ctx.frameIndex);
        c.grain = settings.grain;
        c.power = settings.lookPower;
        c.saturation = settings.lookSaturation;

        nvrhi::GraphicsState state;
        state.pipeline = m_Pipeline;
        state.framebuffer = target;
        state.bindings = { bindings };
        state.viewport.addViewportAndScissorRect(fbInfo.getViewport());
        cmd->setGraphicsState(state);
        cmd->setPushConstants(&c, sizeof(c));
        cmd->draw(nvrhi::DrawArguments().setVertexCount(4));
    }
}
