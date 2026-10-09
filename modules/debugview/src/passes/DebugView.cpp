#include <hyper/passes/DebugView.h>

#include <hyper/App.h>

namespace hyper
{
    namespace
    {
        struct DebugConstants
        {
            uint32_t channel;
            uint32_t outputLinear;
            float scale;
            float pad;
        };
    }

    DebugView::DebugView(Context& ctx) : m_Ctx(ctx)
    {
        m_Layout = ctx.device->createBindingLayout(nvrhi::BindingLayoutDesc()
            .setVisibility(nvrhi::ShaderType::Pixel)
            .addItem(nvrhi::BindingLayoutItem::PushConstants(0, sizeof(DebugConstants)))
            .addItem(nvrhi::BindingLayoutItem::Texture_SRV(0))
            .addItem(nvrhi::BindingLayoutItem::Sampler(0)));
        CreatePipelines();
    }

    void DebugView::CreatePipelines()
    {
        auto ps = m_Ctx.shaders->CreateShader("hyper/passes/debug_view_ps", "main", nullptr, nvrhi::ShaderType::Pixel);
        if (!ps)
            return;
        m_PixelShader = ps;
        m_Pipeline = nullptr;
    }

    void DebugView::Render(nvrhi::ICommandList* cmd, nvrhi::ITexture* texture, nvrhi::IFramebuffer* target,
                           DebugChannel channel, float scale)
    {
        if (!m_PixelShader || !texture)
            return;

        const auto& fbInfo = target->getFramebufferInfo();
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

        auto bindings = m_Ctx.bindingCache->GetOrCreateBindingSet(nvrhi::BindingSetDesc()
            .addItem(nvrhi::BindingSetItem::PushConstants(0, sizeof(DebugConstants)))
            .addItem(nvrhi::BindingSetItem::Texture_SRV(0, texture))
            .addItem(nvrhi::BindingSetItem::Sampler(0, m_Ctx.common->m_PointClampSampler)), m_Layout);

        DebugConstants c{};
        c.channel = uint32_t(channel);
        c.outputLinear = nvrhi::getFormatInfo(fbInfo.colorFormats.empty() ? nvrhi::Format::UNKNOWN : fbInfo.colorFormats[0]).isSRGB ? 1u : 0u;
        c.scale = scale;

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
