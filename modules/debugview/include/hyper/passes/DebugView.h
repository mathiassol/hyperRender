#pragma once

#include <hyper/platform.h>

#include <nvrhi/nvrhi.h>

namespace hyper
{
    struct Context;

    enum class DebugChannel
    {
        Color,    // rgb * scale, Reinhard-compressed for HDR input
        Red,      // .r as grey (AO, masks)
        Alpha,    // .a as grey (confidence, roughness)
        Normal,   // xyz * 0.5 + 0.5
    };

    // Draws one texture over a whole framebuffer, for inspecting intermediates
    // (AO, reflections, G-buffer channels). Output is sRGB-correct for either
    // kind of target.
    class DebugView
    {
    public:
        explicit DebugView(Context& ctx);

        void CreatePipelines();
        void Render(nvrhi::ICommandList* cmd, nvrhi::ITexture* texture, nvrhi::IFramebuffer* target,
                    DebugChannel channel, float scale = 1.f);

    private:
        Context& m_Ctx;
        nvrhi::ShaderHandle m_PixelShader;
        nvrhi::BindingLayoutHandle m_Layout;
        nvrhi::GraphicsPipelineHandle m_Pipeline;
        nvrhi::FramebufferInfo m_PipelineFbInfo;
    };
}
