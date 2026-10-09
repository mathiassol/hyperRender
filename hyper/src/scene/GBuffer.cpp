#include <hyper/scene/GBuffer.h>

#include <donut/engine/FramebufferFactory.h>
#include <nvrhi/utils.h>

namespace hyper
{
    void GBuffer::Init(nvrhi::IDevice* device, dm::uint2 size, dm::uint sampleCount,
                       bool enableMotionVectors, bool useReverseProjection)
    {
        GBufferRenderTargets::Init(device, size, sampleCount, enableMotionVectors, useReverseProjection);

        // Float depth: Donut takes D24S8 first, where reverse-Z
        // gains nothing and everything past ~1,700 km (near plane 0.1 m) rounds to the
        // cleared 0. D32S8 keeps the stencil the motion vectors need.
        const nvrhi::Format floatDepth[] = { nvrhi::Format::D32S8 };
        const nvrhi::Format format = nvrhi::utils::ChooseFormat(device,
            nvrhi::FormatSupport::Texture | nvrhi::FormatSupport::DepthStencil | nvrhi::FormatSupport::ShaderLoad,
            floatDepth, 1);
        if (format != nvrhi::Format::UNKNOWN && format != Depth->getDesc().format)
        {
            nvrhi::TextureDesc desc = Depth->getDesc();
            desc.format = format;
            Depth = device->createTexture(desc);
            GBufferFramebuffer->DepthTarget = Depth;
        }

        HdrColor = device->createTexture(nvrhi::TextureDesc()
            .setDebugName("hyper/gbuffer-hdr")
            .setWidth(size.x)
            .setHeight(size.y)
            .setFormat(nvrhi::Format::RGBA16_FLOAT)
            .setIsRenderTarget(true)
            .setIsUAV(true)
            .setClearValue(nvrhi::Color(0.f))
            .setInitialState(nvrhi::ResourceStates::RenderTarget)
            .setKeepInitialState(true));

        HdrFramebuffer = std::make_shared<donut::engine::FramebufferFactory>(device);
        HdrFramebuffer->RenderTargets = { HdrColor };
        HdrFramebuffer->DepthTarget = Depth;
    }

    nvrhi::DepthStencilState GBuffer::DepthStencil(bool reverseZ)
    {
        const auto mark = nvrhi::DepthStencilState::StencilOpDesc().setPassOp(nvrhi::StencilOp::Replace);
        nvrhi::DepthStencilState state;
        state.enableDepthTest().enableDepthWrite()
            .setDepthFunc(reverseZ ? nvrhi::ComparisonFunc::GreaterOrEqual : nvrhi::ComparisonFunc::LessOrEqual)
            .enableStencil()
            .setStencilReadMask(0)
            .setStencilWriteMask(MotionVectorStencilBit)
            .setStencilRefValue(MotionVectorStencilBit)
            .setFrontFaceStencil(mark)
            .setBackFaceStencil(mark);
        return state;
    }

    void GBuffer::Clear(nvrhi::ICommandList* commandList)
    {
        GBufferRenderTargets::Clear(commandList);
        commandList->clearTextureFloat(HdrColor, nvrhi::AllSubresources, nvrhi::Color(0.f));
    }
}
