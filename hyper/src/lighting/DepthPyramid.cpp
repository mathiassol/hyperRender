#include <hyper/lighting/DepthPyramid.h>

#include <hyper/App.h>

#include <algorithm>

namespace hyper
{
    namespace
    {
        struct HizConstants
        {
            uint32_t dstSize[2];
            uint32_t srcSize[2];
        };
    }

    DepthPyramid::DepthPyramid(Context& ctx) : m_Ctx(ctx)
    {
        m_Layout = ctx.device->createBindingLayout(nvrhi::BindingLayoutDesc()
            .setVisibility(nvrhi::ShaderType::Compute)
            .addItem(nvrhi::BindingLayoutItem::PushConstants(0, sizeof(HizConstants)))
            .addItem(nvrhi::BindingLayoutItem::Texture_SRV(0))
            .addItem(nvrhi::BindingLayoutItem::Texture_SRV(1))
            .addItem(nvrhi::BindingLayoutItem::Texture_UAV(0)));
    }

    void DepthPyramid::CreatePipelines()
    {
        if (!m_Used)
            return;   // built on first use
        auto make = [&](const char* entry) -> nvrhi::ComputePipelineHandle
        {
            auto shader = m_Ctx.shaders->CreateShader("hyper/lighting/hiz_cs", entry, nullptr, nvrhi::ShaderType::Compute);
            return shader ? m_Ctx.device->createComputePipeline(
                nvrhi::ComputePipelineDesc().setComputeShader(shader).addBindingLayout(m_Layout)) : nullptr;
        };
        auto copy = make("copy");
        auto reduce = make("reduce");
        if (copy && reduce)
        {
            m_Copy = copy;
            m_Reduce = reduce;
        }
    }

    void DepthPyramid::Build(nvrhi::ICommandList* cmd, nvrhi::ITexture* depth)
    {
        if (!m_Used)
        {
            m_Used = true;
            CreatePipelines();
        }
        if (!m_Copy)
            return;

        const auto& desc = depth->getDesc();
        if (!m_Pyramid || desc.width != m_Width || desc.height != m_Height)
        {
            m_Width = desc.width;
            m_Height = desc.height;
            m_Mips = 1;
            while ((std::max(m_Width, m_Height) >> m_Mips) > 0)
                ++m_Mips;
            m_Pyramid = m_Ctx.device->createTexture(nvrhi::TextureDesc()
                .setDebugName("hyper/hiz")
                .setWidth(m_Width).setHeight(m_Height).setMipLevels(m_Mips)
                .setFormat(nvrhi::Format::RG32_FLOAT)
                .setIsUAV(true)
                .setInitialState(nvrhi::ResourceStates::ShaderResource)
                .setKeepInitialState(true));
        }

        auto scope = m_Ctx.profiler->Measure(cmd, "hi-z");
        for (uint32_t mip = 0; mip < m_Mips; ++mip)
        {
            const uint32_t w = std::max(1u, m_Width >> mip), h = std::max(1u, m_Height >> mip);
            const uint32_t sw = std::max(1u, m_Width >> (mip ? mip - 1 : 0)), sh = std::max(1u, m_Height >> (mip ? mip - 1 : 0));
            HizConstants c{ { w, h }, { sw, sh } };

            nvrhi::BindingSetDesc set;
            set.addItem(nvrhi::BindingSetItem::PushConstants(0, sizeof(c)));
            set.addItem(nvrhi::BindingSetItem::Texture_SRV(0, mip == 0 ? depth : m_Ctx.common->m_BlackTexture.Get()));
            set.addItem(mip == 0
                ? nvrhi::BindingSetItem::Texture_SRV(1, m_Ctx.common->m_BlackTexture)
                : nvrhi::BindingSetItem::Texture_SRV(1, m_Pyramid, nvrhi::Format::UNKNOWN, nvrhi::TextureSubresourceSet(mip - 1, 1, 0, 1)));
            set.addItem(nvrhi::BindingSetItem::Texture_UAV(0, m_Pyramid, nvrhi::Format::UNKNOWN, nvrhi::TextureSubresourceSet(mip, 1, 0, 1)));

            cmd->setComputeState(nvrhi::ComputeState()
                .setPipeline(mip == 0 ? m_Copy : m_Reduce)
                .addBindingSet(m_Ctx.bindingCache->GetOrCreateBindingSet(set, m_Layout)));
            cmd->setPushConstants(&c, sizeof(c));
            cmd->dispatch((w + 7) / 8, (h + 7) / 8);
        }
    }
}
