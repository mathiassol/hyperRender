#include <hyper/passes/Bloom.h>

#include <hyper/App.h>

#include <donut/core/math/math.h>

using namespace donut;

namespace hyper
{
    namespace
    {
        struct BloomConstants
        {
            float srcTexelSize[2];
            float dstSize[2];
            float filterRadius;
            float pad[3];
        };

        constexpr uint32_t c_MaxLevels = 7;
        constexpr uint32_t c_MinLevelSize = 8;
    }

    Bloom::Bloom(Context& ctx) : m_Ctx(ctx)
    {
        m_Layout = ctx.device->createBindingLayout(nvrhi::BindingLayoutDesc()
            .setVisibility(nvrhi::ShaderType::Compute)
            .addItem(nvrhi::BindingLayoutItem::PushConstants(0, sizeof(BloomConstants)))
            .addItem(nvrhi::BindingLayoutItem::Texture_SRV(0))
            .addItem(nvrhi::BindingLayoutItem::Texture_UAV(0))
            .addItem(nvrhi::BindingLayoutItem::Sampler(0)));
        CreatePipelines();
    }

    void Bloom::CreatePipelines()
    {
        auto make = [&](const char* entry, const char* karis) -> nvrhi::ComputePipelineHandle
        {
            std::vector<engine::ShaderMacro> defines;
            if (karis)
                defines.emplace_back("KARIS", karis);
            auto shader = m_Ctx.shaders->CreateShader("hyper/passes/bloom_cs", entry, &defines, nvrhi::ShaderType::Compute);
            if (!shader)
                return nullptr;
            return m_Ctx.device->createComputePipeline(
                nvrhi::ComputePipelineDesc().setComputeShader(shader).addBindingLayout(m_Layout));
        };

        auto downKaris = make("downsample", "1");
        auto down = make("downsample", "0");
        auto up = make("upsample", nullptr);
        if (downKaris && down && up)
        {
            m_DownsampleKaris = downKaris;
            m_Downsample = down;
            m_Upsample = up;
        }
    }

    void Bloom::Dispatch(nvrhi::ICommandList* cmd, nvrhi::IComputePipeline* pipeline,
                         nvrhi::ITexture* src, uint32_t srcMip, uint32_t dstMip, float radius)
    {
        const auto& srcDesc = src->getDesc();
        const uint32_t srcW = std::max(1u, srcDesc.width >> srcMip);
        const uint32_t srcH = std::max(1u, srcDesc.height >> srcMip);
        const auto& chainDesc = m_Chain->getDesc();
        const uint32_t dstW = std::max(1u, chainDesc.width >> dstMip);
        const uint32_t dstH = std::max(1u, chainDesc.height >> dstMip);

        auto bindings = m_Ctx.bindingCache->GetOrCreateBindingSet(nvrhi::BindingSetDesc()
            .addItem(nvrhi::BindingSetItem::PushConstants(0, sizeof(BloomConstants)))
            .addItem(nvrhi::BindingSetItem::Texture_SRV(0, src, nvrhi::Format::UNKNOWN, nvrhi::TextureSubresourceSet(srcMip, 1, 0, 1)))
            .addItem(nvrhi::BindingSetItem::Texture_UAV(0, m_Chain, nvrhi::Format::UNKNOWN, nvrhi::TextureSubresourceSet(dstMip, 1, 0, 1)))
            .addItem(nvrhi::BindingSetItem::Sampler(0, m_Ctx.common->m_LinearClampSampler)), m_Layout);

        BloomConstants c{};
        c.srcTexelSize[0] = 1.f / float(srcW);
        c.srcTexelSize[1] = 1.f / float(srcH);
        c.dstSize[0] = float(dstW);
        c.dstSize[1] = float(dstH);
        c.filterRadius = radius;

        cmd->setComputeState(nvrhi::ComputeState().setPipeline(pipeline).addBindingSet(bindings));
        cmd->setPushConstants(&c, sizeof(c));
        cmd->dispatch((dstW + 7) / 8, (dstH + 7) / 8);
    }

    void Bloom::Render(nvrhi::ICommandList* cmd, nvrhi::ITexture* source, float radius)
    {
        if (!m_Downsample)
            return;

        const auto& srcDesc = source->getDesc();
        if (!m_Chain || srcDesc.width != m_SourceWidth || srcDesc.height != m_SourceHeight)
        {
            m_SourceWidth = srcDesc.width;
            m_SourceHeight = srcDesc.height;
            const uint32_t w = std::max(1u, m_SourceWidth / 2);
            const uint32_t h = std::max(1u, m_SourceHeight / 2);

            m_Levels = 1;
            while (m_Levels < c_MaxLevels && (w >> m_Levels) >= c_MinLevelSize && (h >> m_Levels) >= c_MinLevelSize)
                ++m_Levels;

            m_Chain = m_Ctx.device->createTexture(nvrhi::TextureDesc()
                .setDebugName("hyper/bloom")
                .setWidth(w)
                .setHeight(h)
                .setMipLevels(m_Levels)
                .setFormat(nvrhi::Format::RGBA16_FLOAT)
                .setIsUAV(true)
                .setInitialState(nvrhi::ResourceStates::ShaderResource)
                .setKeepInitialState(true));
        }

        // Down: source -> level 0 -> ... -> level N-1
        Dispatch(cmd, m_DownsampleKaris, source, 0, 0, radius);
        for (uint32_t i = 1; i < m_Levels; ++i)
            Dispatch(cmd, m_Downsample, m_Chain, i - 1, i, radius);

        // Up: each level accumulates the one below it, so level 0 ends up with the sum of all.
        for (uint32_t i = m_Levels - 1; i > 0; --i)
            Dispatch(cmd, m_Upsample, m_Chain, i, i - 1, radius);
    }
}
