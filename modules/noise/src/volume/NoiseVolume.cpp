#include <hyper/volume/NoiseVolume.h>

#include <hyper/App.h>

namespace hyper
{
    namespace
    {
        struct NoiseConstants
        {
            uint32_t size;
            uint32_t period;
            float pad[2];
        };
    }

    NoiseVolume::NoiseVolume(Context& ctx) : m_Ctx(ctx)
    {
        m_Layout = ctx.device->createBindingLayout(nvrhi::BindingLayoutDesc()
            .setVisibility(nvrhi::ShaderType::Compute)
            .addItem(nvrhi::BindingLayoutItem::PushConstants(0, sizeof(NoiseConstants)))
            .addItem(nvrhi::BindingLayoutItem::Texture_UAV(0))
            .addItem(nvrhi::BindingLayoutItem::Texture_UAV(1)));
    }

    void NoiseVolume::CreatePipelines()
    {
        if (!m_Used)
            return;   // built on first use
        const char* entries[] = { "gradient", "perlin_worley", "worley" };
        for (int i = 0; i < 3; ++i)
        {
            auto shader = m_Ctx.shaders->CreateShader("hyper/volume/noise_bake_cs", entries[i], nullptr, nvrhi::ShaderType::Compute);
            if (shader)
                m_Pipelines[i] = m_Ctx.device->createComputePipeline(
                    nvrhi::ComputePipelineDesc().setComputeShader(shader).addBindingLayout(m_Layout));
        }
    }

    nvrhi::TextureHandle NoiseVolume::Bake(nvrhi::ICommandList* cmd, NoiseType type, uint32_t size, uint32_t period, const char* name)
    {
        auto texture = m_Ctx.device->createTexture(nvrhi::TextureDesc()
            .setDebugName(name)
            .setDimension(nvrhi::TextureDimension::Texture3D)
            .setWidth(size).setHeight(size).setDepth(size)
            .setFormat(type == NoiseType::Gradient ? nvrhi::Format::R16_FLOAT : nvrhi::Format::RGBA8_UNORM)
            .setIsUAV(true)
            .setInitialState(nvrhi::ResourceStates::ShaderResource)
            .setKeepInitialState(true));
        BakeInto(cmd, texture, type, period);
        return texture;
    }

    void NoiseVolume::BakeInto(nvrhi::ICommandList* cmd, nvrhi::ITexture* texture, NoiseType type, uint32_t period)
    {
        if (!m_Used)
        {
            m_Used = true;
            CreatePipelines();
        }
        nvrhi::IComputePipeline* pipeline = m_Pipelines[int(type)];
        if (!pipeline || !texture)
            return;
        const uint32_t size = texture->getDesc().width;
        const bool scalar = type == NoiseType::Gradient;
        // The unused slot gets a small dummy of the other kind.
        nvrhi::ITexture* scalarTarget = scalar ? texture : nullptr;
        nvrhi::ITexture* colorTarget = scalar ? nullptr : texture;
        nvrhi::TextureHandle& dummyScalar = m_DummyScalar;
        nvrhi::TextureHandle& dummyColor = m_DummyColor;
        auto dummy = [&](nvrhi::TextureHandle& h, nvrhi::Format f)
        {
            if (!h)
                h = m_Ctx.device->createTexture(nvrhi::TextureDesc().setDebugName("hyper/noise-dummy")
                    .setDimension(nvrhi::TextureDimension::Texture3D).setWidth(1).setHeight(1).setDepth(1)
                    .setFormat(f).setIsUAV(true).setInitialState(nvrhi::ResourceStates::UnorderedAccess).setKeepInitialState(true));
            return h.Get();
        };
        if (!scalarTarget)
            scalarTarget = dummy(dummyScalar, nvrhi::Format::R16_FLOAT);
        if (!colorTarget)
            colorTarget = dummy(dummyColor, nvrhi::Format::RGBA8_UNORM);

        const NoiseConstants c{ size, period, {} };
        auto set = m_Ctx.device->createBindingSet(nvrhi::BindingSetDesc()
            .addItem(nvrhi::BindingSetItem::PushConstants(0, sizeof(NoiseConstants)))
            .addItem(nvrhi::BindingSetItem::Texture_UAV(0, scalarTarget))
            .addItem(nvrhi::BindingSetItem::Texture_UAV(1, colorTarget)), m_Layout);
        cmd->setComputeState(nvrhi::ComputeState().setPipeline(pipeline).addBindingSet(set));
        cmd->setPushConstants(&c, sizeof(c));
        cmd->dispatch((size + 3) / 4, (size + 3) / 4, (size + 3) / 4);
    }
}
