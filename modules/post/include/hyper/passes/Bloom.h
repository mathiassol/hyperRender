#pragma once

#include <hyper/platform.h>

#include <nvrhi/nvrhi.h>

namespace hyper
{
    struct Context;

    // Energy-conserving mip-chain bloom (CoD: Advanced Warfare, 2014).
    // Render() builds the chain from an HDR texture; Output() is the half-res
    // result to blend over the scene: lerp(scene, Output() * Normalization(), intensity).
    // No threshold: everything blooms a little, bright things a lot.
    class Bloom
    {
    public:
        explicit Bloom(Context& ctx);

        void CreatePipelines();

        // radius: upsample filter radius in texels of each level (1 = default).
        void Render(nvrhi::ICommandList* cmd, nvrhi::ITexture* source, float radius = 1.f);

        [[nodiscard]] nvrhi::ITexture* Output() const { return m_Chain; }
        [[nodiscard]] float Normalization() const { return m_Levels ? 1.f / float(m_Levels) : 0.f; }

    private:
        Context& m_Ctx;
        nvrhi::BindingLayoutHandle m_Layout;
        nvrhi::ComputePipelineHandle m_DownsampleKaris;
        nvrhi::ComputePipelineHandle m_Downsample;
        nvrhi::ComputePipelineHandle m_Upsample;
        nvrhi::TextureHandle m_Chain;
        uint32_t m_Levels = 0;
        uint32_t m_SourceWidth = 0;
        uint32_t m_SourceHeight = 0;

        void Dispatch(nvrhi::ICommandList* cmd, nvrhi::IComputePipeline* pipeline,
                      nvrhi::ITexture* src, uint32_t srcMip, uint32_t dstMip, float radius);
    };
}
