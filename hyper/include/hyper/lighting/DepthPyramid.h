#pragma once

#include <hyper/platform.h>

#include <nvrhi/nvrhi.h>

namespace hyper
{
    struct Context;

    // Hi-Z pyramid of a reverse-Z depth buffer: RG32_FLOAT, full mip chain,
    // x = nearest depth (max), y = farthest (min) per footprint. Built once per
    // frame and shared by SSR and (later) occlusion culling.
    class DepthPyramid
    {
    public:
        explicit DepthPyramid(Context& ctx);

        void Build(nvrhi::ICommandList* cmd, nvrhi::ITexture* depth);
        void CreatePipelines();

        [[nodiscard]] nvrhi::ITexture* Texture() const { return m_Pyramid; }
        [[nodiscard]] uint32_t MipLevels() const { return m_Mips; }

    private:
        bool m_Used = false;   // pipelines are built on first use
        Context& m_Ctx;
        nvrhi::BindingLayoutHandle m_Layout;
        nvrhi::ComputePipelineHandle m_Copy;
        nvrhi::ComputePipelineHandle m_Reduce;
        nvrhi::TextureHandle m_Pyramid;
        uint32_t m_Width = 0;
        uint32_t m_Height = 0;
        uint32_t m_Mips = 0;
    };
}
