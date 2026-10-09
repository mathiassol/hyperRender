#pragma once

#include <hyper/platform.h>

#include <nvrhi/nvrhi.h>

#include <algorithm>
#include <cmath>

namespace hyper
{
    // A texture that follows the output size. Call Resize() every frame with
    // the framebuffer size; it recreates the texture only when the size changes.
    //
    //     hyper::RenderTarget m_Hdr{ nvrhi::TextureDesc().setFormat(RGBA16_FLOAT).setIsUAV(true) };
    //     m_Hdr.Resize(device, fbInfo.width, fbInfo.height);
    //
    // Width/height/dimension default sensibly; debug name, format and usage
    // flags come from the desc. Binding sets that reference it should come
    // from Context::bindingCache, which hyper clears on resize.
    class RenderTarget
    {
    public:
        RenderTarget() = default;
        explicit RenderTarget(nvrhi::TextureDesc desc, float scale = 1.f) : m_Desc(std::move(desc)), m_Scale(scale)
        {
            if (m_Desc.isUAV || m_Desc.isRenderTarget)
            {
                m_Desc.keepInitialState = true;
                if (m_Desc.initialState == nvrhi::ResourceStates::Unknown)
                    m_Desc.initialState = nvrhi::ResourceStates::ShaderResource;
            }
        }

        // Returns true if the texture was (re)created.
        bool Resize(nvrhi::IDevice* device, uint32_t width, uint32_t height)
        {
            const uint32_t w = std::max(1u, uint32_t(std::lround(float(width) * m_Scale)));
            const uint32_t h = std::max(1u, uint32_t(std::lround(float(height) * m_Scale)));
            if (m_Texture && w == m_Desc.width && h == m_Desc.height)
                return false;

            m_Desc.width = w;
            m_Desc.height = h;
            m_Texture = device->createTexture(m_Desc);
            return true;
        }

        void SetScale(float scale) { m_Scale = scale; }

        [[nodiscard]] nvrhi::ITexture* Get() const { return m_Texture; }
        operator nvrhi::ITexture*() const { return m_Texture; }
        [[nodiscard]] uint32_t Width() const { return m_Desc.width; }
        [[nodiscard]] uint32_t Height() const { return m_Desc.height; }
        [[nodiscard]] const nvrhi::TextureDesc& Desc() const { return m_Desc; }

    private:
        nvrhi::TextureDesc m_Desc;
        float m_Scale = 1.f;
        nvrhi::TextureHandle m_Texture;
    };
}
