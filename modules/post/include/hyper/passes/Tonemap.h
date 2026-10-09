#pragma once

#include <hyper/platform.h>

#include <nvrhi/nvrhi.h>

#include <memory>

namespace donut::engine { class PlanarView; class FramebufferFactory; }
namespace donut::render { class ToneMappingPass; }

namespace hyper
{
    struct Context;

    enum class TonemapOperator : uint32_t
    {
        AgX,            // default: graceful highlight desaturation, no hue skews
        AcesFitted,     // punchy filmic (Hill's fit)
        KhronosNeutral, // keeps base colours accurate, for product/PBR checks
        Reinhard,
        Clamp,          // no curve: see raw values
    };

    struct TonemapSettings
    {
        TonemapOperator op = TonemapOperator::AgX;
        float exposureEV = 0.f;         // manual, and a bias on top of auto exposure
        bool autoExposure = true;       // Donut's luminance histogram, maps the average to middle grey
        float adaptSpeedUp = 3.f;
        float adaptSpeedDown = 2.f;
        float minLuminance = 0.001f;    // clamp for the adapted luminance
        float maxLuminance = 16.f;
        float histogramLow = 0.6f;      // percentiles of the histogram that count
        float histogramHigh = 0.95f;
        bool dither = true;
        float grain = 0.f;              // temporal film grain, 0..1
        // A look after the curve, on its output with a 2.2 gamma (where Blender's AgX looks
        // work): a power (contrast; over 1 deepens the mid-tones) and a saturation about
        // the Rec. 709 luminance. 1, 1 is none; Blender's "Punchy" is 1.35, 1.4.
        float lookPower = 1.f;
        float lookSaturation = 1.f;
    };

    // HDR -> display. Composites bloom, applies exposure and the curve, writes
    // into any framebuffer (an sRGB swap chain, or a UNORM one).
    class Tonemap
    {
    public:
        explicit Tonemap(Context& ctx);
        ~Tonemap();

        void CreatePipelines();

        struct BloomInput
        {
            nvrhi::ITexture* texture = nullptr;
            float intensity = 0.f;
            float normalization = 0.f;
        };

        // sRGB targets get linear values (the hardware encodes); other targets get
        // sRGB-encoded values.
        void Render(nvrhi::ICommandList* cmd, const TonemapSettings& settings, const donut::engine::PlanarView& view,
                    nvrhi::ITexture* hdr, const BloomInput& bloom, nvrhi::IFramebuffer* target);

        // The adapted luminance, as Donut writes it (R32_UINT holding a float).
        [[nodiscard]] nvrhi::IBuffer* ExposureBuffer() const { return m_ExposureBuffer; }

    private:
        Context& m_Ctx;
        nvrhi::ShaderHandle m_PixelShader;
        nvrhi::BindingLayoutHandle m_Layout;
        nvrhi::GraphicsPipelineHandle m_Pipeline;
        nvrhi::FramebufferInfo m_PipelineFbInfo;
        nvrhi::BufferHandle m_ExposureBuffer;
        bool m_ExposureReset = false;

        // Donut's histogram + exposure passes; rebuilt when the HDR texture changes
        // (it caches binding sets per texture).
        std::unique_ptr<donut::render::ToneMappingPass> m_Exposure;
        std::shared_ptr<donut::engine::FramebufferFactory> m_DummyFramebuffer;
        nvrhi::ITexture* m_ExposureSource = nullptr;
    };
}
