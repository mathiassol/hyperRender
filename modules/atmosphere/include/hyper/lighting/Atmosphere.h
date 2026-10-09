#pragma once

#include <hyper/platform.h>

#include <donut/core/math/math.h>
#include <nvrhi/nvrhi.h>

#include <memory>

namespace donut::engine { class PlanarView; class DirectionalLight; }
namespace nvrhi { class IFramebuffer; }

namespace hyper
{
    struct Context;

    struct AtmosphereSettings
    {
        bool enabled = true;
        float altitudeKm = 0.1f;          // height of world y = 0 above the ground
        float metersPerUnit = 1.f;        // scene scale
        bool aerialPerspective = true;
        float aerialScale = 1.f;          // exaggerates distances for haze in small scenes
        float aerialMaxKm = 32.f;         // range of the aerial-perspective volume
        float rayleighScale = 1.f;        // air density
        float mieScale = 1.f;             // haze / aerosols
        float ozoneScale = 1.f;
        float mieAnisotropy = 0.8f;
        dm::float3 groundAlbedo = dm::float3(0.3f);
        float sunDiameterDeg = 0.53f;
        float sunDiskScale = 1.f;         // 0 hides the disk
    };

    // Physically based sky and aerial perspective (Hillaire 2020): a few small
    // LUTs, rebuilt per frame except the two that only depend on the settings.
    // Replaces the procedural sky, tints the sun by the atmosphere's
    // transmittance, and provides the sky for the IBL capture.
    class Atmosphere
    {
    public:
        explicit Atmosphere(Context& ctx);
        ~Atmosphere();

        AtmosphereSettings settings;

        // Builds the LUTs for this view and sun. Call once per frame, before Apply/Capture.
        void Update(nvrhi::ICommandList* cmd, const donut::engine::PlanarView& view, const donut::engine::DirectionalLight& sun);

        // The sky and sun disk where the framebuffer's (reverse-Z) depth is empty.
        void RenderSky(nvrhi::ICommandList* cmd, nvrhi::IFramebuffer* framebuffer);

        // Aerial perspective over the geometry of a lit HDR image. color needs
        // UAV access.
        void Apply(nvrhi::ICommandList* cmd, nvrhi::ITexture* color, nvrhi::ITexture* depth);

        // The sky without the sun disk into mip 0 of a cubemap (RGBA16F, UAV).
        void Capture(nvrhi::ICommandList* cmd, nvrhi::ITexture* cube);

        // Sun transmittance through the atmosphere to the camera; the sun colour.
        [[nodiscard]] dm::float3 SunTransmittance(const dm::float3& towardsSun, float cameraWorldY) const;

        // Bumped when a setting that changes the sky changes (for re-capturing IBL).
        [[nodiscard]] uint32_t Revision() const { return m_Revision; }

        // This frame's aerial-perspective volume (screen uv x distance / AerialMaxKm()),
        // for effects that composite over the sky, like clouds (RaymarchedVolumes::Inputs).
        // Luminance per unit sun; null while the atmosphere or its aerial perspective is off.
        [[nodiscard]] nvrhi::ITexture* AerialVolume() const
        {
            return settings.enabled && settings.aerialPerspective ? m_AerialVolume.Get() : nullptr;
        }
        [[nodiscard]] float AerialUnitKm() const { return settings.metersPerUnit * 0.001f * settings.aerialScale; }
        [[nodiscard]] float AerialMaxKm() const { return settings.aerialMaxKm > 0.1f ? settings.aerialMaxKm : 0.1f; }

        void CreatePipelines();
        void BuildUI();

    private:
        bool m_Used = false;   // pipelines are built on first use
        Context& m_Ctx;
        nvrhi::BufferHandle m_Constants;
        nvrhi::BindingLayoutHandle m_LutLayout, m_AerialLayout, m_ApplyLayout, m_CaptureLayout, m_SkyLayout;
        nvrhi::ShaderHandle m_SkyShader;
        nvrhi::GraphicsPipelineHandle m_SkyPipeline;
        nvrhi::FramebufferInfo m_SkyFbInfo;
        nvrhi::ComputePipelineHandle m_Transmittance, m_MultiScatter, m_SkyView, m_Aerial, m_Apply, m_Capture;
        nvrhi::TextureHandle m_TransmittanceLut, m_MultiScatterLut, m_SkyViewLut, m_AerialVolume;

        AtmosphereSettings m_Built;
        bool m_StaticLutsValid = false;
        uint32_t m_Revision = 1;

        // The constants of the last Update, reused (with a new size) by Apply and Capture.
        struct Constants;
        std::unique_ptr<Constants> m_Last;

        void Fill(Constants& c, const donut::engine::PlanarView& view, const donut::engine::DirectionalLight& sun) const;
        void Write(nvrhi::ICommandList* cmd, uint32_t width, uint32_t height);
    };
}
