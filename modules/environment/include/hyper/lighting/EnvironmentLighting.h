#pragma once

#include <hyper/platform.h>

#include <donut/core/math/math.h>
#include <donut/render/SkyPass.h>
#include <nvrhi/nvrhi.h>

#include <filesystem>
#include <functional>
#include <memory>

namespace donut::engine { struct LightProbe; class FramebufferFactory; class TextureCache; class DirectionalLight; }
namespace donut::render { class LightProbeProcessingPass; }

namespace hyper
{
    struct Context;

    // A sky for the Sky source to capture instead of Donut's procedural one. The
    // atmosphere's:
    //     hyper::SkyCapture sky{ [&](nvrhi::ICommandList* c, nvrhi::ITexture* cube) { atmosphere.Capture(c, cube); },
    //                            atmosphere.Revision() };
    struct SkyCapture
    {
        std::function<void(nvrhi::ICommandList*, nvrhi::ITexture*)> render;   // into mip 0 of the cubemap
        uint32_t revision = 0;   // changes whenever the sky does
    };

    enum class EnvironmentSource
    {
        Sky,    // capture the sky: a SkyCapture if given, else Donut's procedural sky (follows the sun)
        File,   // an equirectangular HDR image (.hdr / .exr), also drawn as the background
    };

    struct EnvironmentSettings
    {
        bool enabled = true;
        EnvironmentSource source = EnvironmentSource::Sky;
        std::filesystem::path file;       // for EnvironmentSource::File
        float fileIntensity = 1.f;
        float fileRotationDeg = 0.f;
        float diffuseScale = 1.f;
        float specularScale = 1.f;
    };

    // Image-based lighting: one global light probe (diffuse irradiance +
    // prefiltered specular + BRDF LUT) built from the sky or an HDR image, with
    // Donut's LightProbeProcessingPass. It re-captures only when the sky, sun
    // or source changes. Deferred and forward shading both sample the probe.
    class EnvironmentLighting
    {
    public:
        explicit EnvironmentLighting(Context& ctx);
        ~EnvironmentLighting();

        EnvironmentSettings settings;

        // capture: when given (and up to date this frame), the Sky source captures it.
        void Update(nvrhi::ICommandList* cmd, const donut::engine::DirectionalLight& sun, const donut::render::SkyParameters& sky,
                    const SkyCapture* capture = nullptr);

        // The probe to hand to lighting passes; null when disabled.
        [[nodiscard]] std::shared_ptr<donut::engine::LightProbe> Probe() const;
        // The captured environment cubemap (mipmapped), e.g. for a background; null until
        // the first Update with the module enabled.
        [[nodiscard]] nvrhi::ITexture* EnvironmentMap() const { return m_EnvCube; }
        [[nodiscard]] bool UsesFile() const { return settings.enabled && settings.source == EnvironmentSource::File && m_FileTexture; }
        // Increases with every capture, so dependents know the environment changed.
        [[nodiscard]] uint32_t Revision() const { return m_Revision; }

        void CreatePipelines();
        void BuildUI();
        void Invalidate() { m_Dirty = true; }

    private:
        bool m_Used = false;   // resources and pipelines are made on first use
        void Allocate();
        Context& m_Ctx;
        std::unique_ptr<donut::render::LightProbeProcessingPass> m_ProbePass;     // mips, specular, BRDF LUT
        std::unique_ptr<donut::render::LightProbeProcessingPass> m_DiffusePass;   // smaller intermediate
        std::shared_ptr<donut::engine::LightProbe> m_Probe;
        std::shared_ptr<donut::engine::TextureCache> m_TextureCache;
        std::shared_ptr<donut::engine::FramebufferFactory> m_CubeFramebuffer;
        std::unique_ptr<donut::render::SkyPass> m_SkyPass;
        nvrhi::TextureHandle m_EnvCube;
        nvrhi::TextureHandle m_EnvDepth;
        nvrhi::TextureHandle m_FileTexture;
        std::filesystem::path m_LoadedFile;
        nvrhi::BindingLayoutHandle m_EquirectLayout;
        nvrhi::ComputePipelineHandle m_EquirectPipeline;
        bool m_BrdfDone = false;
        bool m_Dirty = true;
        uint32_t m_Revision = 0;

        // Last captured inputs, to detect changes.
        dm::double3 m_SunDir{ 0.0 };
        float m_SunIrradiance = -1.f;
        donut::render::SkyParameters m_Sky;
        EnvironmentSettings m_Captured;
        uint32_t m_CaptureRevision = 0;   // 0: captured without one

        bool NeedsCapture(const donut::engine::DirectionalLight& sun, const donut::render::SkyParameters& sky,
                          const SkyCapture* capture) const;
    };
}
