#pragma once

#include <hyper/platform.h>

#include <hyper/volume/NoiseVolume.h>

#include <donut/core/math/math.h>
#include <nvrhi/nvrhi.h>

#include <memory>
#include <vector>

namespace donut::engine { class PlanarView; struct LightProbe; }

namespace hyper
{
    struct Context;

    // A bounded volume of noisy media: smoke, dust, a nebula. Emission makes it glow.
    struct MediaVolume
    {
        dm::float3 centre = dm::float3(0.f);
        dm::float3 halfExtent = dm::float3(1.f);
        float density = 1.f;                 // extinction per world unit
        dm::float3 albedo = dm::float3(0.9f);
        dm::float3 emission = dm::float3(0.f);   // radiance per unit density
        float noiseStrength = 0.8f;
        float noiseScale = 0.5f;             // cycles per world unit
        bool sphere = true;
    };

    // A cumulus layer (Schneider's Nubis model, simplified).
    struct CloudLayerSettings
    {
        bool enabled = false;
        float bottom = 1500.f;               // world units (metres by default)
        float top = 4000.f;
        float coverage = 0.45f;              // 0 clear, 1 overcast
        float density = 0.02f;               // extinction per world unit
        float shapeScale = 1.f / 3000.f;     // shape noise cycles per world unit
        float detailScale = 1.f / 400.f;
        float weatherScale = 1.f / 20000.f;  // coverage map cycles per world unit
        dm::float3 wind = dm::float3(12.f, 0.f, 4.f);   // units per second
        dm::float3 albedo = dm::float3(1.f);
        float powder = 0.6f;
        float forwardG = 0.8f, backG = 0.3f, lobeBlend = 0.3f;
    };

    struct VolumeSettings
    {
        bool enabled = true;
        int steps = 64;
        int lightSteps = 6;
        float maxDistance = 30000.f;
        float temporalWeight = 0.9f;
        float sunIntensity = 1.f;
        float ambientIntensity = 1.f;
        std::vector<MediaVolume> volumes;    // up to 8
        CloudLayerSettings clouds;
    };

    // Ray-marched participating media at half resolution: MediaVolumes and a
    // cloud layer, lit by the sun (a short light march: self-shadowing), the sky
    // (IBL) and their own emission, accumulated over frames and upsampled with
    // depth awareness. Works without a hyper::Scene (see apps/clouds).
    class RaymarchedVolumes
    {
    public:
        explicit RaymarchedVolumes(Context& ctx);
        ~RaymarchedVolumes();

        VolumeSettings settings;

        struct Inputs
        {
            nvrhi::ITexture* color = nullptr;       // lit HDR, UAV, full resolution
            nvrhi::ITexture* depth = nullptr;       // reverse-Z, single sample
            dm::float3 sunDirection = dm::float3(0.f, -1.f, 0.f);   // travel direction, like Donut's lights
            dm::float3 sunRadiance = dm::float3(3.f);               // colour * irradiance
            float sunIrradiance = 3.f;
            const donut::engine::LightProbe* probe = nullptr;       // sky ambient
            // Aerial perspective over distant media: Atmosphere::AerialVolume(), AerialMaxKm(), AerialUnitKm().
            nvrhi::ITexture* aerialVolume = nullptr;
            float aerialMaxKm = 1.f;
            float aerialUnitKm = 0.001f;
        };

        [[nodiscard]] bool HasWork() const;
        void Render(nvrhi::ICommandList* cmd, const donut::engine::PlanarView& view,
                    const donut::engine::PlanarView& previous, const Inputs& inputs);
        void CreatePipelines();
        void BuildUI();

    private:
        Context& m_Ctx;
        NoiseVolume m_NoiseBaker;
        nvrhi::TextureHandle m_Shape, m_Detail;
        nvrhi::TextureHandle m_March, m_Distances, m_History[2];
        nvrhi::TextureHandle m_DummyUav;   // for UAV slots a pass leaves unused
        nvrhi::BufferHandle m_Constants;
        nvrhi::BindingLayoutHandle m_Layout;
        nvrhi::ComputePipelineHandle m_MarchPipeline, m_Temporal, m_Composite;
        uint32_t m_Frame = 0;
        bool m_HistoryValid = false;
        bool m_Used = false;   // pipelines exist once there is work
        float m_Time = 0.f;

        struct Constants;
        std::unique_ptr<Constants> m_Data;
    };
}
