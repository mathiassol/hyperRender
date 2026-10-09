#pragma once

#include <hyper/platform.h>

#include <hyper/rt/RtScene.h>
#include <hyper/volume/NoiseVolume.h>

#include <donut/core/math/math.h>
#include <nvrhi/nvrhi.h>

#include <memory>
#include <vector>

namespace donut::engine { class PlanarView; struct LightProbe; }

namespace hyper
{
    struct Context;
    class Scene;
    class RtLighting;

    // A local fog volume: extra density inside a box or sphere, soft at the edges.
    struct FogVolume
    {
        dm::float3 centre = dm::float3(0.f);
        dm::float3 halfExtent = dm::float3(1.f);
        float density = 0.2f;
        bool sphere = false;
    };

    struct FogSettings
    {
        bool enabled = false;            // opt in
        float density = 0.01f;           // extinction per world unit at baseHeight and below
        float heightFalloff = 0.2f;      // per unit above baseHeight
        float baseHeight = 0.f;
        dm::float3 albedo = dm::float3(0.9f);
        float anisotropy = 0.6f;         // Henyey-Greenstein g: forward scattering gives light shafts
        float noiseStrength = 0.5f;
        float noiseScale = 0.25f;        // noise cycles per world unit
        dm::float3 wind = dm::float3(0.6f, 0.f, 0.2f);   // units per second
        float sunIntensity = 1.f;
        float ambientIntensity = 1.f;
        float localLightIntensity = 1.f;
        float maxDistance = 64.f;        // grid range; beyond it the atmosphere's aerial perspective takes over
        int depthSlices = 64;
        int tileSize = 8;                // screen pixels per froxel
        float temporalWeight = 0.9f;
        bool shadows = true;
        std::vector<FogVolume> volumes;  // up to 8
    };

    // Froxel volumetric fog: height fog + noise + local volumes, lit by the sun
    // (shadowed by ray queries or the cascaded shadow map), point and spot lights
    // (unshadowed) and ambient light (DDGI or IBL), reprojected over frames and
    // applied to a lit HDR image. Apply it after aerial perspective: fog is the
    // nearer medium.
    class VolumetricFog
    {
    public:
        explicit VolumetricFog(Context& ctx);
        ~VolumetricFog();

        FogSettings settings;

        struct Inputs
        {
            nvrhi::ITexture* color = nullptr;       // lit HDR, UAV
            nvrhi::ITexture* depth = nullptr;       // reverse-Z
            const donut::engine::LightProbe* probe = nullptr;   // ambient without ray tracing
            const RtScene* rt = nullptr;            // ray-traced shadows, and with lighting its ambient
            const RtLighting* lighting = nullptr;   // (a DDGI volume's, when RtLighting has one)
        };

        void Render(nvrhi::ICommandList* cmd, const donut::engine::PlanarView& view,
                    const donut::engine::PlanarView& previous, Scene& scene, const Inputs& inputs);
        void CreatePipelines();
        void BuildUI();

    private:
        Context& m_Ctx;
        NoiseVolume m_NoiseBaker;
        nvrhi::TextureHandle m_Noise;
        nvrhi::TextureHandle m_Froxels[2];
        nvrhi::TextureHandle m_Integrated;
        nvrhi::BufferHandle m_Constants;
        nvrhi::SamplerHandle m_ShadowCompare;
        nvrhi::BindingLayoutHandle m_InjectLayout, m_InjectRtLayout, m_IntegrateLayout, m_ApplyLayout;
        nvrhi::IBindingLayout* m_BindlessLayout = nullptr;
        nvrhi::ComputePipelineHandle m_Inject, m_InjectRt, m_Integrate, m_Apply;
        RtBindingSets m_Sets;
        dm::uint3 m_Grid = dm::uint3(0u);
        uint32_t m_Frame = 0;
        bool m_HistoryValid = false;
        bool m_Used = false;   // pipelines exist once enabled
        float m_Time = 0.f;

        struct Constants;
        std::unique_ptr<Constants> m_Data;
    };
}
