#pragma once

#include <hyper/platform.h>

#include <donut/core/math/math.h>
#include <nvrhi/nvrhi.h>

namespace donut::engine { struct LightProbe; }

namespace hyper
{
    struct Context;
    class Scene;

    // A probe volume that lights ray hits (hyper::Ddgi fills one in each frame).
    struct RtProbeVolume
    {
        nvrhi::ITexture* irradiance = nullptr;    // octahedral atlases, see hyper/rt_lighting.hlsli
        nvrhi::ITexture* distance = nullptr;
        nvrhi::ITexture* probeData = nullptr;
        dm::float3 origin = dm::float3(0.f);
        dm::float3 spacing = dm::float3(1.f);
        dm::int3 counts = dm::int3(0);
        uint32_t tilesX = 1;
        float normalBias = 0.f, viewBias = 0.f, intensity = 1.f, irradianceGamma = 5.f;
        // The volume's own update passes read these too.
        uint32_t raysPerProbe = 0;
        float rayRotation[3][4] = {};
        float hysteresis = 0.f, minFrontfaceDistance = 0.f;
        bool reset = false;
    };

    // The light ray-traced passes shade their hits with (hyper/rt_lighting.hlsli): the
    // sun with a shadow ray, the environment for rays that miss, and indirect light from
    // a probe volume when there is one, else from the IBL probe. Passes bind it with
    // AddLayoutItems / AddSetItems (b50, t50..t55, s50).
    //
    //     lighting.Prepare(cmd, scene, env.EnvironmentMap(), env.Probe().get(), scale, ddgi.Prepare(cmd, scene));
    class RtLighting
    {
    public:
        explicit RtLighting(Context& ctx);

        // Every frame rays are traced, before them. volume: null for the IBL fallback.
        void Prepare(nvrhi::ICommandList* cmd, Scene& scene, nvrhi::ITexture* environment,
                     const donut::engine::LightProbe* probe, float probeDiffuseScale, const RtProbeVolume* volume = nullptr);

        static void AddLayoutItems(nvrhi::BindingLayoutDesc& desc);
        void AddSetItems(nvrhi::BindingSetDesc& desc) const;
        [[nodiscard]] nvrhi::IBuffer* Constants() const { return m_Constants; }

    private:
        Context& m_Ctx;
        nvrhi::BufferHandle m_Constants;
        nvrhi::ITexture* m_Environment = nullptr;
        const donut::engine::LightProbe* m_Probe = nullptr;
        RtProbeVolume m_Volume;
        bool m_HasVolume = false;
    };
}
