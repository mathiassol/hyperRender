#include <hyper/rt/RtLighting.h>

#include <hyper/App.h>
#include <hyper/scene/Scene.h>

#include <donut/engine/SceneGraph.h>
#include <donut/engine/SceneTypes.h>
#include <nvrhi/utils.h>

#include <algorithm>
#include <cstring>

using namespace donut;
using namespace donut::math;

namespace hyper
{
    namespace
    {
        // Mirrors GiConstants in rt_lighting.hlsli.
        struct GiConstants
        {
            float3 volumeOrigin; uint32_t enabled;
            float3 spacing; uint32_t raysPerProbe;
            int32_t counts[3]; uint32_t tilesX;
            float irradianceInvSize[2]; float distanceInvSize[2];
            float normalBias, viewBias, intensity, maxRayDistance;
            float3 sunDir; uint32_t hasProbe;
            float3 sunRadiance; float envMip;
            float rayRotation[3][4];
            float hysteresis, minFrontfaceDistance; uint32_t frameIndex, reset;
            float probeDiffuseScale, irradianceGamma, probeSpecularScale, probeSpecularMips;
        };
    }

    RtLighting::RtLighting(Context& ctx) : m_Ctx(ctx)
    {
        m_Constants = ctx.device->createBuffer(nvrhi::utils::CreateStaticConstantBufferDesc(sizeof(GiConstants), "hyper/rt-lighting")
            .setInitialState(nvrhi::ResourceStates::ConstantBuffer).setKeepInitialState(true));
    }

    void RtLighting::Prepare(nvrhi::ICommandList* cmd, Scene& scene, nvrhi::ITexture* environment,
                             const engine::LightProbe* probe, float probeDiffuseScale, const RtProbeVolume* volume)
    {
        m_Environment = environment;
        m_Probe = probe;
        m_HasVolume = volume && volume->irradiance;
        m_Volume = m_HasVolume ? *volume : RtProbeVolume{};

        const engine::DirectionalLight& sun = scene.Sun();
        const RtProbeVolume& v = m_Volume;
        GiConstants c{};
        c.volumeOrigin = v.origin;
        c.enabled = m_HasVolume ? 1u : 0u;
        c.spacing = v.spacing;
        c.raysPerProbe = v.raysPerProbe;
        c.counts[0] = v.counts.x; c.counts[1] = v.counts.y; c.counts[2] = v.counts.z;
        c.tilesX = std::max(v.tilesX, 1u);
        if (m_HasVolume)
        {
            c.irradianceInvSize[0] = 1.f / float(v.irradiance->getDesc().width);
            c.irradianceInvSize[1] = 1.f / float(v.irradiance->getDesc().height);
            c.distanceInvSize[0] = 1.f / float(v.distance->getDesc().width);
            c.distanceInvSize[1] = 1.f / float(v.distance->getDesc().height);
        }
        c.normalBias = v.normalBias;
        c.viewBias = v.viewBias;
        c.intensity = v.intensity;
        c.maxRayDistance = 1e4f;
        c.sunDir = normalize(-float3(sun.GetDirection()));
        c.hasProbe = probe && probe->diffuseMap ? 1u : 0u;
        c.sunRadiance = sun.color * sun.irradiance;
        c.envMip = 3.f;
        std::memcpy(c.rayRotation, v.rayRotation, sizeof(c.rayRotation));
        c.hysteresis = v.hysteresis;
        c.minFrontfaceDistance = v.minFrontfaceDistance;
        c.frameIndex = uint32_t(m_Ctx.frameIndex);
        c.reset = v.reset ? 1u : 0u;
        c.probeDiffuseScale = probeDiffuseScale;
        c.probeSpecularScale = probe ? probe->specularScale : 0.f;
        c.probeSpecularMips = probe && probe->specularMap ? float(probe->specularMap->getDesc().mipLevels) : 1.f;
        c.irradianceGamma = v.irradianceGamma;
        cmd->writeBuffer(m_Constants, &c, sizeof(c));
    }

    void RtLighting::AddLayoutItems(nvrhi::BindingLayoutDesc& desc)
    {
        desc.addItem(nvrhi::BindingLayoutItem::ConstantBuffer(50))
            .addItem(nvrhi::BindingLayoutItem::Texture_SRV(50))
            .addItem(nvrhi::BindingLayoutItem::Texture_SRV(51))
            .addItem(nvrhi::BindingLayoutItem::Texture_SRV(52))
            .addItem(nvrhi::BindingLayoutItem::Texture_SRV(53))
            .addItem(nvrhi::BindingLayoutItem::Texture_SRV(54))
            .addItem(nvrhi::BindingLayoutItem::Texture_SRV(55))
            .addItem(nvrhi::BindingLayoutItem::Sampler(50));
    }

    void RtLighting::AddSetItems(nvrhi::BindingSetDesc& desc) const
    {
        const auto& common = *m_Ctx.common;
        nvrhi::ITexture* black = common.m_BlackTexture;
        nvrhi::ITexture* env = m_Environment ? m_Environment : common.m_BlackCubeMapArray.Get();
        nvrhi::ITexture* probeDiffuse = m_Probe && m_Probe->diffuseMap ? m_Probe->diffuseMap.Get() : common.m_BlackCubeMapArray.Get();
        nvrhi::ITexture* probeSpecular = m_Probe && m_Probe->specularMap ? m_Probe->specularMap.Get() : common.m_BlackCubeMapArray.Get();
        desc.addItem(nvrhi::BindingSetItem::ConstantBuffer(50, m_Constants))
            .addItem(nvrhi::BindingSetItem::Texture_SRV(50, m_HasVolume ? m_Volume.irradiance : black))
            .addItem(nvrhi::BindingSetItem::Texture_SRV(51, m_HasVolume ? m_Volume.distance : black))
            .addItem(nvrhi::BindingSetItem::Texture_SRV(52, m_HasVolume ? m_Volume.probeData : black))
            .addItem(nvrhi::BindingSetItem::Texture_SRV(53, env, nvrhi::Format::UNKNOWN,
                nvrhi::TextureSubresourceSet(0, nvrhi::TextureSubresourceSet::AllMipLevels, 0, 6), nvrhi::TextureDimension::TextureCube))
            .addItem(nvrhi::BindingSetItem::Texture_SRV(54, probeDiffuse, nvrhi::Format::UNKNOWN,
                nvrhi::AllSubresources, nvrhi::TextureDimension::TextureCubeArray))
            .addItem(nvrhi::BindingSetItem::Texture_SRV(55, probeSpecular, nvrhi::Format::UNKNOWN,
                nvrhi::AllSubresources, nvrhi::TextureDimension::TextureCubeArray))
            .addItem(nvrhi::BindingSetItem::Sampler(50, common.m_LinearClampSampler));
    }
}
