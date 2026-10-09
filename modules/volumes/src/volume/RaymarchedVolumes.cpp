#include <hyper/volume/RaymarchedVolumes.h>

#include <hyper/App.h>

#include <donut/engine/SceneTypes.h>
#include <donut/engine/View.h>
#include <nvrhi/utils.h>

#include <imgui.h>

#include <algorithm>

using namespace donut;
using namespace donut::math;
#include <donut/shaders/view_cb.h>

namespace hyper
{
    // Mirrors VolumeConstants in volumes_cs.hlsl.
    struct RaymarchedVolumes::Constants
    {
        PlanarViewConstants view;
        PlanarViewConstants viewPrev;
        uint32_t fullSize[2]; uint32_t halfSize[2];
        float3 cameraPos; uint32_t frameIndex;
        float3 sunDir; uint32_t steps;
        float3 sunRadiance; uint32_t lightSteps;
        float maxDistance, temporalWeight; uint32_t historyValid, numBoxes;
        uint32_t cloudsEnabled; float cloudBottom, cloudTop, cloudCoverage;
        float cloudDensity, cloudShapeScale, cloudDetailScale, cloudWeatherScale;
        float3 cloudWind; float cloudPowder;
        float gForward, gBack, lobeBlend, ambientIntensity;
        float3 cloudAlbedo; uint32_t hasProbe;
        uint32_t hasAtmosphere; float aerialMaxKm, unitKm, time;
        float airIrradiance; float pad[3];
        float4 boxCentre[8];
        float4 boxExtent[8];
        float4 boxAlbedo[8];
        float4 boxEmission[8];
    };

    namespace
    {
        nvrhi::TextureHandle Dummy(nvrhi::IDevice* device, nvrhi::TextureDimension dim, const char* name)
        {
            return device->createTexture(nvrhi::TextureDesc().setDebugName(name).setDimension(dim)
                .setWidth(1).setHeight(1).setDepth(1).setFormat(nvrhi::Format::RGBA16_FLOAT).setIsUAV(true)
                .setInitialState(nvrhi::ResourceStates::ShaderResource).setKeepInitialState(true));
        }
    }

    RaymarchedVolumes::RaymarchedVolumes(Context& ctx) : m_Ctx(ctx), m_NoiseBaker(ctx), m_Data(std::make_unique<Constants>())
    {
        nvrhi::BindingLayoutDesc d;
        d.setVisibility(nvrhi::ShaderType::Compute).addItem(nvrhi::BindingLayoutItem::VolatileConstantBuffer(0));
        for (uint32_t t = 0; t < 8; ++t)
            d.addItem(nvrhi::BindingLayoutItem::Texture_SRV(t));
        d.addItem(nvrhi::BindingLayoutItem::Sampler(0)).addItem(nvrhi::BindingLayoutItem::Sampler(1));
        for (uint32_t u = 0; u < 4; ++u)
            d.addItem(nvrhi::BindingLayoutItem::Texture_UAV(u));
        m_Layout = ctx.device->createBindingLayout(d);
        m_Constants = ctx.device->createBuffer(
            nvrhi::utils::CreateVolatileConstantBufferDesc(sizeof(Constants), "hyper/volume-constants", 16));
    }

    RaymarchedVolumes::~RaymarchedVolumes() = default;

    void RaymarchedVolumes::CreatePipelines()
    {
        if (!m_Used)
            return;   // built on first use
        auto make = [&](const char* entry) -> nvrhi::ComputePipelineHandle
        {
            auto shader = m_Ctx.shaders->CreateShader("hyper/volume/volumes_cs", entry, nullptr, nvrhi::ShaderType::Compute);
            return shader ? m_Ctx.device->createComputePipeline(
                nvrhi::ComputePipelineDesc().setComputeShader(shader).addBindingLayout(m_Layout)) : nullptr;
        };
        auto march = make("march");
        auto temporal = make("temporal");
        auto composite = make("composite");
        if (march && temporal && composite)
        {
            m_MarchPipeline = march;
            m_Temporal = temporal;
            m_Composite = composite;
        }
        m_NoiseBaker.CreatePipelines();
    }

    bool RaymarchedVolumes::HasWork() const
    {
        return settings.enabled && (settings.clouds.enabled || !settings.volumes.empty());
    }

    void RaymarchedVolumes::Render(nvrhi::ICommandList* cmd, const engine::PlanarView& view,
                                   const engine::PlanarView& previous, const Inputs& in)
    {
        if (HasWork() && !m_Used)
        {
            m_Used = true;
            CreatePipelines();
        }
        if (!HasWork() || !m_MarchPipeline || !in.color || !in.depth)
        {
            m_HistoryValid = false;
            return;
        }
        auto scope = m_Ctx.profiler->Measure(cmd, "volumes");

        if (!m_Shape)
        {
            m_Shape = m_NoiseBaker.Bake(cmd, NoiseType::PerlinWorley, 128, 4, "hyper/volume-shape-noise");
            m_Detail = m_NoiseBaker.Bake(cmd, NoiseType::Worley, 32, 2, "hyper/volume-detail-noise");
        }

        const auto& cd = in.color->getDesc();
        const uint32_t hw = (cd.width + 1) / 2, hh = (cd.height + 1) / 2;
        if (!m_March || m_March->getDesc().width != hw || m_March->getDesc().height != hh)
        {
            auto tex = [&](const char* name, nvrhi::Format f)
            {
                return m_Ctx.device->createTexture(nvrhi::TextureDesc().setDebugName(name).setWidth(hw).setHeight(hh)
                    .setFormat(f).setIsUAV(true).setInitialState(nvrhi::ResourceStates::ShaderResource).setKeepInitialState(true));
            };
            m_March = tex("hyper/volume-march", nvrhi::Format::RGBA16_FLOAT);
            m_Distances = tex("hyper/volume-distances", nvrhi::Format::RG32_FLOAT);
            m_History[0] = tex("hyper/volume-history0", nvrhi::Format::RGBA16_FLOAT);
            m_History[1] = tex("hyper/volume-history1", nvrhi::Format::RGBA16_FLOAT);
            m_HistoryValid = false;
        }
        m_Time += m_Ctx.deltaTime;

        const CloudLayerSettings& cl = settings.clouds;
        Constants& c = *m_Data;
        c = Constants{};
        view.FillPlanarViewConstants(c.view);
        previous.FillPlanarViewConstants(c.viewPrev);
        c.fullSize[0] = cd.width; c.fullSize[1] = cd.height;
        c.halfSize[0] = hw; c.halfSize[1] = hh;
        c.cameraPos = view.GetViewOrigin();
        c.frameIndex = m_Frame++;
        c.sunDir = normalize(-in.sunDirection);
        c.steps = uint32_t(std::clamp(settings.steps, 8, 256));
        c.sunRadiance = in.sunRadiance * settings.sunIntensity;
        c.lightSteps = uint32_t(std::clamp(settings.lightSteps, 1, 16));
        c.maxDistance = settings.maxDistance;
        c.temporalWeight = std::clamp(settings.temporalWeight, 0.f, 0.98f);
        c.historyValid = m_HistoryValid ? 1u : 0u;
        c.cloudsEnabled = cl.enabled ? 1u : 0u;
        c.cloudBottom = cl.bottom;
        c.cloudTop = std::max(cl.top, cl.bottom + 1.f);
        c.cloudCoverage = std::clamp(cl.coverage, 0.f, 1.f);
        c.cloudDensity = cl.density;
        c.cloudShapeScale = cl.shapeScale;
        c.cloudDetailScale = cl.detailScale;
        c.cloudWeatherScale = cl.weatherScale;
        c.cloudWind = cl.wind * m_Time;
        c.cloudPowder = cl.powder;
        c.gForward = cl.forwardG;
        c.gBack = cl.backG;
        c.lobeBlend = cl.lobeBlend;
        c.ambientIntensity = settings.ambientIntensity;
        c.cloudAlbedo = cl.albedo;
        c.hasProbe = in.probe && in.probe->diffuseMap ? 1u : 0u;
        const bool air = in.aerialVolume != nullptr;
        c.hasAtmosphere = air ? 1u : 0u;
        c.aerialMaxKm = air ? in.aerialMaxKm : 1.f;
        c.unitKm = air ? in.aerialUnitKm : 0.001f;
        c.time = m_Time;
        c.airIrradiance = in.sunIrradiance;
        const uint32_t boxes = uint32_t(std::min<size_t>(settings.volumes.size(), 8));
        for (uint32_t i = 0; i < boxes; ++i)
        {
            const MediaVolume& v = settings.volumes[i];
            c.boxCentre[i] = float4(v.centre, v.density);
            c.boxExtent[i] = float4(v.halfExtent, v.sphere ? 1.f : 0.f);
            c.boxAlbedo[i] = float4(v.albedo, v.noiseStrength);
            c.boxEmission[i] = float4(v.emission, v.noiseScale);
        }
        c.numBoxes = boxes;
        cmd->writeBuffer(m_Constants, &c, sizeof(c));

        if (!m_DummyUav)
            m_DummyUav = Dummy(m_Ctx.device, nvrhi::TextureDimension::Texture2D, "hyper/volume-dummy-uav");
        // One layout for all three passes; slots a pass doesn't use get placeholders.
        nvrhi::ITexture* black = m_Ctx.common->m_BlackTexture;
        nvrhi::ITexture* probe = c.hasProbe ? in.probe->diffuseMap.Get() : m_Ctx.common->m_BlackCubeMapArray.Get();
        nvrhi::ITexture* aerial = air ? in.aerialVolume : m_Detail.Get();   // unread without the atmosphere
        const uint32_t cur = m_Frame & 1u;

        auto pass = [&](nvrhi::IComputePipeline* pipeline, nvrhi::ITexture* t5, nvrhi::ITexture* t6, nvrhi::ITexture* t7,
                        nvrhi::ITexture* u0, nvrhi::ITexture* u1, nvrhi::ITexture* u2, nvrhi::ITexture* u3, uint32_t w, uint32_t h)
        {
            auto set = m_Ctx.bindingCache->GetOrCreateBindingSet(nvrhi::BindingSetDesc()
                .addItem(nvrhi::BindingSetItem::ConstantBuffer(0, m_Constants))
                .addItem(nvrhi::BindingSetItem::Texture_SRV(0, m_Shape))
                .addItem(nvrhi::BindingSetItem::Texture_SRV(1, m_Detail))
                .addItem(nvrhi::BindingSetItem::Texture_SRV(2, in.depth))
                .addItem(nvrhi::BindingSetItem::Texture_SRV(3, probe, nvrhi::Format::UNKNOWN, nvrhi::AllSubresources,
                    nvrhi::TextureDimension::TextureCubeArray))
                .addItem(nvrhi::BindingSetItem::Texture_SRV(4, aerial))
                .addItem(nvrhi::BindingSetItem::Texture_SRV(5, t5))
                .addItem(nvrhi::BindingSetItem::Texture_SRV(6, t6))
                .addItem(nvrhi::BindingSetItem::Texture_SRV(7, t7))
                .addItem(nvrhi::BindingSetItem::Sampler(0, m_Ctx.common->m_LinearWrapSampler))
                .addItem(nvrhi::BindingSetItem::Sampler(1, m_Ctx.common->m_LinearClampSampler))
                .addItem(nvrhi::BindingSetItem::Texture_UAV(0, u0))
                .addItem(nvrhi::BindingSetItem::Texture_UAV(1, u1))
                .addItem(nvrhi::BindingSetItem::Texture_UAV(2, u2))
                .addItem(nvrhi::BindingSetItem::Texture_UAV(3, u3)), m_Layout);
            cmd->setComputeState(nvrhi::ComputeState().setPipeline(pipeline).addBindingSet(set));
            cmd->dispatch((w + 7) / 8, (h + 7) / 8);
        };

        nvrhi::ITexture* spare = m_DummyUav;
        {
            auto t = m_Ctx.profiler->Measure(cmd, "volume march");
            pass(m_MarchPipeline, black, black, black, m_March, m_Distances, spare, spare, hw, hh);
        }
        pass(m_Temporal, m_March, m_History[1 - cur], m_Distances, spare, spare, m_History[cur], spare, hw, hh);
        pass(m_Composite, m_History[cur], black, m_Distances, spare, spare, spare, in.color, cd.width, cd.height);
        m_HistoryValid = true;
    }

    void RaymarchedVolumes::BuildUI()
    {
        ImGui::Checkbox("ray-marched volumes", &settings.enabled);
        if (!settings.enabled)
            return;
        ImGui::SliderInt("volume steps", &settings.steps, 16, 256);
        ImGui::SliderInt("volume light steps", &settings.lightSteps, 1, 12);
        ImGui::SliderFloat("volume history", &settings.temporalWeight, 0.f, 0.98f);
        ImGui::SliderFloat("volume sun", &settings.sunIntensity, 0.f, 4.f);
        ImGui::SliderFloat("volume ambient", &settings.ambientIntensity, 0.f, 4.f);
        ImGui::TextDisabled("%zu media volume(s)", settings.volumes.size());
        CloudLayerSettings& cl = settings.clouds;
        ImGui::Checkbox("cloud layer", &cl.enabled);
        if (cl.enabled)
        {
            ImGui::SliderFloat("cloud coverage", &cl.coverage, 0.f, 1.f);
            ImGui::SliderFloat("cloud density", &cl.density, 0.001f, 0.2f, "%.4f", ImGuiSliderFlags_Logarithmic);
            ImGui::SliderFloat("cloud base", &cl.bottom, 0.f, 8000.f, "%.0f");
            ImGui::SliderFloat("cloud top", &cl.top, 100.f, 12000.f, "%.0f");
            ImGui::SliderFloat3("cloud wind", &cl.wind.x, -60.f, 60.f);
            ImGui::SliderFloat("cloud powder", &cl.powder, 0.f, 1.f);
            ImGui::SliderFloat("cloud silver lining (g)", &cl.forwardG, 0.f, 0.95f);
        }
    }
}
