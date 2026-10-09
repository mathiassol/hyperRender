#include <hyper/volume/VolumetricFog.h>

#include <hyper/App.h>
#include <hyper/rt/RtLighting.h>
#include <hyper/scene/Scene.h>

#include <donut/engine/SceneGraph.h>
#include <donut/engine/SceneTypes.h>
#include <donut/engine/ShaderFactory.h>
#include <donut/engine/ShadowMap.h>
#include <donut/engine/View.h>
#include <nvrhi/utils.h>

#include <imgui.h>

#include <algorithm>
#include <cmath>

using namespace donut;
using namespace donut::math;
#include <donut/shaders/light_cb.h>
#include <donut/shaders/view_cb.h>

namespace hyper
{
    // Mirrors FogConstants in fog_cs.hlsl.
    struct VolumetricFog::Constants
    {
        PlanarViewConstants view;
        PlanarViewConstants viewPrev;
        uint32_t grid[3]; uint32_t frameIndex;
        float nearZ, farZ, temporalWeight; uint32_t historyValid;
        float density, heightFalloff, baseHeight, anisotropy;
        float3 albedo; float noiseStrength;
        float3 windOffset; float noiseScale;
        float3 sunDir; uint32_t shadowMode;
        float3 sunRadiance; uint32_t numLights;
        float ambientIntensity, localIntensity; uint32_t numVolumes, numCascades;
        float3 cameraPos; uint32_t hasProbe;
        uint32_t size[2]; float pad[2];
        float4 volumeCentre[8];
        float4 volumeExtent[8];
        ShadowConstants cascades[4];
        LightConstants lights[16];
    };

    VolumetricFog::VolumetricFog(Context& ctx) : m_Ctx(ctx), m_NoiseBaker(ctx), m_Data(std::make_unique<Constants>())
    {
        auto cs = nvrhi::ShaderType::Compute;
        auto injectBase = [&]
        {
            nvrhi::BindingLayoutDesc d;
            d.setVisibility(cs)
                .addItem(nvrhi::BindingLayoutItem::VolatileConstantBuffer(0))
                .addItem(nvrhi::BindingLayoutItem::Texture_SRV(0))
                .addItem(nvrhi::BindingLayoutItem::Texture_SRV(1))
                .addItem(nvrhi::BindingLayoutItem::Texture_SRV(5))
                .addItem(nvrhi::BindingLayoutItem::Sampler(0))
                .addItem(nvrhi::BindingLayoutItem::Sampler(1))
                .addItem(nvrhi::BindingLayoutItem::Texture_UAV(0));
            return d;
        };
        nvrhi::BindingLayoutDesc inject = injectBase();
        inject.addItem(nvrhi::BindingLayoutItem::Texture_SRV(6)).addItem(nvrhi::BindingLayoutItem::Sampler(2));
        m_InjectLayout = ctx.device->createBindingLayout(inject);
        if (ctx.rayQuery)
        {
            nvrhi::BindingLayoutDesc injectRt = injectBase();
            RtScene::AddLayoutItems(injectRt);
            RtLighting::AddLayoutItems(injectRt);
            m_InjectRtLayout = ctx.device->createBindingLayout(injectRt);
        }
        m_IntegrateLayout = ctx.device->createBindingLayout(nvrhi::BindingLayoutDesc().setVisibility(cs)
            .addItem(nvrhi::BindingLayoutItem::VolatileConstantBuffer(0))
            .addItem(nvrhi::BindingLayoutItem::Texture_SRV(2))
            .addItem(nvrhi::BindingLayoutItem::Texture_UAV(1)));
        m_ApplyLayout = ctx.device->createBindingLayout(nvrhi::BindingLayoutDesc().setVisibility(cs)
            .addItem(nvrhi::BindingLayoutItem::VolatileConstantBuffer(0))
            .addItem(nvrhi::BindingLayoutItem::Texture_SRV(3))
            .addItem(nvrhi::BindingLayoutItem::Texture_SRV(4))
            .addItem(nvrhi::BindingLayoutItem::Sampler(1))
            .addItem(nvrhi::BindingLayoutItem::Texture_UAV(2)));

        m_Constants = ctx.device->createBuffer(
            nvrhi::utils::CreateVolatileConstantBufferDesc(sizeof(Constants), "hyper/fog-constants", 16));
        m_ShadowCompare = ctx.device->createSampler(nvrhi::SamplerDesc()
            .setAllAddressModes(nvrhi::SamplerAddressMode::Border).setBorderColor(1.f)
            .setReductionType(nvrhi::SamplerReductionType::Comparison));
    }

    VolumetricFog::~VolumetricFog() = default;

    void VolumetricFog::CreatePipelines()
    {
        if (!m_Used)
            return;   // built on first use
        auto make = [&](const char* entry, std::vector<engine::ShaderMacro> defines, nvrhi::IBindingLayout* layout,
                        nvrhi::IBindingLayout* bindless) -> nvrhi::ComputePipelineHandle
        {
            if (!layout)
                return nullptr;
            auto shader = m_Ctx.shaders->CreateShader("hyper/volume/fog_cs", entry, &defines, nvrhi::ShaderType::Compute);
            if (!shader)
                return nullptr;
            nvrhi::ComputePipelineDesc desc;
            desc.setComputeShader(shader).addBindingLayout(layout);
            if (bindless)
                desc.addBindingLayout(bindless);
            return m_Ctx.device->createComputePipeline(desc);
        };
        using M = engine::ShaderMacro;
        auto inject = make("inject", { M("FOG_RT", "0") }, m_InjectLayout, nullptr);
        auto integrate = make("integrate", { M("FOG_RT", "0") }, m_IntegrateLayout, nullptr);
        auto apply = make("apply", { M("FOG_RT", "0") }, m_ApplyLayout, nullptr);
        if (inject && integrate && apply)
        {
            m_Inject = inject;
            m_Integrate = integrate;
            m_Apply = apply;
        }
        if (m_BindlessLayout)
            m_InjectRt = make("inject", { M("FOG_RT", "1") }, m_InjectRtLayout, m_BindlessLayout);
        m_NoiseBaker.CreatePipelines();
    }

    void VolumetricFog::Render(nvrhi::ICommandList* cmd, const engine::PlanarView& view,
                               const engine::PlanarView& previous, Scene& scene, const Inputs& in)
    {
        const bool newBindless = m_Ctx.rayQuery && scene.BindlessLayout() && scene.BindlessLayout() != m_BindlessLayout;
        if (settings.enabled && (!m_Used || newBindless))
        {
            m_Used = true;
            m_BindlessLayout = m_Ctx.rayQuery ? scene.BindlessLayout() : nullptr;
            CreatePipelines();
        }
        if (!settings.enabled || !m_Inject || !in.color || !in.depth)
        {
            m_HistoryValid = false;
            return;
        }

        auto scope = m_Ctx.profiler->Measure(cmd, "volumetric fog");

        const auto& cd = in.color->getDesc();
        const uint32_t tile = uint32_t(std::clamp(settings.tileSize, 4, 32));
        const uint3 grid((cd.width + tile - 1) / tile, (cd.height + tile - 1) / tile, uint32_t(std::clamp(settings.depthSlices, 16, 256)));
        if (any(grid != m_Grid))
        {
            auto volume = [&](const char* name)
            {
                return m_Ctx.device->createTexture(nvrhi::TextureDesc()
                    .setDebugName(name).setDimension(nvrhi::TextureDimension::Texture3D)
                    .setWidth(grid.x).setHeight(grid.y).setDepth(grid.z)
                    .setFormat(nvrhi::Format::RGBA16_FLOAT).setIsUAV(true)
                    .setInitialState(nvrhi::ResourceStates::ShaderResource).setKeepInitialState(true));
            };
            m_Froxels[0] = volume("hyper/fog-froxels0");
            m_Froxels[1] = volume("hyper/fog-froxels1");
            m_Integrated = volume("hyper/fog-integrated");
            m_Grid = grid;
            m_HistoryValid = false;
            m_Sets.Clear();
        }
        if (!m_Noise)
            m_Noise = m_NoiseBaker.Bake(cmd, NoiseType::Gradient, 64, 8, "hyper/fog-noise");

        m_Time += m_Ctx.deltaTime;
        const engine::DirectionalLight& sun = scene.Sun();
        const bool rtShadows = settings.shadows && in.rt && in.rt->Ready() && in.lighting && m_InjectRt;
        const bool csm = settings.shadows && !rtShadows && sun.shadowMap;

        Constants& c = *m_Data;
        c = Constants{};
        view.FillPlanarViewConstants(c.view);
        previous.FillPlanarViewConstants(c.viewPrev);
        c.grid[0] = grid.x; c.grid[1] = grid.y; c.grid[2] = grid.z;
        c.frameIndex = m_Frame++;
        c.nearZ = 0.5f;
        c.farZ = std::max(settings.maxDistance, 1.f);
        c.temporalWeight = std::clamp(settings.temporalWeight, 0.f, 0.98f);
        c.historyValid = m_HistoryValid ? 1u : 0u;
        c.density = settings.density;
        c.heightFalloff = settings.heightFalloff;
        c.baseHeight = settings.baseHeight;
        c.anisotropy = std::clamp(settings.anisotropy, -0.95f, 0.95f);
        c.albedo = settings.albedo;
        c.noiseStrength = settings.noiseStrength;
        c.windOffset = -settings.wind * m_Time;
        c.noiseScale = settings.noiseScale;
        c.sunDir = normalize(-float3(sun.GetDirection()));
        c.shadowMode = rtShadows ? 2u : csm ? 1u : 0u;
        c.sunRadiance = sun.color * sun.irradiance * settings.sunIntensity;
        c.ambientIntensity = settings.ambientIntensity;
        c.localIntensity = settings.localLightIntensity;
        c.cameraPos = view.GetViewOrigin();
        c.hasProbe = in.probe && in.probe->diffuseMap ? 1u : 0u;
        c.size[0] = cd.width;
        c.size[1] = cd.height;

        const uint32_t volumes = uint32_t(std::min<size_t>(settings.volumes.size(), 8));
        for (uint32_t i = 0; i < volumes; ++i)
        {
            const FogVolume& v = settings.volumes[i];
            c.volumeCentre[i] = float4(v.centre, v.density);
            c.volumeExtent[i] = float4(v.halfExtent, v.sphere ? 1.f : 0.f);
        }
        c.numVolumes = volumes;

        // Comparison sampling needs a depth format even when unused.
        nvrhi::ITexture* shadowTexture = m_Ctx.common->m_BlackDepthStencilTexture2DArray;
        if (csm)
        {
            c.numCascades = std::min(sun.shadowMap->GetNumberOfCascades(), 4u);
            for (uint32_t i = 0; i < c.numCascades; ++i)
                sun.shadowMap->GetCascade(i)->FillShadowConstants(c.cascades[i]);
            shadowTexture = sun.shadowMap->GetTexture();
        }

        uint32_t lights = 0;
        for (const auto& light : scene.Graph()->GetLights())
        {
            if (lights >= 16)
                break;
            if (light->GetLightType() == LightType_Point || light->GetLightType() == LightType_Spot)
                light->FillLightConstants(c.lights[lights++]);
        }
        c.numLights = lights;
        cmd->writeBuffer(m_Constants, &c, sizeof(c));

        const uint32_t cur = m_Frame & 1u;
        nvrhi::ITexture* probeDiffuse = c.hasProbe ? in.probe->diffuseMap.Get() : m_Ctx.common->m_BlackCubeMapArray.Get();

        nvrhi::BindingSetDesc inject;
        inject.addItem(nvrhi::BindingSetItem::ConstantBuffer(0, m_Constants))
            .addItem(nvrhi::BindingSetItem::Texture_SRV(0, m_Noise))
            .addItem(nvrhi::BindingSetItem::Texture_SRV(1, m_Froxels[1 - cur]))
            .addItem(nvrhi::BindingSetItem::Texture_SRV(5, probeDiffuse, nvrhi::Format::UNKNOWN, nvrhi::AllSubresources,
                nvrhi::TextureDimension::TextureCubeArray))
            .addItem(nvrhi::BindingSetItem::Sampler(0, m_Ctx.common->m_LinearWrapSampler))
            .addItem(nvrhi::BindingSetItem::Sampler(1, m_Ctx.common->m_LinearClampSampler))
            .addItem(nvrhi::BindingSetItem::Texture_UAV(0, m_Froxels[cur]));
        if (rtShadows)
        {
            in.rt->AddSetItems(inject, scene);
            in.lighting->AddSetItems(inject);
            cmd->setComputeState(nvrhi::ComputeState().setPipeline(m_InjectRt)
                .addBindingSet(m_Sets.Get(m_Ctx.device, inject, m_InjectRtLayout)).addBindingSet(scene.DescriptorTable()));
        }
        else
        {
            inject.addItem(nvrhi::BindingSetItem::Texture_SRV(6, shadowTexture))
                .addItem(nvrhi::BindingSetItem::Sampler(2, m_ShadowCompare));
            cmd->setComputeState(nvrhi::ComputeState().setPipeline(m_Inject)
                .addBindingSet(m_Ctx.bindingCache->GetOrCreateBindingSet(inject, m_InjectLayout)));
        }
        cmd->dispatch((grid.x + 3) / 4, (grid.y + 3) / 4, (grid.z + 3) / 4);

        cmd->setComputeState(nvrhi::ComputeState().setPipeline(m_Integrate)
            .addBindingSet(m_Ctx.bindingCache->GetOrCreateBindingSet(nvrhi::BindingSetDesc()
                .addItem(nvrhi::BindingSetItem::ConstantBuffer(0, m_Constants))
                .addItem(nvrhi::BindingSetItem::Texture_SRV(2, m_Froxels[cur]))
                .addItem(nvrhi::BindingSetItem::Texture_UAV(1, m_Integrated)), m_IntegrateLayout)));
        cmd->dispatch((grid.x + 7) / 8, (grid.y + 7) / 8);

        cmd->setComputeState(nvrhi::ComputeState().setPipeline(m_Apply)
            .addBindingSet(m_Ctx.bindingCache->GetOrCreateBindingSet(nvrhi::BindingSetDesc()
                .addItem(nvrhi::BindingSetItem::ConstantBuffer(0, m_Constants))
                .addItem(nvrhi::BindingSetItem::Texture_SRV(3, m_Integrated))
                .addItem(nvrhi::BindingSetItem::Texture_SRV(4, in.depth))
                .addItem(nvrhi::BindingSetItem::Sampler(1, m_Ctx.common->m_LinearClampSampler))
                .addItem(nvrhi::BindingSetItem::Texture_UAV(2, in.color)), m_ApplyLayout)));
        cmd->dispatch((cd.width + 7) / 8, (cd.height + 7) / 8);

        m_HistoryValid = true;
    }

    void VolumetricFog::BuildUI()
    {
        ImGui::Checkbox("volumetric fog", &settings.enabled);
        if (!settings.enabled)
            return;
        ImGui::SliderFloat("fog density", &settings.density, 0.f, 0.5f, "%.4f", ImGuiSliderFlags_Logarithmic);
        ImGui::SliderFloat("fog height falloff", &settings.heightFalloff, 0.f, 2.f);
        ImGui::SliderFloat("fog base height", &settings.baseHeight, -50.f, 50.f);
        ImGui::SliderFloat("fog anisotropy", &settings.anisotropy, -0.9f, 0.95f);
        ImGui::ColorEdit3("fog albedo", &settings.albedo.x);
        ImGui::SliderFloat("fog noise", &settings.noiseStrength, 0.f, 1.f);
        ImGui::SliderFloat("fog noise scale", &settings.noiseScale, 0.01f, 2.f, "%.3f", ImGuiSliderFlags_Logarithmic);
        ImGui::SliderFloat3("fog wind", &settings.wind.x, -5.f, 5.f);
        ImGui::SliderFloat("fog sun", &settings.sunIntensity, 0.f, 4.f);
        ImGui::SliderFloat("fog ambient", &settings.ambientIntensity, 0.f, 4.f);
        ImGui::SliderFloat("fog range", &settings.maxDistance, 8.f, 512.f, "%.0f", ImGuiSliderFlags_Logarithmic);
        ImGui::SliderFloat("fog history", &settings.temporalWeight, 0.f, 0.98f);
        ImGui::Checkbox("fog shadows", &settings.shadows);
        ImGui::TextDisabled("%u x %u x %u froxels, %zu volume(s)", m_Grid.x, m_Grid.y, m_Grid.z, settings.volumes.size());
    }
}
