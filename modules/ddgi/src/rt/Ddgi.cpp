#include <hyper/rt/Ddgi.h>

#include <hyper/App.h>
#include <hyper/scene/GBuffer.h>
#include <hyper/scene/Scene.h>

#include <donut/core/log.h>
#include <donut/engine/SceneGraph.h>
#include <donut/engine/SceneTypes.h>
#include <donut/engine/View.h>
#include <nvrhi/utils.h>

#include <imgui.h>

#include <algorithm>
#include <cmath>
#include <random>

using namespace donut;
using namespace donut::math;
#include <donut/shaders/view_cb.h>

namespace hyper
{
    namespace
    {
        constexpr uint32_t c_IrradianceTile = 8;    // 6 x 6 interior + border
        constexpr uint32_t c_DistanceTile = 16;     // 14 x 14 interior + border

        struct GatherConstants
        {
            PlanarViewConstants view;
            uint32_t size[2]; uint32_t pad[2];
        };

        // Uniformly random rotation (Shoemake's quaternion).
        void RandomRotation(uint64_t seed, float out[3][4])
        {
            std::mt19937 rng(uint32_t(seed * 2654435761ull + 1));
            std::uniform_real_distribution<float> u(0.f, 1.f);
            const float u1 = u(rng), u2 = u(rng), u3 = u(rng);
            const float a = std::sqrt(1.f - u1), b = std::sqrt(u1);
            const float x = a * std::sin(2.f * PI_f * u2), y = a * std::cos(2.f * PI_f * u2);
            const float z = b * std::sin(2.f * PI_f * u3), w = b * std::cos(2.f * PI_f * u3);
            const float m[3][3] = {
                { 1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w) },
                { 2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w) },
                { 2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y) },
            };
            for (int r = 0; r < 3; ++r)
            {
                for (int c = 0; c < 3; ++c)
                    out[r][c] = m[r][c];
                out[r][3] = 0.f;
            }
        }
    }

    Ddgi::Ddgi(Context& ctx) : m_Ctx(ctx)
    {
        m_GatherConstants = ctx.device->createBuffer(
            nvrhi::utils::CreateVolatileConstantBufferDesc(sizeof(GatherConstants), "hyper/ddgi-gather-constants", 16));

        auto cs = nvrhi::ShaderType::Compute;
        nvrhi::BindingLayoutDesc trace;
        trace.setVisibility(cs).addItem(nvrhi::BindingLayoutItem::Texture_UAV(0));
        RtLighting::AddLayoutItems(trace);
        RtScene::AddLayoutItems(trace);
        m_TraceLayout = ctx.device->createBindingLayout(trace);

        m_BlendLayout = ctx.device->createBindingLayout(nvrhi::BindingLayoutDesc().setVisibility(cs)
            .addItem(nvrhi::BindingLayoutItem::ConstantBuffer(50))
            .addItem(nvrhi::BindingLayoutItem::Texture_SRV(0))
            .addItem(nvrhi::BindingLayoutItem::Texture_SRV(1))
            .addItem(nvrhi::BindingLayoutItem::Texture_UAV(1)));
        m_RelocateLayout = ctx.device->createBindingLayout(nvrhi::BindingLayoutDesc().setVisibility(cs)
            .addItem(nvrhi::BindingLayoutItem::ConstantBuffer(50))
            .addItem(nvrhi::BindingLayoutItem::Texture_SRV(0))
            .addItem(nvrhi::BindingLayoutItem::Texture_UAV(2)));

        nvrhi::BindingLayoutDesc gather;
        gather.setVisibility(cs)
            .addItem(nvrhi::BindingLayoutItem::VolatileConstantBuffer(0))
            .addItem(nvrhi::BindingLayoutItem::Texture_SRV(2))
            .addItem(nvrhi::BindingLayoutItem::Texture_SRV(3))
            .addItem(nvrhi::BindingLayoutItem::Texture_SRV(4))
            .addItem(nvrhi::BindingLayoutItem::Texture_SRV(5))
            .addItem(nvrhi::BindingLayoutItem::Texture_UAV(3));
        RtLighting::AddLayoutItems(gather);
        m_GatherLayout = ctx.device->createBindingLayout(gather);
    }

    void Ddgi::CreatePipelines()
    {
        auto make = [&](const char* entry, nvrhi::IBindingLayout* layout, bool bindless) -> nvrhi::ComputePipelineHandle
        {
            auto shader = m_Ctx.shaders->CreateShader("hyper/rt/ddgi_cs", entry, nullptr, nvrhi::ShaderType::Compute);
            if (!shader)
                return nullptr;
            nvrhi::ComputePipelineDesc desc;
            desc.setComputeShader(shader).addBindingLayout(layout);
            if (bindless)
                desc.addBindingLayout(m_BindlessLayout);
            return m_Ctx.device->createComputePipeline(desc);
        };
        if (!m_BindlessLayout)
            return;
        auto trace = make("trace", m_TraceLayout, true);
        auto blendIrradiance = make("blend_irradiance", m_BlendLayout, false);
        auto blendDistance = make("blend_distance", m_BlendLayout, false);
        auto borderIrradiance = make("border_irradiance", m_BlendLayout, false);
        auto borderDistance = make("border_distance", m_BlendLayout, false);
        auto relocate = make("relocate", m_RelocateLayout, false);
        auto gather = make("gather", m_GatherLayout, false);
        if (trace && blendIrradiance && blendDistance && borderIrradiance && borderDistance && relocate && gather)
        {
            m_Trace = trace;
            m_BlendIrradiance = blendIrradiance;
            m_BlendDistance = blendDistance;
            m_BorderIrradiance = borderIrradiance;
            m_BorderDistance = borderDistance;
            m_Relocate = relocate;
            m_Gather = gather;
        }
    }

    void Ddgi::Allocate(Scene& scene)
    {
        const int budget = std::clamp(settings.probeBudget, 8, 65536);
        const int rays = std::clamp((settings.raysPerProbe + 31) / 32 * 32, 32, 512);
        settings.raysPerProbe = rays;

        const box3 bounds = scene.Bounds();
        float3 extent = bounds.diagonal();
        const float largest = std::max(extent.x, std::max(extent.y, extent.z));
        extent = max(extent, float3(largest * 0.05f));
        // Near-uniform spacing that fits the budget and the atlas size limits.
        float s = std::cbrt(extent.x * extent.y * extent.z / float(budget));
        int3 counts;
        for (;;)
        {
            counts = int3(std::max(2, int(std::ceil(extent.x / s)) + 1), std::max(2, int(std::ceil(extent.y / s)) + 1),
                std::max(2, int(std::ceil(extent.z / s)) + 1));
            if (counts.x * counts.y * counts.z <= budget && counts.x * counts.z <= 1024 && counts.y <= 1024)
                break;
            s *= 1.05f;
        }
        m_Counts = counts;
        m_Origin = bounds.m_mins;
        m_Spacing = extent / float3(counts - 1);
        m_TilesX = uint32_t(counts.x * counts.z);
        const uint32_t probes = uint32_t(counts.x * counts.y * counts.z);

        auto tex = [&](const char* name, uint32_t w, uint32_t h, nvrhi::Format format)
        {
            return m_Ctx.device->createTexture(nvrhi::TextureDesc()
                .setDebugName(name).setWidth(w).setHeight(h).setFormat(format).setIsUAV(true)
                .setInitialState(nvrhi::ResourceStates::ShaderResource).setKeepInitialState(true));
        };
        m_Rays = tex("hyper/ddgi-rays", uint32_t(rays), probes, nvrhi::Format::RGBA16_FLOAT);
        m_Irradiance = tex("hyper/ddgi-irradiance", m_TilesX * c_IrradianceTile, uint32_t(counts.y) * c_IrradianceTile, nvrhi::Format::RGBA16_FLOAT);
        m_Distance = tex("hyper/ddgi-distance", m_TilesX * c_DistanceTile, uint32_t(counts.y) * c_DistanceTile, nvrhi::Format::RG16_FLOAT);
        m_ProbeData = tex("hyper/ddgi-probes", m_TilesX, uint32_t(counts.y), nvrhi::Format::RGBA16_FLOAT);
        m_Sets.Clear();
        m_Ctx.bindingCache->Clear();

        m_BuiltBudget = settings.probeBudget;
        m_BuiltRays = rays;
        m_Reset = true;
        m_FramesSinceReset = 0;
        log::info("hyper: DDGI %d x %d x %d probes (spacing %.2f %.2f %.2f), %d rays each",
            counts.x, counts.y, counts.z, m_Spacing.x, m_Spacing.y, m_Spacing.z, rays);
    }

    const RtProbeVolume* Ddgi::Prepare(nvrhi::ICommandList* cmd, Scene& scene)
    {
        if (scene.BindlessLayout() != m_BindlessLayout)
        {
            m_BindlessLayout = scene.BindlessLayout();
            CreatePipelines();
        }
        if (settings.enabled && (scene.Generation() != m_Generation || settings.probeBudget != m_BuiltBudget
            || settings.raysPerProbe != m_BuiltRays || !m_Irradiance))
        {
            m_Generation = scene.Generation();
            Allocate(scene);
            cmd->clearTextureFloat(m_Irradiance, nvrhi::AllSubresources, nvrhi::Color(0.f));
            cmd->clearTextureFloat(m_Distance, nvrhi::AllSubresources, nvrhi::Color(0.f));
            cmd->clearTextureFloat(m_ProbeData, nvrhi::AllSubresources, nvrhi::Color(0.f, 0.f, 0.f, 1.f));
        }
        if (!Active())
            return nullptr;

        RtProbeVolume& v = m_Volume;
        v.irradiance = m_Irradiance;
        v.distance = m_Distance;
        v.probeData = m_ProbeData;
        v.origin = m_Origin;
        v.spacing = m_Spacing;
        v.counts = m_Counts;
        v.tilesX = m_TilesX;
        v.normalBias = settings.normalBias;
        v.viewBias = settings.viewBias;
        v.intensity = settings.intensity;
        v.irradianceGamma = 5.f;
        v.raysPerProbe = uint32_t(m_BuiltRays);
        RandomRotation(m_Ctx.frameIndex, v.rayRotation);
        // Plain average for the first frames after a reset, then the set hysteresis.
        v.hysteresis = std::min(std::clamp(settings.hysteresis, 0.f, 0.999f), 1.f - 1.f / float(m_FramesSinceReset + 1));
        v.minFrontfaceDistance = 0.25f * std::min(m_Spacing.x, std::min(m_Spacing.y, m_Spacing.z));
        v.reset = m_Reset;
        return &v;
    }

    void Ddgi::Update(nvrhi::ICommandList* cmd, Scene& scene, const RtScene& rt, const RtLighting& lighting)
    {
        if (!Active() || !rt.Ready())
            return;
        auto scope = m_Ctx.profiler->Measure(cmd, "ddgi");
        const uint32_t probes = uint32_t(m_Counts.x * m_Counts.y * m_Counts.z);

        {
            nvrhi::BindingSetDesc d;
            d.addItem(nvrhi::BindingSetItem::Texture_UAV(0, m_Rays));
            lighting.AddSetItems(d);
            rt.AddSetItems(d, scene);
            cmd->setComputeState(nvrhi::ComputeState().setPipeline(m_Trace)
                .addBindingSet(m_Sets.Get(m_Ctx.device, d, m_TraceLayout)).addBindingSet(scene.DescriptorTable()));
            cmd->dispatch(uint32_t(m_BuiltRays) / 32, probes);
        }

        if (settings.relocate)
        {
            auto set = m_Ctx.bindingCache->GetOrCreateBindingSet(nvrhi::BindingSetDesc()
                .addItem(nvrhi::BindingSetItem::ConstantBuffer(50, lighting.Constants()))
                .addItem(nvrhi::BindingSetItem::Texture_SRV(0, m_Rays))
                .addItem(nvrhi::BindingSetItem::Texture_UAV(2, m_ProbeData)), m_RelocateLayout);
            cmd->setComputeState(nvrhi::ComputeState().setPipeline(m_Relocate).addBindingSet(set));
            cmd->dispatch((probes + 31) / 32);
        }

        auto blend = [&](nvrhi::IComputePipeline* pipeline, nvrhi::ITexture* atlas)
        {
            auto set = m_Ctx.bindingCache->GetOrCreateBindingSet(nvrhi::BindingSetDesc()
                .addItem(nvrhi::BindingSetItem::ConstantBuffer(50, lighting.Constants()))
                .addItem(nvrhi::BindingSetItem::Texture_SRV(0, m_Rays))
                .addItem(nvrhi::BindingSetItem::Texture_SRV(1, m_ProbeData))
                .addItem(nvrhi::BindingSetItem::Texture_UAV(1, atlas)), m_BlendLayout);
            cmd->setComputeState(nvrhi::ComputeState().setPipeline(pipeline).addBindingSet(set));
            cmd->dispatch(m_TilesX, uint32_t(m_Counts.y));
        };
        blend(m_BlendIrradiance, m_Irradiance);
        blend(m_BlendDistance, m_Distance);
        blend(m_BorderIrradiance, m_Irradiance);
        blend(m_BorderDistance, m_Distance);

        m_Reset = false;
        ++m_FramesSinceReset;
    }

    nvrhi::ITexture* Ddgi::Gather(nvrhi::ICommandList* cmd, const engine::PlanarView& view, GBuffer& gbuffer,
                                  const RtLighting& lighting)
    {
        if (!Active() || !m_Gather)
            return nullptr;
        const auto& d = gbuffer.Depth->getDesc();
        if (!m_Indirect || m_Indirect->getDesc().width != d.width || m_Indirect->getDesc().height != d.height)
        {
            m_Indirect = m_Ctx.device->createTexture(nvrhi::TextureDesc()
                .setDebugName("hyper/ddgi-indirect").setWidth(d.width).setHeight(d.height)
                .setFormat(nvrhi::Format::RGBA16_FLOAT).setIsUAV(true)
                .setInitialState(nvrhi::ResourceStates::ShaderResource).setKeepInitialState(true));
        }
        auto scope = m_Ctx.profiler->Measure(cmd, "ddgi gather");

        GatherConstants c{};
        view.FillPlanarViewConstants(c.view);
        c.size[0] = d.width;
        c.size[1] = d.height;
        cmd->writeBuffer(m_GatherConstants, &c, sizeof(c));

        nvrhi::BindingSetDesc s;
        s.addItem(nvrhi::BindingSetItem::ConstantBuffer(0, m_GatherConstants))
            .addItem(nvrhi::BindingSetItem::Texture_SRV(2, gbuffer.Depth))
            .addItem(nvrhi::BindingSetItem::Texture_SRV(3, gbuffer.GBufferNormals))
            .addItem(nvrhi::BindingSetItem::Texture_SRV(4, gbuffer.GBufferDiffuse))
            .addItem(nvrhi::BindingSetItem::Texture_SRV(5, gbuffer.GBufferSpecular))
            .addItem(nvrhi::BindingSetItem::Texture_UAV(3, m_Indirect));
        lighting.AddSetItems(s);
        cmd->setComputeState(nvrhi::ComputeState().setPipeline(m_Gather).addBindingSet(m_Sets.Get(m_Ctx.device, s, m_GatherLayout)));
        cmd->dispatch((d.width + 7) / 8, (d.height + 7) / 8);
        return m_Indirect;
    }

    void Ddgi::BuildUI()
    {
        ImGui::Checkbox("DDGI (ray-traced probe GI)", &settings.enabled);
        if (!settings.enabled)
            return;
        if (m_Irradiance)
            ImGui::TextDisabled("%d x %d x %d probes, spacing %.2f", m_Counts.x, m_Counts.y, m_Counts.z,
                std::min(m_Spacing.x, std::min(m_Spacing.y, m_Spacing.z)));
        const int budgets[] = { 512, 1024, 2048, 4096, 8192, 16384 };
        const char* budgetNames[] = { "512", "1024", "2048", "4096", "8192", "16384" };
        int b = 3;
        for (int i = 0; i < 6; ++i)
            if (budgets[i] == settings.probeBudget)
                b = i;
        if (ImGui::Combo("probe budget", &b, budgetNames, 6))
            settings.probeBudget = budgets[b];
        const int rays[] = { 64, 128, 192, 256 };
        const char* rayNames[] = { "64", "128", "192", "256" };
        int r = 1;
        for (int i = 0; i < 4; ++i)
            if (rays[i] == settings.raysPerProbe)
                r = i;
        if (ImGui::Combo("rays per probe", &r, rayNames, 4))
            settings.raysPerProbe = rays[r];
        ImGui::SliderFloat("GI hysteresis", &settings.hysteresis, 0.f, 0.995f);
        ImGui::SliderFloat("GI intensity", &settings.intensity, 0.f, 4.f);
        ImGui::SliderFloat("GI normal bias", &settings.normalBias, 0.f, 1.f);
        ImGui::SliderFloat("GI view bias", &settings.viewBias, 0.f, 1.f);
        ImGui::Checkbox("probe relocation", &settings.relocate);
    }
}
