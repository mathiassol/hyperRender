#include <hyper/rt/ReferencePathTracer.h>

#include <hyper/App.h>
#include <hyper/rt/RtLighting.h>
#include <hyper/scene/Scene.h>

#include <donut/engine/SceneGraph.h>
#include <donut/engine/View.h>
#include <nvrhi/utils.h>

#include <imgui.h>

#include <algorithm>
#include <cmath>
#include <cstring>

using namespace donut;
using namespace donut::math;
#include <donut/shaders/view_cb.h>

namespace hyper
{
    namespace
    {
        struct ReferenceConstants
        {
            PlanarViewConstants view;
            uint32_t size[2]; uint32_t sampleIndex; uint32_t samplesPerFrame;
            uint32_t maxBounces; float clampLuminance; float sunCosMax; float sunPdf;
        };

        // Everything the image depends on, compared bytewise frame to frame.
        struct Key
        {
            float4x4 worldToView, viewToClip;
            double3 sunDirection;
            float3 sunColor;
            float sunIrradiance, sunAngularSize;
            uint32_t width, height, generation, environmentRevision;
            int maxBounces;
            float clamp;
        };
    }

    ReferencePathTracer::ReferencePathTracer(Context& ctx) : m_Ctx(ctx)
    {
        if (!ctx.rayQuery)
            return;
        nvrhi::BindingLayoutDesc desc;
        desc.setVisibility(nvrhi::ShaderType::Compute)
            .addItem(nvrhi::BindingLayoutItem::VolatileConstantBuffer(0))
            .addItem(nvrhi::BindingLayoutItem::Texture_UAV(0));
        RtScene::AddLayoutItems(desc);
        RtLighting::AddLayoutItems(desc);
        m_Layout = ctx.device->createBindingLayout(desc);
        m_Constants = ctx.device->createBuffer(
            nvrhi::utils::CreateVolatileConstantBufferDesc(sizeof(ReferenceConstants), "hyper/reference-pt-constants", 16));
    }

    void ReferencePathTracer::CreatePipelines()
    {
        if (!m_BindlessLayout || !m_Layout)
            return;
        auto shader = m_Ctx.shaders->CreateShader("hyper/rt/reference_pt_cs", "main", nullptr, nvrhi::ShaderType::Compute);
        if (shader)
            m_Pipeline = m_Ctx.device->createComputePipeline(nvrhi::ComputePipelineDesc()
                .setComputeShader(shader).addBindingLayout(m_Layout).addBindingLayout(m_BindlessLayout));
        m_Samples = 0;   // a changed shader makes a different image
    }

    nvrhi::ITexture* ReferencePathTracer::Render(nvrhi::ICommandList* cmd, const engine::PlanarView& view, Scene& scene,
                                                 const RtScene& rt, const RtLighting& lighting, uint32_t width, uint32_t height,
                                                 uint32_t environmentRevision)
    {
        if (!m_Layout)
            return nullptr;
        if (scene.BindlessLayout() != m_BindlessLayout)
        {
            m_BindlessLayout = scene.BindlessLayout();
            CreatePipelines();
        }
        if (!m_Pipeline || !rt.Ready())
            return nullptr;

        if (!m_Accumulation || m_Accumulation->getDesc().width != width || m_Accumulation->getDesc().height != height)
        {
            m_Accumulation = m_Ctx.device->createTexture(nvrhi::TextureDesc()
                .setDebugName("hyper/reference-pt").setWidth(width).setHeight(height)
                .setFormat(nvrhi::Format::RGBA32_FLOAT).setIsUAV(true)
                .setInitialState(nvrhi::ResourceStates::ShaderResource).setKeepInitialState(true));
            m_Sets.Clear();
            m_Samples = 0;
        }

        const auto& sun = scene.Sun();
        Key key;
        std::memset(&key, 0, sizeof(key));   // compared bytewise, padding included
        key.worldToView = affineToHomogeneous(view.GetViewMatrix());
        key.viewToClip = view.GetProjectionMatrix(false);
        key.sunDirection = sun.GetDirection();
        key.sunColor = sun.color;
        key.sunIrradiance = sun.irradiance;
        key.sunAngularSize = sun.angularSize;
        key.width = width;
        key.height = height;
        key.generation = scene.Revision();   // restart when what's in the scene changes
        key.environmentRevision = environmentRevision;
        key.maxBounces = settings.maxBounces;
        key.clamp = settings.clamp;
        const auto* bytes = reinterpret_cast<const uint8_t*>(&key);
        if (m_Key.size() != sizeof(key) || std::memcmp(m_Key.data(), bytes, sizeof(key)) != 0 || !scene.IsStatic())
        {
            m_Key.assign(bytes, bytes + sizeof(key));
            m_Samples = 0;
        }

        const uint32_t spp = uint32_t(std::clamp(settings.samplesPerFrame, 1, 64));
        if (settings.maxSamples > 0 && m_Samples >= uint32_t(settings.maxSamples))
            return m_Accumulation;   // converged enough: keep showing it

        auto scope = m_Ctx.profiler->Measure(cmd, "reference path tracer");

        const double halfAngle = radians(double(std::max(sun.angularSize, 0.01f)) * 0.5);
        ReferenceConstants c{};
        view.FillPlanarViewConstants(c.view);
        c.size[0] = width;
        c.size[1] = height;
        c.sampleIndex = m_Samples;
        c.samplesPerFrame = spp;
        c.maxBounces = uint32_t(std::clamp(settings.maxBounces, 0, 64));
        c.clampLuminance = std::max(settings.clamp, 0.f);
        c.sunCosMax = float(std::cos(halfAngle));
        c.sunPdf = float(1.0 / (2.0 * PI_d * (1.0 - std::cos(halfAngle))));
        cmd->writeBuffer(m_Constants, &c, sizeof(c));

        nvrhi::BindingSetDesc setDesc;
        setDesc.addItem(nvrhi::BindingSetItem::ConstantBuffer(0, m_Constants))
            .addItem(nvrhi::BindingSetItem::Texture_UAV(0, m_Accumulation));
        rt.AddSetItems(setDesc, scene);
        lighting.AddSetItems(setDesc);

        cmd->setComputeState(nvrhi::ComputeState().setPipeline(m_Pipeline)
            .addBindingSet(m_Sets.Get(m_Ctx.device, setDesc, m_Layout)).addBindingSet(scene.DescriptorTable()));
        cmd->dispatch((width + 7) / 8, (height + 7) / 8);
        m_Samples += spp;
        return m_Accumulation;
    }

    void ReferencePathTracer::BuildUI()
    {
        ImGui::Text("%u samples per pixel", m_Samples);
        ImGui::SameLine();
        if (ImGui::SmallButton("restart"))
            Reset();
        ImGui::SliderInt("bounces", &settings.maxBounces, 0, 16);
        ImGui::SliderInt("samples per frame", &settings.samplesPerFrame, 1, 16);
        ImGui::InputInt("stop after", &settings.maxSamples, 256, 1024);
        ImGui::SliderFloat("firefly clamp", &settings.clamp, 0.f, 100.f, settings.clamp > 0.f ? "%.1f" : "off");
        ImGui::TextDisabled("sun, sky and emission; no point/spot lights, glass or fog");
    }
}
