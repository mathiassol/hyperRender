#include <hyper/rt/RtReflections.h>

#include <hyper/App.h>
#include <hyper/rt/RtLighting.h>
#include <hyper/scene/Scene.h>

#include <donut/engine/View.h>
#include <nvrhi/utils.h>

#include <imgui.h>

using namespace donut;
using namespace donut::math;
#include <donut/shaders/view_cb.h>

namespace hyper
{
    namespace
    {
        struct RtReflectionConstants
        {
            PlanarViewConstants view;
            uint32_t size[2]; uint32_t frameIndex; float maxRoughness;
            float normalBias; float pad[3];
        };
    }

    RtReflections::RtReflections(Context& ctx) : m_Ctx(ctx)
    {
        nvrhi::BindingLayoutDesc desc;
        desc.setVisibility(nvrhi::ShaderType::Compute)
            .addItem(nvrhi::BindingLayoutItem::VolatileConstantBuffer(0))
            .addItem(nvrhi::BindingLayoutItem::Texture_SRV(0))
            .addItem(nvrhi::BindingLayoutItem::Texture_SRV(1))
            .addItem(nvrhi::BindingLayoutItem::Texture_UAV(0));
        RtScene::AddLayoutItems(desc);
        RtLighting::AddLayoutItems(desc);
        m_Layout = ctx.device->createBindingLayout(desc);
        m_Constants = ctx.device->createBuffer(
            nvrhi::utils::CreateVolatileConstantBufferDesc(sizeof(RtReflectionConstants), "hyper/rt-reflection-constants", 16));
    }

    void RtReflections::CreatePipelines()
    {
        if (!m_BindlessLayout)
            return;
        auto shader = m_Ctx.shaders->CreateShader("hyper/rt/rt_reflections_cs", "main", nullptr, nvrhi::ShaderType::Compute);
        if (shader)
            m_Pipeline = m_Ctx.device->createComputePipeline(nvrhi::ComputePipelineDesc()
                .setComputeShader(shader).addBindingLayout(m_Layout).addBindingLayout(m_BindlessLayout));
    }

    nvrhi::ITexture* RtReflections::Trace(nvrhi::ICommandList* cmd, const engine::PlanarView& view, Scene& scene,
                                          const RtScene& rt, const RtLighting& lighting, nvrhi::ITexture* depth, nvrhi::ITexture* normals)
    {
        if (scene.BindlessLayout() != m_BindlessLayout)
        {
            m_BindlessLayout = scene.BindlessLayout();
            CreatePipelines();
        }
        if (!m_Pipeline || !rt.Ready())
            return nullptr;

        const auto& d = depth->getDesc();
        if (!m_Output || m_Output->getDesc().width != d.width || m_Output->getDesc().height != d.height)
        {
            m_Output = m_Ctx.device->createTexture(nvrhi::TextureDesc()
                .setDebugName("hyper/rt-reflections").setWidth(d.width).setHeight(d.height)
                .setFormat(nvrhi::Format::RGBA16_FLOAT).setIsUAV(true)
                .setInitialState(nvrhi::ResourceStates::ShaderResource).setKeepInitialState(true));
            m_Sets.Clear();
        }

        auto scope = m_Ctx.profiler->Measure(cmd, "rt reflections");

        RtReflectionConstants c{};
        view.FillPlanarViewConstants(c.view);
        c.size[0] = d.width;
        c.size[1] = d.height;
        c.frameIndex = uint32_t(m_Ctx.frameIndex);
        c.maxRoughness = settings.maxRoughness;
        c.normalBias = settings.normalBias;
        cmd->writeBuffer(m_Constants, &c, sizeof(c));

        nvrhi::BindingSetDesc setDesc;
        setDesc.addItem(nvrhi::BindingSetItem::ConstantBuffer(0, m_Constants))
            .addItem(nvrhi::BindingSetItem::Texture_SRV(0, depth))
            .addItem(nvrhi::BindingSetItem::Texture_SRV(1, normals))
            .addItem(nvrhi::BindingSetItem::Texture_UAV(0, m_Output));
        rt.AddSetItems(setDesc, scene);
        lighting.AddSetItems(setDesc);

        cmd->setComputeState(nvrhi::ComputeState().setPipeline(m_Pipeline)
            .addBindingSet(m_Sets.Get(m_Ctx.device, setDesc, m_Layout)).addBindingSet(scene.DescriptorTable()));
        cmd->dispatch((d.width + 7) / 8, (d.height + 7) / 8);
        return m_Output;
    }

    void RtReflections::BuildUI()
    {
        ImGui::Checkbox("ray-traced reflections", &settings.enabled);
        if (!settings.enabled)
            return;
        ImGui::SliderFloat("RT reflection max roughness", &settings.maxRoughness, 0.05f, 1.f);
        ImGui::TextDisabled("(intensity / history: the SSR settings below)");
    }
}
