#include <hyper/rt/RtShadows.h>

#include <hyper/App.h>
#include <hyper/rt/RtScene.h>
#include <hyper/scene/Scene.h>

#include <donut/engine/SceneGraph.h>
#include <donut/engine/View.h>
#include <nvrhi/utils.h>

#include <imgui.h>

#include <cmath>

using namespace donut;
using namespace donut::math;
#include <donut/shaders/view_cb.h>

namespace hyper
{
    namespace
    {
        struct RtShadowConstants
        {
            PlanarViewConstants view;
            float3 sunDir; float tanRadius;
            uint32_t size[2]; uint32_t frameIndex; float normalBias;
        };
    }

    RtShadows::RtShadows(Context& ctx) : m_Ctx(ctx)
    {
        nvrhi::BindingLayoutDesc desc;
        desc.setVisibility(nvrhi::ShaderType::Compute)
            .addItem(nvrhi::BindingLayoutItem::VolatileConstantBuffer(0))
            .addItem(nvrhi::BindingLayoutItem::Texture_SRV(0))
            .addItem(nvrhi::BindingLayoutItem::Texture_SRV(1))
            .addItem(nvrhi::BindingLayoutItem::Texture_UAV(0));
        RtScene::AddLayoutItems(desc);
        m_Layout = ctx.device->createBindingLayout(desc);
        m_Constants = ctx.device->createBuffer(
            nvrhi::utils::CreateVolatileConstantBufferDesc(sizeof(RtShadowConstants), "hyper/rt-shadow-constants", 16));
    }

    void RtShadows::CreatePipelines()
    {
        if (!m_BindlessLayout)
            return;
        auto shader = m_Ctx.shaders->CreateShader("hyper/rt/rt_shadows_cs", "main", nullptr, nvrhi::ShaderType::Compute);
        if (shader)
            m_Pipeline = m_Ctx.device->createComputePipeline(nvrhi::ComputePipelineDesc()
                .setComputeShader(shader).addBindingLayout(m_Layout).addBindingLayout(m_BindlessLayout));
    }

    nvrhi::ITexture* RtShadows::Render(nvrhi::ICommandList* cmd, const engine::PlanarView& view, Scene& scene,
                                       const RtScene& rt, nvrhi::ITexture* depth, nvrhi::ITexture* normals)
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
                .setDebugName("hyper/rt-shadows").setWidth(d.width).setHeight(d.height)
                .setFormat(nvrhi::Format::RGBA8_UNORM).setIsUAV(true)
                .setInitialState(nvrhi::ResourceStates::ShaderResource).setKeepInitialState(true));
        }

        auto scope = m_Ctx.profiler->Measure(cmd, "rt shadows");

        const engine::DirectionalLight& sun = scene.Sun();
        RtShadowConstants c{};
        view.FillPlanarViewConstants(c.view);
        c.sunDir = normalize(-float3(sun.GetDirection()));
        c.tanRadius = std::tan(radians(std::max(sun.angularSize, 0.01f) * 0.5f * std::max(settings.softness, 0.f)));
        c.size[0] = d.width;
        c.size[1] = d.height;
        c.frameIndex = uint32_t(m_Ctx.frameIndex);
        c.normalBias = settings.normalBias;
        cmd->writeBuffer(m_Constants, &c, sizeof(c));

        nvrhi::BindingSetDesc setDesc;
        setDesc.addItem(nvrhi::BindingSetItem::ConstantBuffer(0, m_Constants))
            .addItem(nvrhi::BindingSetItem::Texture_SRV(0, depth))
            .addItem(nvrhi::BindingSetItem::Texture_SRV(1, normals))
            .addItem(nvrhi::BindingSetItem::Texture_UAV(0, m_Output));
        rt.AddSetItems(setDesc, scene);
        nvrhi::IBindingSet* set = m_Sets.Get(m_Ctx.device, setDesc, m_Layout);

        cmd->setComputeState(nvrhi::ComputeState().setPipeline(m_Pipeline)
            .addBindingSet(set).addBindingSet(scene.DescriptorTable()));
        cmd->dispatch((d.width + 7) / 8, (d.height + 7) / 8);
        return m_Output;
    }

    void RtShadows::BuildUI()
    {
        ImGui::Checkbox("ray-traced sun shadows", &settings.enabled);
        if (!settings.enabled)
            return;
        ImGui::SliderFloat("RT shadow softness", &settings.softness, 0.f, 10.f);
        ImGui::SliderFloat("RT shadow bias", &settings.normalBias, 0.f, 0.1f, "%.3f");
    }
}
