// hello: one app touching every level hyper offers.
//   L0  raw Vulkan   -- reads the GPU's limits straight from VkPhysicalDevice
//   L1  NVRHI        -- compute pipeline tracing a small scene into HDR colour + depth
//   L2  hyper passes -- PostProcess: TAA, bloom, auto exposure, AgX
//   dev loop         -- fly camera, profiler scopes, shader hot reload, F12 screenshots

#include <hyper/hyper.h>
#include <hyper/passes/PostProcess.h>

#include <donut/core/log.h>
#include <donut/core/math/math.h>
#include <nvrhi/utils.h>

#include <imgui.h>

using namespace donut;
using namespace donut::math;
#include <donut/shaders/view_cb.h>

struct SceneConstants
{
    float time;
    float sunSpeed;
    float pad[2];
};

class Hello : public hyper::App
{
public:
    explicit Hello(hyper::Context& ctx) : App(ctx), m_Post(ctx) { }

    bool Init() override
    {
        auto& ctx = Ctx();

        if (hyper::vk::IsVulkan(ctx.device))
        {
            auto props = ::vk::PhysicalDevice(hyper::vk::PhysicalDevice(ctx.device)).getProperties();
            log::info("raw Vulkan: %s, API %u.%u, max compute workgroup %u",
                props.deviceName.data(), VK_API_VERSION_MAJOR(props.apiVersion), VK_API_VERSION_MINOR(props.apiVersion),
                props.limits.maxComputeWorkGroupInvocations);
        }

        m_Layout = ctx.device->createBindingLayout(nvrhi::BindingLayoutDesc()
            .setVisibility(nvrhi::ShaderType::Compute)
            .addItem(nvrhi::BindingLayoutItem::VolatileConstantBuffer(0))
            .addItem(nvrhi::BindingLayoutItem::PushConstants(1, sizeof(SceneConstants)))
            .addItem(nvrhi::BindingLayoutItem::Texture_UAV(0))
            .addItem(nvrhi::BindingLayoutItem::Texture_UAV(1)));

        m_ViewBuffer = ctx.device->createBuffer(
            nvrhi::utils::CreateVolatileConstantBufferDesc(sizeof(PlanarViewConstants), "hello/view", 16));

        m_CommandList = ctx.device->createCommandList();
        return CreatePipeline();
    }

    void ShadersReloaded() override
    {
        CreatePipeline();
        m_Post.ShadersReloaded();
    }

    void Render(nvrhi::IFramebuffer* framebuffer) override
    {
        auto& ctx = Ctx();
        const auto& fbInfo = framebuffer->getFramebufferInfo();

        m_Color.Resize(ctx.device, fbInfo.width, fbInfo.height);
        m_Depth.Resize(ctx.device, fbInfo.width, fbInfo.height);

        ctx.camera.GetView(m_View, fbInfo.width, fbInfo.height);
        m_Post.PrepareView(m_View);
        PlanarViewConstants viewConstants;
        m_View.FillPlanarViewConstants(viewConstants);

        SceneConstants constants{ ctx.time, m_SunSpeed, {} };

        auto bindings = ctx.bindingCache->GetOrCreateBindingSet(nvrhi::BindingSetDesc()
            .addItem(nvrhi::BindingSetItem::ConstantBuffer(0, m_ViewBuffer))
            .addItem(nvrhi::BindingSetItem::PushConstants(1, sizeof(SceneConstants)))
            .addItem(nvrhi::BindingSetItem::Texture_UAV(0, m_Color))
            .addItem(nvrhi::BindingSetItem::Texture_UAV(1, m_Depth)), m_Layout);

        m_CommandList->open();
        {
            auto t = ctx.profiler->Measure(m_CommandList, "hello");
            {
                auto ts = ctx.profiler->Measure(m_CommandList, "scene");
                m_CommandList->writeBuffer(m_ViewBuffer, &viewConstants, sizeof(viewConstants));
                m_CommandList->setComputeState(nvrhi::ComputeState().setPipeline(m_Pipeline).addBindingSet(bindings));
                m_CommandList->setPushConstants(&constants, sizeof(constants));
                m_CommandList->dispatch((fbInfo.width + 7) / 8, (fbInfo.height + 7) / 8);
            }
            m_Post.Render(m_CommandList, { m_Color, m_Depth }, m_View, framebuffer);
        }
        m_CommandList->close();
        ctx.device->executeCommandList(m_CommandList);
    }

    void BuildUI() override
    {
        ImGui::SetNextWindowPos(ImVec2(10.f, 260.f), ImGuiCond_FirstUseEver);
        ImGui::Begin("hello", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
        ImGui::SliderFloat("sun speed", &m_SunSpeed, 0.f, 2.f);
        const float3 p = Ctx().camera.fly.GetPosition();
        ImGui::Text("camera %.2f %.2f %.2f", p.x, p.y, p.z);
        ImGui::TextDisabled("LMB drag: look, WASD/QE: move, F12: screenshot");
        ImGui::Separator();
        m_Post.BuildUI();
        ImGui::End();
    }

private:
    hyper::PostProcess m_Post;
    nvrhi::BindingLayoutHandle m_Layout;
    nvrhi::ComputePipelineHandle m_Pipeline;
    nvrhi::BufferHandle m_ViewBuffer;
    nvrhi::CommandListHandle m_CommandList;
    engine::PlanarView m_View;
    hyper::RenderTarget m_Color{ nvrhi::TextureDesc()
        .setDebugName("hello/color")
        .setFormat(nvrhi::Format::RGBA16_FLOAT)
        .setIsUAV(true) };
    hyper::RenderTarget m_Depth{ nvrhi::TextureDesc()
        .setDebugName("hello/depth")
        .setFormat(nvrhi::Format::R32_FLOAT)
        .setIsUAV(true) };
    float m_SunSpeed = 0.3f;

    // Keeps the old pipeline if the new shader can't be created.
    bool CreatePipeline()
    {
        auto& ctx = Ctx();
        nvrhi::ShaderHandle shader = ctx.shaders->CreateShader("app/scene_cs", "main", nullptr, nvrhi::ShaderType::Compute);
        if (!shader)
            return false;

        auto pipeline = ctx.device->createComputePipeline(
            nvrhi::ComputePipelineDesc().setComputeShader(shader).addBindingLayout(m_Layout));
        if (!pipeline)
            return false;

        m_Pipeline = pipeline;
        return true;
    }
};

int main(int argc, const char* const* argv)
{
    hyper::AppDesc desc;
    desc.title = "hyper: hello";
    desc.shaderDir = "hello";
    return hyper::Run<Hello>(argc, argv, desc);
}
