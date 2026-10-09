// clouds: hyper's sky on its own, no scene: the Hillaire atmosphere, a Nubis-
// style cloud layer (hyper::RaymarchedVolumes), sky ambient from the IBL capture,
// and the post stack. Each module is driven directly, as an app without
// SceneRenderer would.
//
//   clouds.exe [--set time=16.5 coverage=0.5 density=0.02 base=1500 top=4000 ...]
//
// Fly camera: hold LMB + drag to look, WASD/QE to move, Shift faster.

#include <hyper/hyper.h>
#include <hyper/lighting/Atmosphere.h>
#include <hyper/lighting/EnvironmentLighting.h>
#include <hyper/passes/PostProcess.h>
#include <hyper/volume/RaymarchedVolumes.h>

#include <donut/engine/SceneGraph.h>

#include <imgui.h>

#include <cmath>

using namespace donut;
using namespace donut::math;

class CloudsApp : public hyper::App
{
public:
    explicit CloudsApp(hyper::Context& ctx)
        : App(ctx), m_Post(ctx), m_Atmosphere(ctx), m_Environment(ctx), m_Volumes(ctx) { }

    bool Init() override
    {
        auto& ctx = Ctx();
        // Donut's lights live in a scene graph: a root and one directional light.
        m_Graph = std::make_shared<engine::SceneGraph>();
        m_Graph->SetRootNode(std::make_shared<engine::SceneGraphNode>());
        m_Sun = std::make_shared<engine::DirectionalLight>();
        auto node = std::make_shared<engine::SceneGraphNode>();
        node->SetLeaf(m_Sun);
        m_Graph->Attach(m_Graph->GetRootNode(), node);
        m_Sun->SetName("Sun");
        m_Sun->angularSize = 0.53f;

        auto& clouds = m_Volumes.settings.clouds;
        clouds.enabled = true;
        m_Atmosphere.settings.altitudeKm = 0.3f;
        m_Post.settings.tonemap.autoExposure = true;

        hyper::Overrides o(ctx.args);
        o.Apply("time", m_TimeOfDay);
        o.Apply("heading", m_HeadingDeg);
        o.Apply("sun", m_SunIrradiance);
        o.Apply("coverage", clouds.coverage);
        o.Apply("density", clouds.density);
        o.Apply("base", clouds.bottom);
        o.Apply("top", clouds.top);
        o.Apply("steps", m_Volumes.settings.steps);
        o.Apply("animate", m_Animate);
        o.WarnUnused();

        // On the ground, looking along the heading a little above the horizon.
        const float h = radians(m_HeadingDeg);
        ctx.camera.fly.LookAt(float3(0.f, 2.f, 0.f), float3(std::sin(h), 2.f + 0.25f, std::cos(h)));
        ctx.camera.fly.SetMoveSpeed(50.f);
        m_CommandList = ctx.device->createCommandList();
        return true;
    }

    void ShadersReloaded() override
    {
        m_Post.ShadersReloaded();
        m_Atmosphere.CreatePipelines();
        m_Environment.CreatePipelines();
        m_Volumes.CreatePipelines();
    }

    void Animate(float) override
    {
        if (m_Animate)
            m_TimeOfDay = std::fmod(m_TimeOfDay + Ctx().deltaTime * 0.1f, 24.f);   // an hour every 10 s
    }

    void Render(nvrhi::IFramebuffer* framebuffer) override
    {
        auto& ctx = Ctx();
        const auto& fb = framebuffer->getFramebufferInfo();
        EnsureTargets(fb.width, fb.height);

        ctx.camera.GetView(m_View, fb.width, fb.height);
        m_Post.PrepareView(m_View);

        // Sun from the time of day: rises at 6, 60 degrees at noon, sets at 18.
        const float elevation = radians(60.f * std::sin(PI_f * (m_TimeOfDay - 6.f) / 12.f));
        const float azimuth = radians(m_HeadingDeg + (m_TimeOfDay - 12.f) * 15.f);
        const float3 towardsSun(std::cos(elevation) * std::sin(azimuth), std::sin(elevation), std::cos(elevation) * std::cos(azimuth));
        m_Sun->SetDirection(double3(-towardsSun));
        m_Sun->irradiance = m_SunIrradiance;
        m_Sun->color = m_Atmosphere.SunTransmittance(towardsSun, m_View.GetViewOrigin().y);
        m_Graph->Refresh(uint32_t(ctx.frameIndex));

        m_CommandList->open();
        {
            auto t = ctx.profiler->Measure(m_CommandList, "clouds app");
            m_Atmosphere.Update(m_CommandList, m_View, *m_Sun);
            const hyper::SkyCapture sky{ [this](nvrhi::ICommandList* c, nvrhi::ITexture* cube) { m_Atmosphere.Capture(c, cube); },
                                         m_Atmosphere.Revision() };
            m_Environment.Update(m_CommandList, *m_Sun, render::SkyParameters(), &sky);

            m_CommandList->clearTextureFloat(m_Color, nvrhi::AllSubresources, nvrhi::Color(0.f));
            m_CommandList->clearDepthStencilTexture(m_Depth, nvrhi::AllSubresources, true, 0.f, false, 0);
            m_Atmosphere.RenderSky(m_CommandList, m_SkyFramebuffer);

            hyper::RaymarchedVolumes::Inputs in;
            in.color = m_Color;
            in.depth = m_Depth;
            in.sunDirection = float3(m_Sun->GetDirection());
            in.sunRadiance = m_Sun->color * m_Sun->irradiance;
            in.sunIrradiance = m_Sun->irradiance;
            in.probe = m_Environment.Probe().get();
            in.aerialVolume = m_Atmosphere.AerialVolume();
            in.aerialMaxKm = m_Atmosphere.AerialMaxKm();
            in.aerialUnitKm = m_Atmosphere.AerialUnitKm();
            m_Volumes.Render(m_CommandList, m_View, m_PreviousValid ? m_Previous : m_View, in);

            m_Post.Render(m_CommandList, { m_Color, m_Depth }, m_View, framebuffer);
        }
        m_CommandList->close();
        ctx.device->executeCommandList(m_CommandList);
        m_Previous = m_View;
        m_PreviousValid = true;
    }

    void BuildUI() override
    {
        ImGui::SetNextWindowPos(ImVec2(10.f, 200.f), ImGuiCond_FirstUseEver);
        ImGui::Begin("clouds", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
        ImGui::SliderFloat("time of day", &m_TimeOfDay, 5.f, 19.f, "%.2f h");
        ImGui::Checkbox("animate time", &m_Animate);
        ImGui::SliderFloat("sun irradiance", &m_SunIrradiance, 0.f, 20.f);
        if (ImGui::CollapsingHeader("clouds", ImGuiTreeNodeFlags_DefaultOpen))
            m_Volumes.BuildUI();
        if (ImGui::CollapsingHeader("atmosphere"))
            m_Atmosphere.BuildUI();
        if (ImGui::CollapsingHeader("post"))
            m_Post.BuildUI();
        ImGui::TextDisabled("LMB drag: look, WASD/QE: move, Shift: faster");
        ImGui::End();
    }

private:
    hyper::PostProcess m_Post;
    hyper::Atmosphere m_Atmosphere;
    hyper::EnvironmentLighting m_Environment;
    hyper::RaymarchedVolumes m_Volumes;
    std::shared_ptr<engine::SceneGraph> m_Graph;
    std::shared_ptr<engine::DirectionalLight> m_Sun;
    nvrhi::CommandListHandle m_CommandList;
    nvrhi::TextureHandle m_Color, m_Depth;
    nvrhi::FramebufferHandle m_SkyFramebuffer;
    engine::PlanarView m_View, m_Previous;
    bool m_PreviousValid = false;
    float m_TimeOfDay = 16.5f;
    float m_HeadingDeg = 200.f;
    float m_SunIrradiance = 6.f;
    bool m_Animate = false;

    void EnsureTargets(uint32_t width, uint32_t height)
    {
        if (m_Color && m_Color->getDesc().width == width && m_Color->getDesc().height == height)
            return;
        auto& ctx = Ctx();
        m_Color = ctx.device->createTexture(nvrhi::TextureDesc()
            .setDebugName("clouds/color").setWidth(width).setHeight(height)
            .setFormat(nvrhi::Format::RGBA16_FLOAT).setIsRenderTarget(true).setIsUAV(true)
            .setClearValue(nvrhi::Color(0.f))
            .setInitialState(nvrhi::ResourceStates::ShaderResource).setKeepInitialState(true));
        m_Depth = ctx.device->createTexture(nvrhi::TextureDesc()
            .setDebugName("clouds/depth").setWidth(width).setHeight(height)
            .setFormat(nvrhi::Format::D32).setIsRenderTarget(true).setIsTypeless(true)
            .setClearValue(nvrhi::Color(0.f))
            .setInitialState(nvrhi::ResourceStates::DepthWrite).setKeepInitialState(true));
        m_SkyFramebuffer = ctx.device->createFramebuffer(nvrhi::FramebufferDesc()
            .addColorAttachment(m_Color).setDepthAttachment(m_Depth));
        ctx.bindingCache->Clear();
        m_PreviousValid = false;
    }
};

int main(int argc, const char* const* argv)
{
    hyper::AppDesc desc;
    desc.title = "hyper: clouds";
    return hyper::Run<CloudsApp>(argc, argv, desc);
}
