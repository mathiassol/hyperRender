#include <hyper/lighting/SunShadows.h>

#include <hyper/App.h>
#include <hyper/gpu/GpuDrivenRenderer.h>
#include <hyper/scene/Scene.h>

#include <donut/engine/FramebufferFactory.h>
#include <donut/engine/SceneGraph.h>
#include <donut/engine/View.h>
#include <donut/render/CascadedShadowMap.h>
#include <donut/render/DepthPass.h>
#include <donut/render/DrawStrategy.h>
#include <donut/render/GeometryPasses.h>
#include <nvrhi/utils.h>

#include <imgui.h>

#include <algorithm>

using namespace donut;

namespace hyper
{
    SunShadows::SunShadows(Context& ctx) : m_Ctx(ctx)
    {
        m_Strategy = std::make_unique<render::InstancedOpaqueDrawStrategy>();
    }

    SunShadows::~SunShadows() = default;

    void SunShadows::CreatePipelines()
    {
        render::DepthPass::CreateParameters params;
        params.depthBias = settings.depthBias;
        params.slopeScaledDepthBias = settings.slopeBias;
        m_DepthPass = std::make_unique<render::DepthPass>(m_Ctx.device, m_Ctx.common);
        m_DepthPass->Init(*m_Ctx.shaders, params);
        m_BuiltBias = settings.depthBias;
        m_BuiltSlopeBias = settings.slopeBias;
    }

    void SunShadows::Render(nvrhi::ICommandList* cmd, Scene& scene, const engine::PlanarView& view, GpuDrivenRenderer* gpu)
    {
        engine::DirectionalLight& sun = scene.Sun();
        if (!settings.enabled)
        {
            sun.shadowMap = nullptr;
            return;
        }

        auto scope = m_Ctx.profiler->Measure(cmd, "shadows");

        settings.resolution = std::clamp(settings.resolution, 256, 8192);
        settings.cascades = std::clamp(settings.cascades, 1, 4);
        if (!m_Csm || m_BuiltResolution != settings.resolution || m_BuiltCascades != settings.cascades)
        {
            const nvrhi::Format formats[] = { nvrhi::Format::D24S8, nvrhi::Format::D32, nvrhi::Format::D16, nvrhi::Format::D32S8 };
            const nvrhi::Format format = nvrhi::utils::ChooseFormat(m_Ctx.device,
                nvrhi::FormatSupport::Texture | nvrhi::FormatSupport::DepthStencil | nvrhi::FormatSupport::ShaderLoad,
                formats, std::size(formats));

            m_Csm = std::make_shared<render::CascadedShadowMap>(m_Ctx.device, settings.resolution, settings.cascades, 0, format);
            m_Csm->SetupProxyViews();
            m_Framebuffer = std::make_shared<engine::FramebufferFactory>(m_Ctx.device);
            m_Framebuffer->DepthTarget = m_Csm->GetTexture();
            m_BuiltResolution = settings.resolution;
            m_BuiltCascades = settings.cascades;
            m_Ctx.bindingCache->Clear();
        }
        if (!m_DepthPass || m_BuiltBias != settings.depthBias || m_BuiltSlopeBias != settings.slopeBias)
            CreatePipelines();

        // Stable cascades: their size doesn't depend on camera orientation, so edges don't crawl.
        const float zRange = length(scene.Bounds().diagonal());
        m_Csm->SetupForPlanarViewStable(sun, view.GetProjectionFrustum(), view.GetInverseViewMatrix(),
            settings.maxDistance, zRange, zRange, settings.exponent);
        m_Csm->Clear(cmd);

        if (gpu)
        {
            for (uint32_t i = 0; i < m_Csm->GetNumberOfCascades(); ++i)
            {
                const auto cascade = m_Csm->GetCascadeView(i);
                gpu->RenderDepth(cmd, *cascade, view, scene, m_Framebuffer->GetFramebuffer(*cascade), i,
                                 settings.depthBias, settings.slopeBias);
            }
        }
        else
        {
            render::DepthPass::Context context;
            render::RenderCompositeView(cmd, &m_Csm->GetView(), nullptr, *m_Framebuffer,
                scene.Graph()->GetRootNode(), *m_Strategy, *m_DepthPass, context, "ShadowMap");
        }

        sun.shadowMap = m_Csm;
    }

    void SunShadows::BuildUI()
    {
        ImGui::Checkbox("sun shadows", &settings.enabled);
        if (!settings.enabled)
            return;
        const char* sizes[] = { "1024", "2048", "4096" };
        int idx = settings.resolution >= 4096 ? 2 : settings.resolution >= 2048 ? 1 : 0;
        if (ImGui::Combo("shadow map size", &idx, sizes, IM_ARRAYSIZE(sizes)))
            settings.resolution = 1024 << idx;
        ImGui::SliderInt("cascades", &settings.cascades, 1, 4);
        ImGui::SliderFloat("shadow distance", &settings.maxDistance, 5.f, 500.f, "%.0f", ImGuiSliderFlags_Logarithmic);
        ImGui::SliderFloat("cascade exponent", &settings.exponent, 1.f, 8.f);
        ImGui::SliderInt("depth bias", &settings.depthBias, 0, 1000);
        ImGui::SliderFloat("slope bias", &settings.slopeBias, 0.f, 10.f);
    }
}
