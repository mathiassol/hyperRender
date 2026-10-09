#include <hyper/lighting/ScreenSpaceReflections.h>

#include <hyper/App.h>
#include <hyper/lighting/DepthPyramid.h>
#include <hyper/scene/GBuffer.h>

#include <donut/core/math/math.h>
#include <donut/engine/SceneTypes.h>
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
        struct SsrConstants
        {
            PlanarViewConstants view;
            float size[2];
            float invSize[2];
            uint32_t hizMips;
            uint32_t maxSteps;
            uint32_t frameIndex;
            uint32_t hasProbe;
            float maxRoughness;
            float thickness;
            float intensity;
            float historyWeight;
            float probeMips;
            float specularScale;
            float zNear;
            uint32_t historyValid;
        };
    }

    ScreenSpaceReflections::ScreenSpaceReflections(Context& ctx) : m_Ctx(ctx)
    {
        m_Layout = ctx.device->createBindingLayout(nvrhi::BindingLayoutDesc()
            .setVisibility(nvrhi::ShaderType::Compute)
            .addItem(nvrhi::BindingLayoutItem::VolatileConstantBuffer(0))
            .addItem(nvrhi::BindingLayoutItem::Texture_SRV(0))
            .addItem(nvrhi::BindingLayoutItem::Texture_SRV(1))
            .addItem(nvrhi::BindingLayoutItem::Texture_SRV(2))
            .addItem(nvrhi::BindingLayoutItem::Texture_SRV(3))
            .addItem(nvrhi::BindingLayoutItem::Texture_SRV(4))
            .addItem(nvrhi::BindingLayoutItem::Texture_SRV(5))
            .addItem(nvrhi::BindingLayoutItem::Sampler(0))
            .addItem(nvrhi::BindingLayoutItem::Sampler(1))
            .addItem(nvrhi::BindingLayoutItem::Texture_UAV(0)));
        m_Constants = ctx.device->createBuffer(
            nvrhi::utils::CreateVolatileConstantBufferDesc(sizeof(SsrConstants), "hyper/ssr-constants", 16));
    }

    void ScreenSpaceReflections::CreatePipelines()
    {
        if (!m_Used)
            return;   // built on first use
        auto make = [&](const char* entry) -> nvrhi::ComputePipelineHandle
        {
            auto shader = m_Ctx.shaders->CreateShader("hyper/lighting/ssr_cs", entry, nullptr, nvrhi::ShaderType::Compute);
            return shader ? m_Ctx.device->createComputePipeline(
                nvrhi::ComputePipelineDesc().setComputeShader(shader).addBindingLayout(m_Layout)) : nullptr;
        };
        auto trace = make("trace");
        auto temporal = make("temporal");
        auto composite = make("composite");
        if (trace && temporal && composite)
        {
            m_Trace = trace;
            m_Temporal = temporal;
            m_Composite = composite;
        }
    }

    void ScreenSpaceReflections::Render(nvrhi::ICommandList* cmd, const engine::PlanarView& view, GBuffer& gbuffer,
                                        DepthPyramid* hiz, const engine::LightProbe* probe, float zNear,
                                        nvrhi::ITexture* externalTrace, nvrhi::ITexture* specularOcclusion)
    {
        if (!m_Used)
        {
            m_Used = true;
            CreatePipelines();
        }
        const bool traceHere = externalTrace == nullptr;
        if (!m_Trace || (traceHere && (!settings.enabled || !hiz || !hiz->Texture())))
        {
            m_HistoryValid = false;
            return;
        }

        const auto& desc = gbuffer.HdrColor->getDesc();
        if (!m_Traced || desc.width != m_Width || desc.height != m_Height)
        {
            m_Width = desc.width;
            m_Height = desc.height;
            auto tex = [&](const char* name)
            {
                return m_Ctx.device->createTexture(nvrhi::TextureDesc()
                    .setDebugName(name).setWidth(m_Width).setHeight(m_Height)
                    .setFormat(nvrhi::Format::RGBA16_FLOAT).setIsUAV(true)
                    .setInitialState(nvrhi::ResourceStates::ShaderResource).setKeepInitialState(true));
            };
            m_Traced = tex("hyper/ssr-traced");
            m_History[0] = tex("hyper/ssr-history0");
            m_History[1] = tex("hyper/ssr-history1");
            m_HistoryValid = false;
        }

        auto scope = m_Ctx.profiler->Measure(cmd, traceHere ? "ssr" : "reflection resolve");

        SsrConstants c{};
        view.FillPlanarViewConstants(c.view);
        c.size[0] = float(m_Width);
        c.size[1] = float(m_Height);
        c.invSize[0] = 1.f / float(m_Width);
        c.invSize[1] = 1.f / float(m_Height);
        c.hizMips = traceHere ? hiz->MipLevels() : 1u;
        c.maxSteps = uint32_t(std::max(settings.maxSteps, 8));
        c.frameIndex = uint32_t(m_Ctx.frameIndex);
        c.hasProbe = probe && probe->specularMap ? 1u : 0u;
        c.maxRoughness = settings.maxRoughness;
        c.thickness = settings.thickness;
        c.intensity = settings.intensity;
        c.historyWeight = settings.historyWeight;
        c.probeMips = probe && probe->specularMap ? float(probe->specularMap->getDesc().mipLevels) : 1.f;
        c.specularScale = probe ? probe->specularScale : 0.f;
        c.zNear = zNear;
        c.historyValid = m_HistoryValid ? 1u : 0u;
        cmd->writeBuffer(m_Constants, &c, sizeof(c));

        nvrhi::ITexture* black = m_Ctx.common->m_BlackTexture;
        nvrhi::ITexture* probeTex = c.hasProbe ? probe->specularMap.Get() : m_Ctx.common->m_BlackCubeMapArray.Get();
        auto dispatch = [&](nvrhi::IComputePipeline* pipeline, nvrhi::ITexture* t0, nvrhi::ITexture* t1,
                            nvrhi::ITexture* t2, nvrhi::ITexture* t3, nvrhi::ITexture* out)
        {
            auto set = m_Ctx.bindingCache->GetOrCreateBindingSet(nvrhi::BindingSetDesc()
                .addItem(nvrhi::BindingSetItem::ConstantBuffer(0, m_Constants))
                .addItem(nvrhi::BindingSetItem::Texture_SRV(0, t0 ? t0 : black))
                .addItem(nvrhi::BindingSetItem::Texture_SRV(1, t1 ? t1 : black))
                .addItem(nvrhi::BindingSetItem::Texture_SRV(2, t2 ? t2 : black))
                .addItem(nvrhi::BindingSetItem::Texture_SRV(3, t3 ? t3 : black))
                .addItem(nvrhi::BindingSetItem::Texture_SRV(4, probeTex))
                .addItem(nvrhi::BindingSetItem::Texture_SRV(5, specularOcclusion ? specularOcclusion : m_Ctx.common->m_WhiteTexture.Get()))
                .addItem(nvrhi::BindingSetItem::Sampler(0, m_Ctx.common->m_LinearClampSampler))
                .addItem(nvrhi::BindingSetItem::Sampler(1, m_Ctx.common->m_PointClampSampler))
                .addItem(nvrhi::BindingSetItem::Texture_UAV(0, out)), m_Layout);
            cmd->setComputeState(nvrhi::ComputeState().setPipeline(pipeline).addBindingSet(set));
            cmd->dispatch((m_Width + 7) / 8, (m_Height + 7) / 8);
        };

        const uint32_t cur = m_Frame & 1u;
        nvrhi::ITexture* traced = externalTrace;
        if (traceHere)
        {
            dispatch(m_Trace, gbuffer.Depth, gbuffer.GBufferNormals, hiz->Texture(), gbuffer.HdrColor, m_Traced);
            traced = m_Traced;
        }
        dispatch(m_Temporal, traced, m_History[1 - cur], gbuffer.MotionVectors, gbuffer.Depth, m_History[cur]);
        dispatch(m_Composite, m_History[cur], gbuffer.Depth, gbuffer.GBufferNormals, gbuffer.GBufferSpecular, gbuffer.HdrColor);

        ++m_Frame;
        m_HistoryValid = true;
    }

    void ScreenSpaceReflections::BuildUI()
    {
        ImGui::Checkbox("screen-space reflections", &settings.enabled);
        if (!settings.enabled)
            return;
        ImGui::SliderFloat("SSR max roughness", &settings.maxRoughness, 0.05f, 1.f);
        ImGui::SliderFloat("SSR thickness", &settings.thickness, 0.01f, 2.f, "%.2f", ImGuiSliderFlags_Logarithmic);
        ImGui::SliderInt("SSR max steps", &settings.maxSteps, 16, 256);
        ImGui::SliderFloat("SSR intensity", &settings.intensity, 0.f, 2.f);
        ImGui::SliderFloat("SSR history weight", &settings.historyWeight, 0.02f, 1.f);
    }
}
