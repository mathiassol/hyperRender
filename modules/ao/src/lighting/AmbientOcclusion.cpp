#include <hyper/lighting/AmbientOcclusion.h>

#include <hyper/App.h>

#include <donut/engine/View.h>
#include <nvrhi/utils.h>

#include <imgui.h>

#include <algorithm>
#include <cmath>

#include "../../thirdparty/xegtao/XeGTAO.h"

using namespace donut;

namespace hyper
{
    namespace
    {
        struct GtaoExtra
        {
            float worldToView[3][4];
            uint32_t qualitySlices;
            uint32_t qualitySteps;
            uint32_t size[2];
            uint32_t multiBounce;
            uint32_t pad[3];
        };

        // Slices / steps per slice for XeGTAO's quality presets.
        constexpr uint32_t c_Slices[4] = { 1, 2, 3, 9 };
        constexpr uint32_t c_Steps[4] = { 2, 2, 3, 3 };
    }

    AmbientOcclusion::AmbientOcclusion(Context& ctx) : m_Ctx(ctx)
    {
        auto layout = [&](std::initializer_list<nvrhi::BindingLayoutItem> items)
        {
            nvrhi::BindingLayoutDesc desc;
            desc.setVisibility(nvrhi::ShaderType::Compute);
            desc.addItem(nvrhi::BindingLayoutItem::VolatileConstantBuffer(0));
            desc.addItem(nvrhi::BindingLayoutItem::PushConstants(1, sizeof(GtaoExtra)));
            desc.addItem(nvrhi::BindingLayoutItem::Sampler(0));
            for (const auto& item : items)
                desc.addItem(item);
            return ctx.device->createBindingLayout(desc);
        };
        using Item = nvrhi::BindingLayoutItem;
        m_PrefilterLayout = layout({ Item::Texture_SRV(0), Item::Texture_UAV(0), Item::Texture_UAV(1), Item::Texture_UAV(2), Item::Texture_UAV(3), Item::Texture_UAV(4) });
        m_MainLayout = layout({ Item::Texture_SRV(0), Item::Texture_SRV(1), Item::Texture_UAV(0), Item::Texture_UAV(1) });
        m_DenoiseLayout = layout({ Item::Texture_SRV(0), Item::Texture_SRV(1), Item::Texture_UAV(0) });
        m_ResolveLayout = layout({ Item::Texture_SRV(0), Item::Texture_SRV(2), Item::Texture_UAV(1) });

        m_Constants = ctx.device->createBuffer(
            nvrhi::utils::CreateVolatileConstantBufferDesc(sizeof(XeGTAO::GTAOConstants), "hyper/gtao-constants", 16));
    }

    void AmbientOcclusion::CreatePipelines()
    {
        if (!m_Used)
            return;   // built on first use
        auto make = [&](const char* entry, nvrhi::IBindingLayout* layout) -> nvrhi::ComputePipelineHandle
        {
            auto shader = m_Ctx.shaders->CreateShader("hyper/lighting/gtao_cs", entry, nullptr, nvrhi::ShaderType::Compute);
            return shader ? m_Ctx.device->createComputePipeline(
                nvrhi::ComputePipelineDesc().setComputeShader(shader).addBindingLayout(layout)) : nullptr;
        };
        auto prefilter = make("prefilter", m_PrefilterLayout);
        auto main = make("main", m_MainLayout);
        auto denoise = make("denoise", m_DenoiseLayout);
        auto denoiseLast = make("denoise_last", m_DenoiseLayout);
        auto resolve = make("resolve", m_ResolveLayout);
        if (prefilter && main && denoise && denoiseLast && resolve)
        {
            m_Prefilter = prefilter;
            m_Main = main;
            m_Denoise = denoise;
            m_DenoiseLast = denoiseLast;
            m_Resolve = resolve;
        }
    }

    nvrhi::ITexture* AmbientOcclusion::Render(nvrhi::ICommandList* cmd, const engine::PlanarView& view,
                                              nvrhi::ITexture* depth, nvrhi::ITexture* normals, nvrhi::ITexture* albedo)
    {
        if (!m_Used)
        {
            m_Used = true;
            CreatePipelines();
        }
        if (!settings.enabled || !m_Main)
            return nullptr;

        const auto& desc = depth->getDesc();
        if (!m_Output || desc.width != m_Width || desc.height != m_Height)
        {
            m_Width = desc.width;
            m_Height = desc.height;
            auto tex = [&](const char* name, nvrhi::Format format, uint32_t mips = 1)
            {
                return m_Ctx.device->createTexture(nvrhi::TextureDesc()
                    .setDebugName(name).setWidth(m_Width).setHeight(m_Height).setMipLevels(mips)
                    .setFormat(format).setIsUAV(true)
                    .setInitialState(nvrhi::ResourceStates::ShaderResource).setKeepInitialState(true));
            };
            m_WorkingDepth = tex("hyper/gtao-depth", nvrhi::Format::R32_FLOAT, XE_GTAO_DEPTH_MIP_LEVELS);
            m_Term[0] = tex("hyper/gtao-term0", nvrhi::Format::R8_UINT);
            m_Term[1] = tex("hyper/gtao-term1", nvrhi::Format::R8_UINT);
            m_Edges = tex("hyper/gtao-edges", nvrhi::Format::R8_UNORM);
            m_Output = tex("hyper/gtao-ao", nvrhi::Format::R8_UNORM);
        }

        auto scope = m_Ctx.profiler->Measure(cmd, "ao");

        XeGTAO::GTAOSettings gtao;
        gtao.QualityLevel = std::clamp(settings.quality, 0, 3);
        gtao.DenoisePasses = std::clamp(settings.denoisePasses, 1, 3);
        gtao.Radius = settings.radius;
        gtao.FalloffRange = settings.falloff;
        gtao.FinalValuePower = settings.power;
        gtao.ThinOccluderCompensation = std::clamp(settings.thinOccluders, 0.f, 0.7f);

        // Row-major, row-vector projection without the TAA jitter; XeGTAO reads it with rowMajor = true.
        const dm::float4x4 proj = view.GetProjectionMatrix(false);
        float projArray[16];
        for (int r = 0; r < 4; ++r)
            for (int c = 0; c < 4; ++c)
                projArray[r * 4 + c] = proj[r][c];
        XeGTAO::GTAOConstants constants{};
        XeGTAO::GTAOUpdateConstants(constants, int(m_Width), int(m_Height), gtao, projArray, true, uint32_t(m_Ctx.frameIndex));
        cmd->writeBuffer(m_Constants, &constants, sizeof(constants));

        GtaoExtra extra{};
        const dm::affine3 worldToView = view.GetViewMatrix();
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                extra.worldToView[r][c] = worldToView.m_linear[c][r];   // row r of the rotation applied to column vectors
        extra.qualitySlices = c_Slices[gtao.QualityLevel];
        extra.qualitySteps = c_Steps[gtao.QualityLevel];
        extra.size[0] = m_Width;
        extra.size[1] = m_Height;
        extra.multiBounce = settings.multiBounce && albedo ? 1u : 0u;

        auto dispatch = [&](nvrhi::IComputePipeline* pipeline, nvrhi::IBindingLayout* layout,
                            std::initializer_list<nvrhi::BindingSetItem> items, uint32_t x, uint32_t y)
        {
            nvrhi::BindingSetDesc set;
            set.addItem(nvrhi::BindingSetItem::ConstantBuffer(0, m_Constants));
            set.addItem(nvrhi::BindingSetItem::PushConstants(1, sizeof(GtaoExtra)));
            set.addItem(nvrhi::BindingSetItem::Sampler(0, m_Ctx.common->m_PointClampSampler));
            for (const auto& item : items)
                set.addItem(item);
            cmd->setComputeState(nvrhi::ComputeState().setPipeline(pipeline)
                .addBindingSet(m_Ctx.bindingCache->GetOrCreateBindingSet(set, layout)));
            cmd->setPushConstants(&extra, sizeof(extra));
            cmd->dispatch(x, y);
        };
        using Set = nvrhi::BindingSetItem;
        auto uav = [&](uint32_t slot, nvrhi::ITexture* t, uint32_t level)
        {
            return Set::Texture_UAV(slot, t, nvrhi::Format::UNKNOWN, nvrhi::TextureSubresourceSet(level, 1, 0, 1));
        };

        dispatch(m_Prefilter, m_PrefilterLayout,
            { Set::Texture_SRV(0, depth), uav(0, m_WorkingDepth, 0), uav(1, m_WorkingDepth, 1), uav(2, m_WorkingDepth, 2),
              uav(3, m_WorkingDepth, 3), uav(4, m_WorkingDepth, 4) },
            (m_Width + 15) / 16, (m_Height + 15) / 16);

        dispatch(m_Main, m_MainLayout,
            { Set::Texture_SRV(0, m_WorkingDepth), Set::Texture_SRV(1, normals), Set::Texture_UAV(0, m_Term[0]), Set::Texture_UAV(1, m_Edges) },
            (m_Width + XE_GTAO_NUMTHREADS_X - 1) / XE_GTAO_NUMTHREADS_X, (m_Height + XE_GTAO_NUMTHREADS_Y - 1) / XE_GTAO_NUMTHREADS_Y);

        int src = 0;
        for (int pass = 0; pass < gtao.DenoisePasses; ++pass)
        {
            const bool last = pass == gtao.DenoisePasses - 1;
            dispatch(last ? m_DenoiseLast : m_Denoise, m_DenoiseLayout,
                { Set::Texture_SRV(0, m_Term[src]), Set::Texture_SRV(1, m_Edges), Set::Texture_UAV(0, m_Term[1 - src]) },
                ((m_Width + 1) / 2 + XE_GTAO_NUMTHREADS_X - 1) / XE_GTAO_NUMTHREADS_X, (m_Height + XE_GTAO_NUMTHREADS_Y - 1) / XE_GTAO_NUMTHREADS_Y);
            src = 1 - src;
        }

        dispatch(m_Resolve, m_ResolveLayout,
            { Set::Texture_SRV(0, m_Term[src]), Set::Texture_SRV(2, albedo ? albedo : m_Ctx.common->m_BlackTexture.Get()),
              Set::Texture_UAV(1, m_Output) },
            (m_Width + 7) / 8, (m_Height + 7) / 8);

        return m_Output;
    }

    void AmbientOcclusion::BuildUI()
    {
        ImGui::Checkbox("ambient occlusion (XeGTAO)", &settings.enabled);
        if (!settings.enabled)
            return;
        const char* qualities[] = { "low", "medium", "high", "ultra" };
        ImGui::Combo("AO quality", &settings.quality, qualities, IM_ARRAYSIZE(qualities));
        ImGui::SliderFloat("AO radius", &settings.radius, 0.05f, 5.f, "%.2f", ImGuiSliderFlags_Logarithmic);
        ImGui::SliderFloat("AO falloff", &settings.falloff, 0.05f, 1.f);
        ImGui::SliderFloat("AO power", &settings.power, 0.5f, 5.f);
        ImGui::SliderInt("AO denoise passes", &settings.denoisePasses, 1, 3);
        ImGui::SliderFloat("AO thin occluders", &settings.thinOccluders, 0.f, 0.7f);
        ImGui::Checkbox("AO multi-bounce", &settings.multiBounce);
    }
}
