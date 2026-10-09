#include <hyper/lighting/Atmosphere.h>

#include <hyper/App.h>

#include <donut/engine/SceneGraph.h>
#include <donut/engine/ShaderFactory.h>
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
        constexpr float c_GroundRadius = 6360.f;   // km
        constexpr float c_TopRadius = 6460.f;
        const float3 c_Rayleigh(5.802e-3f, 13.558e-3f, 33.1e-3f);   // per km, at sea level
        constexpr float c_MieScattering = 3.996e-3f;
        constexpr float c_MieExtinction = 4.440e-3f;
        const float3 c_Ozone(0.650e-3f, 1.881e-3f, 0.085e-3f);
        constexpr uint32_t c_AerialSize = 32;
    }

    // Mirrors AtmosphereConstants in atmosphere_cs.hlsl.
    struct Atmosphere::Constants
    {
        PlanarViewConstants view;
        float3 sunDir; float sunIrradiance;
        float3 rayleighScattering; float mieScattering;
        float3 ozoneAbsorption; float mieExtinction;
        float3 groundAlbedo; float mieG;
        float3 cameraKm; float groundRadius;
        float3 cameraWorld; float topRadius;
        float unitKm; float aerialMaxKm; float sunCosRadius; float sunDiskRadiance;
        uint32_t size[2]; uint32_t pad[2];
    };

    Atmosphere::Atmosphere(Context& ctx) : m_Ctx(ctx), m_Last(std::make_unique<Constants>())
    {
        auto layout = [&](int srvs, const int* srvSlots, nvrhi::BindingLayoutItem uav)
        {
            nvrhi::BindingLayoutDesc desc;
            desc.setVisibility(nvrhi::ShaderType::Compute)
                .addItem(nvrhi::BindingLayoutItem::VolatileConstantBuffer(0))
                .addItem(nvrhi::BindingLayoutItem::Sampler(0))
                .addItem(uav);
            for (int i = 0; i < srvs; ++i)
                desc.addItem(nvrhi::BindingLayoutItem::Texture_SRV(srvSlots[i]));
            return ctx.device->createBindingLayout(desc);
        };
        const int lut[] = { 0, 1 };
        const int apply[] = { 0, 2, 3, 4 };
        const int capture[] = { 0, 2 };
        m_LutLayout = layout(2, lut, nvrhi::BindingLayoutItem::Texture_UAV(0));
        m_AerialLayout = layout(2, lut, nvrhi::BindingLayoutItem::Texture_UAV(1));
        m_ApplyLayout = layout(4, apply, nvrhi::BindingLayoutItem::Texture_UAV(0));
        m_CaptureLayout = layout(2, capture, nvrhi::BindingLayoutItem::Texture_UAV(2));
        m_SkyLayout = ctx.device->createBindingLayout(nvrhi::BindingLayoutDesc()
            .setVisibility(nvrhi::ShaderType::Pixel)
            .addItem(nvrhi::BindingLayoutItem::VolatileConstantBuffer(0))
            .addItem(nvrhi::BindingLayoutItem::Sampler(0))
            .addItem(nvrhi::BindingLayoutItem::Texture_SRV(0))
            .addItem(nvrhi::BindingLayoutItem::Texture_SRV(2)));

        m_Constants = ctx.device->createBuffer(
            nvrhi::utils::CreateVolatileConstantBufferDesc(sizeof(Constants), "hyper/atmosphere-constants", 16));

        auto tex = [&](const char* name, uint32_t w, uint32_t h, uint32_t d)
        {
            return ctx.device->createTexture(nvrhi::TextureDesc()
                .setDebugName(name)
                .setDimension(d > 1 ? nvrhi::TextureDimension::Texture3D : nvrhi::TextureDimension::Texture2D)
                .setWidth(w).setHeight(h).setDepth(d)
                .setFormat(nvrhi::Format::RGBA16_FLOAT).setIsUAV(true)
                .setInitialState(nvrhi::ResourceStates::ShaderResource).setKeepInitialState(true));
        };
        m_TransmittanceLut = tex("hyper/atmosphere-transmittance", 256, 64, 1);
        m_MultiScatterLut = tex("hyper/atmosphere-multiscatter", 32, 32, 1);
        m_SkyViewLut = tex("hyper/atmosphere-skyview", 192, 108, 1);
        m_AerialVolume = tex("hyper/atmosphere-aerial", c_AerialSize, c_AerialSize, c_AerialSize);
    }

    Atmosphere::~Atmosphere() = default;

    void Atmosphere::CreatePipelines()
    {
        if (!m_Used)
            return;   // built on first use
        auto make = [&](const char* entry, nvrhi::IBindingLayout* layout, const std::vector<engine::ShaderMacro>* defines = nullptr)
            -> nvrhi::ComputePipelineHandle
        {
            auto shader = m_Ctx.shaders->CreateShader("hyper/lighting/atmosphere_cs", entry, defines, nvrhi::ShaderType::Compute);
            return shader ? m_Ctx.device->createComputePipeline(
                nvrhi::ComputePipelineDesc().setComputeShader(shader).addBindingLayout(layout)) : nullptr;
        };
        auto transmittance = make("transmittance", m_LutLayout);
        auto multiScatter = make("multiscatter", m_LutLayout);
        auto skyView = make("skyview", m_LutLayout);
        auto aerial = make("aerial", m_AerialLayout);
        auto apply = make("apply", m_ApplyLayout);
        auto capture = make("capture", m_CaptureLayout);
        auto sky = m_Ctx.shaders->CreateShader("hyper/lighting/atmosphere_cs", "sky", nullptr, nvrhi::ShaderType::Pixel);
        if (transmittance && multiScatter && skyView && aerial && apply && capture && sky)
        {
            m_SkyShader = sky;
            m_SkyPipeline = nullptr;
            m_Transmittance = transmittance;
            m_MultiScatter = multiScatter;
            m_SkyView = skyView;
            m_Aerial = aerial;
            m_Apply = apply;
            m_Capture = capture;
            m_StaticLutsValid = false;
        }
    }

    void Atmosphere::Fill(Constants& c, const engine::PlanarView& view, const engine::DirectionalLight& sun) const
    {
        const AtmosphereSettings& s = settings;
        view.FillPlanarViewConstants(c.view);
        c.sunDir = normalize(-float3(sun.GetDirection()));
        c.sunIrradiance = sun.irradiance;
        c.rayleighScattering = c_Rayleigh * s.rayleighScale;
        c.mieScattering = c_MieScattering * s.mieScale;
        c.mieExtinction = c_MieExtinction * s.mieScale;
        c.ozoneAbsorption = c_Ozone * s.ozoneScale;
        c.groundAlbedo = s.groundAlbedo;
        c.mieG = std::clamp(s.mieAnisotropy, 0.f, 0.99f);
        c.groundRadius = c_GroundRadius;
        c.topRadius = c_TopRadius;

        const float3 origin = view.GetViewOrigin();
        const float kmPerUnit = s.metersPerUnit * 0.001f;
        c.cameraWorld = origin;
        c.cameraKm = float3(0.f, c_GroundRadius + std::max(s.altitudeKm + origin.y * kmPerUnit, 0.001f), 0.f);
        c.unitKm = kmPerUnit * s.aerialScale;
        c.aerialMaxKm = std::max(s.aerialMaxKm, 0.1f);

        const float radius = radians(std::max(s.sunDiameterDeg, 0.01f) * 0.5f);
        c.sunCosRadius = std::cos(radius);
        // Radiance of a uniform disk with the sun's irradiance (times the scale).
        c.sunDiskRadiance = s.sunDiskScale / (2.f * PI_f * (1.f - c.sunCosRadius));
        c.size[0] = c.size[1] = 0;
        c.pad[0] = c.pad[1] = 0;
    }

    void Atmosphere::Write(nvrhi::ICommandList* cmd, uint32_t width, uint32_t height)
    {
        m_Last->size[0] = width;
        m_Last->size[1] = height;
        cmd->writeBuffer(m_Constants, m_Last.get(), sizeof(Constants));
    }

    void Atmosphere::Update(nvrhi::ICommandList* cmd, const engine::PlanarView& view, const engine::DirectionalLight& sun)
    {
        if (!m_Used)
        {
            m_Used = true;
            CreatePipelines();
        }
        if (!m_Transmittance)
            return;
        auto scope = m_Ctx.profiler->Measure(cmd, "atmosphere");

        if (std::memcmp(&settings, &m_Built, sizeof(settings)) != 0)
        {
            // Only the medium feeds the static LUTs, but any change alters the sky.
            const bool medium = settings.rayleighScale != m_Built.rayleighScale || settings.mieScale != m_Built.mieScale
                || settings.ozoneScale != m_Built.ozoneScale || any(settings.groundAlbedo != m_Built.groundAlbedo);
            if (medium)
                m_StaticLutsValid = false;
            m_Built = settings;
            ++m_Revision;
        }

        Fill(*m_Last, view, sun);
        Write(cmd, 0, 0);

        nvrhi::ITexture* black = m_Ctx.common->m_BlackTexture;
        auto run = [&](nvrhi::IComputePipeline* pipeline, nvrhi::IBindingLayout* layout, nvrhi::BindingSetItem uav,
                       nvrhi::ITexture* t0, nvrhi::ITexture* t1, uint32_t gx, uint32_t gy)
        {
            auto set = m_Ctx.bindingCache->GetOrCreateBindingSet(nvrhi::BindingSetDesc()
                .addItem(nvrhi::BindingSetItem::ConstantBuffer(0, m_Constants))
                .addItem(nvrhi::BindingSetItem::Sampler(0, m_Ctx.common->m_LinearClampSampler))
                .addItem(uav)
                .addItem(nvrhi::BindingSetItem::Texture_SRV(0, t0))
                .addItem(nvrhi::BindingSetItem::Texture_SRV(1, t1)), layout);
            cmd->setComputeState(nvrhi::ComputeState().setPipeline(pipeline).addBindingSet(set));
            cmd->dispatch(gx, gy);
        };

        if (!m_StaticLutsValid)
        {
            run(m_Transmittance, m_LutLayout, nvrhi::BindingSetItem::Texture_UAV(0, m_TransmittanceLut), black, black, 256 / 8, 64 / 8);
            run(m_MultiScatter, m_LutLayout, nvrhi::BindingSetItem::Texture_UAV(0, m_MultiScatterLut), m_TransmittanceLut, black, 32 / 8, 32 / 8);
            m_StaticLutsValid = true;
        }
        run(m_SkyView, m_LutLayout, nvrhi::BindingSetItem::Texture_UAV(0, m_SkyViewLut), m_TransmittanceLut, m_MultiScatterLut,
            (192 + 7) / 8, (108 + 7) / 8);
        if (settings.aerialPerspective)
            run(m_Aerial, m_AerialLayout, nvrhi::BindingSetItem::Texture_UAV(1, m_AerialVolume), m_TransmittanceLut, m_MultiScatterLut,
                c_AerialSize / 8, c_AerialSize / 8);
    }

    void Atmosphere::RenderSky(nvrhi::ICommandList* cmd, nvrhi::IFramebuffer* framebuffer)
    {
        if (!m_Used)
        {
            m_Used = true;
            CreatePipelines();
        }
        if (!m_SkyShader)
            return;
        auto scope = m_Ctx.profiler->Measure(cmd, "sky");

        const auto& fbInfo = framebuffer->getFramebufferInfo();
        if (!m_SkyPipeline || !(m_SkyFbInfo == fbInfo))
        {
            nvrhi::GraphicsPipelineDesc desc;
            desc.primType = nvrhi::PrimitiveType::TriangleStrip;
            desc.VS = m_Ctx.common->m_FullscreenVS;   // z = 0: the far plane under reverse-Z
            desc.PS = m_SkyShader;
            desc.bindingLayouts = { m_SkyLayout };
            desc.renderState.rasterState.setCullNone();
            desc.renderState.depthStencilState.enableDepthTest().disableDepthWrite().disableStencil()
                .setDepthFunc(nvrhi::ComparisonFunc::GreaterOrEqual);
            m_SkyPipeline = m_Ctx.device->createGraphicsPipeline(desc, fbInfo);
            m_SkyFbInfo = fbInfo;
        }

        Write(cmd, fbInfo.width, fbInfo.height);
        auto set = m_Ctx.bindingCache->GetOrCreateBindingSet(nvrhi::BindingSetDesc()
            .addItem(nvrhi::BindingSetItem::ConstantBuffer(0, m_Constants))
            .addItem(nvrhi::BindingSetItem::Sampler(0, m_Ctx.common->m_LinearClampSampler))
            .addItem(nvrhi::BindingSetItem::Texture_SRV(0, m_TransmittanceLut))
            .addItem(nvrhi::BindingSetItem::Texture_SRV(2, m_SkyViewLut)), m_SkyLayout);

        nvrhi::GraphicsState state;
        state.pipeline = m_SkyPipeline;
        state.framebuffer = framebuffer;
        state.bindings = { set };
        state.viewport.addViewportAndScissorRect(fbInfo.getViewport());
        cmd->setGraphicsState(state);
        cmd->draw(nvrhi::DrawArguments().setVertexCount(4));
    }

    void Atmosphere::Apply(nvrhi::ICommandList* cmd, nvrhi::ITexture* color, nvrhi::ITexture* depth)
    {
        if (!m_Used)
        {
            m_Used = true;
            CreatePipelines();
        }
        if (!m_Apply || !settings.aerialPerspective)
            return;
        auto scope = m_Ctx.profiler->Measure(cmd, "aerial perspective");

        const auto& desc = color->getDesc();
        Write(cmd, desc.width, desc.height);
        auto set = m_Ctx.bindingCache->GetOrCreateBindingSet(nvrhi::BindingSetDesc()
            .addItem(nvrhi::BindingSetItem::ConstantBuffer(0, m_Constants))
            .addItem(nvrhi::BindingSetItem::Sampler(0, m_Ctx.common->m_LinearClampSampler))
            .addItem(nvrhi::BindingSetItem::Texture_UAV(0, color))
            .addItem(nvrhi::BindingSetItem::Texture_SRV(0, m_TransmittanceLut))
            .addItem(nvrhi::BindingSetItem::Texture_SRV(2, m_SkyViewLut))
            .addItem(nvrhi::BindingSetItem::Texture_SRV(3, m_AerialVolume))
            .addItem(nvrhi::BindingSetItem::Texture_SRV(4, depth)), m_ApplyLayout);
        cmd->setComputeState(nvrhi::ComputeState().setPipeline(m_Apply).addBindingSet(set));
        cmd->dispatch((desc.width + 7) / 8, (desc.height + 7) / 8);
    }

    void Atmosphere::Capture(nvrhi::ICommandList* cmd, nvrhi::ITexture* cube)
    {
        if (!m_Used)
        {
            m_Used = true;
            CreatePipelines();
        }
        if (!m_Capture)
            return;
        const uint32_t size = cube->getDesc().width;
        Write(cmd, size, size);
        auto set = m_Ctx.bindingCache->GetOrCreateBindingSet(nvrhi::BindingSetDesc()
            .addItem(nvrhi::BindingSetItem::ConstantBuffer(0, m_Constants))
            .addItem(nvrhi::BindingSetItem::Sampler(0, m_Ctx.common->m_LinearClampSampler))
            .addItem(nvrhi::BindingSetItem::Texture_UAV(2, cube, nvrhi::Format::UNKNOWN,
                nvrhi::TextureSubresourceSet(0, 1, 0, 6), nvrhi::TextureDimension::Texture2DArray))
            .addItem(nvrhi::BindingSetItem::Texture_SRV(0, m_TransmittanceLut))
            .addItem(nvrhi::BindingSetItem::Texture_SRV(2, m_SkyViewLut)), m_CaptureLayout);
        cmd->setComputeState(nvrhi::ComputeState().setPipeline(m_Capture).addBindingSet(set));
        cmd->dispatch((size + 7) / 8, (size + 7) / 8, 6);
    }

    dm::float3 Atmosphere::SunTransmittance(const dm::float3& towardsSun, float cameraWorldY) const
    {
        // The transmittance LUT's integral, on the CPU: 40 steps to the top of the atmosphere.
        const AtmosphereSettings& s = settings;
        const float3 dir = normalize(towardsSun);
        const float3 p(0.f, c_GroundRadius + std::max(s.altitudeKm + cameraWorldY * s.metersPerUnit * 0.001f, 0.001f), 0.f);

        auto raySphere = [](const float3& ro, const float3& rd, float radius)
        {
            const float b = dot(ro, rd);
            const float c = dot(ro, ro) - radius * radius;
            if (c > 0.f && b > 0.f)
                return -1.f;
            const float disc = b * b - c;
            if (disc < 0.f)
                return -1.f;
            const float sq = std::sqrt(disc);
            return -b - sq > 0.f ? -b - sq : -b + sq;
        };
        if (raySphere(p, dir, c_GroundRadius) > 0.f)
            return float3(0.f);

        const float tMax = raySphere(p, dir, c_TopRadius);
        const int steps = 40;
        float3 depth(0.f);
        for (int i = 0; i < steps; ++i)
        {
            const float3 pos = p + dir * ((float(i) + 0.5f) / float(steps) * tMax);
            const float h = std::max(length(pos) - c_GroundRadius, 0.f);
            const float rd = std::exp(-h / 8.f);
            const float md = std::exp(-h / 1.2f);
            const float od = std::max(0.f, 1.f - std::abs(h - 25.f) / 15.f);
            depth += c_Rayleigh * s.rayleighScale * rd + float3(c_MieExtinction * s.mieScale * md) + c_Ozone * s.ozoneScale * od;
        }
        const float dt = tMax / float(steps);
        return float3(std::exp(-depth.x * dt), std::exp(-depth.y * dt), std::exp(-depth.z * dt));
    }

    void Atmosphere::BuildUI()
    {
        ImGui::Checkbox("atmosphere (Hillaire)", &settings.enabled);
        if (!settings.enabled)
            return;
        ImGui::SliderFloat("altitude km", &settings.altitudeKm, 0.f, 30.f, "%.2f", ImGuiSliderFlags_Logarithmic);
        ImGui::SliderFloat("air density", &settings.rayleighScale, 0.f, 4.f);
        ImGui::SliderFloat("haze", &settings.mieScale, 0.f, 20.f, "%.2f", ImGuiSliderFlags_Logarithmic);
        ImGui::SliderFloat("ozone", &settings.ozoneScale, 0.f, 4.f);
        ImGui::SliderFloat("haze anisotropy", &settings.mieAnisotropy, 0.f, 0.99f);
        ImGui::ColorEdit3("ground albedo", &settings.groundAlbedo.x);
        ImGui::SliderFloat("sun disk", &settings.sunDiskScale, 0.f, 2.f);
        ImGui::SliderFloat("sun diameter", &settings.sunDiameterDeg, 0.1f, 5.f);
        ImGui::Checkbox("aerial perspective", &settings.aerialPerspective);
        if (settings.aerialPerspective)
        {
            ImGui::SliderFloat("aerial scale", &settings.aerialScale, 1.f, 1000.f, "%.0f", ImGuiSliderFlags_Logarithmic);
            ImGui::SliderFloat("aerial range km", &settings.aerialMaxKm, 1.f, 200.f, "%.0f", ImGuiSliderFlags_Logarithmic);
        }
    }
}
