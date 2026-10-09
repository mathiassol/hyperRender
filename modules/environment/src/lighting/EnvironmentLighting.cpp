#include <hyper/lighting/EnvironmentLighting.h>

#include <hyper/App.h>

#include <donut/core/log.h>
#include <donut/core/vfs/VFS.h>
#include <donut/engine/FramebufferFactory.h>
#include <donut/engine/SceneGraph.h>
#include <donut/engine/SceneTypes.h>
#include <donut/engine/TextureCache.h>
#include <donut/engine/View.h>
#include <donut/render/LightProbeProcessingPass.h>

#include <imgui.h>

#include <cmath>
#include <cstring>

using namespace donut;

namespace hyper
{
    namespace
    {
        constexpr uint32_t c_EnvSize = 512;
        constexpr uint32_t c_EnvMips = 10;       // 512 .. 1
        // Donut convolves at up to 4x the output size; these keep a capture at a
        // few ms, so dragging the sun stays interactive. The sky is smooth, and
        // SSR covers sharp on-screen reflections.
        constexpr uint32_t c_DiffuseSize = 16;
        constexpr uint32_t c_DiffuseIntermediate = 64;
        constexpr uint32_t c_SpecularSize = 128;
        constexpr uint32_t c_SpecularMips = 6;   // 128 .. 4

        struct EquirectConstants
        {
            uint32_t size;
            float intensity;
            float rotation;   // radians
            float pad;
        };

        nvrhi::TextureHandle MakeCubeArray(nvrhi::IDevice* device, const char* name, uint32_t size, uint32_t mips)
        {
            return device->createTexture(nvrhi::TextureDesc()
                .setDebugName(name)
                .setDimension(nvrhi::TextureDimension::TextureCubeArray)
                .setWidth(size).setHeight(size).setArraySize(6).setMipLevels(mips)
                .setFormat(nvrhi::Format::RGBA16_FLOAT)
                .setIsRenderTarget(true)
                .setInitialState(nvrhi::ResourceStates::ShaderResource)
                .setKeepInitialState(true));
        }
    }

    EnvironmentLighting::EnvironmentLighting(Context& ctx) : m_Ctx(ctx) { }

    // The cubemaps, probe passes and pipelines, on first use: about 23 MB and a few ms
    // that a disabled module shouldn't cost.
    void EnvironmentLighting::Allocate()
    {
        auto& ctx = m_Ctx;
        m_ProbePass = std::make_unique<render::LightProbeProcessingPass>(ctx.device, ctx.shaders, ctx.common, c_SpecularSize);
        m_DiffusePass = std::make_unique<render::LightProbeProcessingPass>(ctx.device, ctx.shaders, ctx.common, c_DiffuseIntermediate);
        m_TextureCache = std::make_shared<engine::TextureCache>(ctx.device, std::make_shared<vfs::NativeFileSystem>(), nullptr);

        m_EnvCube = ctx.device->createTexture(nvrhi::TextureDesc()
            .setDebugName("hyper/env-cube")
            .setDimension(nvrhi::TextureDimension::TextureCube)
            .setWidth(c_EnvSize).setHeight(c_EnvSize).setArraySize(6).setMipLevels(c_EnvMips)
            .setFormat(nvrhi::Format::RGBA16_FLOAT)
            .setIsRenderTarget(true).setIsUAV(true)
            .setClearValue(nvrhi::Color(0.f))
            .setInitialState(nvrhi::ResourceStates::ShaderResource)
            .setKeepInitialState(true));
        m_EnvDepth = ctx.device->createTexture(nvrhi::TextureDesc()
            .setDebugName("hyper/env-depth")
            .setDimension(nvrhi::TextureDimension::TextureCube)
            .setWidth(c_EnvSize).setHeight(c_EnvSize).setArraySize(6)
            .setFormat(nvrhi::Format::D32)
            .setIsRenderTarget(true).setIsTypeless(true)
            .setClearValue(nvrhi::Color(0.f))
            .setInitialState(nvrhi::ResourceStates::DepthWrite)
            .setKeepInitialState(true));
        m_CubeFramebuffer = std::make_shared<engine::FramebufferFactory>(ctx.device);
        m_CubeFramebuffer->RenderTargets = { m_EnvCube };
        m_CubeFramebuffer->DepthTarget = m_EnvDepth;

        m_Probe = std::make_shared<engine::LightProbe>();
        m_Probe->name = "hyper/environment";
        m_Probe->diffuseMap = MakeCubeArray(ctx.device, "hyper/env-diffuse", c_DiffuseSize, 1);
        m_Probe->specularMap = MakeCubeArray(ctx.device, "hyper/env-specular", c_SpecularSize, c_SpecularMips);
        m_Probe->bounds = dm::frustum::infinite();

        m_EquirectLayout = ctx.device->createBindingLayout(nvrhi::BindingLayoutDesc()
            .setVisibility(nvrhi::ShaderType::Compute)
            .addItem(nvrhi::BindingLayoutItem::PushConstants(0, sizeof(EquirectConstants)))
            .addItem(nvrhi::BindingLayoutItem::Texture_SRV(0))
            .addItem(nvrhi::BindingLayoutItem::Sampler(0))
            .addItem(nvrhi::BindingLayoutItem::Texture_UAV(0)));
        m_Used = true;
        CreatePipelines();
    }

    EnvironmentLighting::~EnvironmentLighting() = default;

    void EnvironmentLighting::CreatePipelines()
    {
        if (!m_Used)
            return;   // built on first use
        auto shader = m_Ctx.shaders->CreateShader("hyper/lighting/equirect_cs", "main", nullptr, nvrhi::ShaderType::Compute);
        if (shader)
            m_EquirectPipeline = m_Ctx.device->createComputePipeline(
                nvrhi::ComputePipelineDesc().setComputeShader(shader).addBindingLayout(m_EquirectLayout));

        engine::CubemapView view;
        view.SetArrayViewports(int(c_EnvSize), 0);
        view.SetTransform(dm::affine3::identity(), 0.1f, 100.f);
        view.UpdateCache();
        m_SkyPass = std::make_unique<render::SkyPass>(m_Ctx.device, m_Ctx.shaders, m_Ctx.common, m_CubeFramebuffer, view);
        m_Dirty = true;
    }

    bool EnvironmentLighting::NeedsCapture(const engine::DirectionalLight& sun, const render::SkyParameters& sky,
                                           const SkyCapture* capture) const
    {
        if (m_Dirty || settings.source != m_Captured.source || settings.file != m_Captured.file
            || settings.fileIntensity != m_Captured.fileIntensity || settings.fileRotationDeg != m_Captured.fileRotationDeg)
            return true;
        if (settings.source == EnvironmentSource::File && m_FileTexture)
            return false;   // the image doesn't follow the sun
        if ((capture ? capture->revision : 0u) != m_CaptureRevision)
            return true;
        const dm::double3 dir = sun.GetDirection();
        return dir.x != m_SunDir.x || dir.y != m_SunDir.y || dir.z != m_SunDir.z
            || sun.irradiance != m_SunIrradiance || std::memcmp(&sky, &m_Sky, sizeof(sky)) != 0;
    }

    void EnvironmentLighting::Update(nvrhi::ICommandList* cmd, const engine::DirectionalLight& sun, const render::SkyParameters& sky,
                                     const SkyCapture* capture)
    {
        if (!settings.enabled)
            return;
        if (!m_Used)
            Allocate();

        m_Probe->diffuseScale = settings.diffuseScale;
        m_Probe->specularScale = settings.specularScale;

        if (!m_BrdfDone)
        {
            m_ProbePass->RenderEnvironmentBrdfTexture(cmd);
            m_Probe->environmentBrdf = m_ProbePass->GetEnvironmentBrdfTexture();
            m_BrdfDone = true;
        }

        if (settings.source == EnvironmentSource::File && settings.file != m_LoadedFile)
        {
            m_LoadedFile = settings.file;
            m_FileTexture = nullptr;
            if (!settings.file.empty())
            {
                auto loaded = m_TextureCache->LoadTextureFromFile(settings.file,
                    engine::TextureLoadOptions{ engine::SRGBMode::ForceLinear }, m_Ctx.common.get(), cmd);
                if (loaded && loaded->texture)
                    m_FileTexture = loaded->texture;
                else
                    log::warning("hyper: couldn't load environment %s; using the sky", settings.file.string().c_str());
            }
        }

        if (!NeedsCapture(sun, sky, capture))
            return;

        auto scope = m_Ctx.profiler->Measure(cmd, "environment capture");

        if (settings.source == EnvironmentSource::File && m_FileTexture && m_EquirectPipeline)
        {
            EquirectConstants c{ c_EnvSize, settings.fileIntensity, dm::radians(settings.fileRotationDeg), 0.f };
            auto set = m_Ctx.bindingCache->GetOrCreateBindingSet(nvrhi::BindingSetDesc()
                .addItem(nvrhi::BindingSetItem::PushConstants(0, sizeof(c)))
                .addItem(nvrhi::BindingSetItem::Texture_SRV(0, m_FileTexture))
                .addItem(nvrhi::BindingSetItem::Sampler(0, m_Ctx.common->m_LinearWrapSampler))
                .addItem(nvrhi::BindingSetItem::Texture_UAV(0, m_EnvCube, nvrhi::Format::UNKNOWN,
                    nvrhi::TextureSubresourceSet(0, 1, 0, 6), nvrhi::TextureDimension::Texture2DArray)), m_EquirectLayout);
            cmd->setComputeState(nvrhi::ComputeState().setPipeline(m_EquirectPipeline).addBindingSet(set));
            cmd->setPushConstants(&c, sizeof(c));
            cmd->dispatch((c_EnvSize + 7) / 8, (c_EnvSize + 7) / 8, 6);
        }
        else if (capture)
        {
            capture->render(cmd, m_EnvCube);
        }
        else
        {
            engine::CubemapView view;
            view.SetArrayViewports(int(c_EnvSize), 0);
            view.SetTransform(dm::affine3::identity(), 0.1f, 100.f);
            view.UpdateCache();
            cmd->clearTextureFloat(m_EnvCube, nvrhi::TextureSubresourceSet(0, 1, 0, 6), nvrhi::Color(0.f));
            cmd->clearDepthStencilTexture(m_EnvDepth, nvrhi::AllSubresources, true, 0.f, false, 0);
            m_SkyPass->Render(cmd, view, sun, sky);
        }

        m_ProbePass->GenerateCubemapMips(cmd, m_EnvCube, 0, 0, c_EnvMips - 1);
        m_DiffusePass->RenderDiffuseMap(cmd, m_EnvCube, nvrhi::AllSubresources, m_Probe->diffuseMap, 0, 0);
        for (uint32_t mip = 0; mip < c_SpecularMips; ++mip)
        {
            const float roughness = std::pow(float(mip) / float(c_SpecularMips - 1), 2.f);
            m_ProbePass->RenderSpecularMap(cmd, roughness, m_EnvCube, nvrhi::AllSubresources, m_Probe->specularMap, 0, mip);
        }

        m_SunDir = sun.GetDirection();
        m_SunIrradiance = sun.irradiance;
        m_Sky = sky;
        m_Captured = settings;
        m_CaptureRevision = capture ? capture->revision : 0u;
        m_Dirty = false;
        ++m_Revision;
    }

    std::shared_ptr<engine::LightProbe> EnvironmentLighting::Probe() const
    {
        return settings.enabled && m_BrdfDone ? m_Probe : nullptr;   // m_BrdfDone: allocated and drawn
    }

    void EnvironmentLighting::BuildUI()
    {
        ImGui::Checkbox("environment lighting (IBL)", &settings.enabled);
        if (!settings.enabled)
            return;
        int source = int(settings.source);
        ImGui::RadioButton("from sky", &source, int(EnvironmentSource::Sky));
        ImGui::SameLine();
        ImGui::RadioButton("from HDR file", &source, int(EnvironmentSource::File));
        settings.source = EnvironmentSource(source);
        if (settings.source == EnvironmentSource::File)
        {
            ImGui::TextDisabled("%s", settings.file.empty() ? "(set with --env <file.hdr>)" : settings.file.filename().string().c_str());
            ImGui::SliderFloat("HDR intensity", &settings.fileIntensity, 0.f, 10.f, "%.2f", ImGuiSliderFlags_Logarithmic);
            ImGui::SliderFloat("HDR rotation", &settings.fileRotationDeg, -180.f, 180.f);
        }
        ImGui::SliderFloat("IBL diffuse", &settings.diffuseScale, 0.f, 4.f);
        ImGui::SliderFloat("IBL specular", &settings.specularScale, 0.f, 4.f);
    }
}
