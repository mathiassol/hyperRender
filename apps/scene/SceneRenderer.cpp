#include "SceneRenderer.h"

#include <donut/core/log.h>
#include <donut/engine/FramebufferFactory.h>
#include <donut/engine/SceneGraph.h>
#include <donut/render/DeferredLightingPass.h>
#include <donut/render/DrawStrategy.h>
#include <donut/render/EnvironmentMapPass.h>
#include <donut/render/ForwardShadingPass.h>
#include <donut/render/GeometryPasses.h>
// (Donut's shader structs, for C++: its math types in scope, as Donut includes them.)
using namespace donut::math;
#include <donut/shaders/light_cb.h>
#include <donut/shaders/light_types.h>

#include <imgui.h>

#include <cmath>

using namespace donut;
using namespace hyper;

namespace scene
{
    // How many of the scene's material textures have arrived. Donut's raster passes cache
    // each material's binding set the first time they draw it, textures and all: one drawn
    // while its textures still stream keeps the fallbacks for good (a blended decal drew
    // as a white box), so the caches are dropped whenever this changes.
    static uint32_t ArrivedTextures(Scene& scene)
    {
        uint32_t n = 0;
        for (const auto& m : scene.Graph()->GetMaterials())
            for (const auto* t : { &m->baseOrDiffuseTexture, &m->metalRoughOrSpecularTexture, &m->normalTexture,
                                   &m->emissiveTexture, &m->occlusionTexture, &m->transmissionTexture, &m->opacityTexture })
                n += *t && (*t)->texture ? 1u : 0u;
        return n;
    }

    SceneRenderer::SceneRenderer(Context& ctx) : m_Ctx(ctx), m_Post(ctx), m_DebugView(ctx)
    {
        m_TransparentStrategy = std::make_unique<render::TransparentDrawStrategy>();

        m_Shadows = std::make_unique<SunShadows>(ctx);
        m_Atmosphere = std::make_unique<Atmosphere>(ctx);
        m_Environment = std::make_unique<EnvironmentLighting>(ctx);
        m_HiZ = std::make_unique<DepthPyramid>(ctx);
        m_Ao = std::make_unique<AmbientOcclusion>(ctx);
        m_Reflections = std::make_unique<ScreenSpaceReflections>(ctx);
        m_RtScene = std::make_unique<RtScene>(ctx);
        m_RtShadows = std::make_unique<RtShadows>(ctx);
        m_RtReflections = std::make_unique<RtReflections>(ctx);
        m_Ddgi = std::make_unique<Ddgi>(ctx);
        m_RtLighting = std::make_unique<RtLighting>(ctx);
        m_Fog = std::make_unique<VolumetricFog>(ctx);
        m_Volumes = std::make_unique<RaymarchedVolumes>(ctx);
        m_GpuDriven = std::make_unique<GpuDrivenRenderer>(ctx);
        m_Reference = std::make_unique<ReferencePathTracer>(ctx);
        m_ManyPasses = std::make_unique<ComputePasses>(ctx,
            std::vector<nvrhi::BindingLayoutItem>{ nvrhi::BindingLayoutItem::StructuredBuffer_SRV(20), nvrhi::BindingLayoutItem::StructuredBuffer_SRV(21) },
            [this](nvrhi::BindingSetDesc& d)
            {
                d.addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(20, m_ManyLightBuffer))
                    .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(21, m_ManyLightSpheres));
            },
            nullptr);
    }

    SceneRenderer::~SceneRenderer() = default;

    void SceneRenderer::ShadersReloaded()
    {
        m_Post.ShadersReloaded();
        m_DebugView.CreatePipelines();
        m_Shadows->CreatePipelines();
        m_Atmosphere->CreatePipelines();
        m_Environment->CreatePipelines();
        m_HiZ->CreatePipelines();
        m_Ao->CreatePipelines();
        m_Reflections->CreatePipelines();
        m_RtShadows->CreatePipelines();
        m_RtReflections->CreatePipelines();
        m_Ddgi->CreatePipelines();
        m_Fog->CreatePipelines();
        m_Volumes->CreatePipelines();
        m_GpuDriven->CreatePipelines();
        m_Reference->CreatePipelines();
        m_ManyPasses->Pipeline(m_ManyPipeline, "app/many_lights_cs", "main", nullptr);
        m_PassesDirty = true;
    }

    // Donut's deferred and forward passes take 16 lights. With more, they get the others (the
    // sun, spots) and every point light goes to the tiled pass, its constants as Donut fills
    // them. A light with no range gets one: where its light falls under the cutoff.
    void SceneRenderer::SortLights(nvrhi::ICommandList* cmd, Scene& scene)
    {
        const auto& lights = scene.Graph()->GetLights();
        m_PassLights.clear();
        m_ManyLights = 0;
        if (lights.size() <= 16)
        {
            m_PassLights.assign(lights.begin(), lights.end());
            return;
        }
        std::vector<LightConstants> constants;
        std::vector<dm::float4> spheres;
        for (const auto& light : lights)
        {
            if (light->GetLightType() != LightType_Point)
            {
                if (m_PassLights.size() < 16)
                    m_PassLights.push_back(light);
                continue;
            }
            if (settings.pointLightScale <= 0.f)
                continue;
            LightConstants c{};
            light->FillLightConstants(c);
            c.intensity *= settings.pointLightScale;
            const float brightest = c.intensity * std::max(c.color.x, std::max(c.color.y, c.color.z));
            if (brightest <= 0.f)
                continue;
            float range = c.angularSizeOrInvRange > 0.f ? 1.f / c.angularSizeOrInvRange : 0.f;
            if (range <= 0.f)
            {
                range = std::sqrt(brightest / std::max(settings.pointLightCutoff, 1e-6f));
                c.angularSizeOrInvRange = 1.f / range;
            }
            // A light with no size is infinitely bright at a wall it touches: give it a small
            // one (pointLightRadius).
            c.radius = std::max(c.radius, settings.pointLightRadius);
            c.shadowChannel = dm::int4(-1);
            c.shadowCascades = dm::int4(-1);
            c.perObjectShadows = dm::int4(-1);
            constants.push_back(c);
            spheres.push_back(dm::float4(c.position, range));
        }
        m_ManyLights = uint32_t(constants.size());
        if (!m_ManyLights)
            return;
        if (!m_ManyLightBuffer || m_ManyLightBuffer->getDesc().byteSize < constants.size() * sizeof(LightConstants))
        {
            m_ManyLightBuffer = m_Ctx.device->createBuffer(nvrhi::BufferDesc().setDebugName("scene/many-lights")
                .setByteSize(constants.size() * sizeof(LightConstants)).setStructStride(sizeof(LightConstants))
                .setInitialState(nvrhi::ResourceStates::ShaderResource).setKeepInitialState(true));
            m_ManyLightSpheres = m_Ctx.device->createBuffer(nvrhi::BufferDesc().setDebugName("scene/many-light-spheres")
                .setByteSize(spheres.size() * sizeof(dm::float4)).setStructStride(sizeof(dm::float4))
                .setInitialState(nvrhi::ResourceStates::ShaderResource).setKeepInitialState(true));
            m_ManyPasses->ClearSets();
        }
        cmd->writeBuffer(m_ManyLightBuffer, constants.data(), constants.size() * sizeof(LightConstants));
        cmd->writeBuffer(m_ManyLightSpheres, spheres.data(), spheres.size() * sizeof(dm::float4));
    }

    void SceneRenderer::PrepareView(engine::PlanarView& view)
    {
        if (settings.reference)
        {
            // The reference jitters its own samples: no TAA jitter.
            view.SetPixelOffset(dm::float2::zero());
            view.UpdateCache();
        }
        else
            m_Post.PrepareView(view);
    }

    void SceneRenderer::EnsureTargets(uint32_t width, uint32_t height)
    {
        const dm::uint2 size(width, height);
        if (m_GBuffer && all(size == m_Size))
            return;
        m_Size = size;
        m_GBuffer = std::make_unique<GBuffer>();
        m_GBuffer->Init(m_Ctx.device, size, 1, true, true);
        m_Ctx.bindingCache->Clear();
        m_ManyPasses->ClearSets();
        m_PreviousValid = false;
        m_PassesDirty = true;
    }

    void SceneRenderer::CreatePasses()
    {
        // Pipelines depend on the framebuffer formats, so the passes follow the targets
        // (and shader reloads).
        render::ForwardShadingPass::CreateParameters forwardParams;
        m_ForwardPass = std::make_unique<render::ForwardShadingPass>(m_Ctx.device, m_Ctx.common);
        m_ForwardPass->Init(*m_Ctx.shaders, forwardParams);

        m_DeferredPass = std::make_unique<render::DeferredLightingPass>(m_Ctx.device, m_Ctx.common);
        m_DeferredPass->Init(m_Ctx.shaders);

        // SkyPass picks its depth test and quad depth from this view's projection,
        // so it must be a reverse-Z one like the camera's.
        engine::PlanarView view;
        m_Ctx.camera.GetView(view, m_Size.x, m_Size.y);
        m_SkyPass = std::make_unique<render::SkyPass>(m_Ctx.device, m_Ctx.shaders, m_Ctx.common, m_GBuffer->HdrFramebuffer, view);
        // The background for an HDR-file environment is made when first drawn: the
        // environment's cube exists once it has captured.
        m_EnvBackground.reset();
        m_PassesDirty = false;
    }

    void SceneRenderer::UpdateSun(Scene& scene, const engine::PlanarView& view)
    {
        const double az = dm::radians(double(settings.sunAzimuthDeg));
        const double el = dm::radians(double(settings.sunElevationDeg));
        // Direction the light travels: from the sun, down into the scene.
        const dm::double3 dir = -dm::double3(std::cos(el) * std::sin(az), std::sin(el), std::cos(el) * std::cos(az));
        scene.Sun().irradiance = settings.sunIrradiance;
        // Sunlight arrives through the atmosphere: reddened near the horizon, gone below it.
        scene.Sun().color = m_Atmosphere->settings.enabled
            ? m_Atmosphere->SunTransmittance(dm::float3(-dir), view.GetViewOrigin().y) : dm::float3(1.f);
        scene.SetSunDirection(dir);
    }

    // What one frame runs, decided once from the settings and from what the device
    // and the scene allow; the stages below read it rather than asking again.
    struct SceneRenderer::Frame
    {
        const engine::PlanarView* view = nullptr;
        const engine::PlanarView* previous = nullptr;   // == view on the first frame
        bool rt = false;              // RtScene is current (ray queries, a module wants it)
        bool rtShadows = false;       // replace the shadow cascades
        bool rtReflections = false;   // replace the SSR trace
        bool gi = false;              // the DDGI volume is live
        bool gpuDriven = false;       // G-buffer and cascades drawn by GpuDrivenRenderer
        bool reference = false;       // the frame is ReferencePathTracer's image
        bool atmosphere = false;
        std::vector<std::shared_ptr<engine::LightProbe>> probes;   // the IBL probe, if any
        dm::float3 ambientTop, ambientBottom;                      // flat ambient, when there is no probe
        render::ForwardShadingPass::Context forward;               // lights for transparent drawing

        [[nodiscard]] const engine::LightProbe* Probe() const { return probes.empty() ? nullptr : probes[0].get(); }
    };

    void SceneRenderer::Render(nvrhi::ICommandList* cmd, Scene& scene, const engine::PlanarView& view, nvrhi::IFramebuffer* target)
    {
        auto scope = m_Ctx.profiler->Measure(cmd, "scene");

        if (!scene.IsReady())
        {
            cmd->clearTextureFloat(target->getDesc().colorAttachments[0].texture, nvrhi::AllSubresources, nvrhi::Color(0.02f));
            return;
        }

        const auto& fbInfo = target->getFramebufferInfo();
        EnsureTargets(fbInfo.width, fbInfo.height);
        if (m_PassesDirty)
            CreatePasses();
        UpdateSun(scene, view);
        scene.RefreshBuffers(cmd);

        Frame f;
        f.view = &view;
        f.previous = m_PreviousValid ? &m_PreviousView : &view;
        PrepareLighting(cmd, scene, f);

        const PostProcess::Inputs post = f.reference ? RenderReference(cmd, scene, f) : RenderDeferred(cmd, scene, f);

        // The reference's accumulated jitter is its own anti-aliasing.
        const AntiAliasing userAa = m_Post.settings.aa;
        if (f.reference)
            m_Post.settings.aa = AntiAliasing::None;
        m_Post.Render(cmd, post, view, target);
        m_Post.settings.aa = userAa;

        if (!f.reference)
            RenderDebugView(cmd, target);
        m_PreviousView = view;
        m_PreviousValid = true;
    }

    // Everything the frame lights with: which RT modules run, shadows, sky, probe, DDGI, the transparent pass's lights.
    void SceneRenderer::PrepareLighting(nvrhi::ICommandList* cmd, Scene& scene, Frame& f)
    {
        const bool rtWanted = m_Ctx.rayQuery
            && (settings.reference || m_RtShadows->settings.enabled || m_RtReflections->settings.enabled || m_Ddgi->settings.enabled);
        f.rt = rtWanted && m_RtScene->Update(cmd, scene);
        f.reference = f.rt && settings.reference;
        f.rtShadows = f.rt && !f.reference && m_RtShadows->settings.enabled;
        f.rtReflections = f.rt && !f.reference && m_RtReflections->settings.enabled;
        f.gpuDriven = !f.reference && m_GpuDriven->BeginFrame(cmd, scene);

        if (f.reference || f.rtShadows)
            scene.Sun().shadowMap = nullptr;   // traced instead
        else
            m_Shadows->Render(cmd, scene, *f.view, f.gpuDriven ? m_GpuDriven.get() : nullptr);
        f.atmosphere = m_Atmosphere->settings.enabled;
        if (f.atmosphere)
            m_Atmosphere->Update(cmd, *f.view, scene.Sun());
        const SkyCapture sky{ [this](nvrhi::ICommandList* c, nvrhi::ITexture* cube) { m_Atmosphere->Capture(c, cube); },
                              m_Atmosphere->Revision() };
        m_Environment->Update(cmd, scene.Sun(), settings.skyParams, f.atmosphere ? &sky : nullptr);

        // The probe replaces the flat ambient; Donut would add both.
        if (auto probe = m_Environment->Probe())
            f.probes.push_back(probe);
        const float ambient = f.probes.empty() ? settings.ambient : 0.f;
        f.ambientTop = ambient * settings.skyParams.skyColor * settings.skyParams.brightness;
        f.ambientBottom = ambient * settings.skyParams.groundColor * settings.skyParams.brightness;

        // The light every RT pass shades hits with (with the DDGI volume), then the volume itself.
        if (f.rt)
        {
            const engine::LightProbe* probe = f.Probe();
            const RtProbeVolume* volume = f.reference ? nullptr : m_Ddgi->Prepare(cmd, scene);
            m_RtLighting->Prepare(cmd, scene, probe ? m_Environment->EnvironmentMap() : nullptr, probe,
                                  probe ? probe->diffuseScale : 0.f, volume);
            if (volume)
            {
                m_Ddgi->Update(cmd, scene, *m_RtScene, *m_RtLighting);
                f.gi = true;
            }
        }

        SortLights(cmd, scene);
        if (!f.reference && settings.transparency)
            m_ForwardPass->PrepareLights(f.forward, cmd, m_PassLights, f.ambientTop, f.ambientBottom, f.probes);
    }

    PostProcess::Inputs SceneRenderer::RenderDeferred(nvrhi::ICommandList* cmd, Scene& scene, Frame& f)
    {
        const engine::PlanarView& view = *f.view;
        m_GBuffer->Clear(cmd);
        {
            auto t = m_Ctx.profiler->Measure(cmd, "gbuffer");
            // Blended / transmissive geometry isn't in it; the transparent pass draws it.
            m_GpuDriven->RenderGBuffer(cmd, view, *f.previous, scene, *m_GBuffer);
        }

        m_Last.ao = m_Ao->Render(cmd, view, m_GBuffer->Depth, m_GBuffer->GBufferNormals, m_GBuffer->GBufferDiffuse);
        m_Last.shadows = f.rtShadows
            ? m_RtShadows->Render(cmd, view, scene, *m_RtScene, m_GBuffer->Depth, m_GBuffer->GBufferNormals) : nullptr;
        nvrhi::ITexture* indirect = f.gi ? m_Ddgi->Gather(cmd, view, *m_GBuffer, *m_RtLighting) : nullptr;
        m_Last.indirect = indirect;
        {
            auto t = m_Ctx.profiler->Measure(cmd, "lighting");
            render::DeferredLightingPass::Inputs inputs;
            inputs.SetGBuffer(*m_GBuffer);
            inputs.ambientOcclusion = m_Last.ao;
            inputs.ambientColorTop = f.ambientTop;
            inputs.ambientColorBottom = f.ambientBottom;
            inputs.lights = &m_PassLights;
            // An indirect texture holds all indirect light, the probe's specular included.
            static const std::vector<std::shared_ptr<engine::LightProbe>> noProbes;
            inputs.lightProbes = indirect ? &noProbes : &f.probes;
            inputs.indirectDiffuse = indirect;
            inputs.output = m_GBuffer->HdrColor;
            // Donut's sun reads its shadow from channel 0 of this texture, for this pass only.
            inputs.shadowChannels = m_Last.shadows;
            scene.Sun().shadowChannel = m_Last.shadows ? 0 : -1;
            m_DeferredPass->Render(cmd, view, inputs);
            scene.Sun().shadowChannel = -1;
        }
        if (m_ManyLights && !m_ManyPipeline)
            m_ManyPasses->Pipeline(m_ManyPipeline, "app/many_lights_cs", "main", nullptr);
        if (m_ManyLights && m_ManyPipeline)
        {
            auto t = m_Ctx.profiler->Measure(cmd, "many lights");
            m_ManyPasses->BeginFrame(cmd, view, *f.previous);
            const uint32_t push[4] = { m_ManyLights, 0, 0, 0 };
            m_ManyPasses->Dispatch(cmd, m_ManyPipeline,
                { m_GBuffer->Depth.Get(), m_GBuffer->GBufferDiffuse.Get(), m_GBuffer->GBufferSpecular.Get(), m_GBuffer->GBufferNormals.Get(),
                  m_GBuffer->GBufferEmissive.Get() },
                { m_GBuffer->HdrColor.Get() }, m_Size.x, m_Size.y, push, sizeof(push), 16);
        }
        DrawBackground(cmd, scene, f);

        if (f.rtReflections)
        {
            if (nvrhi::ITexture* traced = m_RtReflections->Trace(cmd, view, scene, *m_RtScene, *m_RtLighting,
                    m_GBuffer->Depth, m_GBuffer->GBufferNormals))
                m_Reflections->Render(cmd, view, *m_GBuffer, nullptr, f.Probe(), m_Ctx.camera.zNear, traced, indirect);
        }
        else if (m_Reflections->settings.enabled)
        {
            m_HiZ->Build(cmd, m_GBuffer->Depth);
            m_Reflections->Render(cmd, view, *m_GBuffer, m_HiZ.get(), f.Probe(), m_Ctx.camera.zNear, nullptr, indirect);
        }

        ApplyMedia(cmd, scene, f);
        DrawTransparent(cmd, scene, f);
        ApplyFog(cmd, scene, f);

        PostProcess::Inputs post;
        post.color = m_GBuffer->HdrColor;
        post.depth = m_GBuffer->Depth;
        post.motionVectors = m_GBuffer->MotionVectors;
        post.motionVectorStencilMask = GBuffer::MotionVectorStencilBit;
        return post;
    }

    // Ground truth: the path tracer's running mean, straight to the post stack.
    PostProcess::Inputs SceneRenderer::RenderReference(nvrhi::ICommandList* cmd, Scene& scene, Frame& f)
    {
        PostProcess::Inputs post;
        post.color = m_Reference->Render(cmd, *f.view, scene, *m_RtScene, *m_RtLighting, m_Size.x, m_Size.y, m_Environment->Revision());
        if (!post.color)
        {
            cmd->clearTextureFloat(m_GBuffer->HdrColor, nvrhi::AllSubresources, nvrhi::Color(0.f));
            post.color = m_GBuffer->HdrColor;
        }
        return post;
    }

    // The atmosphere when it is on, else Donut's procedural sky, or the HDR image
    // when the environment comes from a file.
    void SceneRenderer::DrawBackground(nvrhi::ICommandList* cmd, Scene& scene, const Frame& f)
    {
        if (m_Environment->UsesFile())
        {
            auto t = m_Ctx.profiler->Measure(cmd, "sky");
            if (!m_EnvBackground)
                m_EnvBackground = std::make_unique<render::EnvironmentMapPass>(m_Ctx.device, m_Ctx.shaders, m_Ctx.common,
                    m_GBuffer->HdrFramebuffer, *f.view, m_Environment->EnvironmentMap());
            m_EnvBackground->Render(cmd, *f.view);
        }
        else if (settings.sky && f.atmosphere)
            m_Atmosphere->RenderSky(cmd, m_GBuffer->HdrFramebuffer->GetFramebuffer(*f.view));
        else if (settings.sky)
        {
            auto t = m_Ctx.profiler->Measure(cmd, "sky");
            m_SkyPass->Render(cmd, *f.view, scene.Sun(), settings.skyParams);
        }
    }

    void SceneRenderer::DrawTransparent(nvrhi::ICommandList* cmd, Scene& scene, Frame& f)
    {
        if (!settings.transparency)
            return;
        auto t = m_Ctx.profiler->Measure(cmd, "transparent");
        if (const uint32_t arrived = ArrivedTextures(scene); arrived != m_TexturesSeen)
        {
            m_ForwardPass->ResetBindingCache();
            m_TexturesSeen = arrived;
        }
        render::RenderCompositeView(cmd, f.view, f.previous, *m_GBuffer->HdrFramebuffer, scene.Graph()->GetRootNode(),
            *m_TransparentStrategy, *m_ForwardPass, f.forward, "Transparent");
    }

    // Aerial perspective over the lit opaque image, then media volumes and clouds.
    void SceneRenderer::ApplyMedia(nvrhi::ICommandList* cmd, Scene& scene, const Frame& f)
    {
        if (f.atmosphere)
            m_Atmosphere->Apply(cmd, m_GBuffer->HdrColor, m_GBuffer->Depth);
        if (m_Volumes->HasWork())
        {
            RaymarchedVolumes::Inputs v;
            v.color = m_GBuffer->HdrColor;
            v.depth = m_GBuffer->Depth;
            v.sunDirection = dm::float3(scene.Sun().GetDirection());
            v.sunRadiance = scene.Sun().color * scene.Sun().irradiance;
            v.sunIrradiance = scene.Sun().irradiance;
            v.probe = f.Probe();
            v.aerialVolume = f.atmosphere ? m_Atmosphere->AerialVolume() : nullptr;
            v.aerialMaxKm = m_Atmosphere->AerialMaxKm();
            v.aerialUnitKm = m_Atmosphere->AerialUnitKm();
            m_Volumes->Render(cmd, *f.view, *f.previous, v);
        }
    }

    void SceneRenderer::ApplyFog(nvrhi::ICommandList* cmd, Scene& scene, const Frame& f)
    {
        VolumetricFog::Inputs fog;
        fog.color = m_GBuffer->HdrColor;
        fog.depth = m_GBuffer->Depth;
        fog.probe = f.Probe();
        fog.rt = f.rt ? m_RtScene.get() : nullptr;
        fog.lighting = m_RtLighting.get();
        m_Fog->Render(cmd, *f.view, *f.previous, scene, fog);
    }

    void SceneRenderer::RenderDebugView(nvrhi::ICommandList* cmd, nvrhi::IFramebuffer* target)
    {
        if (settings.debugView == DebugView::None)
            return;
        nvrhi::ITexture* black = m_Ctx.common->m_BlackTexture;
        nvrhi::ITexture* white = m_Ctx.common->m_WhiteTexture;
        nvrhi::ITexture* reflections = m_Reflections->Output() ? m_Reflections->Output() : black;
        switch (settings.debugView)
        {
        case DebugView::AmbientOcclusion:
            m_DebugView.Render(cmd, m_Ao->settings.enabled ? m_Last.ao : white, target, DebugChannel::Red);
            break;
        case DebugView::Shadows:
            m_DebugView.Render(cmd, m_Last.shadows ? m_Last.shadows : white, target, DebugChannel::Red);
            break;
        case DebugView::IndirectDiffuse:
            m_DebugView.Render(cmd, m_Last.indirect ? m_Last.indirect : black, target, DebugChannel::Color);
            break;
        case DebugView::Reflections:
            m_DebugView.Render(cmd, reflections, target, DebugChannel::Color, 2.f);
            break;
        case DebugView::ReflectionConfidence:
            m_DebugView.Render(cmd, reflections, target, DebugChannel::Alpha);
            break;
        case DebugView::Normals:
            m_DebugView.Render(cmd, m_GBuffer->GBufferNormals, target, DebugChannel::Normal);
            break;
        case DebugView::Roughness:
            m_DebugView.Render(cmd, m_GBuffer->GBufferNormals, target, DebugChannel::Alpha);
            break;
        default:
            break;
        }
    }

    void SceneRenderer::BuildUI(Scene& scene)
    {
        Settings& s = settings;

        if (scene.GetState() == Scene::State::Loading)
            ImGui::Text("loading...");
        else if (scene.TextureProgress() < 1.f)
            ImGui::ProgressBar(scene.TextureProgress(), ImVec2(-1.f, 0.f), "textures");

        if (m_Ctx.rayQuery)
        {
            ImGui::Checkbox("reference path tracer", &s.reference);
            if (s.reference && ImGui::TreeNodeEx("reference", ImGuiTreeNodeFlags_DefaultOpen))
            {
                m_Reference->BuildUI();
                ImGui::TreePop();
            }
        }
        ImGui::Checkbox("transparency", &s.transparency);
        if (!s.reference)
        {
            if (ImGui::TreeNode("GPU-driven drawing"))
            {
                m_GpuDriven->BuildUI();
                ImGui::TreePop();
            }
            const char* views[] = { "none", "ambient occlusion", "RT shadows", "GI (DDGI)", "reflection colour", "reflection confidence", "normals", "roughness" };
            int v = int(s.debugView);
            if (ImGui::Combo("debug view", &v, views, IM_ARRAYSIZE(views)))
                s.debugView = DebugView(v);
        }

        if (ImGui::CollapsingHeader("sun & sky"))
        {
            ImGui::PushID("atmosphere");
            m_Atmosphere->BuildUI();
            ImGui::PopID();
            ImGui::Separator();
            ImGui::SliderFloat("sun azimuth", &s.sunAzimuthDeg, -180.f, 180.f);
            ImGui::SliderFloat("sun elevation", &s.sunElevationDeg, -10.f, 90.f);
            ImGui::SliderFloat("sun irradiance", &s.sunIrradiance, 0.f, 20.f);
            if (m_ManyLights || s.pointLightScale != 1.f)
                ImGui::SliderFloat("point lights (tiled)", &s.pointLightScale, 0.f, 4.f);
            ImGui::Checkbox("draw sky", &s.sky);
            if (!m_Atmosphere->settings.enabled)
            {
                ImGui::SliderFloat("sky brightness", &s.skyParams.brightness, 0.f, 1.f);
                ImGui::ColorEdit3("sky colour", &s.skyParams.skyColor.x);
                ImGui::ColorEdit3("horizon colour", &s.skyParams.horizonColor.x);
                ImGui::ColorEdit3("ground colour", &s.skyParams.groundColor.x);
            }
            if (m_Environment->Probe())
                ImGui::TextDisabled("flat ambient is replaced by IBL");
            else
                ImGui::SliderFloat("ambient", &s.ambient, 0.f, 4.f);
        }
        if (ImGui::CollapsingHeader("lighting", ImGuiTreeNodeFlags_DefaultOpen))
        {
            ImGui::PushID("shadows");
            if (m_Ctx.rayQuery)
            {
                m_RtShadows->BuildUI();
                if (m_RtShadows->settings.enabled)
                    ImGui::TextDisabled("(replaces the shadow maps below)");
            }
            m_Shadows->BuildUI();
            ImGui::PopID();
            ImGui::Separator();
            ImGui::PushID("ibl");
            m_Environment->BuildUI();
            ImGui::PopID();
            ImGui::Separator();
            ImGui::PushID("ao");
            m_Ao->BuildUI();
            ImGui::PopID();
            ImGui::Separator();
            ImGui::PushID("ssr");
            if (m_Ctx.rayQuery)
                m_RtReflections->BuildUI();
            m_Reflections->BuildUI();
            ImGui::PopID();
            if (m_Ctx.rayQuery)
            {
                ImGui::Separator();
                ImGui::PushID("gi");
                m_Ddgi->BuildUI();
                ImGui::PopID();
            }
        }
        if (ImGui::CollapsingHeader("volumetrics"))
        {
            ImGui::PushID("fog");
            m_Fog->BuildUI();
            ImGui::PopID();
            ImGui::Separator();
            ImGui::PushID("volumes");
            m_Volumes->BuildUI();
            ImGui::PopID();
        }
        if (ImGui::CollapsingHeader("post"))
            m_Post.BuildUI();
    }
}
