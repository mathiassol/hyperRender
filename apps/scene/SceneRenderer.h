#pragma once

// The scene app's frame: most of hyper's modules assembled for a loaded hyper::Scene. It
// lives here, not in hyper, because a frame belongs to its app; this one is the example of
// assembling the modules (each has its own settings, reachable through the accessors below;
// one left off costs nothing).
//
// On by default: the raster set, i.e. CSM, atmosphere, IBL probe, XeGTAO, SSR, GPU-driven
// drawing and the post stack. Opt in: ray-traced shadows and reflections, DDGI, volumetric
// fog, media volumes / clouds, meshlets. The ray-traced modules need ctx.rayQuery; without
// it they stay off and the raster modules take their place.
//
//   every frame  sun shadows (CSM), atmosphere LUTs, environment lighting (IBL probe)
//   deferred     G-buffer -> XeGTAO -> deferred lighting -> Hi-Z + SSR -> sky / atmosphere
//                -> volumes / clouds -> transparent -> fog -> PostProcess
//   reference    ReferencePathTracer's accumulated image -> PostProcess (no TAA)
//
// The sky is the Atmosphere when it is enabled, else Donut's procedural sky, or the HDR
// image when the environment comes from a file.
//
//     ctx.camera.GetView(view, w, h);
//     renderer.PrepareView(view);
//     renderer.Render(cmd, scene, view, framebuffer);

#include <hyper/hyper.h>
#include <hyper/gpu/GpuDrivenRenderer.h>
#include <hyper/lighting/AmbientOcclusion.h>
#include <hyper/lighting/Atmosphere.h>
#include <hyper/lighting/EnvironmentLighting.h>
#include <hyper/lighting/ScreenSpaceReflections.h>
#include <hyper/lighting/SunShadows.h>
#include <hyper/passes/ComputePasses.h>
#include <hyper/passes/DebugView.h>
#include <hyper/passes/PostProcess.h>
#include <hyper/rt/Ddgi.h>
#include <hyper/rt/ReferencePathTracer.h>
#include <hyper/rt/RtLighting.h>
#include <hyper/rt/RtReflections.h>
#include <hyper/rt/RtShadows.h>
#include <hyper/volume/RaymarchedVolumes.h>
#include <hyper/volume/VolumetricFog.h>

#include <donut/engine/View.h>
#include <donut/render/SkyPass.h>

#include <memory>

namespace donut::render
{
    class DeferredLightingPass;
    class ForwardShadingPass;
    class TransparentDrawStrategy;
    class EnvironmentMapPass;
}
namespace donut::engine { class FramebufferFactory; class Light; }

namespace scene
{
    enum class DebugView
    {
        None,
        AmbientOcclusion,
        Shadows,                // RT shadow visibility
        IndirectDiffuse,        // DDGI: albedo * irradiance
        Reflections,            // SSR colour
        ReflectionConfidence,   // SSR hit weight
        Normals,
        Roughness,
    };

    struct Settings
    {
        bool reference = false;         // ReferencePathTracer: ground truth, accumulated (needs ray queries)
        bool transparency = true;       // blended geometry, drawn forward over the lit frame
        bool sky = true;                // draw a background at all
        donut::render::SkyParameters skyParams;   // Donut's procedural sky (atmosphere off)
        float ambient = 1.f;            // flat sky-coloured ambient; only used while IBL is off
        float sunAzimuthDeg = 30.f;
        float sunElevationDeg = 55.f;
        float sunIrradiance = 3.f;
        // With more lights than Donut's deferred pass takes (16), every point light is lit by
        // the tiled pass instead (many_lights_cs, up to 16,384 of them), its intensity times
        // this, out to where its light falls under pointLightCutoff (for a light with no range
        // of its own). No shadows.
        float pointLightScale = 1.f;
        float pointLightCutoff = 0.16f;
        float pointLightRadius = 0.03f;   // metres: the least size a point light is given (a candle flame)
        DebugView debugView = DebugView::None;
    };

    class SceneRenderer
    {
    public:
        explicit SceneRenderer(hyper::Context& ctx);
        ~SceneRenderer();

        Settings settings;

        // Applies the TAA jitter (none for the reference, which jitters its own samples).
        void PrepareView(donut::engine::PlanarView& view);

        // Renders the scene through PostProcess into target. Before the scene is ready it
        // just clears the target.
        void Render(nvrhi::ICommandList* cmd, hyper::Scene& scene, const donut::engine::PlanarView& view, nvrhi::IFramebuffer* target);

        void ShadersReloaded();

        // Controls for `settings`, the sun and every module; call inside a window.
        void BuildUI(hyper::Scene& scene);

        [[nodiscard]] hyper::PostProcess& Post() { return m_Post; }
        [[nodiscard]] hyper::SunShadows& Shadows() { return *m_Shadows; }
        [[nodiscard]] hyper::Atmosphere& Sky() { return *m_Atmosphere; }
        [[nodiscard]] hyper::EnvironmentLighting& Environment() { return *m_Environment; }
        [[nodiscard]] hyper::AmbientOcclusion& Ao() { return *m_Ao; }
        [[nodiscard]] hyper::ScreenSpaceReflections& Reflections() { return *m_Reflections; }
        // Ray tracing (when ctx.rayQuery): replaces CSM / SSR where enabled.
        [[nodiscard]] hyper::RtShadows& RayShadows() { return *m_RtShadows; }
        [[nodiscard]] hyper::RtReflections& RayReflections() { return *m_RtReflections; }
        [[nodiscard]] hyper::Ddgi& Gi() { return *m_Ddgi; }
        [[nodiscard]] hyper::VolumetricFog& Fog() { return *m_Fog; }
        [[nodiscard]] hyper::GpuDrivenRenderer& GpuDriven() { return *m_GpuDriven; }
        [[nodiscard]] hyper::ReferencePathTracer& Reference() { return *m_Reference; }
        [[nodiscard]] hyper::RaymarchedVolumes& Volumes() { return *m_Volumes; }

    private:
        struct Frame;

        hyper::Context& m_Ctx;
        hyper::PostProcess m_Post;
        hyper::DebugView m_DebugView;
        struct
        {
            nvrhi::ITexture* ao = nullptr;
            nvrhi::ITexture* shadows = nullptr;    // RT shadow visibility
            nvrhi::ITexture* indirect = nullptr;   // DDGI gather
        } m_Last;                                  // this frame's intermediates, for the debug view

        std::unique_ptr<hyper::GBuffer> m_GBuffer;
        dm::uint2 m_Size = dm::uint2::zero();

        std::unique_ptr<donut::render::DeferredLightingPass> m_DeferredPass;
        std::unique_ptr<donut::render::ForwardShadingPass> m_ForwardPass;   // the transparent pass
        std::unique_ptr<donut::render::SkyPass> m_SkyPass;
        std::unique_ptr<donut::render::EnvironmentMapPass> m_EnvBackground;

        std::unique_ptr<hyper::SunShadows> m_Shadows;
        std::unique_ptr<hyper::Atmosphere> m_Atmosphere;
        std::unique_ptr<hyper::EnvironmentLighting> m_Environment;
        std::unique_ptr<hyper::DepthPyramid> m_HiZ;
        std::unique_ptr<hyper::AmbientOcclusion> m_Ao;
        std::unique_ptr<hyper::ScreenSpaceReflections> m_Reflections;
        std::unique_ptr<hyper::RtScene> m_RtScene;
        std::unique_ptr<hyper::RtShadows> m_RtShadows;
        std::unique_ptr<hyper::RtReflections> m_RtReflections;
        std::unique_ptr<hyper::Ddgi> m_Ddgi;
        std::unique_ptr<hyper::RtLighting> m_RtLighting;
        std::unique_ptr<hyper::VolumetricFog> m_Fog;
        std::unique_ptr<hyper::RaymarchedVolumes> m_Volumes;
        std::unique_ptr<hyper::GpuDrivenRenderer> m_GpuDriven;
        std::unique_ptr<hyper::ReferencePathTracer> m_Reference;
        std::unique_ptr<donut::render::TransparentDrawStrategy> m_TransparentStrategy;
        bool m_PassesDirty = true;
        uint32_t m_TexturesSeen = ~0u;   // material textures arrived when the transparent pass last drew

        // The lights Donut's passes take (at most 16), and the point lights the tiled pass does.
        std::vector<std::shared_ptr<donut::engine::Light>> m_PassLights;
        uint32_t m_ManyLights = 0;
        nvrhi::BufferHandle m_ManyLightBuffer, m_ManyLightSpheres;
        std::unique_ptr<hyper::ComputePasses> m_ManyPasses;
        nvrhi::ComputePipelineHandle m_ManyPipeline;
        void SortLights(nvrhi::ICommandList* cmd, hyper::Scene& scene);

        donut::engine::PlanarView m_PreviousView;
        bool m_PreviousValid = false;

        void EnsureTargets(uint32_t width, uint32_t height);
        void CreatePasses();
        void UpdateSun(hyper::Scene& scene, const donut::engine::PlanarView& view);

        // Render's stages, in order.
        void PrepareLighting(nvrhi::ICommandList* cmd, hyper::Scene& scene, Frame& f);
        hyper::PostProcess::Inputs RenderDeferred(nvrhi::ICommandList* cmd, hyper::Scene& scene, Frame& f);
        hyper::PostProcess::Inputs RenderReference(nvrhi::ICommandList* cmd, hyper::Scene& scene, Frame& f);
        void DrawBackground(nvrhi::ICommandList* cmd, hyper::Scene& scene, const Frame& f);
        void DrawTransparent(nvrhi::ICommandList* cmd, hyper::Scene& scene, Frame& f);
        void ApplyMedia(nvrhi::ICommandList* cmd, hyper::Scene& scene, const Frame& f);
        void ApplyFog(nvrhi::ICommandList* cmd, hyper::Scene& scene, const Frame& f);
        void RenderDebugView(nvrhi::ICommandList* cmd, nvrhi::IFramebuffer* target);
    };
}
