#pragma once

#include <hyper/platform.h>

#include <hyper/Readback.h>
#include <hyper/gpu/GpuScene.h>
#include <hyper/lighting/DepthPyramid.h>

#include <nvrhi/nvrhi.h>

#include <memory>
#include <unordered_map>

namespace donut::engine { class PlanarView; }
namespace donut::render { class GBufferFillPass; class InstancedOpaqueDrawStrategy; }

namespace hyper
{
    struct Context;
    class Scene;
    class GBuffer;

    struct GpuDrivenSettings
    {
        bool enabled = true;
        bool occlusionCulling = true;
        bool frustumCulling = true;
        bool meshlets = false;           // opt in: mesh shading where supported (task shader culls
                                         // meshlets); pays off on dense, partly hidden meshes
        bool clusterLod = false;         // opt in (needs meshlets): a LOD DAG of clusters, cut per view;
                                         // building it takes a moment at load
        float lodErrorPixels = 1.f;      // screen-space error the cut allows
    };

    // GPU-driven rendering: the draw list lives on the GPU
    // (hyper::GpuScene), a compute pass culls it per view and writes indirect
    // arguments, and one drawIndexedIndirectCount per bucket draws the survivors
    // with vertex pulling and bindless materials. The camera uses two-phase Hi-Z
    // occlusion culling; shadow cascades are frustum culled.
    //
    // With mesh shader support, static geometry is drawn by meshlets:
    // the cull emits task work, a task shader culls each meshlet (frustum, normal
    // cone, Hi-Z in phase 2) and a mesh shader draws the rest. Skinned geometry
    // and shadow cascades use the indexed path.
    //
    // Covers opaque and alpha-tested geometry; blended and transmissive stay on
    // Donut's forward pass. Needs a bindless scene (Vulkan / D3D12); without one, or
    // with settings.enabled off, RenderGBuffer draws with Donut's CPU-driven pass, and
    // SunShadows with Donut's depth pass, so apps draw one way either way.
    class GpuDrivenRenderer
    {
    public:
        explicit GpuDrivenRenderer(Context& ctx);
        ~GpuDrivenRenderer();

        GpuDrivenSettings settings;

        // Once per frame after scene.RefreshBuffers(cmd): updates the GPU scene. False if
        // GPU-driven drawing isn't running (off, not bindless, or the GPU scene still building).
        bool BeginFrame(nvrhi::ICommandList* cmd, Scene& scene);

        // Fills Donut's G-buffer (the depth must be cleared) with the two-phase cull, or
        // with Donut's pass when GPU-driven drawing is off or can't run. Draws nothing in
        // the frame or two the GPU scene takes to build.
        void RenderGBuffer(nvrhi::ICommandList* cmd, const donut::engine::PlanarView& view,
                           const donut::engine::PlanarView& previous, Scene& scene, GBuffer& gbuffer);

        // Depth only into framebuffer, frustum culled: a shadow cascade (0..3). camera: the
        // view the LOD cut is chosen for, so shadows match what the camera draws.
        void RenderDepth(nvrhi::ICommandList* cmd, const donut::engine::PlanarView& view, const donut::engine::PlanarView& camera,
                         Scene& scene, nvrhi::IFramebuffer* framebuffer, uint32_t cascade, int depthBias, float slopeBias);

        struct Stats
        {
            uint32_t draws = 0;          // instance x geometry records
            uint32_t phase1 = 0;         // drawn, visible last frame
            uint32_t phase2 = 0;         // drawn after the occlusion test
            uint32_t frustumCulled = 0;
            uint32_t occluded = 0;
            uint32_t triangles = 0;      // in the scene
            uint32_t meshlets = 0;       // in the scene
            uint32_t meshletsTested = 0; // by the task shader (both phases)
            uint32_t meshletsDrawn = 0;
            uint32_t trianglesDrawn = 0; // by meshlets (both phases)
            uint32_t lodLevels = 0;      // 0: no cluster LOD
        };
        // From three frames ago (read back without stalling).
        [[nodiscard]] const Stats& LastStats() const { return m_Stats; }
        [[nodiscard]] GpuScene& Geometry() { return m_Scene; }

        void CreatePipelines();
        void BuildUI();

    private:
        Context& m_Ctx;
        GpuScene m_Scene;
        DepthPyramid m_HiZ;
        nvrhi::BindingLayoutHandle m_CullLayout, m_DrawLayout;
        nvrhi::IBindingLayout* m_BindlessLayout = nullptr;
        nvrhi::ComputePipelineHandle m_Cull;
        nvrhi::ShaderHandle m_Vs, m_Ps, m_PsAlpha, m_VsDepth, m_PsDepthAlpha, m_Task, m_Mesh, m_MeshDepth;
        std::unordered_map<size_t, nvrhi::GraphicsPipelineHandle> m_Pipelines;
        std::unordered_map<size_t, nvrhi::MeshletPipelineHandle> m_MeshPipelines;
        nvrhi::BindingLayoutHandle m_MeshLayout;
        nvrhi::BufferHandle m_MeshCullConstants, m_Tasks, m_MeshArgs;
        bool m_MeshShading = false;
        nvrhi::BufferHandle m_CullConstants, m_DrawConstants;
        nvrhi::BufferHandle m_Visibility, m_Args, m_Counts;
        Readback m_StatsReadback;
        uint32_t m_MaxDraws = 0;
        uint32_t m_Frame = 0;
        uint32_t m_Revision = ~0u;
        Stats m_Stats;
        bool m_Ready = false;
        std::unique_ptr<donut::render::GBufferFillPass> m_GBufferPass;    // the fallback
        uint32_t m_TexturesSeen = ~0u;   // material textures arrived when the fallback last drew
        std::unique_ptr<donut::render::InstancedOpaqueDrawStrategy> m_Opaque;

        enum class Mode { GBuffer, Depth };

        void EnsureBuffers(nvrhi::ICommandList* cmd);
        nvrhi::IGraphicsPipeline* Pipeline(nvrhi::IFramebuffer* fb, Mode mode, uint32_t variant, bool reverseZ,
                                           int depthBias, float slopeBias);
        nvrhi::IMeshletPipeline* MeshPipeline(nvrhi::IFramebuffer* fb, Mode mode, uint32_t variant, bool reverseZ,
                                              int depthBias, float slopeBias);
        [[nodiscard]] bool UseMeshlets() const { return m_MeshShading && settings.meshlets && m_Task && m_Scene.MeshletCount() > 0; }
        void Cull(nvrhi::ICommandList* cmd, const donut::engine::PlanarView& view, Scene& scene, uint32_t pass,
                  uint32_t phase, bool occlusion);
        void Draw(nvrhi::ICommandList* cmd, const donut::engine::PlanarView& view, const donut::engine::PlanarView& previous,
                  const donut::engine::PlanarView& camera, Scene& scene, nvrhi::IFramebuffer* fb, uint32_t pass,
                  Mode mode, int depthBias, float slopeBias);
        void ReadStats(nvrhi::ICommandList* cmd);
    };
}
