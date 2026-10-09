#include <hyper/gpu/GpuDrivenRenderer.h>

#include <hyper/App.h>
#include <hyper/scene/GBuffer.h>
#include <hyper/scene/Scene.h>

#include <donut/engine/FramebufferFactory.h>
#include <donut/engine/SceneGraph.h>
#include <donut/engine/View.h>
#include <donut/render/DrawStrategy.h>
#include <donut/render/GBufferFillPass.h>
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
        // Mirrors CullConstants in cull_cs.hlsl.
        struct CullConstants
        {
            float4x4 viewProj;
            float4 planes[6];
            uint32_t drawCount, phase, argsBase, countBase;
            uint32_t maxDraws, useOcclusion, hizSize[2];
            uint32_t hizMips, reverseZ, useMeshlets, taskCapacity;
        };
        struct MeshCullConstants
        {
            float4x4 viewProj;
            float4 planes[6];
            float3 cameraPos; uint32_t phase;
            uint32_t hizSize[2]; uint32_t hizMips, useOcclusion;
            float lodScale, lodThreshold, lodNear; uint32_t useCone;
            float3 lodOrigin; uint32_t countStats;
        };
        struct DrawConstants
        {
            PlanarViewConstants view;
            PlanarViewConstants viewPrev;
        };

        constexpr uint32_t c_Passes = 6;          // camera phase 1, phase 2, 4 shadow cascades
        constexpr uint32_t c_ArgStride = 20;      // DrawIndexedIndirectArguments
        constexpr uint32_t c_CountWords = 64;     // pass x bucket counters, then statistics
        constexpr uint32_t c_StatTrianglesDrawn = 57, c_StatDrawnPhase1 = 58, c_StatDrawnPhase2 = 59, c_StatFrustumCulled = 60, c_StatOccluded = 61, c_StatMeshletsTested = 62, c_StatMeshletsDrawn = 63;
        constexpr uint32_t c_MeshArgStride = 12;  // DispatchMesh indirect (x, y, z)
    }

    GpuDrivenRenderer::GpuDrivenRenderer(Context& ctx)
        : m_Ctx(ctx), m_Scene(ctx), m_HiZ(ctx), m_StatsReadback(ctx, c_CountWords * 4, "hyper/gpu-draw-stats")
    {
        auto cs = nvrhi::ShaderType::Compute;
        nvrhi::BindingLayoutDesc cull;
        cull.setVisibility(cs)
            .addItem(nvrhi::BindingLayoutItem::VolatileConstantBuffer(0))
            .addItem(nvrhi::BindingLayoutItem::Texture_SRV(0))
            .addItem(nvrhi::BindingLayoutItem::RawBuffer_UAV(0))
            .addItem(nvrhi::BindingLayoutItem::RawBuffer_UAV(1))
            .addItem(nvrhi::BindingLayoutItem::RawBuffer_UAV(2))
            .addItem(nvrhi::BindingLayoutItem::RawBuffer_UAV(3))
            .addItem(nvrhi::BindingLayoutItem::RawBuffer_UAV(4));
        GpuScene::AddLayoutItems(cull);
        m_CullLayout = ctx.device->createBindingLayout(cull);

        nvrhi::BindingLayoutDesc draw;
        draw.setVisibility(nvrhi::ShaderType::Vertex | nvrhi::ShaderType::Pixel)
            .addItem(nvrhi::BindingLayoutItem::VolatileConstantBuffer(0))
            .addItem(nvrhi::BindingLayoutItem::Sampler(0));
        GpuScene::AddLayoutItems(draw);
        m_DrawLayout = ctx.device->createBindingLayout(draw);

        m_MeshShading = ctx.device->queryFeatureSupport(nvrhi::Feature::Meshlets);
        if (m_MeshShading)
        {
            nvrhi::BindingLayoutDesc mesh;
            mesh.setVisibility(nvrhi::ShaderType::Amplification | nvrhi::ShaderType::Mesh | nvrhi::ShaderType::Pixel)
                .addItem(nvrhi::BindingLayoutItem::VolatileConstantBuffer(0))
                .addItem(nvrhi::BindingLayoutItem::PushConstants(1, 16))
                .addItem(nvrhi::BindingLayoutItem::VolatileConstantBuffer(2))
                .addItem(nvrhi::BindingLayoutItem::Sampler(0))
                .addItem(nvrhi::BindingLayoutItem::RawBuffer_SRV(72))
                .addItem(nvrhi::BindingLayoutItem::Texture_SRV(73))
                .addItem(nvrhi::BindingLayoutItem::RawBuffer_UAV(0));
            GpuScene::AddLayoutItems(mesh);
            m_MeshLayout = ctx.device->createBindingLayout(mesh);
            m_MeshCullConstants = ctx.device->createBuffer(
                nvrhi::utils::CreateVolatileConstantBufferDesc(sizeof(MeshCullConstants), "hyper/gpu-mesh-cull-constants", 32));
            m_MeshArgs = ctx.device->createBuffer(nvrhi::BufferDesc().setDebugName("hyper/gpu-mesh-args")
                .setByteSize(uint64_t(c_Passes) * GpuScene::c_Buckets * c_MeshArgStride)
                .setCanHaveUAVs(true).setCanHaveRawViews(true).setIsDrawIndirectArgs(true)
                .setInitialState(nvrhi::ResourceStates::UnorderedAccess).setKeepInitialState(true));
        }

        m_CullConstants = ctx.device->createBuffer(
            nvrhi::utils::CreateVolatileConstantBufferDesc(sizeof(CullConstants), "hyper/gpu-cull-constants", 32));
        m_DrawConstants = ctx.device->createBuffer(
            nvrhi::utils::CreateVolatileConstantBufferDesc(sizeof(DrawConstants), "hyper/gpu-draw-constants", 32));
        m_Counts = ctx.device->createBuffer(nvrhi::BufferDesc().setDebugName("hyper/gpu-draw-counts")
            .setByteSize(c_CountWords * 4).setCanHaveUAVs(true).setCanHaveRawViews(true).setIsDrawIndirectArgs(true)
            .setInitialState(nvrhi::ResourceStates::UnorderedAccess).setKeepInitialState(true));
    }

    GpuDrivenRenderer::~GpuDrivenRenderer() = default;

    void GpuDrivenRenderer::CreatePipelines()
    {
        m_GBufferPass.reset();
        m_HiZ.CreatePipelines();
        m_Scene.CreatePipelines();
        auto cull = m_Ctx.shaders->CreateShader("hyper/gpu/cull_cs", "main", nullptr, nvrhi::ShaderType::Compute);
        if (cull)
            m_Cull = m_Ctx.device->createComputePipeline(nvrhi::ComputePipelineDesc().setComputeShader(cull).addBindingLayout(m_CullLayout));

        const std::vector<engine::ShaderMacro> opaque = { engine::ShaderMacro("ALPHA_TESTED", "0") };
        const std::vector<engine::ShaderMacro> alpha = { engine::ShaderMacro("ALPHA_TESTED", "1") };
        auto vs = m_Ctx.shaders->CreateShader("hyper/gpu/draw", "vs", &opaque, nvrhi::ShaderType::Vertex);
        auto ps = m_Ctx.shaders->CreateShader("hyper/gpu/draw", "ps", &opaque, nvrhi::ShaderType::Pixel);
        auto psAlpha = m_Ctx.shaders->CreateShader("hyper/gpu/draw", "ps", &alpha, nvrhi::ShaderType::Pixel);
        auto vsDepth = m_Ctx.shaders->CreateShader("hyper/gpu/draw", "vs_depth", &opaque, nvrhi::ShaderType::Vertex);
        auto psDepthAlpha = m_Ctx.shaders->CreateShader("hyper/gpu/draw", "ps_depth", &alpha, nvrhi::ShaderType::Pixel);
        if (m_MeshShading)
        {
            auto task = m_Ctx.shaders->CreateShader("hyper/gpu/draw", "task", &opaque, nvrhi::ShaderType::Amplification);
            auto mesh = m_Ctx.shaders->CreateShader("hyper/gpu/draw", "mesh", &opaque, nvrhi::ShaderType::Mesh);
            auto meshDepth = m_Ctx.shaders->CreateShader("hyper/gpu/draw", "mesh_depth", &opaque, nvrhi::ShaderType::Mesh);
            if (task && mesh && meshDepth)
            {
                m_Task = task;
                m_Mesh = mesh;
                m_MeshDepth = meshDepth;
                m_MeshPipelines.clear();
            }
        }
        if (vs && ps && psAlpha && vsDepth && psDepthAlpha)
        {
            m_Vs = vs;
            m_Ps = ps;
            m_PsAlpha = psAlpha;
            m_VsDepth = vsDepth;
            m_PsDepthAlpha = psDepthAlpha;
            m_Pipelines.clear();
        }
    }

    void GpuDrivenRenderer::EnsureBuffers(nvrhi::ICommandList* cmd)
    {
        const uint32_t draws = std::max(m_Scene.DrawCount(), 1u);
        if (draws != m_MaxDraws)
        {
            m_MaxDraws = draws;
            m_Visibility = m_Ctx.device->createBuffer(nvrhi::BufferDesc().setDebugName("hyper/gpu-visibility")
                .setByteSize(uint64_t(draws) * 4).setCanHaveUAVs(true).setCanHaveRawViews(true)
                .setInitialState(nvrhi::ResourceStates::UnorderedAccess).setKeepInitialState(true));
            m_Args = m_Ctx.device->createBuffer(nvrhi::BufferDesc().setDebugName("hyper/gpu-draw-args")
                .setByteSize(uint64_t(c_Passes) * GpuScene::c_Buckets * draws * c_ArgStride)
                .setCanHaveUAVs(true).setCanHaveRawViews(true).setIsDrawIndirectArgs(true)
                .setInitialState(nvrhi::ResourceStates::UnorderedAccess).setKeepInitialState(true));
            m_Ctx.bindingCache->Clear();
        }
        const uint32_t tasks = std::max(m_Scene.TaskCapacity(), 1u);
        if (!m_Tasks || m_Tasks->getDesc().byteSize < uint64_t(c_Passes) * GpuScene::c_Buckets * tasks * 8)
        {
            m_Tasks = m_Ctx.device->createBuffer(nvrhi::BufferDesc().setDebugName("hyper/gpu-tasks")
                .setByteSize(uint64_t(c_Passes) * GpuScene::c_Buckets * tasks * 8)
                .setCanHaveUAVs(true).setCanHaveRawViews(true)
                .setInitialState(nvrhi::ResourceStates::ShaderResource).setKeepInitialState(true));
            m_Ctx.bindingCache->Clear();
        }
        // New scene: nothing was visible last frame.
        cmd->clearBufferUInt(m_Visibility, 0);
    }

    bool GpuDrivenRenderer::BeginFrame(nvrhi::ICommandList* cmd, Scene& scene)
    {
        m_Ready = false;
        if (!settings.enabled)
            return false;
        if (scene.BindlessLayout() && scene.BindlessLayout() != m_BindlessLayout)
        {
            m_BindlessLayout = scene.BindlessLayout();
            CreatePipelines();
        }
        if (!m_Cull || !m_Vs || !m_BindlessLayout || !m_Scene.Update(cmd, scene, settings.clusterLod && settings.meshlets && m_MeshShading))
            return false;
        if (m_Scene.Revision() != m_Revision)
        {
            m_Revision = m_Scene.Revision();
            EnsureBuffers(cmd);
        }
        m_Ready = true;
        return true;
    }

    nvrhi::IGraphicsPipeline* GpuDrivenRenderer::Pipeline(nvrhi::IFramebuffer* fb, Mode mode, uint32_t variant,
                                                          bool reverseZ, int depthBias, float slopeBias)
    {
        const auto& info = fb->getFramebufferInfo();
        size_t key = std::hash<nvrhi::FramebufferInfo>()(info);
        nvrhi::hash_combine(key, int(mode));
        nvrhi::hash_combine(key, variant);
        nvrhi::hash_combine(key, reverseZ);
        nvrhi::hash_combine(key, depthBias);
        nvrhi::hash_combine(key, slopeBias);
        auto& pipeline = m_Pipelines[key];
        if (pipeline)
            return pipeline;

        const bool alpha = variant == 2;
        nvrhi::GraphicsPipelineDesc desc;
        desc.primType = nvrhi::PrimitiveType::TriangleList;
        switch (mode)
        {
        case Mode::GBuffer:    desc.VS = m_Vs;      desc.PS = alpha ? m_PsAlpha : m_Ps; break;
        case Mode::Depth:      desc.VS = m_VsDepth; desc.PS = alpha ? m_PsDepthAlpha : nullptr; break;
        }
        desc.bindingLayouts = { m_DrawLayout, m_BindlessLayout };
        // As Donut's GBufferFillPass: glTF winding, back faces culled unless double-sided or alpha-tested.
        desc.renderState.rasterState.setFrontCounterClockwise(true)
            .setCullMode(variant == 0 ? nvrhi::RasterCullMode::Back : nvrhi::RasterCullMode::None);
        desc.renderState.rasterState.depthBias = depthBias;
        desc.renderState.rasterState.slopeScaledDepthBias = slopeBias;
        desc.renderState.depthStencilState = GBuffer::DepthStencil(reverseZ);
        if (mode == Mode::Depth)
            desc.renderState.depthStencilState.disableStencil();
        pipeline = m_Ctx.device->createGraphicsPipeline(desc, info);
        return pipeline;
    }

    nvrhi::IMeshletPipeline* GpuDrivenRenderer::MeshPipeline(nvrhi::IFramebuffer* fb, Mode mode, uint32_t variant,
                                                             bool reverseZ, int depthBias, float slopeBias)
    {
        const auto& info = fb->getFramebufferInfo();
        size_t key = std::hash<nvrhi::FramebufferInfo>()(info);
        nvrhi::hash_combine(key, variant);
        nvrhi::hash_combine(key, int(mode));
        nvrhi::hash_combine(key, reverseZ);
        nvrhi::hash_combine(key, depthBias);
        nvrhi::hash_combine(key, slopeBias);
        auto& pipeline = m_MeshPipelines[key];
        if (pipeline)
            return pipeline;
        const bool alpha = variant == 2;
        nvrhi::MeshletPipelineDesc desc;
        desc.setPrimType(nvrhi::PrimitiveType::TriangleList).setTaskShader(m_Task)
            .addBindingLayout(m_MeshLayout).addBindingLayout(m_BindlessLayout);
        switch (mode)
        {
        case Mode::GBuffer:    desc.setMeshShader(m_Mesh).setPixelShader(alpha ? m_PsAlpha : m_Ps); break;
        case Mode::Depth:      desc.setMeshShader(m_MeshDepth).setPixelShader(alpha ? m_PsDepthAlpha : nullptr); break;
        }
        // The same state as the indexed pipelines.
        desc.renderState.rasterState.setFrontCounterClockwise(true)
            .setCullMode(variant == 0 ? nvrhi::RasterCullMode::Back : nvrhi::RasterCullMode::None);
        desc.renderState.rasterState.depthBias = depthBias;
        desc.renderState.rasterState.slopeScaledDepthBias = slopeBias;
        desc.renderState.depthStencilState = GBuffer::DepthStencil(reverseZ);
        if (mode == Mode::Depth)
            desc.renderState.depthStencilState.disableStencil();
        pipeline = m_Ctx.device->createMeshletPipeline(desc, info);
        return pipeline;
    }

    void GpuDrivenRenderer::Cull(nvrhi::ICommandList* cmd, const engine::PlanarView& view, Scene& scene, uint32_t pass,
                                 uint32_t phase, bool occlusion)
    {
        const bool meshlets = UseMeshlets();
        if (m_MeshArgs)
        {
            uint32_t args[GpuScene::c_Buckets * 3];
            for (uint32_t b = 0; b < GpuScene::c_Buckets; ++b)
            {
                args[b * 3 + 0] = 0;
                args[b * 3 + 1] = 1;
                args[b * 3 + 2] = 1;
            }
            cmd->writeBuffer(m_MeshArgs, args, sizeof(args), uint64_t(pass) * GpuScene::c_Buckets * c_MeshArgStride);
        }
        // Reset this pass's counters (and the statistics, with the camera's first phase).
        const uint32_t zeros[4] = {};
        cmd->writeBuffer(m_Counts, zeros, sizeof(zeros), uint64_t(pass) * GpuScene::c_Buckets * 4);
        if (phase == 1 || (pass == 0 && !occlusion))
        {
            const uint32_t stats[7] = {};
            cmd->writeBuffer(m_Counts, stats, sizeof(stats), uint64_t(c_StatTrianglesDrawn) * 4);
        }

        CullConstants c{};
        c.viewProj = view.GetViewProjectionMatrix(false);
        const dm::frustum f = view.GetViewFrustum();
        for (int i = 0; i < 6; ++i)
        {
            const dm::plane& p = f.planes[i];
            const bool finite = std::isfinite(p.normal.x) && std::isfinite(p.normal.y) && std::isfinite(p.normal.z)
                && std::isfinite(p.distance);
            c.planes[i] = (finite && settings.frustumCulling) ? float4(p.normal, p.distance) : float4(0.f);
        }
        c.drawCount = m_Scene.DrawCount();
        c.phase = phase;
        c.argsBase = pass * GpuScene::c_Buckets * m_MaxDraws;
        c.countBase = pass * GpuScene::c_Buckets;
        c.maxDraws = m_MaxDraws;
        c.useOcclusion = occlusion ? 1u : 0u;
        nvrhi::ITexture* hiz = m_HiZ.Texture() ? m_HiZ.Texture() : m_Ctx.common->m_BlackTexture.Get();
        c.hizSize[0] = hiz->getDesc().width;
        c.hizSize[1] = hiz->getDesc().height;
        c.hizMips = std::max(m_HiZ.MipLevels(), 1u);
        c.reverseZ = view.IsReverseDepth() ? 1u : 0u;
        c.useMeshlets = meshlets ? 1u : 0u;
        c.taskCapacity = std::max(m_Scene.TaskCapacity(), 1u);
        cmd->writeBuffer(m_CullConstants, &c, sizeof(c));

        nvrhi::BindingSetDesc set;
        set.addItem(nvrhi::BindingSetItem::ConstantBuffer(0, m_CullConstants))
            .addItem(nvrhi::BindingSetItem::Texture_SRV(0, hiz))
            .addItem(nvrhi::BindingSetItem::RawBuffer_UAV(0, m_Visibility))
            .addItem(nvrhi::BindingSetItem::RawBuffer_UAV(1, m_Args))
            .addItem(nvrhi::BindingSetItem::RawBuffer_UAV(2, m_Counts))
            .addItem(nvrhi::BindingSetItem::RawBuffer_UAV(3, m_Tasks))
            .addItem(nvrhi::BindingSetItem::RawBuffer_UAV(4, m_MeshArgs ? m_MeshArgs.Get() : m_Counts.Get()));
        m_Scene.AddSetItems(set, scene);
        cmd->setComputeState(nvrhi::ComputeState().setPipeline(m_Cull)
            .addBindingSet(m_Ctx.bindingCache->GetOrCreateBindingSet(set, m_CullLayout)));
        cmd->dispatch((c.drawCount + 63) / 64);
    }

    void GpuDrivenRenderer::Draw(nvrhi::ICommandList* cmd, const engine::PlanarView& view, const engine::PlanarView& previous,
                                 const engine::PlanarView& camera, Scene& scene, nvrhi::IFramebuffer* fb, uint32_t pass,
                                 Mode mode, int depthBias, float slopeBias)
    {
        const bool depthOnly = mode == Mode::Depth;
        DrawConstants c{};
        view.FillPlanarViewConstants(c.view);
        previous.FillPlanarViewConstants(c.viewPrev);
        cmd->writeBuffer(m_DrawConstants, &c, sizeof(c));

        nvrhi::BindingSetDesc setDesc;
        setDesc.addItem(nvrhi::BindingSetItem::ConstantBuffer(0, m_DrawConstants))
            .addItem(nvrhi::BindingSetItem::Sampler(0, m_Ctx.common->m_AnisotropicWrapSampler));
        m_Scene.AddSetItems(setDesc, scene);
        auto set = m_Ctx.bindingCache->GetOrCreateBindingSet(setDesc, m_DrawLayout);

        // Bucket -> pipeline variant: 0 opaque, 1 opaque double-sided, 2 alpha-tested (either).
        const uint32_t variants[GpuScene::c_Buckets] = { 0, 1, 2, 2 };
        for (uint32_t bucket = 0; bucket < GpuScene::c_Buckets; ++bucket)
        {
            nvrhi::GraphicsState state;
            state.pipeline = Pipeline(fb, mode, variants[bucket], view.IsReverseDepth(), depthBias, slopeBias);
            state.framebuffer = fb;
            state.viewport = view.GetViewportState();
            state.bindings = { set, scene.DescriptorTable() };
            state.indexBuffer = nvrhi::IndexBufferBinding().setBuffer(m_Scene.IndexBuffer()).setFormat(nvrhi::Format::R32_UINT);
            state.indirectParams = m_Args;
            state.indirectCountBuffer = m_Counts;
            cmd->setGraphicsState(state);
            const uint32_t slot = pass * GpuScene::c_Buckets + bucket;
            cmd->drawIndexedIndirectCount(slot * m_MaxDraws * c_ArgStride, slot * 4, m_MaxDraws);
        }

        if (!UseMeshlets())
            return;

        // Meshlet draws: the task shader culls meshlets against this view.
        MeshCullConstants mc{};
        mc.viewProj = view.GetViewProjectionMatrix(false);
        const dm::frustum f = view.GetViewFrustum();
        for (int i = 0; i < 6; ++i)
        {
            const dm::plane& p = f.planes[i];
            const bool finite = std::isfinite(p.normal.x) && std::isfinite(p.normal.y) && std::isfinite(p.normal.z)
                && std::isfinite(p.distance);
            mc.planes[i] = (finite && settings.frustumCulling) ? float4(p.normal, p.distance) : float4(0.f);
        }
        mc.cameraPos = view.GetViewOrigin();
        mc.phase = pass == 1 ? 2u : pass == 0 ? 1u : 0u;
        nvrhi::ITexture* hiz = m_HiZ.Texture() ? m_HiZ.Texture() : m_Ctx.common->m_BlackTexture.Get();
        mc.hizSize[0] = hiz->getDesc().width;
        mc.hizSize[1] = hiz->getDesc().height;
        mc.hizMips = std::max(m_HiZ.MipLevels(), 1u);
        mc.useOcclusion = settings.occlusionCulling ? 1u : 0u;
        // Error in object units over distance, times this, is pixels (clusterlod's estimate).
        // Always the camera's: shadow cascades draw the same cut.
        mc.lodScale = camera.GetProjectionMatrix(false)[1][1] * 0.5f * camera.GetViewExtent().height();
        mc.lodThreshold = std::max(settings.lodErrorPixels, 0.f);
        mc.lodNear = m_Ctx.camera.zNear;
        mc.useCone = depthOnly ? 0u : 1u;
        mc.lodOrigin = camera.GetViewOrigin();
        mc.countStats = pass < 2 ? 1u : 0u;
        cmd->writeBuffer(m_MeshCullConstants, &mc, sizeof(mc));

        nvrhi::BindingSetDesc meshDesc;
        meshDesc.addItem(nvrhi::BindingSetItem::ConstantBuffer(0, m_DrawConstants))
            .addItem(nvrhi::BindingSetItem::PushConstants(1, 16))
            .addItem(nvrhi::BindingSetItem::ConstantBuffer(2, m_MeshCullConstants))
            .addItem(nvrhi::BindingSetItem::Sampler(0, m_Ctx.common->m_AnisotropicWrapSampler))
            .addItem(nvrhi::BindingSetItem::RawBuffer_SRV(72, m_Tasks))
            .addItem(nvrhi::BindingSetItem::Texture_SRV(73, hiz))
            .addItem(nvrhi::BindingSetItem::RawBuffer_UAV(0, m_Counts));
        m_Scene.AddSetItems(meshDesc, scene);
        auto meshSet = m_Ctx.bindingCache->GetOrCreateBindingSet(meshDesc, m_MeshLayout);
        const uint32_t capacity = std::max(m_Scene.TaskCapacity(), 1u);
        for (uint32_t bucket = 0; bucket < GpuScene::c_Buckets; ++bucket)
        {
            nvrhi::MeshletState state;
            state.pipeline = MeshPipeline(fb, mode, variants[bucket], view.IsReverseDepth(), depthBias, slopeBias);
            state.framebuffer = fb;
            state.viewport = view.GetViewportState();
            state.bindings = { meshSet, scene.DescriptorTable() };
            state.indirectParams = m_MeshArgs;
            cmd->setMeshletState(state);
            const uint32_t slot = pass * GpuScene::c_Buckets + bucket;
            const uint32_t push[4] = { slot * capacity, 0, 0, 0 };
            cmd->setPushConstants(push, sizeof(push));
            cmd->dispatchMeshIndirect(slot * c_MeshArgStride, 1);
        }
    }

    void GpuDrivenRenderer::RenderGBuffer(nvrhi::ICommandList* cmd, const engine::PlanarView& view,
                                          const engine::PlanarView& previous, Scene& scene, GBuffer& gbuffer)
    {
        if (!m_Ready)
        {
            if (settings.enabled && scene.BindlessLayout())
                return;   // the GPU scene is being built
            if (!scene.IsReady())
                return;
            auto scope = m_Ctx.profiler->Measure(cmd, "donut gbuffer");
            if (!m_GBufferPass)
            {
                render::GBufferFillPass::CreateParameters params;
                params.enableMotionVectors = true;
                params.stencilWriteMask = GBuffer::MotionVectorStencilBit;
                m_GBufferPass = std::make_unique<render::GBufferFillPass>(m_Ctx.device, m_Ctx.common);
                m_GBufferPass->Init(*m_Ctx.shaders, params);
                m_Opaque = std::make_unique<render::InstancedOpaqueDrawStrategy>();
            }
            // Donut caches each material's binding set, textures and all, the first time it
            // draws it: drawn while its textures stream, it would keep the fallbacks for good.
            uint32_t arrived = 0;
            for (const auto& m : scene.Graph()->GetMaterials())
                for (const auto* t : { &m->baseOrDiffuseTexture, &m->metalRoughOrSpecularTexture, &m->normalTexture,
                                       &m->emissiveTexture, &m->occlusionTexture, &m->transmissionTexture, &m->opacityTexture })
                    arrived += *t && (*t)->texture ? 1u : 0u;
            if (arrived != m_TexturesSeen)
            {
                m_GBufferPass->ResetBindingCache();
                m_TexturesSeen = arrived;
            }
            render::GBufferFillPass::Context context;
            render::RenderCompositeView(cmd, &view, &previous, *gbuffer.GBufferFramebuffer, scene.Graph()->GetRootNode(),
                *m_Opaque, *m_GBufferPass, context, "GBufferFill");
            return;
        }
        auto scope = m_Ctx.profiler->Measure(cmd, "gpu-driven gbuffer");
        const Mode mode = Mode::GBuffer;
        nvrhi::IFramebuffer* fb = gbuffer.GBufferFramebuffer->GetFramebuffer(view);
        const bool occlusion = settings.occlusionCulling;
        {
            auto t = m_Ctx.profiler->Measure(cmd, "phase 1");
            // Without occlusion culling phase 1 draws everything in the frustum.
            {
                auto c = m_Ctx.profiler->Measure(cmd, "cull");
                Cull(cmd, view, scene, 0, occlusion ? 1u : 0u, false);
            }
            Draw(cmd, view, previous, view, scene, fb, 0, mode, 0, 0.f);
        }
        if (occlusion)
        {
            auto t = m_Ctx.profiler->Measure(cmd, "phase 2");
            {
                auto c = m_Ctx.profiler->Measure(cmd, "hi-z + cull");
                m_HiZ.Build(cmd, gbuffer.Depth);
                Cull(cmd, view, scene, 1, 2, true);
            }
            Draw(cmd, view, previous, view, scene, fb, 1, mode, 0, 0.f);
        }
        ReadStats(cmd);
    }

    void GpuDrivenRenderer::RenderDepth(nvrhi::ICommandList* cmd, const engine::PlanarView& view, const engine::PlanarView& camera,
                                        Scene& scene, nvrhi::IFramebuffer* framebuffer, uint32_t cascade, int depthBias, float slopeBias)
    {
        if (!m_Ready || cascade > 3)
            return;
        const uint32_t pass = 2 + cascade;
        Cull(cmd, view, scene, pass, 0, false);
        Draw(cmd, view, view, camera, scene, framebuffer, pass, Mode::Depth, depthBias, slopeBias);
    }

    void GpuDrivenRenderer::ReadStats(nvrhi::ICommandList* cmd)
    {
        m_StatsReadback.Copy(cmd, m_Counts);
        if (const uint32_t* words = m_StatsReadback.Latest<uint32_t>())
        {
            m_Stats.phase1 = words[c_StatDrawnPhase1];
            m_Stats.phase2 = words[c_StatDrawnPhase2];
            m_Stats.frustumCulled = words[c_StatFrustumCulled];
            m_Stats.occluded = words[c_StatOccluded];
            m_Stats.meshletsTested = words[c_StatMeshletsTested];
            m_Stats.meshletsDrawn = words[c_StatMeshletsDrawn];
            m_Stats.trianglesDrawn = words[c_StatTrianglesDrawn];
        }
        m_Stats.draws = m_Scene.DrawCount();
        m_Stats.triangles = m_Scene.TriangleCount();
        m_Stats.meshlets = m_Scene.MeshletCount();
        m_Stats.lodLevels = m_Scene.HasClusterLod() ? m_Scene.LodLevels() : 0u;
        ++m_Frame;
    }

    void GpuDrivenRenderer::BuildUI()
    {
        ImGui::Checkbox("GPU-driven (indirect, GPU culling)", &settings.enabled);
        if (!settings.enabled)
            return;
        ImGui::Checkbox("frustum culling", &settings.frustumCulling);
        ImGui::Checkbox("occlusion culling (two-phase Hi-Z)", &settings.occlusionCulling);
        if (m_MeshShading)
        {
            ImGui::Checkbox("meshlets (task + mesh shaders)", &settings.meshlets);
            if (settings.meshlets)
            {
                ImGui::Checkbox("cluster LOD", &settings.clusterLod);
                if (settings.clusterLod)
                    ImGui::SliderFloat("LOD error (pixels)", &settings.lodErrorPixels, 0.f, 8.f);
            }
        }
        else
            ImGui::TextDisabled("meshlets: no mesh shader support");
        const Stats& s = m_Stats;
        ImGui::TextDisabled("%u draws, %u triangles in the scene", s.draws, s.triangles);
        ImGui::TextDisabled("drawn: %u (phase 1) + %u (phase 2)", s.phase1, s.phase2);
        ImGui::TextDisabled("culled: %u frustum, %u occluded", s.frustumCulled, s.occluded);
        if (UseMeshlets())
        {
            ImGui::TextDisabled("meshlets: %u in the scene, %u tested, %u drawn", s.meshlets, s.meshletsTested, s.meshletsDrawn);
            ImGui::TextDisabled("triangles drawn by meshlets: %u%s", s.trianglesDrawn,
                s.lodLevels ? (" (LOD DAG: " + std::to_string(s.lodLevels) + " levels)").c_str() : "");
        }
    }
}
