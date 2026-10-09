// scene: a glTF scene through most of hyper's modules, assembled in SceneRenderer.
//
//   scene.exe [--scene path/to/file.gltf] [--env path/to/sky.hdr] [--set name=value ...]
//
// Default scene: Sponza from tools/fetch_assets.ps1. --set cam=N looks through the file's
// camera N, --set eyex= eyey= eyez= yaw= pitch= (degrees) puts the camera anywhere. The raster defaults (atmosphere or
// HDR environment, sun shadows, IBL, XeGTAO, SSR, GPU-driven drawing, the post stack) plus
// the opt-in modules: ray-traced shadows and reflections, DDGI and fog; or the reference
// path tracer. `--set name=0` turns any of them off. Fly camera: hold LMB + drag to look,
// WASD/QE to move, Shift faster.

#include "SceneRenderer.h"

#include <donut/core/log.h>
#include <donut/engine/SceneGraph.h>

#include <imgui.h>

#include <algorithm>
#include <cctype>
#include <string>

using namespace donut;
using namespace donut::math;

class SceneApp : public hyper::App
{
public:
    explicit SceneApp(hyper::Context& ctx) : App(ctx), m_Scene(ctx), m_Renderer(ctx) { }

    bool Init() override
    {
        auto& ctx = Ctx();
        m_Renderer.Post().settings.tonemap.autoExposure = true;
        // Opt in to everything; the ray-traced modules only run with ray queries.
        m_Renderer.RayShadows().settings.enabled = true;
        m_Renderer.RayReflections().settings.enabled = true;
        m_Renderer.Gi().settings.enabled = true;
        m_Renderer.Fog().settings.enabled = true;

        std::filesystem::path file = hyper::AssetPath("downloaded/Sponza/glTF/Sponza.gltf");
        const auto& args = ctx.args;
        for (size_t i = 0; i + 1 < args.size(); ++i)
        {
            if (args[i] == "--scene")
                file = hyper::AssetPath(args[i + 1]);
            else if (args[i] == "--env")
            {
                auto& env = m_Renderer.Environment().settings;
                env.file = hyper::AssetPath(args[i + 1]);
                env.source = hyper::EnvironmentSource::File;
            }
        }

        if (!std::filesystem::exists(file))
        {
            log::error("scene: %s not found. Run `pwsh tools/fetch_assets.ps1` for the sample scenes, or pass --scene <file>.",
                file.string().c_str());
            return false;
        }

        std::string lower = file.filename().string();
        std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) { return char(std::tolower(c)); });
        m_Interior = lower.find("sponza") != std::string::npos;
        if (m_Interior)
        {
            // The nave is narrow: the sun reaches its floor only when it is high.
            m_Renderer.settings.sunElevationDeg = 72.f;
            m_Renderer.settings.sunAzimuthDeg = 60.f;
            // Dusty air: light shafts through the open roof.
            auto& fog = m_Renderer.Fog().settings;
            fog.density = 0.02f;
            fog.heightFalloff = 0.f;
            fog.noiseStrength = 0.35f;
            fog.anisotropy = 0.7f;
        }

        ApplyOverrides();
        m_Scene.Load(file);
        m_CommandList = ctx.device->createCommandList();
        return true;
    }

    void ShadersReloaded() override { m_Renderer.ShadersReloaded(); }

    void Animate(float) override
    {
        auto& ctx = Ctx();
        if (m_Edit && m_Scene.IsReady())
            Edit(ctx.frameIndex);
        m_Scene.Update(ctx.deltaTime);
        if (m_Scene.IsReady() && !m_CameraPlaced)
        {
            PlaceCamera();
            m_CameraPlaced = true;
        }
    }

    void Render(nvrhi::IFramebuffer* framebuffer) override
    {
        auto& ctx = Ctx();
        const auto& fbInfo = framebuffer->getFramebufferInfo();

        ctx.camera.GetView(m_View, fbInfo.width, fbInfo.height);
        m_Renderer.PrepareView(m_View);

        m_CommandList->open();
        m_Renderer.Render(m_CommandList, m_Scene, m_View, framebuffer);
        m_CommandList->close();
        ctx.device->executeCommandList(m_CommandList);
    }

    void BuildUI() override
    {
        ImGui::SetNextWindowPos(ImVec2(10.f, 200.f), ImGuiCond_FirstUseEver);
        ImGui::Begin("scene", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
        m_Renderer.BuildUI(m_Scene);
        ImGui::Checkbox("animate", &m_Scene.animate);
        if (ImGui::Button("frame camera"))
            PlaceCamera();
        {
            // Where the camera is, as the overrides that put it back (for reports and captures).
            const float3 p = Ctx().camera.fly.GetPosition(), d = Ctx().camera.fly.GetDir();
            ImGui::TextDisabled("--set eyex=%.2f --set eyey=%.2f --set eyez=%.2f --set yaw=%.1f --set pitch=%.1f", p.x, p.y, p.z,
                                dm::degrees(std::atan2(d.x, d.z)), dm::degrees(std::asin(std::clamp(d.y, -1.f, 1.f))));
        }
        ImGui::TextDisabled("LMB drag: look, WASD/QE: move, Shift: faster");
        ImGui::End();
    }

    ~SceneApp() override
    {
        // Culling statistics for captures (UI shows them live).
        const auto& st = m_Renderer.GpuDriven().LastStats();
        if (st.draws)
            log::info("scene: GPU-driven %u draws: %u + %u drawn, %u frustum culled, %u occluded; meshlets %u tested, %u drawn (%u in the scene, %u LOD levels), %u triangles",
                st.draws, st.phase1, st.phase2, st.frustumCulled, st.occluded, st.meshletsTested, st.meshletsDrawn, st.meshlets,
                st.lodLevels, st.trianglesDrawn);
    }

private:
    hyper::Scene m_Scene;
    scene::SceneRenderer m_Renderer;
    nvrhi::CommandListHandle m_CommandList;
    engine::PlanarView m_View;
    bool m_Interior = false;
    bool m_CameraPlaced = false;
    int m_Camera = 0;                         // --set cam=N: the file's camera N
    float3 m_Eye = float3(-1e30f);            // --set eyex= eyey= eyez=: the camera here instead
    float m_Yaw = 0.f, m_Pitch = 0.f;         // degrees: from +z towards +x, and up
    int m_Edit = 0;
    std::vector<std::shared_ptr<engine::SceneGraphNode>> m_Edited;

    // --set edit=1 takes everything out of the scene at frame 10 and puts it back at 20;
    // edit=2 adds a second instance of the first mesh at 10 and removes it at 20. Either
    // way the capture should look as without: it checks that the GPU scene and the rays
    // follow edits (and the log, what an edit costs).
    void Edit(uint64_t frame)
    {
        auto graph = m_Scene.Graph();
        if (frame == 10 && m_Edit == 1)
        {
            const auto& root = graph->GetRootNode();
            for (size_t i = 0; i < root->GetNumChildren(); ++i)
                m_Edited.push_back(root->GetChild(i)->shared_from_this());
            for (const auto& node : m_Edited)
                graph->Detach(node);
        }
        else if (frame == 10 && m_Edit == 2 && !graph->GetMeshInstances().empty())
        {
            const auto& first = graph->GetMeshInstances().front();
            auto node = std::make_shared<engine::SceneGraphNode>();
            node->SetLeaf(std::make_shared<engine::MeshInstance>(first->GetMesh()));
            node->SetTranslation(dm::double3(0.0, 1000.0, 0.0));   // out of view
            graph->Attach(graph->GetRootNode(), node);
            m_Edited.push_back(node);
        }
        else if (frame == 20)
        {
            for (const auto& node : m_Edited)
                m_Edit == 1 ? (void)graph->Attach(graph->GetRootNode(), node) : (void)graph->Detach(node);
            m_Edited.clear();
        }
    }


    void ApplyOverrides()
    {
        auto& s = m_Renderer.settings;
        auto& post = m_Renderer.Post().settings;
        hyper::Overrides o(Ctx().args);
        bool taa = post.aa == hyper::AntiAliasing::Taa;
        o.Apply("reference", s.reference);
        if (o.Apply("taa", taa)) post.aa = taa ? hyper::AntiAliasing::Taa : hyper::AntiAliasing::None;
        o.Apply("transparency", s.transparency);
        o.Apply("sky", s.sky);
        o.Apply("sunaz", s.sunAzimuthDeg);
        o.Apply("sunel", s.sunElevationDeg);
        o.Apply("sun", s.sunIrradiance);
        o.Apply("ambient", s.ambient);
        o.Apply("ev", post.tonemap.exposureEV);
        o.Apply("autoexposure", post.tonemap.autoExposure);
        o.Apply("bloom", post.bloomIntensity);
        o.Apply("shadows", m_Renderer.Shadows().settings.enabled);
        o.Apply("ibl", m_Renderer.Environment().settings.enabled);
        o.Apply("ao", m_Renderer.Ao().settings.enabled);
        o.Apply("aoradius", m_Renderer.Ao().settings.radius);
        o.Apply("aopower", m_Renderer.Ao().settings.power);
        o.Apply("aothin", m_Renderer.Ao().settings.thinOccluders);
        o.Apply("aobounce", m_Renderer.Ao().settings.multiBounce);
        o.Apply("ssr", m_Renderer.Reflections().settings.enabled);
        auto& atmosphere = m_Renderer.Sky().settings;
        o.Apply("atmosphere", atmosphere.enabled);
        o.Apply("altitude", atmosphere.altitudeKm);
        o.Apply("haze", atmosphere.mieScale);
        o.Apply("aerial", atmosphere.aerialScale);
        int debug = int(s.debugView);
        if (o.Apply("debug", debug)) s.debugView = scene::DebugView(debug);   // 1 AO, 2 RT shadows, 3 GI, 4 refl, 5 refl weight, 6 normals, 7 roughness
        o.Apply("rtshadows", m_Renderer.RayShadows().settings.enabled);
        o.Apply("rtreflections", m_Renderer.RayReflections().settings.enabled);
        o.Apply("gi", m_Renderer.Gi().settings.enabled);
        o.Apply("gibudget", m_Renderer.Gi().settings.probeBudget);
        o.Apply("girays", m_Renderer.Gi().settings.raysPerProbe);
        auto& fog = m_Renderer.Fog().settings;
        o.Apply("fog", fog.enabled);
        o.Apply("fogdensity", fog.density);
        o.Apply("fogheight", fog.heightFalloff);
        auto& volumes = m_Renderer.Volumes().settings;
        o.Apply("clouds", volumes.clouds.enabled);
        auto& reference = m_Renderer.Reference().settings;
        o.Apply("refbounces", reference.maxBounces);
        o.Apply("refspp", reference.samplesPerFrame);
        o.Apply("refmax", reference.maxSamples);
        o.Apply("refclamp", reference.clamp);
        o.Apply("gpudriven", m_Renderer.GpuDriven().settings.enabled);
        o.Apply("occlusion", m_Renderer.GpuDriven().settings.occlusionCulling);
        o.Apply("meshlets", m_Renderer.GpuDriven().settings.meshlets);
        o.Apply("lod", m_Renderer.GpuDriven().settings.clusterLod);
        o.Apply("lodpixels", m_Renderer.GpuDriven().settings.lodErrorPixels);
        o.Apply("coverage", volumes.clouds.coverage);
        o.Apply("edit", m_Edit);
        o.Apply("cam", m_Camera);
        o.Apply("eyex", m_Eye.x);
        o.Apply("eyey", m_Eye.y);
        o.Apply("eyez", m_Eye.z);
        o.Apply("yaw", m_Yaw);
        o.Apply("pitch", m_Pitch);
        bool smoke = false;
        if (o.Apply("smoke", smoke) && smoke)
        {
            // A glowing smoke ball in front of the camera's start, for testing.
            hyper::MediaVolume v;
            v.centre = m_Interior ? dm::float3(0.f, 2.f, 0.f) : dm::float3(0.f);
            v.halfExtent = dm::float3(m_Interior ? 1.5f : 0.5f);
            v.density = m_Interior ? 0.5f : 4.f;
            v.albedo = dm::float3(0.5f);
            v.emission = dm::float3(0.05f, 0.015f, 0.003f);
            v.noiseScale = m_Interior ? 3.f : 8.f;
            volumes.volumes.push_back(v);
        }
        o.WarnUnused();
    }

    // --set eyex= eyey= eyez= yaw= pitch=, if given; else the file's camera (--set cam=N, the
    // first by default) if it has one; inside for interiors like Sponza (down the nave, at
    // eye height); otherwise an outside view of the whole bounds.
    void PlaceCamera()
    {
        auto& camera = Ctx().camera;
        if (m_Eye.x > -1e29f)
        {
            const float yaw = dm::radians(m_Yaw), pitch = dm::radians(m_Pitch);
            const float3 dir(std::cos(pitch) * std::sin(yaw), std::sin(pitch), std::cos(pitch) * std::cos(yaw));
            camera.fly.LookAt(m_Eye, m_Eye + dir);
            camera.fly.SetMoveSpeed(std::max(length(m_Scene.Bounds().diagonal()) * 0.15f, 1.f));
            return;
        }
        const auto& cameras = m_Scene.Graph()->GetCameras();
        if (m_Camera > 0 && size_t(m_Camera) < cameras.size())
        {
            // As Scene::UseSceneCamera, for another of the file's cameras.
            const dm::affine3 viewToWorld = cameras[size_t(m_Camera)]->GetViewToWorldMatrix();
            const float3 position = viewToWorld.m_translation;
            camera.fly.LookAt(position, position + viewToWorld.m_linear.row2);
            if (auto perspective = std::dynamic_pointer_cast<engine::PerspectiveCamera>(cameras[size_t(m_Camera)]))
                camera.verticalFov = perspective->verticalFov;
            camera.fly.SetMoveSpeed(std::max(length(m_Scene.Bounds().diagonal()) * 0.15f, 1.f));
            return;
        }
        if (m_Scene.UseSceneCamera(camera))
            return;
        if (!m_Interior)
        {
            m_Scene.FrameCamera(camera);
            return;
        }
        const box3 b = m_Scene.Bounds();
        const float3 size = b.diagonal();
        const float3 from(b.m_mins.x + size.x * 0.12f, b.m_mins.y + size.y * 0.12f, b.center().z);
        const float3 to(b.m_maxs.x, b.m_mins.y + size.y * 0.35f, b.center().z);
        camera.fly.LookAt(from, to);
        camera.fly.SetMoveSpeed(std::max(size.x * 0.1f, 1.f));
    }
};

int main(int argc, const char* const* argv)
{
    hyper::AppDesc desc;
    desc.title = "hyper: scene";
    desc.shaderDir = "scene";
    return hyper::Run<SceneApp>(argc, argv, desc);
}
