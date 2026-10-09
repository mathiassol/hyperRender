#pragma once

#include <hyper/platform.h>

#include <donut/core/math/math.h>
#include <nvrhi/nvrhi.h>

#include <atomic>
#include <filesystem>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace donut::engine
{
    struct Material;
    class Scene;
    class SceneGraph;
    class TextureCache;
    class DirectionalLight;
    class DescriptorTableManager;
}

namespace hyper
{
    struct Context;
    struct Camera;

    // A glTF 2.0 (.gltf/.glb) or Donut .scene.json scene: meshes, materials,
    // textures, lights, cameras, animations. A thin layer over Donut's Scene
    // that loads on a background thread and streams textures in afterwards.
    //
    //     hyper::Scene scene(ctx);
    //     scene.Load(hyper::AssetPath("downloaded/Sponza/glTF/Sponza.gltf"));
    //     ...each frame:
    //     scene.Update(ctx.deltaTime);          // before the frame's command list
    //     scene.RefreshBuffers(cmd);            // inside it, before drawing
    //
    // Under --capture, Load() is synchronous and all textures are uploaded
    // before the first frame, so captures are deterministic.
    //
    // On Vulkan and D3D12 the scene is also bindless: every mesh buffer
    // and texture has an index in DescriptorTable() (layout: BindlessLayout(),
    // raw buffers at binding 0, textures at binding 1), which Donut's geometry
    // and material buffers refer to. hyper/rt.hlsli and the GPU-driven passes read them.
    class Scene
    {
    public:
        enum class State { Empty, Loading, Ready, Failed };

        explicit Scene(Context& ctx);
        ~Scene();
        Scene(const Scene&) = delete;
        Scene& operator=(const Scene&) = delete;

        void Load(const std::filesystem::path& file);

        // Finishes loading, streams textures, animates, refreshes transforms.
        void Update(float deltaTime);
        // Uploads changed instance/material/geometry data. Needs an open command list.
        void RefreshBuffers(nvrhi::ICommandList* cmd);

        [[nodiscard]] State GetState() const { return m_State; }
        [[nodiscard]] bool IsReady() const { return m_State == State::Ready; }
        // Textures loaded / requested, for a loading bar.
        [[nodiscard]] float TextureProgress() const;

        [[nodiscard]] donut::engine::Scene* Get() const { return m_Scene.get(); }
        [[nodiscard]] std::shared_ptr<donut::engine::SceneGraph> Graph() const;
        [[nodiscard]] donut::math::box3 Bounds() const;

        // The scene's first directional light; one is added if the file has none.
        [[nodiscard]] donut::engine::DirectionalLight& Sun() const { return *m_Sun; }
        // Points the sun (the direction its light travels). Takes effect this frame even
        // after Update(), which has already refreshed the graph: shadows, sky and IBL read it.
        void SetSunDirection(const donut::math::double3& direction);

        // Places the fly camera so the whole scene is in view.
        void FrameCamera(Camera& camera) const;
        // Moves the fly camera to the file's first camera; false if it has none.
        bool UseSceneCamera(Camera& camera) const;

        // Bindless tables (Vulkan / D3D12; null on D3D11).
        [[nodiscard]] nvrhi::IBindingLayout* BindlessLayout() const;
        [[nodiscard]] nvrhi::IDescriptorTable* DescriptorTable() const;
        // Increases with every successful load, so dependents know to rebuild.
        [[nodiscard]] uint32_t Generation() const { return m_Generation; }
        // Increases whenever what's in the scene changes: a load, or nodes attached to or
        // detached from the graph (Update notices). GpuScene and RtScene follow it,
        // adding and dropping only what changed.
        [[nodiscard]] uint32_t Revision() const { return m_Revision; }

        // The app's own data on each material, from the glTF material's extras:
        //     "extras": { "hyper": { "data": 3, "rays": false } }
        // data: a uint whose meaning is the app's (0 without one), by material ID, in
        // MaterialDataBuffer() (StructuredBuffer<uint>; hyper/rt.hlsli: MaterialData(id)).
        // rays: false hides the material from ray tracing (RtScene masks instances whose
        // materials are all hidden).
        [[nodiscard]] nvrhi::IBuffer* MaterialDataBuffer() const { return m_MaterialDataBuffer; }
        [[nodiscard]] bool SeenByRays(uint32_t materialID) const
        {
            return materialID >= m_MaterialRays.size() || m_MaterialRays[materialID] != 0;
        }
        // Nothing animates or streams in: an image accumulated over frames stays valid.
        [[nodiscard]] bool IsStatic() const;

        bool animate = true;
        float animationTime = 0.f;

    private:
        Context& m_Ctx;
        std::shared_ptr<donut::engine::TextureCache> m_TextureCache;
        nvrhi::BindingLayoutHandle m_BindlessLayout;
        std::shared_ptr<donut::engine::DescriptorTableManager> m_DescriptorTable;
        uint32_t m_Generation = 0;
        uint32_t m_Revision = 0;
        std::shared_ptr<donut::engine::Scene> m_Scene;
        std::shared_ptr<donut::engine::Scene> m_Pending;   // written by the loader thread
        std::shared_ptr<donut::engine::DirectionalLight> m_Sun;
        std::unique_ptr<std::thread> m_Loader;
        std::atomic<bool> m_LoaderDone{ false };
        std::atomic<bool> m_LoaderOk{ false };
        State m_State = State::Empty;
        bool m_TexturesFinished = false;
        // Materials with textures still streaming in, and how many of them had arrived
        // when each was last written.
        std::vector<std::pair<std::shared_ptr<donut::engine::Material>, uint32_t>> m_Streaming;
        uint32_t m_ReadyUpdates = 0;     // a streaming capture's updates since the scene was in
        bool m_Counted = false;          // counted in Context::loading
        std::filesystem::path m_File;
        std::unordered_map<std::string, std::pair<uint32_t, bool>> m_Extras;   // material name -> data, seen by rays
        std::vector<uint32_t> m_MaterialData;   // by material ID
        std::vector<uint8_t> m_MaterialRays;    // by material ID; 0: hidden from rays
        nvrhi::BufferHandle m_MaterialDataBuffer;
        bool m_MaterialDataDirty = false;

        void OnLoaded();
        void ReadMaterialExtras();
        void MapMaterialData();
        void RefreshStreamedMaterials();
        void DoneLoading();
    };
}
