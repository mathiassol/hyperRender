#pragma once

// Level 3/2 entry point: a window, a device and the shared services every
// pass needs. Nothing here hides the layers below it -- App is a Donut
// IRenderPass, Context hands out the nvrhi::IDevice, and hyper/native.h
// reaches the raw Vulkan objects.

#include <hyper/platform.h>

#include <hyper/Camera.h>
#include <hyper/Profiler.h>
#include <hyper/ShaderHotReload.h>

#include <donut/app/DeviceManager.h>
#include <donut/core/vfs/VFS.h>
#include <donut/engine/BindingCache.h>
#include <donut/engine/CommonRenderPasses.h>
#include <donut/engine/ShaderFactory.h>
#include <nvrhi/nvrhi.h>

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace hyper
{
    struct AppDesc
    {
        std::string title = "hyperRender";
        uint32_t width = 1600;
        uint32_t height = 900;
        bool vsync = true;
#ifdef HYPER_DEBUG
        bool debug = true;   // Vulkan validation layers + NVRHI validation
#else
        bool debug = false;
#endif
        // Folder under bin/shaders/ holding this app's compiled shaders
        // (the NAME given to hyper_add_shaders). Mounted at /shaders/app.
        std::string shaderDir;
        // Tweak anything Donut exposes before the device is created:
        // extra Vulkan extensions, depth format, queues, ray tracing...
        std::function<void(donut::app::DeviceCreationParameters&)> configure;
    };

    // Services shared by every pass. Plain members, no getters: take what you need.
    struct Context
    {
        donut::app::DeviceManager* deviceManager = nullptr;
        nvrhi::IDevice* device = nullptr;
        nvrhi::GraphicsAPI api = nvrhi::GraphicsAPI::VULKAN;
        bool rayQuery = false;           // inline ray tracing (acceleration structures + RayQuery) is available
        std::vector<std::string> args;   // command line, without argv[0]; hyper's own flags included
        bool capturing = false;          // --capture: load synchronously, fixed timestep, then exit
        bool asyncLoad = false;          // --async-load: a capture loads as interactively, textures last (a cold disk's order)
        uint32_t loading = 0;            // scenes still loading or streaming textures; captures wait for 0
        bool showUi = true;              // F1 toggles: the hyper overlay and the app's BuildUI()

        // /shaders/donut, /shaders/hyper -> the libraries' shaders, /shaders/app -> AppDesc::shaderDir.
        std::shared_ptr<donut::vfs::RootFileSystem> fs;
        std::shared_ptr<donut::engine::ShaderFactory> shaders;
        std::shared_ptr<donut::engine::CommonRenderPasses> common;
        // Cleared by hyper whenever the back buffer resizes.
        std::shared_ptr<donut::engine::BindingCache> bindingCache;

        std::shared_ptr<Profiler> profiler;
        std::shared_ptr<ShaderHotReload> hotReload;
        Camera camera;

        // Use these rather than Animate's argument: under --capture they advance
        // by a fixed 1/60 s per frame, so captures don't depend on timing.
        float time = 0.f;         // seconds since start
        float deltaTime = 0.f;    // seconds since the previous frame
        uint64_t frameIndex = 0;
    };

    // Subclass this for a project. Override Donut's IRenderPass hooks directly
    // (Render, Animate, BackBufferResized, KeyboardUpdate, ...) plus the ones below.
    class App : public donut::app::IRenderPass
    {
    public:
        explicit App(Context& ctx) : IRenderPass(ctx.deviceManager), m_Ctx(ctx) { }

        // Keep animating while another window (your editor) has focus.
        bool ShouldAnimateUnfocused() override { return true; }

        // Called once after the device exists. Return false to exit.
        virtual bool Init() { return true; }

        // Called after a successful shader hot reload, before the frame renders.
        // Recreate shaders and pipelines here; resources can stay.
        virtual void ShadersReloaded() { }

        // ImGui calls go here; runs every frame inside the UI pass.
        virtual void BuildUI() { }

        Context& Ctx() { return m_Ctx; }
        const Context& Ctx() const { return m_Ctx; }

    private:
        Context& m_Ctx;
    };

    using AppFactory = std::function<std::unique_ptr<App>(Context&)>;

    // Creates the window and device, runs the loop, tears everything down.
    //
    // Command line:
    //   -dx12                      D3D12 (when built with HYPER_WITH_DX12)
    //   --no-vsync                 present unthrottled (also implied by --capture)
    //   --capture <file.png>       save frame N (without UI) and exit; the exit code
    //   [--frames N]               is 2 if errors or validation messages were logged
    //
    // Keys: F1 hides / shows the UI, F12 screenshot to bin/screenshots/, F5 force shader rebuild.
    int Run(int argc, const char* const* argv, const AppDesc& desc, const AppFactory& factory);

    template <typename T>
    int Run(int argc, const char* const* argv, const AppDesc& desc)
    {
        return Run(argc, argv, desc, [](Context& ctx) { return std::make_unique<T>(ctx); });
    }
}
