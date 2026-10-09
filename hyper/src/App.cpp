#include <hyper/App.h>
#include <hyper/native.h>

#include <donut/app/ApplicationBase.h>
#include <donut/app/imgui_renderer.h>
#include <donut/core/log.h>
#include <donut/engine/TextureCache.h>

#include <GLFW/glfw3.h>
#include <imgui.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

using namespace donut;

namespace hyper
{
    namespace
    {
        // Every shader set compiled into a folder (hyper's modules share one: hotreload-<module>.txt).
        void AddManifests(ShaderHotReload& hotReload, const std::filesystem::path& folder)
        {
            std::error_code ec;
            for (const auto& entry : std::filesystem::directory_iterator(folder, ec))
                if (entry.path().extension() == ".txt" && entry.path().stem().string().rfind("hotreload", 0) == 0)
                    hotReload.AddManifest(entry.path());
        }

        // Errors, and every Vulkan validation message, logged so far. --capture
        // turns a non-zero count into a failing exit code.
        std::atomic<int> g_Problems{ 0 };

        void InstallLogCounter()
        {
            log::Callback previous = log::GetCallback();
            log::SetCallback([previous](log::Severity severity, const char* message)
            {
                if (severity >= log::Severity::Error ||
                    (severity == log::Severity::Warning && std::strstr(message, "[Vulkan") != nullptr))
                    ++g_Problems;
                previous(severity, message);
            });
        }

        // First pass in the list: advances time, runs the profiler frame,
        // hot reload and the camera. Input reaches it last, after ImGui and the app.
        class CorePass : public app::IRenderPass
        {
        public:
            CorePass(Context& ctx, App& app, bool fixedTimestep)
                : IRenderPass(ctx.deviceManager), m_Ctx(ctx), m_App(app), m_FixedTimestep(fixedTimestep) { }

            bool ShouldAnimateUnfocused() override { return true; }
            bool ShouldRenderUnfocused() override { return true; }   // keep drawing while you edit shaders

            void Animate(float realDt) override
            {
                // A capture's clock starts when its scenes are in, however long they stream,
                // so a streaming capture shows the same moment as a synchronous one.
                const float dt = m_FixedTimestep ? (m_Ctx.loading ? 0.f : 1.f / 60.f) : realDt;
                m_Ctx.deltaTime = dt;
                m_Ctx.time += dt;
                // (The frame index seeds noise and jitter: a capture's waits too, so how long
                // the load took never shows in the image.)
                if (!m_FixedTimestep || !m_Ctx.loading)
                    ++m_Ctx.frameIndex;
                m_Ctx.profiler->BeginFrame();

                if (m_Ctx.hotReload->Poll())
                {
                    m_Ctx.shaders->ClearCache();
                    m_App.ShadersReloaded();
                }

                const bool focused = glfwGetWindowAttrib(m_Ctx.deviceManager->GetWindow(), GLFW_FOCUSED) != 0;
                if (m_Ctx.camera.enableInput && focused)
                    m_Ctx.camera.fly.Animate(dt);
            }

            void BackBufferResizing() override { m_Ctx.bindingCache->Clear(); }

            bool KeyboardUpdate(int key, int scancode, int action, int mods) override
            {
                if (key == GLFW_KEY_F5 && action == GLFW_PRESS)
                {
                    m_Ctx.hotReload->RequestRebuild();
                    return true;
                }
                if (key == GLFW_KEY_F1 && action == GLFW_PRESS)
                {
                    m_Ctx.showUi = !m_Ctx.showUi;
                    return true;
                }
                if (m_Ctx.camera.enableInput)
                    m_Ctx.camera.fly.KeyboardUpdate(key, scancode, action, mods);
                return false;
            }
            bool MousePosUpdate(double x, double y) override
            {
                m_Ctx.camera.fly.MousePosUpdate(x, y);
                return false;
            }
            bool MouseButtonUpdate(int button, int action, int mods) override
            {
                if (m_Ctx.camera.enableInput)
                    m_Ctx.camera.fly.MouseButtonUpdate(button, action, mods);
                return false;
            }
            bool MouseScrollUpdate(double dx, double dy) override
            {
                if (m_Ctx.camera.enableInput)
                    m_Ctx.camera.fly.MouseScrollUpdate(dx, dy);
                return false;
            }

        private:
            Context& m_Ctx;
            App& m_App;
            bool m_FixedTimestep;
        };

        // Saves the back buffer after the app rendered and before the UI:
        // on F12, and once at frame N for --capture (then exits).
        class CapturePass : public app::IRenderPass
        {
        public:
            CapturePass(Context& ctx, std::string captureFile, int captureFrames, std::filesystem::path screenshotDir, std::string appName)
                : IRenderPass(ctx.deviceManager), m_Ctx(ctx), m_CaptureFile(std::move(captureFile)), m_FramesLeft(captureFrames)
                , m_ScreenshotDir(std::move(screenshotDir)), m_AppName(std::move(appName)) { }

            bool SupportsDepthBuffer() override { return false; }

            bool KeyboardUpdate(int key, int, int action, int) override
            {
                if (key != GLFW_KEY_F12 || action != GLFW_PRESS)
                    return false;

                char stamp[32];
                std::time_t t = std::time(nullptr);
                std::tm tm{};
#ifdef _WIN32
                localtime_s(&tm, &t);
#else
                localtime_r(&t, &tm);
#endif
                std::strftime(stamp, sizeof(stamp), "%Y%m%d-%H%M%S", &tm);
                std::error_code ec;
                std::filesystem::create_directories(m_ScreenshotDir, ec);
                m_Screenshot = (m_ScreenshotDir / (m_AppName + "-" + stamp + ".png")).string();
                return true;
            }

            void Render(nvrhi::IFramebuffer*) override
            {
                if (!m_Screenshot.empty())
                {
                    Save(m_Screenshot);
                    m_Screenshot.clear();
                }

                // Frames count once everything has loaded (see CorePass::Animate).
                if (m_CaptureFile.empty() || m_Ctx.loading || --m_FramesLeft != 0)
                    return;

                Save(m_CaptureFile);
                glfwSetWindowShouldClose(m_Ctx.deviceManager->GetWindow(), GLFW_TRUE);
            }

        private:
            Context& m_Ctx;
            std::string m_CaptureFile;
            int m_FramesLeft;
            std::filesystem::path m_ScreenshotDir;
            std::string m_AppName;
            std::string m_Screenshot;

            void Save(const std::string& file)
            {
                auto* backBuffer = m_Ctx.deviceManager->GetCurrentBackBuffer();
                if (engine::SaveTextureToFile(m_Ctx.device, m_Ctx.common.get(), backBuffer,
                                              nvrhi::ResourceStates::Present, file.c_str(), false))
                    log::info("hyper: saved %s", file.c_str());
                else
                    log::error("hyper: saving %s failed", file.c_str());
            }
        };

        // The hyper overlay (device, frame time, GPU scopes, shader status) plus the app's BuildUI().
        class UiPass : public app::ImGui_Renderer
        {
        public:
            UiPass(Context& ctx, App& app) : ImGui_Renderer(ctx.deviceManager), m_Ctx(ctx), m_App(app) { }

        protected:
            void buildUI() override
            {
                if (!m_Ctx.showUi)
                    return;
                ImGui::SetNextWindowPos(ImVec2(10.f, 10.f), ImGuiCond_FirstUseEver);
                ImGui::Begin("hyper", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
                ImGui::TextUnformatted(m_Ctx.deviceManager->GetRendererString());
                const double frameTime = m_Ctx.deviceManager->GetAverageFrameTimeSeconds();
                if (frameTime > 0.0)
                    ImGui::Text("%.2f ms  (%.0f fps)", frameTime * 1e3, 1.0 / frameTime);

                m_Ctx.profiler->BuildUI();
                if (m_Ctx.deviceManager->GetDeviceParams().vsyncEnabled)
                    ImGui::TextDisabled("vsync on: the first pass that draws to a colour target\nalso counts the wait for the next image (--no-vsync to profile)");
                ShaderStatus();
                ImGui::End();

                m_App.BuildUI();
            }

        private:
            Context& m_Ctx;
            App& m_App;

            void ShaderStatus()
            {
                const ShaderHotReload& hr = *m_Ctx.hotReload;
                if (!hr.IsEnabled())
                    return;

                if (hr.IsBusy())
                {
                    ImGui::TextUnformatted("shaders: compiling...");
                    return;
                }

                const std::string error = hr.LastError();
                if (error.empty())
                {
                    ImGui::TextDisabled("shaders: live (F5 rebuilds)");
                    return;
                }

                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.f, 0.35f, 0.3f, 1.f));
                ImGui::TextUnformatted("shaders: compile failed, running the previous ones");
                ImGui::PushTextWrapPos(ImGui::GetFontSize() * 50.f);
                ImGui::TextUnformatted(error.c_str());
                ImGui::PopTextWrapPos();
                ImGui::PopStyleColor();
            }
        };

        // Donut's CommonRenderPasses gives its placeholder depth textures (bound as
        // the shadow map when a light has none) a *permanent* ShaderResource state.
        // On Vulkan that puts them in SHADER_READ_ONLY_OPTIMAL while NVRHI's depth
        // SRV descriptors expect DEPTH_STENCIL_READ_ONLY_OPTIMAL, which validation
        // flags on every submit. Replace them with normally tracked equivalents.
        void ReplaceBlackDepthTextures(Context& ctx)
        {
            auto make = [&](nvrhi::TextureDimension dim, const char* name)
            {
                return ctx.device->createTexture(nvrhi::TextureDesc()
                    .setDebugName(name)
                    .setDimension(dim)
                    .setFormat(nvrhi::Format::D24S8)
                    .setIsRenderTarget(true)
                    .setIsTypeless(true)
                    .setClearValue(nvrhi::Color(0.f))
                    .setInitialState(nvrhi::ResourceStates::ShaderResource)
                    .setKeepInitialState(true));
            };
            nvrhi::TextureHandle depth = make(nvrhi::TextureDimension::Texture2D, "hyper/black-depth");
            nvrhi::TextureHandle depthArray = make(nvrhi::TextureDimension::Texture2DArray, "hyper/black-depth-array");

            nvrhi::CommandListHandle cmd = ctx.device->createCommandList();
            cmd->open();
            cmd->clearDepthStencilTexture(depth, nvrhi::AllSubresources, true, 0.f, true, 0);
            cmd->clearDepthStencilTexture(depthArray, nvrhi::AllSubresources, true, 0.f, true, 0);
            cmd->close();
            ctx.device->executeCommandList(cmd);

            ctx.common->m_BlackDepthStencilTexture = depth;
            ctx.common->m_BlackDepthStencilTexture2DArray = depthArray;
        }

        struct Args
        {
            nvrhi::GraphicsAPI api = nvrhi::GraphicsAPI::VULKAN;
            std::string captureFile;
            int captureFrames = 60;
            bool noVsync = false;
            int width = 0, height = 0;      // --size WxH: the window's client size, instead of the app's
            bool asyncLoad = false;
        };

        Args ParseArgs(int argc, const char* const* argv)
        {
            Args args;
            for (int i = 1; i < argc; ++i)
            {
                const bool hasValue = i + 1 < argc;
#if HYPER_WITH_DX12
                if (std::strcmp(argv[i], "-dx12") == 0)
                    args.api = nvrhi::GraphicsAPI::D3D12;
#endif
                if (std::strcmp(argv[i], "--capture") == 0 && hasValue)
                    args.captureFile = argv[++i];
                else if (std::strcmp(argv[i], "--frames") == 0 && hasValue)
                    args.captureFrames = std::max(1, std::atoi(argv[++i]));
                else if (std::strcmp(argv[i], "--no-vsync") == 0)
                    args.noVsync = true;
                else if (std::strcmp(argv[i], "--async-load") == 0)
                    args.asyncLoad = true;
                else if (std::strcmp(argv[i], "--size") == 0 && hasValue)
                {
                    int w = 0, h = 0;
                    if (std::sscanf(argv[++i], "%dx%d", &w, &h) == 2 && w > 0 && h > 0)
                    {
                        args.width = w;
                        args.height = h;
                    }
                }
            }
            return args;
        }
    }

    int Run(int argc, const char* const* argv, const AppDesc& desc, const AppFactory& factory)
    {
        log::ConsoleApplicationMode();
        std::setvbuf(stdout, nullptr, _IONBF, 0);   // logs stay readable when piped or the app is killed
        InstallLogCounter();

        const Args args = ParseArgs(argc, argv);
        const bool capturing = !args.captureFile.empty();

        Context ctx;
        ctx.api = args.api;
        for (int i = 1; i < argc; ++i)
            ctx.args.emplace_back(argv[i]);
        ctx.capturing = capturing;
        ctx.asyncLoad = args.asyncLoad;
        ctx.deviceManager = app::DeviceManager::Create(ctx.api);

        app::DeviceCreationParameters params;
        params.backBufferWidth = args.width ? args.width : desc.width;
        params.backBufferHeight = args.height ? args.height : desc.height;
        // Captures run unthrottled: vsync only adds waiting (and that waiting shows up in GPU timers).
        params.vsyncEnabled = desc.vsync && !args.noVsync && !capturing;
        params.enableDebugRuntime = desc.debug;
        params.enableNvrhiValidationLayer = desc.debug;
        params.enableRayTracingExtensions = true;   // enabled when the GPU has them
        // Features Donut doesn't switch on: demote-to-helper (what clip() compiles to for
        // Vulkan 1.3; required by 1.3) and draw-indirect-count (GPU-driven drawing).
        params.deviceCreateInfoCallback = [](VkDeviceCreateInfo& info)
        {
            for (auto* s = static_cast<VkBaseOutStructure*>(const_cast<void*>(info.pNext)); s; s = s->pNext)
            {
                if (s->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES)
                    reinterpret_cast<VkPhysicalDeviceVulkan13Features*>(s)->shaderDemoteToHelperInvocation = VK_TRUE;
                if (s->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES)
                    reinterpret_cast<VkPhysicalDeviceVulkan12Features*>(s)->drawIndirectCount = VK_TRUE;
            }
        };
        params.enablePerMonitorDPI = true;
        if (desc.configure)
            desc.configure(params);

        if (!ctx.deviceManager->CreateWindowDeviceAndSwapChain(params, desc.title.c_str()))
        {
            log::fatal("hyper: cannot create a graphics device");
            delete ctx.deviceManager;
            return 1;
        }

        ctx.device = ctx.deviceManager->GetDevice();
        {
            // The OS may not give the size asked for (a small or remote screen clamps it).
            int w = 0, h = 0;
            ctx.deviceManager->GetWindowDimensions(w, h);
            const int askedW = args.width ? args.width : int(desc.width), askedH = args.height ? args.height : int(desc.height);
            if (w != askedW || h != askedH)
                log::info("hyper: window %dx%d (asked for %dx%d; the screen limits it)", w, h, askedW, askedH);
            else
                log::info("hyper: window %dx%d", w, h);
        }
        ctx.rayQuery = ctx.device->queryFeatureSupport(nvrhi::Feature::RayTracingAccelStruct)
            && ctx.device->queryFeatureSupport(nvrhi::Feature::RayQuery);
        log::info("hyper: ray queries %s", ctx.rayQuery ? "available" : "not available");

        const std::filesystem::path binDir = app::GetDirectoryWithExecutable();
        const std::filesystem::path shaderRoot = binDir / "shaders";
        const char* apiDir = app::GetShaderTypeName(ctx.api);
        ctx.fs = std::make_shared<vfs::RootFileSystem>();
        ctx.fs->mount("/shaders/donut", shaderRoot / "framework" / apiDir);
        ctx.fs->mount("/shaders/hyper", shaderRoot / "hyper" / apiDir);

        ctx.hotReload = std::make_shared<ShaderHotReload>();
        if (!capturing)
            AddManifests(*ctx.hotReload, shaderRoot / "hyper" / apiDir);
        if (!desc.shaderDir.empty())
        {
            const std::filesystem::path appShaders = shaderRoot / desc.shaderDir / apiDir;
            ctx.fs->mount("/shaders/app", appShaders);
            if (!capturing)
                AddManifests(*ctx.hotReload, appShaders);
        }

        ctx.shaders = std::make_shared<engine::ShaderFactory>(ctx.device, ctx.fs, "/shaders");
        ctx.common = std::make_shared<engine::CommonRenderPasses>(ctx.device, ctx.shaders);
        ReplaceBlackDepthTextures(ctx);
        ctx.bindingCache = std::make_shared<engine::BindingCache>(ctx.device);
        ctx.profiler = std::make_shared<Profiler>(ctx.device);

        const std::string appName = argc > 0 ? std::filesystem::path(argv[0]).stem().string() : "app";
        const std::string imguiIni = (binDir / (appName + ".imgui.ini")).string();

        int result = 0;
        {
            std::unique_ptr<App> application = factory(ctx);
            CorePass core(ctx, *application, capturing);
            CapturePass capture(ctx, args.captureFile, args.captureFrames, binDir / "screenshots", appName);
            UiPass ui(ctx, *application);

            if (application->Init() && ui.Init(ctx.shaders))
            {
                // Keep window layout next to the exe, not in whatever folder the app was started from.
                ImGui::GetIO().IniFilename = imguiIni.c_str();

                // Render order: core, app, capture (scene only), UI.
                // Input goes the other way: UI, capture, app, core/camera.
                ctx.deviceManager->AddRenderPassToBack(&core);
                ctx.deviceManager->AddRenderPassToBack(application.get());
                ctx.deviceManager->AddRenderPassToBack(&capture);
                ctx.deviceManager->AddRenderPassToBack(&ui);

                ctx.deviceManager->RunMessageLoop();

                ctx.deviceManager->RemoveRenderPass(&ui);
                ctx.deviceManager->RemoveRenderPass(&capture);
                ctx.deviceManager->RemoveRenderPass(application.get());
                ctx.deviceManager->RemoveRenderPass(&core);
            }
            else
            {
                log::error("hyper: app initialisation failed");
                result = 1;
            }

            ctx.device->waitForIdle();

            if (capturing)
                log::info("hyper: timings\n%s", ctx.profiler->Report().c_str());
        }

        if (capturing && result == 0 && g_Problems > 0)
        {
            std::fprintf(stderr, "hyper: %d error(s) or validation message(s) during capture\n", g_Problems.load());
            result = 2;
        }

        ctx.profiler.reset();
        ctx.bindingCache.reset();
        ctx.common.reset();
        ctx.shaders.reset();
        ctx.deviceManager->Shutdown();
        delete ctx.deviceManager;
        return result;
    }
}
