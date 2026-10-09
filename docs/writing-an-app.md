# Writing an app

This page builds a small app: a compute shader that writes colour bands into an HDR
texture, tone-mapped by the `post` module, with a setting in the UI and on the command
line. Then it shows where to go from there: scenes, modules, ray tracing, raw NVRHI.

The app can live in this repository (`apps/myapp/`, added to `apps/CMakeLists.txt`) or in
your own ([Using hyperRender in your project](using-in-your-project.md)). The code is the
same.

## 1. The files

```
myapp/
  CMakeLists.txt
  main.cpp
  shaders/
    shaders.cfg
    waves_cs.hlsl
```

**CMakeLists.txt**

```cmake
add_executable(myapp main.cpp)
target_link_libraries(myapp PRIVATE hyper::hyper hyper::post)
hyper_app(myapp)
hyper_add_shaders(myapp CONFIG "${CMAKE_CURRENT_SOURCE_DIR}/shaders/shaders.cfg" NAME myapp)
```

- `hyper::hyper` is the core; link a module (`hyper::post`, `hyper::gpu`, ...) for each
  feature you use.
- `hyper_app` puts the exe in `bin/` beside the compiled shaders, names the Debug build
  `myapp-debug.exe`, and matches hyper's MSVC runtime.
- `hyper_add_shaders` compiles every entry point listed in `shaders.cfg` at build time, into
  `bin/shaders/myapp/`.

**shaders/shaders.cfg**: one line per entry point, in ShaderMake's format (file, `-T`
stage, `-E` entry, optional `-D` defines):

```
waves_cs.hlsl -T cs -E main
```

## 2. The shader

```hlsl
// Colour bands that drift over time, written into the app's HDR target.
#include <hyper/pass.hlsli>

struct WavesPush
{
    float time;
    float speed;
    float2 pad;
};
DECLARE_PUSH_CONSTANTS(WavesPush, g_Push, 3, 0);

VK_IMAGE_FORMAT("rgba16f") RWTexture2D<float4> u_Color : register(u0);
VK_IMAGE_FORMAT("r32f") RWTexture2D<float> u_Depth : register(u1);

[numthreads(8, 8, 1)]
void main(uint2 id : SV_DispatchThreadID)
{
    if (any(id >= uint2(g_View.viewportSize)))
        return;
    const float2 uv = (float2(id) + 0.5) * g_View.viewportSizeInv;
    const float t = g_Push.time * g_Push.speed;
    const float3 colour = 0.5 + 0.5 * cos(t + uv.xyx * 6.0 + float3(0.0, 2.0, 4.0));
    u_Color[id] = float4(colour, 1.0);
    u_Depth[id] = 0.0;   // reverse-Z: nothing in front (the post stack's TAA reads it)
}
```

`hyper/pass.hlsli` declares the slots every `hyper::ComputePasses` pass shares: the view at
`b0` (`g_View`), the previous frame's at `b1`, push constants at `b3` (up to 32 bytes),
up to ten textures in (`t0`–`t9`) and four out (`u0`–`u3`) per dispatch, and two samplers.
The full slot map is in [architecture.md](architecture.md#3-shaders).

## 3. The app

```cpp
// myapp: a compute pass that fills an HDR target, tone-mapped by the post module.
#include <hyper/hyper.h>
#include <hyper/passes/PostProcess.h>

#include <imgui.h>

struct WavesPush
{
    float time;
    float speed;
    float pad[2];
};

class MyApp : public hyper::App
{
public:
    explicit MyApp(hyper::Context& ctx)
        : App(ctx), m_Post(ctx), m_Passes(ctx, {}, nullptr, nullptr) { }

    bool Init() override
    {
        hyper::Overrides overrides(Ctx().args);   // --set speed=2
        overrides.Apply("speed", m_Speed);
        overrides.WarnUnused();
        m_CommandList = Ctx().device->createCommandList();
        return CreatePipelines();
    }

    void ShadersReloaded() override
    {
        CreatePipelines();
        m_Post.ShadersReloaded();
    }

    void Render(nvrhi::IFramebuffer* framebuffer) override
    {
        auto& ctx = Ctx();
        const auto& info = framebuffer->getFramebufferInfo();
        m_Color.Resize(ctx.device, info.width, info.height);
        m_Depth.Resize(ctx.device, info.width, info.height);
        ctx.camera.GetView(m_View, info.width, info.height);
        m_Post.PrepareView(m_View);

        m_CommandList->open();
        {
            auto scope = ctx.profiler->Measure(m_CommandList, "waves");
            m_Passes.BeginFrame(m_CommandList, m_View, m_View);
            const WavesPush push{ ctx.time, m_Speed, {} };
            m_Passes.Dispatch(m_CommandList, m_Waves, {}, { m_Color, m_Depth }, info.width, info.height, &push, sizeof(push));
        }
        m_Post.Render(m_CommandList, { m_Color, m_Depth }, m_View, framebuffer);
        m_CommandList->close();
        ctx.device->executeCommandList(m_CommandList);
    }

    void BuildUI() override
    {
        ImGui::Begin("myapp", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
        ImGui::SliderFloat("speed", &m_Speed, 0.f, 4.f);
        m_Post.BuildUI();
        ImGui::End();
    }

private:
    hyper::PostProcess m_Post;
    hyper::ComputePasses m_Passes;
    nvrhi::ComputePipelineHandle m_Waves;
    nvrhi::CommandListHandle m_CommandList;
    donut::engine::PlanarView m_View;
    hyper::RenderTarget m_Color{ nvrhi::TextureDesc().setDebugName("myapp/color").setFormat(nvrhi::Format::RGBA16_FLOAT).setIsUAV(true) };
    hyper::RenderTarget m_Depth{ nvrhi::TextureDesc().setDebugName("myapp/depth").setFormat(nvrhi::Format::R32_FLOAT).setIsUAV(true) };
    float m_Speed = 1.f;

    // Keeps the pipeline it had if the shader doesn't compile (a typo during hot reload).
    bool CreatePipelines()
    {
        return m_Passes.Pipeline(m_Waves, "app/waves_cs", "main", nullptr);
    }
};

int main(int argc, const char* const* argv)
{
    hyper::AppDesc desc;
    desc.title = "myapp";
    desc.shaderDir = "myapp";   // the NAME given to hyper_add_shaders
    return hyper::Run<MyApp>(argc, argv, desc);
}
```

Build it and run `myapp.exe`. Edit `waves_cs.hlsl` and save: the app recompiles it and
`ShadersReloaded()` swaps the pipeline in. `myapp.exe --set speed=3 --capture out.png`
renders 60 frames with the speed changed, saves the image and exits.

## 4. What each part does

**The app's life.** `hyper::Run<T>` creates the window and the device, constructs your
app, then calls:

| Hook | When | Use it for |
|---|---|---|
| `Init()` | once, after the device exists | pipelines, buffers, loading. Return false to exit. |
| `Animate(float)` | every frame, before rendering | simulation, scene updates. Use `Ctx().time` / `Ctx().deltaTime`, not the argument: under `--capture` they step a fixed 1/60 s. |
| `Render(framebuffer)` | every frame | recording and executing command lists into the swap chain's framebuffer |
| `BuildUI()` | every frame, inside the ImGui frame | ImGui windows; modules have their own `BuildUI()` |
| `ShadersReloaded()` | after a successful hot reload | recreating pipelines (resources can stay) |

`hyper::App` is a Donut `IRenderPass`, so its other hooks (`KeyboardUpdate`,
`MouseButtonUpdate`, `BackBufferResized`, ...) are there to override too.

**The context.** `Ctx()` holds what every pass needs: `device` (the NVRHI device),
`shaders` (Donut's shader factory), `common` (Donut's common passes, samplers and
placeholder textures), `bindingCache`, `profiler`, `camera` (the fly camera), `time`,
`deltaTime`, `frameIndex`, `rayQuery` (whether the GPU can trace rays), `args` (the command
line) and `capturing`.

**Shaders** are loaded by name, without extension: `"app/waves_cs"` for your own set,
`"hyper/<path>"` for the core's and modules', `"donut/<file>"` for Donut's. Every set can
include `hyper/shaders/include/` and every module's `shaders/include/`. Matrices are
row-major, as Donut's are. A struct shared with C++ goes in a header that the C++ side
includes after `<hyper/shared.h>`, which gives C++ HLSL's type names.

**Passes.** `hyper::ComputePasses` is the quickest way to run a compute shader: one binding
layout, `Pipeline(handle, shader, entry, bindless)` (keeps the old pipeline on a failed
compile), `BeginFrame(cmd, view, previous)` once a frame, and
`Dispatch(cmd, pipeline, { inputs }, { outputs }, width, height, &push, sizeof(push))`.
Its first two constructor arguments add your own shared resources (buffers, textures) at
`t20`–`t39`; the last is an `RtScene` when the passes trace rays. For anything else
(graphics pipelines, your own layouts) use NVRHI directly through `Ctx().device`, with
binding sets from `Ctx().bindingCache`.

**Render targets.** `hyper::RenderTarget` is a texture that follows the window's size:
call `Resize` every frame and use it as an `nvrhi::ITexture*`.

**The post stack.** `hyper::PostProcess` takes your HDR colour and depth (and motion
vectors, if you have them) and does TAA, bloom, exposure and the tone curve into the
framebuffer. Call `PrepareView(view)` before drawing (it jitters the view for TAA).

**Settings from the command line.** `hyper::Overrides` maps `--set name=value` onto your
settings, and `WarnUnused()` logs the ones nobody asked for (typos). Naming the settings
worth testing makes them reachable from captures and the regression suite
([testing.md](testing.md)).

**Profiling.** `auto scope = ctx.profiler->Measure(cmd, "name");` times a scope on the GPU
and the CPU and marks it for graphics debuggers. Scopes nest. The overlay shows them live,
and every capture prints them.

## 5. Drawing a scene

For glTF content, a frame is usually: `hyper::Scene` loads the file; a G-buffer
(`hyper::GBuffer`) is filled, by the `gpu` module's `GpuDrivenRenderer` or Donut's own
pass; lighting modules read it; `post` finishes the frame.

```cpp
hyper::Scene m_Scene{ ctx };
m_Scene.Load(hyper::AssetPath("downloaded/Sponza/glTF/Sponza.gltf"));   // on a thread

void Animate(float) override { m_Scene.Update(Ctx().deltaTime); }       // finishes loading, streams textures
void Render(nvrhi::IFramebuffer* fb) override
{
    if (!m_Scene.IsReady()) return;
    // ... m_Scene.RefreshBuffers(cmd), the G-buffer, lighting, post
}
```

`apps/scene` is the complete version: `SceneRenderer.cpp` wires the G-buffer, shadows,
AO, the atmosphere, IBL, SSR, ray-traced shadows and reflections, DDGI, fog and the post
stack into one deferred frame. It's the place to copy a frame from; trim what you don't
need. [modules.md](modules.md) describes each module's inputs and settings.

`hyper::AssetPath("x")` resolves `x` against the nearest `assets/` folder above the exe.

## 6. Ray tracing

When `Ctx().rayQuery` is true, `hyper::RtScene` builds acceleration structures for a
`hyper::Scene` (a BLAS per mesh, the TLAS each frame). Shaders include `hyper/rt.hlsli`
for `TraceClosest`, `TraceVisible` and `GetHitSurface`, and the `rtlighting` module's
`hyper/rt_lighting.hlsli` to light what a ray hits. Give `ComputePasses` the `RtScene` and
the scene (`BeginFrame(cmd, view, previous, &scene)`) and its passes can trace. The
`traced` module is a worked system on top: bounce light and reflections where the app
decides what a ray sees.

## 7. Going lower

Nothing stops you below hyper's layer: `Ctx().device` is the NVRHI device, Donut's passes
(`<donut/render/...>`) work as they are, and `<hyper/native.h>` gives the raw Vulkan
objects behind NVRHI's (instance, device, images, buffers, command buffers) and
`hyper::vk::RawCommands`, a scope for recording raw Vulkan commands into an open NVRHI
command list. `apps/hello`
shows all three levels in one file.

## 8. Things to know

- A Windows header does `#define hyper __int64`. Every hyper header includes
  `<hyper/platform.h>` first, which includes `<windows.h>` once and takes the name back,
  so the order you include things in doesn't matter. Its `NOMINMAX` keeps `min` / `max`
  from being macros.
- NVRHI places barriers only when the bound binding sets change: two raw dispatches in a
  row with the same set get none. Call `cmd->setResourceStatesForBindingSet(set)` before
  each (`ComputePasses` does it for you).
- Donut's math vectors don't zero themselves: `float3 v;` is garbage. Set every field of a
  struct shared with HLSL, or `memset` it.
- More in [architecture.md, Pitfalls](architecture.md#6-pitfalls).
