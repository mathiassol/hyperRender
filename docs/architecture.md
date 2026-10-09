# Architecture

How hyperRender is put together: the layers, the core, shaders and binding slots, scenes
and rays, and what it does about Donut's behaviour. The modules have their own page,
[modules.md](modules.md); making an app, [writing-an-app.md](writing-an-app.md).

## 1. Shape

```
apps/<name>/          a project: main.cpp, its own passes, shaders/shaders.cfg
   │ links
   ▼
modules/<name>/       features, one library each (hyper::<name>), built only when linked
   │ built on
   ▼
hyper/                the core (hyper::hyper): loop, captures, scene, rays, pass plumbing
   │ built on
   ▼
Donut → NVRHI → Vulkan   (pinned submodule external/donut, never edited)
```

An app can reach every layer. `ctx.device` is the NVRHI device; `hyper/native.h` gives the
raw Vulkan objects behind it; Donut's passes (`donut/render/*`) are there to use directly.

| Level | You write | Through |
|---|---|---|
| L0 | raw Vulkan on hyper's device and command buffer | `hyper/native.h` (`hyper::vk::RawCommands`) |
| L1 | NVRHI pipelines, binding sets, command lists | `Context::device`, `nvrhi::utils` |
| L2 | hyper's core and modules, Donut's passes | `hyper/hyper.h`, module headers, `donut/render/*` |

There's no level above that: a frame is the app's. `apps/scene/SceneRenderer` is a full
deferred frame built from the modules, kept as the example to copy.

## 2. The core

About 3,800 lines of C++ and HLSL in `hyper/`: only what every app needs. It changes
rarely, and only for a bug or for something real projects show is missing; features go in
modules ([CONTRIBUTING.md](../CONTRIBUTING.md)).

| Piece | Files | What it does |
|---|---|---|
| App loop | `App.h`, `App.cpp` | `hyper::Run<T>()`: window, device (Vulkan 1.3; validation in Debug), the frame loop, ImGui, the profiler overlay, hot reload, F1 / F5 / F12, `--capture` |
| Context | `App.h` | what every pass needs: `device`, `shaders`, `common` (Donut's common passes), `bindingCache`, `profiler`, `camera`, `time` / `deltaTime`, `frameIndex`, `rayQuery`, `args`, `capturing`, `loading` |
| Profiler | `Profiler.h/.cpp` | `ctx.profiler->Measure(cmd, "name")`: GPU + CPU timer and debug marker in one scope, nested; shown in the overlay and printed by every capture, marking scopes that ran `(once)`, stopped running, or are `(unsteady)` |
| Hot reload | `ShaderHotReload.h/.cpp`, `Process.*` | watches shader sources, re-runs the build's own ShaderMake command for every set (`hotreload*.txt` beside the compiled shaders), then calls `App::ShadersReloaded()`; on failure the old shaders keep running and the overlay shows the errors |
| Camera | `Camera.h` | fly camera (`ctx.camera`): LMB-drag to look, WASD / QE, Shift / Ctrl |
| Overrides | `Overrides.h/.cpp` | `--set name=value` parsing; each app maps names to its settings and warns about unused ones |
| Render targets | `RenderTarget.h` | a texture that follows the output size (and a scale of it) |
| Readback | `Readback.h/.cpp` | a GPU buffer read back without stalling: copies into a ring of staging buffers, and the newest copy the GPU is certainly done with (three frames old) |
| Assets | `Assets.h/.cpp` | `hyper::AssetPath("downloaded/...")` → `assets/` |
| Raw Vulkan | `native.h` | `hyper::vk::RawCommands`, native handles |
| Shared structs | `shared.h` | HLSL's type names in C++, so one struct header serves both |
| Scene | `scene/Scene.h/.cpp` | glTF / .glb / Donut .scene.json on a background thread, texture streaming, animation and skinning, a guaranteed sun (`SetSunDirection` applies in the same frame), glTF extras for materials, runtime edits (§4) |
| G-buffer | `scene/GBuffer.h/.cpp` | Donut's G-buffer, with float reverse-Z depth (D32S8), plus the HDR target the frame is lit into; `DepthStencil()`, the state for anything drawing into it (reverse-Z, and the stencil bit that gives TAA its motion vectors) |
| Scene for rays | `rt/RtScene.h/.cpp`, `shaders/include/hyper/rt.hlsli` | a BLAS per mesh, the TLAS rebuilt per frame, hit lookups for shaders (§4) |
| Compute passes | `passes/ComputePasses.h/.cpp`, `shaders/include/hyper/pass.hlsli` | one binding layout with fixed slots for an app's own compute passes, and `Dispatch` (§3) |
| Hi-Z | `lighting/DepthPyramid.h/.cpp`, `shaders/lighting/hiz_cs.hlsl` | min / max depth pyramid (used by `gpu` and `ssr`) |
| Shader library | `shaders/include/hyper/*.hlsli` | `common` (bindings, row-major), `color` (luminance, tone curves), `sampling` (RNG, cosine / GGX sampling, `EnvBrdf`), `noise`, `volume` (phase functions, Beer-powder), `view` (camera rays), `cubemap` |


## 3. Shaders

- HLSL, shader model 6.6, compiled at build time by ShaderMake / DXC into
  `bin/shaders/<set>/spirv/`. Each set has a `.cfg` listing entry points and
  defines. Loaded without extension: `"app/<file>"` (the app's set), `"hyper/<path>"` (the
  core and all modules: they compile into one folder under their own names),
  `"donut/<file>"`.
- **Includes**: every set sees `hyper/shaders/include/` and every module's
  `shaders/include/`. Headers cost nothing; only entry points are compiled.
- **Hot reload** re-runs the exact command the build used, so the two can't disagree.
- **Matrices are row-major** (`common.hlsli`), as Donut's are.
- `hyper_add_shaders(... SLANG)` compiles a set with Slang instead (needs
  `SHADERMAKE_SLANG_PATH`).

**Binding slots.** Register numbers that mean the same thing everywhere:

| Slots | Meaning | Declared by |
|---|---|---|
| b0, b1 | this frame's and the previous frame's view | `pass.hlsli` |
| b2 | the app's own constants | the app |
| b3 | 32 bytes of push constants per dispatch | `pass.hlsli` users |
| t0–t9, u0–u3 | one dispatch's textures in and out | `ComputePasses::Dispatch` |
| t20–t39 | the app's shared resources | the app |
| s0, s1 | linear clamp, linear wrap | `pass.hlsli` |
| t40–t44, s40 | TLAS, instances, geometry, materials, material data; wrap sampler | `rt.hlsli`, `RtScene::AddLayoutItems` |
| b50, t50–t55, s50 | hit lighting: constants, probe volume, environment, IBL | `rt_lighting.hlsli`, `RtLighting::AddLayoutItems` |
| t60–t71 | the GPU scene's buffers | `gpu_scene.hlsli`, `GpuScene::AddLayoutItems` |
| set 1 | the scene's bindless table: binding 0 buffers, 1 textures | `Scene::BindlessLayout()` |

**An app's compute passes** go through `hyper::ComputePasses`: the layout above (views,
push constants, t0–t9 / u0–u3, the app's items, the scene for rays when given an
`RtScene`), `Pipeline(handle, "app/file", "entry", bindless)`, which keeps the old pipeline
if the shader doesn't compile, `BeginFrame(cmd, view, previous, &scene)` once a frame (the
scene only when the passes trace), and `Dispatch(cmd, pipeline, { inputs }, { outputs },
width, height, &push, sizeof(push), groupSize)`, which pads unused slots and places a
barrier on the outputs of the dispatch before (NVRHI only does when the bound sets change,
and two dispatches with the same inputs and outputs share a cached set). An app's own
buffers (raw, structured, UAV) go in as its shared items. `hyper/pass.hlsli` declares the same
slots for the shader, with `WorldFromDepth` and `PixelFromWorld`. A struct shared with
C++ goes in a header that C++ includes after `<hyper/shared.h>`.

## 4. Frames, scenes and rays

**Conventions.**
- Scene colour: linear, scene-referred Rec.709, RGBA16_FLOAT.
- Depth: reverse-Z (1 near, 0 = infinitely far), infinite far plane.
- Motion vectors: RG16_FLOAT, pixels from the current frame to the previous one; stencil
  bit 1 marks pixels that have them (`GBuffer::MotionVectorStencilBit`).
- G-buffer (`hyper::GBuffer`, Donut's): depth + stencil, diffuse albedo, specular F0,
  normals (xyz) with roughness (w), emissive with **material ID + 1** in w, motion
  vectors, and `HdrColor` to light into.

**`hyper::Scene`.**
- `Load(path)` runs on a thread; `Update(dt)` each frame finishes it, streams textures,
  animates, and refreshes the graph; `RefreshBuffers(cmd)` uploads what changed. Under
  `--capture` the load is synchronous, unless `--async-load` asks for the interactive
  order (the scene, one frame without textures, then every texture), which is the same on
  every run. `Context::loading` counts scenes still loading; captures wait for 0.
- **Streaming**: a material is rewritten when one of its own textures has been uploaded
  (Donut only writes dirty materials, and never re-dirties them).
- **Material data**: `"extras": { "hyper": { "data": 3, "rays": false } }` on a glTF
  material. `data` is a uint whose meaning is the app's (`MaterialData(id)` in shaders);
  `rays: false` hides the material from ray tracing. Read from the file, because Donut
  drops extras.
- **Edits**: attach or detach Donut scene-graph nodes and `Revision()` moves on.
  `Generation()` moves only on a new load.
- **A sun** always exists (`Sun()`): the file's first directional light, or one hyper
  adds. `UseSceneCamera` / `FrameCamera` place the fly camera.
- **Compressed textures**: Donut loads a `.dds` beside a `.jpg` / `.png` in its place.
  `tools/compress_textures.py` writes them (BC7 sRGB for colour, BC5 for normals, BC7 for
  the rest, with mips): a quarter of the memory, nothing to decode.

**`hyper::RtScene`** (needs `ctx.rayQuery`).
- A BLAS per mesh, built when the mesh enters the scene; skinned meshes are refit when
  Donut re-skins them. The TLAS is rebuilt every frame; the instance ID is Donut's
  instance index. An instance whose materials are all `rays: false` gets mask 0.
- `hyper/rt.hlsli`: `TraceClosest`, `TraceVisible`, `GetHitSurface` (position, normals,
  albedo, roughness, emission, optionally normal-mapped), `MaterialData`,
  `OffsetRayOrigin`. A ray stops at its 25th alpha-tested candidate
  (`HYPER_RT_CUTOUTS`), deep in foliage by then; `HYPER_RT_EXACT` tests them all, for
  rays that place things.



## 5. Donut, and the workarounds for it

hyper uses Donut's device manager, scene graph and glTF loader, texture cache, shader
factory, TAA, cascaded shadows, light probe processing, G-buffer, deferred and forward
passes. Each place hyper works around Donut's behaviour:

| Behaviour | Where hyper handles it |
|---|---|
| A material's constants are written only while it's dirty, and nothing re-dirties it when a texture arrives | `Scene::RefreshStreamedMaterials` |
| glTF extras are dropped | `Scene::ReadMaterialExtras` reads the file itself |
| CPU copies of geometry are freed after upload; its buffers can't be copied from | `GpuScene` gathers with a compute copy and reads back what it needs for meshlets |
| New meshes' buffers are uploaded in the frame's own command list | `GpuScene` adds a mesh the frame after it appears |
| The binding cache can't hold sets with a TLAS under the validation layer | `hyper::RtBindingSets` |
| One `.dds` stands in for both `x.jpg` and `x.png` | `compress_textures.py` leaves such pairs uncompressed |
| Placeholder depth textures have a state that fails validation | replaced in `hyper::Run` |
| The probe convolution supersamples up to 4× its output | `environment` uses 128² specular, 16² diffuse, a pass each |
| The deferred pass adds ambient and probe specular on top of other indirect light | `apps/scene` zeroes the ambient under IBL; `ssr` composites itself |
| `SkyPass` needs a reverse-Z view at construction | `apps/scene` |

## 6. Pitfalls

- **Windows `#define hyper __int64`**: every hyper header includes `<hyper/platform.h>`
  first, which undefines it.
- **GPU timings include vsync waits** (charged to the first pass that draws to a colour
  target): profile with `--no-vsync`; captures turn vsync off themselves.
- **`discard` doesn't stop a pixel shader** with `shaderDemoteToHelperInvocation` on (DXC
  compiles it to a demote): return right after a `discard` that guards memory reads.
- **ShaderMake's include scanner ignores `#ifdef`**: vendored headers mustn't include files
  it can't find.
- **DXC 1.4.357's SPIR-V legalizer** can crash on innocent code: bisect with `dxc -spirv`
  directly (`-fcgl` passing means it's the legalizer) and rephrase the expression.
- **A second command list open during a frame** must be non-immediate; Vulkan NVRHI
  allows one immediate list open at a time.
- **`nvrhi::RefCountPtr` overloads `operator&`**: `&handle` is the raw pointer's address.
  Use `std::addressof` for a pointer to the handle.
- **NVRHI copies 3D textures to staging one slice at a time.**
- **Donut's math vectors don't zero themselves.** `float3 v;` is garbage, and so is
  `s = {}` on a struct of them: set every field (or `memset` a struct shared with HLSL).
- **NVRHI places barriers only when the bound binding sets change.** Two dispatches in a
  row with the same set get none, even if the second reads what the first wrote. Call
  `cmd->setResourceStatesForBindingSet(set)` before each (`ComputePasses` does).
- **Culling off and `SV_IsFrontFace`**: which side is the front depends on the winding;
  a surface that only faces one way (water seen from above) shouldn't flip its normal by it.
- **Remote sessions** have no visible desktop: windows run hidden at about 1028×749, so use
  `--capture` to see output.
