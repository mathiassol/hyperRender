# Getting started

## 1. Install the tools

| Tool | Notes |
|---|---|
| Visual Studio 2026 | Workload "Desktop development with C++". Visual Studio 2022 has a preset (`vs2022`) but hasn't been tested yet; reports are welcome. |
| CMake 3.28 or newer | Visual Studio's bundled CMake works, or install it from cmake.org. |
| Vulkan SDK | From LunarG. Gives the validation layers (used by Debug builds) and the DXC shader compiler. Install with its defaults; the build finds it through `VULKAN_SDK`. |
| Git | For the clone and its submodules. |
| PowerShell 7 (`pwsh`) | For `tools/fetch_assets.ps1` and the test runner. |
| Python 3 | For the image comparisons in the test runner (`pip install numpy pillow`), and the optional asset tools. |

**The GPU** needs Vulkan 1.3. Ray tracing (inline ray queries) and mesh shaders are used
when it has them: NVIDIA RTX 20-series and newer, AMD RX 6000 and newer, Intel Arc. Without
them, the modules that need them stay off (`ctx.rayQuery` says which), and the raster
modules take their place. hyperRender has been developed and tested on an NVIDIA RTX 3070
Ti; other GPUs should work, but haven't been tried yet.

## 2. Clone and build

```powershell
git clone --recursive https://github.com/mathiassol/hyperRender.git
cd hyperRender
cmake --preset vs2026
cmake --build --preset vs2026-release
```

If you cloned without `--recursive`, run `git submodule update --init --recursive`.

The first configure downloads a few dependencies into `build/` (meshoptimizer, DXC, the
Vulkan headers, zstd), and the first build compiles every shader; both take a few
minutes. Everything ends up in `build/vs2026/bin/`: each exe, and the compiled shaders
beside it in `shaders/`.

| Preset | Gives | Use it for |
|---|---|---|
| `vs2026-debug` | `<app>-debug.exe`, Vulkan validation and NVRHI validation on | developing: any validation message is a bug |
| `vs2026-release` | `<app>.exe`, optimised (RelWithDebInfo, so it still debugs) | timings, and running things |

Both write to the same folder; the Debug build's `-debug` suffix keeps them apart. You can
also open `build/vs2026/hyperRender.sln` in Visual Studio.

## 3. Run the examples

```powershell
build\vs2026\bin\hello.exe
build\vs2026\bin\clouds.exe
```

`scene` needs the test scenes:

```powershell
pwsh tools/fetch_assets.ps1          # Khronos glTF samples and an HDRI, into assets/downloaded/
build\vs2026\bin\scene.exe
build\vs2026\bin\scene.exe --scene downloaded/DamagedHelmet/glTF/DamagedHelmet.gltf --env downloaded/hdri/kloofendal_48d_partly_cloudy_puresky_1k.hdr
```

`--scene` takes any glTF or .glb, as a path relative to `assets/` or an absolute one.

**In every app:**

| Key | Does |
|---|---|
| Left mouse drag | look |
| W A S D, Q / E | move, down / up (Shift faster, Ctrl slower) |
| F1 | hide or show the UI |
| F5 | rebuild the shaders now |
| F12 | screenshot to `bin/screenshots/` |

| Flag | Does |
|---|---|
| `--capture out.png [--frames N]` | render N frames (60 by default) at a fixed 1/60 s step, save the last without UI, print the profiler, exit. The exit code is 2 if any error or validation message was logged. |
| `--async-load` | with `--capture`: load the scene the way the interactive app does (streaming), in a fixed order |
| `--no-vsync` | present as fast as possible (for timing) |
| `--size WxH` | ask for a window size (the OS clamps it to the screen) |
| `--set name=value` | change a setting the app exposes (each app lists its own) |

**`scene`'s settings** (`--set name=value`, and the same in its panel):

| Setting | Does |
|---|---|
| `rtshadows= rtreflections= gi= fog=` | the ray-traced shadows, reflections, DDGI and fog (on by default; off, the raster modules take over) |
| `shadows= ao= ssr= atmosphere= ibl=` | the raster set's modules |
| `gpudriven= occlusion= meshlets= lod= lodpixels=` | GPU-driven drawing and its culling, meshlets and cluster LOD |
| `reference=1 refspp=N` | the reference path tracer (converges after about 2,000 samples) |
| `debug=1..7` | show AO, ray-traced shadows, GI, reflections, reflection weight, normals or roughness |
| `sunel= sunaz= sun=` | the sun's elevation, azimuth (degrees) and irradiance |
| `ev= autoexposure= bloom=` | exposure |
| `cam=N`, `eyex= eyey= eyez= yaw= pitch=` | the file's camera N, or a camera anywhere (the panel shows the current one in this form) |
| `smoke=1 clouds=1` | a test media volume and the cloud layer |

## 4. Edit a shader while it runs

Run `hello.exe`, open `apps/hello/shaders/scene_cs.hlsl`, change a colour and save. The
shader recompiles in the background and the app picks it up. If it doesn't compile, the
overlay shows the error and the old shader keeps running.

## 5. Where next

- [Writing an app](writing-an-app.md) to make your own.
- [Using hyperRender in your project](using-in-your-project.md) to build on it from
  another repository.
- [Testing](testing.md) to run the regression suite.

## Troubleshooting

| Problem | Fix |
|---|---|
| `could not find any instance of Visual Studio` | The preset's Visual Studio version isn't installed: use the other preset, or install the C++ workload. |
| CMake can't find DXC or the validation layers | Install the Vulkan SDK and open a new terminal (it sets `VULKAN_SDK`). |
| `external/donut` is empty | `git submodule update --init --recursive` |
| `scene: ... not found` | `pwsh tools/fetch_assets.ps1` |
| The window is smaller than asked for | Windows clamps it to the screen; captures run at that size too. |
| Timings jump around | Use the release build and `--no-vsync`, and make sure nothing else is using the GPU. |
