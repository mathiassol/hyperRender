# Contributing

Thanks for helping. Bug reports, fixes, new modules and documentation are all welcome.

## Before you start

- For anything bigger than a fix, open an issue first and describe what you want to do,
  so we can agree on where it belongs (see *Core or module* below).
- Build with a Debug preset while you work: any Vulkan or NVRHI validation message is a
  bug.

## Building and testing

[Getting started](docs/getting-started.md) sets up the tools. Before you send a change:

```powershell
cmake --build --preset vs2026-release
cmake --build --preset vs2026-debug
pwsh tools/capture_all.ps1                 # every case passes
pwsh tools/capture_all.ps1 -Suffix -debug  # and under the validation layers
```

If your change alters an image on purpose, look at the new images in `build/captures/`,
then run `-Update` for those cases and include the new references in the change. Say in
the pull request what changed and why. Reference images depend on the GPU and driver:
mention which you used. A new feature comes with a case in `tools/captures.txt` that
shows it ([testing.md](docs/testing.md)).

## Core or module

- **The core (`hyper/`) is only what every app needs**: the loop, captures, the profiler,
  hot reload, `Scene`, `GBuffer`, `RtScene`, `ComputePasses`, `DepthPyramid`, `Readback`.
  It changes for bugs, and for something real apps show is missing. A feature is not a
  reason to change it.
- **A feature is a module** in `modules/<name>/`, declared with `hyper_add_module` in
  `modules/CMakeLists.txt`:
  - `include/hyper/...` for its public headers, `src/` for its C++, `shaders/<name>.cfg`
    and its HLSL, `shaders/include/` for shader headers other code may include.
  - Modules depend on each other only through `DEPENDS`. They hand data across as a
    texture, a callback or a plain struct, not each other's types.
  - One shape: a `settings` struct, `BuildUI()`, `CreatePipelines()`, and a `Render` (or
    similar) taking a small `Inputs` struct.
  - **Off costs nothing.** Build pipelines and size-dependent resources on first use,
    never in the constructor; `CreatePipelines()` does nothing before that; no work while
    disabled. The `scene-off` case constructs every module switched off and checks that
    none of them runs.
  - `CreatePipelines()` is called again after a hot reload. If a shader fails to compile,
    keep the old pipeline.
- **A frame belongs to its app.** There are no presets that assemble modules for you;
  `apps/scene/SceneRenderer` shows how.

## Code style

- C++20. Follow the code around you: four spaces, braces on their own lines, `m_` for
  members, `PascalCase` for types and functions, `camelCase` for locals and settings.
- Every hyper header includes `<hyper/platform.h>` first.
- Comments say *why*, in plain sentences, and cite the paper or source a technique comes
  from. Keep them short.
- No new warnings: hyper and the modules build with `/W4 /permissive-`.
- Don't edit `external/`: Donut is a pinned submodule. When Donut's behaviour needs working
  around, do it in hyper and add a line to the table in
  [architecture.md](docs/architecture.md#5-donut-and-the-workarounds-for-it).

## Shaders

- HLSL, shader model 6.6, matrices row-major. One `.cfg` per shader set.
- Use the shared binding slots ([architecture.md](docs/architecture.md#3-shaders)) so passes
  compose.
- A struct shared with C++ lives in a header the C++ side includes after
  `<hyper/shared.h>`.

## Pull requests

- One change per pull request, with a description of what and why.
- The build is clean, both test runs pass, and the docs describe the change if it's
  visible to users (README, `docs/`, the changelog's *Unreleased* section).

By contributing you agree that your contribution is licensed under the project's MIT
license.
