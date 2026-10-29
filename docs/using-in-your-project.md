# Using hyperRender in your project

Two ways to build on hyperRender:

- **Inside this repository**: put your app in `apps/<name>/` and add it to
  `apps/CMakeLists.txt`. Quickest for experiments; you get the regression suite for free.
- **From your own repository** (this page): hyperRender as a git submodule, pulled in with
  `add_subdirectory`. Your code stays yours, and you update hyperRender when you choose to.

## 1. Add the submodule

```powershell
git submodule add https://github.com/mathiassol/hyperRender.git external/hyperRender
git submodule update --init --recursive
```

`--recursive` matters: hyperRender has Donut as its own submodule.

A submodule is pinned to one commit, so your project builds the same hyperRender until you
move it. Pin it to a release:

```powershell
git -C external/hyperRender checkout v0.1.0
git add external/hyperRender
git commit -m "hyperRender v0.1.0"
```

On GitHub the folder then shows as `hyperRender @ <commit>`, a link to exactly that version.

## 2. CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.28)
project(myproject LANGUAGES C CXX)

add_subdirectory(external/hyperRender)

add_executable(myapp main.cpp)
target_link_libraries(myapp PRIVATE hyper::hyper hyper::post)
hyper_app(myapp)
hyper_add_shaders(myapp CONFIG "${CMAKE_CURRENT_SOURCE_DIR}/shaders/shaders.cfg" NAME myapp)
```

- `add_subdirectory` gives you the core (`hyper::hyper`), every module (`hyper::<name>`),
  and the CMake functions `hyper_app` and `hyper_add_shaders`. A module is only built if
  something links it.
- hyperRender's examples aren't built when it's a subproject. Set
  `-DHYPER_BUILD_EXAMPLES=ON` to build them anyway.
- `-DHYPER_WITH_DX12=ON` also builds the D3D12 backend (experimental).

`main.cpp` and `shaders/` are the app from [Writing an app](writing-an-app.md).

## 3. A preset

Copy hyperRender's `CMakePresets.json` into your project, or use a configure command:

```powershell
cmake -S . -B build -G "Visual Studio 18 2026" -A x64
cmake --build build --config RelWithDebInfo
build\bin\myapp.exe
```

Everything lands in `<build>/bin/`: your exe, hyperRender's shader folders, and Donut's.
Apps find their shaders beside the exe, so run them from anywhere.

## 4. Assets

`hyper::AssetPath("models/thing.gltf")` looks for an `assets/` folder in the exe's folder
and each one above it, and resolves the path inside it. Put an `assets/` folder at your
project's root and `build/bin/` finds it. Keep large downloads in `assets/downloaded/`
and out of git.

Donut loads a `.dds` file in place of a `.png` or `.jpg` beside it.
`python external/hyperRender/tools/compress_textures.py <scene.gltf>` writes block-
compressed `.dds` files with mips for a glTF's textures (`pip install etcpak numpy
pillow`): a quarter of the memory, and nothing to decode at load.

## 5. Testing your app

`tools/capture_all.ps1` reads its cases from `tools/captures.txt` next to it, so copy
the script, `tools/png_stats.py` and a `captures.txt` of your own into your project, with a
`tests/reference/` folder. Run it with `-Bin build/bin`. See [testing.md](testing.md).

## 6. Updating hyperRender

```powershell
cd external/hyperRender
git fetch
git checkout <tag or commit>
git submodule update --init --recursive
cd ../..
git add external/hyperRender
```

Read the [changelog](../CHANGELOG.md) first: before 1.0, a minor version can change the
API.
