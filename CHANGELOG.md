# Changelog

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versions
follow [Semantic Versioning](https://semver.org/); before 1.0, a minor version may change
the API.

## Unreleased

## 0.1.0

The first public release.

- **Core:** the app loop (`hyper::App`, `hyper::Run`), glTF scenes with background loading,
  texture streaming, animation, skinning and runtime edits (`hyper::Scene`), the
  G-buffer with float reverse-Z depth, acceleration structures for ray queries
  (`hyper::RtScene`), compute-pass plumbing (`hyper::ComputePasses`), a Hi-Z pyramid, GPU
  readback, a GPU and CPU profiler, shader hot reload, command-line settings
  (`hyper::Overrides`), and deterministic captures.
- **20 modules:** `post`, `gpu`, `shadows`, `atmosphere`, `environment`, `ao`, `ssr`,
  `traced`, `rtlighting`, `rtshadows`, `rtreflections`, `ddgi`, `reference`, `noise`, `fog`,
  `volumes`, `ocean`, `atmospheres`, `planet`, `debugview`.
- **Examples:** `hello`, `clouds`, `scene`.
- **CMake:** `hyper_app` and `hyper_add_shaders`; hyperRender can be used with
  `add_subdirectory`, without its examples.
- **Tests:** an image regression suite of 20 cases.
