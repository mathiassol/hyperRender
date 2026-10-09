#pragma once

// hyperRender: pick the level you need.
//   L2  modules     features, each a library the app links (modules/; README lists them):
//                   #include its headers, e.g. <hyper/rt/Ddgi.h>, and link hyper::ddgi.
//                   A frame is the app's: apps/scene assembles most of them into one.
//   L2  core        this header: the app loop, the scene and its ray-tracing structures,
//                   compute pass plumbing (below)
//   L1  NVRHI       Context::device, nvrhi::utils, hyper::RenderTarget
//   L0  Vulkan      hyper/native.h

#include <hyper/App.h>
#include <hyper/Assets.h>
#include <hyper/Camera.h>
#include <hyper/Overrides.h>
#include <hyper/Profiler.h>
#include <hyper/Readback.h>
#include <hyper/RenderTarget.h>
#include <hyper/native.h>
#include <hyper/lighting/DepthPyramid.h>
#include <hyper/passes/ComputePasses.h>
#include <hyper/rt/RtScene.h>
#include <hyper/scene/GBuffer.h>
#include <hyper/scene/Scene.h>
