cmake_minimum_required(VERSION 3.28)

project(hyperRender VERSION 0.1.0 LANGUAGES C CXX)

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(HYPER_WITH_DX12 "Also build the D3D12 backend (Vulkan is always on)" OFF)
# The examples build by default only when hyperRender is the top-level project; a project
# that pulls it in with add_subdirectory() gets the libraries alone.
option(HYPER_BUILD_EXAMPLES "Build the examples under apps/" ${PROJECT_IS_TOP_LEVEL})

# ---------------------------------------------------------------------------
# Global settings (must be set before Donut is added so it inherits them)
# ---------------------------------------------------------------------------
if (MSVC)
    # The static CRT, for hyper, Donut and everything built with them (hyper_app sets it on
    # an app's own targets too).
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

# Every exe and its compiled shaders go to one folder: an app finds its shaders beside it.
set(HYPER_BIN_DIR "${CMAKE_BINARY_DIR}/bin" CACHE INTERNAL "")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${HYPER_BIN_DIR}")
foreach (cfg DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${cfg} "${HYPER_BIN_DIR}")
endforeach()

# ---------------------------------------------------------------------------
# Donut (brings NVRHI, ShaderMake, GLFW, ImGui, cgltf, stb, tinyexr, zstd)
# ---------------------------------------------------------------------------
set(DONUT_WITH_DX11   OFF                CACHE BOOL "" FORCE)
set(DONUT_WITH_DX12   ${HYPER_WITH_DX12} CACHE BOOL "" FORCE)
set(DONUT_WITH_VULKAN ON                 CACHE BOOL "" FORCE)
set(DONUT_WITH_KTX    ON                 CACHE BOOL "" FORCE)
set(DONUT_FORCE_DISCRETE_GPU ON          CACHE BOOL "" FORCE)
# Keep fetched sources in the build tree, not inside the submodule.
set(DONUT_ZSTD_FETCH_DIR "${CMAKE_BINARY_DIR}/_deps/zstd-src" CACHE STRING "" FORCE)
set(DONUT_SHADERS_OUTPUT_DIR "${HYPER_BIN_DIR}/shaders/framework")

# SYSTEM: Donut's headers don't take part in our warning level.
add_subdirectory(external/donut SYSTEM)
include(${CMAKE_CURRENT_SOURCE_DIR}/external/donut/compileshaders.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/hyper_shaders.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/hyper_module.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/hyper_app.cmake)

include(FetchContent)

# meshoptimizer (MIT): meshlets for GPU-driven drawing.
FetchContent_Declare(meshoptimizer
    GIT_REPOSITORY https://github.com/zeux/meshoptimizer.git
    GIT_TAG 9e1f07b159d3cb777f1c67ed31fc11fd117986f4   # v1.3
    GIT_SHALLOW FALSE)
FetchContent_MakeAvailable(meshoptimizer)
set_target_properties(meshoptimizer PROPERTIES FOLDER "External")

# ---------------------------------------------------------------------------
# hyper: the core, its modules, the examples
# ---------------------------------------------------------------------------
add_subdirectory(hyper)
add_subdirectory(modules)

if (HYPER_BUILD_EXAMPLES)
    add_subdirectory(apps)
endif()
