# The core: the app loop, captures, profiling, hot reload, the scene (and its ray-tracing
# structures), compute pass plumbing. Features are modules (modules/).
set(HYPER_SHADER_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/include" CACHE INTERNAL "")

file(GLOB_RECURSE hyper_sources CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

add_library(hyper STATIC ${hyper_sources})
add_library(hyper::hyper ALIAS hyper)

target_compile_features(hyper PUBLIC cxx_std_20)
target_include_directories(hyper PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

# Donut is part of hyper's public surface on purpose: every layer below
# hyper (Donut, NVRHI, Vulkan) stays reachable from app code.
target_link_libraries(hyper PUBLIC donut_app donut_engine donut_render donut_core nvrhi nvrhi_vk)

target_compile_definitions(hyper PUBLIC
    HYPER_WITH_DX12=$<BOOL:${HYPER_WITH_DX12}>
    $<$<CONFIG:Debug>:HYPER_DEBUG=1>)

if (MSVC)
    target_compile_options(hyper PRIVATE /W4 /permissive-)
endif()

set_target_properties(hyper PROPERTIES FOLDER "hyper")

# The core's own passes; mounted at /shaders/hyper with the modules', hot-reloaded like app shaders.
hyper_add_shaders(hyper CONFIG "${CMAKE_CURRENT_SOURCE_DIR}/shaders/shaders.cfg" NAME hyper)
