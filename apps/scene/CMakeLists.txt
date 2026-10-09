# Console subsystem on purpose: logs and validation errors land in the terminal.
add_executable(scene main.cpp SceneRenderer.cpp SceneRenderer.h)
target_link_libraries(scene PRIVATE hyper::hyper hyper::post hyper::gpu hyper::ao hyper::atmosphere hyper::environment
    hyper::shadows hyper::ssr hyper::rtshadows hyper::rtreflections hyper::ddgi hyper::reference hyper::fog hyper::volumes
    hyper::debugview)
hyper_app(scene)
hyper_add_shaders(scene CONFIG "${CMAKE_CURRENT_SOURCE_DIR}/shaders/shaders.cfg" NAME scene)
