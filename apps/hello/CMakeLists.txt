# Console subsystem on purpose: logs and validation errors land in the terminal.
add_executable(hello main.cpp)
target_link_libraries(hello PRIVATE hyper::hyper hyper::post)
hyper_app(hello)
hyper_add_shaders(hello CONFIG "${CMAKE_CURRENT_SOURCE_DIR}/shaders/shaders.cfg" NAME hello)
