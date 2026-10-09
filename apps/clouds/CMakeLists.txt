# Console subsystem on purpose: logs and validation errors land in the terminal.
add_executable(clouds main.cpp)
target_link_libraries(clouds PRIVATE hyper::hyper hyper::post hyper::atmosphere hyper::environment hyper::volumes)
hyper_app(clouds)
