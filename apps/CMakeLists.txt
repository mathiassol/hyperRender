# The examples, each a small app on hyper (hyper_app in cmake/hyper_app.cmake). Your own app
# can live here too: copy apps/hello, add its folder below.
set(HYPER_EXAMPLES hello clouds scene)
foreach (app ${HYPER_EXAMPLES})
    add_subdirectory(${app})
endforeach()
