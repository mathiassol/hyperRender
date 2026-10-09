# hyper_add_module(<name> [DEPENDS <module>...] [PRIVATE_LIBS <target>...])
#
# A feature module in modules/<name>/: include/ (its public headers, under hyper/),
# src/ (its C++), shaders/ (<name>.cfg with its .hlsl, and include/ for shader headers
# others may include, and its own C++ too). It becomes the static library hyper_<name>,
# alias hyper::<name>, on top of the core (hyper::hyper) and the modules it DEPENDS on.
# An app links the modules it uses; nothing else of hyper's is built for it, C++ or shaders.
function(hyper_add_module name)
    cmake_parse_arguments(arg "" "" "DEPENDS;PRIVATE_LIBS" ${ARGN})
    set(dir "${CMAKE_CURRENT_SOURCE_DIR}/${name}")
    file(GLOB_RECURSE sources CONFIGURE_DEPENDS "${dir}/include/*.h" "${dir}/src/*.h" "${dir}/src/*.cpp")
    add_library(hyper_${name} STATIC ${sources})
    add_library(hyper::${name} ALIAS hyper_${name})
    target_include_directories(hyper_${name} PUBLIC "${dir}/include")
    if (EXISTS "${dir}/shaders/include")
        # A struct shared with the shaders lives there (after <hyper/shared.h> in C++); apps
        # that link the module fill such structs too.
        target_include_directories(hyper_${name} PUBLIC "${dir}/shaders/include")
    endif()
    target_link_libraries(hyper_${name} PUBLIC hyper::hyper)
    foreach (dep IN LISTS arg_DEPENDS)
        target_link_libraries(hyper_${name} PUBLIC hyper::${dep})
    endforeach()
    if (arg_PRIVATE_LIBS)
        target_link_libraries(hyper_${name} PRIVATE ${arg_PRIVATE_LIBS})
    endif()
    if (MSVC)
        target_compile_options(hyper_${name} PRIVATE /W4 /permissive-)
    endif()
    set_target_properties(hyper_${name} PROPERTIES FOLDER "hyper/modules")
    if (EXISTS "${dir}/shaders/${name}.cfg")
        hyper_add_shaders(hyper_${name} CONFIG "${dir}/shaders/${name}.cfg" NAME ${name} OUT hyper)
    endif()
endfunction()
