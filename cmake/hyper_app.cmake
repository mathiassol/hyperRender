# hyper_app(<target>)
#
# Makes an executable a hyper app: it goes to hyperRender's bin/ folder (where its compiled
# shaders and hyper's are, found beside the exe at run time), its Debug build is named
# <target>-debug so it never overwrites the optimised one, and it uses the same static
# MSVC runtime as hyper. Call it for every app, in this repository or in your own project:
#
#     add_executable(myapp main.cpp)
#     target_link_libraries(myapp PRIVATE hyper::hyper hyper::post)
#     hyper_app(myapp)
#     hyper_add_shaders(myapp CONFIG "${CMAKE_CURRENT_SOURCE_DIR}/shaders/shaders.cfg" NAME myapp)
function(hyper_app target)
    # ($<0:> stops multi-config generators from adding a per-configuration subfolder.)
    set_target_properties(${target} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${HYPER_BIN_DIR}$<0:>"
        DEBUG_POSTFIX "-debug"
        FOLDER "apps")
    if (MSVC)
        set_target_properties(${target} PROPERTIES MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    endif()
endfunction()
