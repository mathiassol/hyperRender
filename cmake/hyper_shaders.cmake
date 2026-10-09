# hyper_add_shaders(<target> CONFIG <shaders.cfg> NAME <name> [OUT <folder>] [SLANG]
#                   [INCLUDES dirs...] [DEFINES NAME=VALUE...] [OPTIONS compiler-options...])
#
# Compiles the shaders listed in CONFIG with ShaderMake into
#   bin/shaders/<OUT, default NAME>/<spirv|dxil>/
# and makes <target> depend on them. At runtime hyper mounts an app's folder at
# /shaders/app (loaded as "app/file") and its own at /shaders/hyper ("hyper/file");
# hyper's modules all compile into that one (OUT hyper), each set under its NAME.
# A target can have several shader sets; each NAME gets its own <target>_<NAME>_shaders.
# Every set sees hyper's shader headers and its modules' (HYPER_SHADER_INCLUDE_DIRS).
#
# The exact ShaderMake command is also written next to the output as
# hotreload.txt (hotreload-<NAME>.txt in a shared folder), which the running app
# re-executes when a watched file changes. Build and hot reload therefore can't
# disagree about flags.
# Donut's register offsets, kept where a project that adds hyperRender as a subdirectory
# can see them (compileshaders.cmake sets them in hyperRender's scope only).
set(HYPER_VK_REGISTER_OFFSETS ${NVRHI_DEFAULT_VK_REGISTER_OFFSETS} CACHE INTERNAL "")

function(hyper_add_shaders target)
    cmake_parse_arguments(arg "SLANG" "CONFIG;NAME;OUT" "INCLUDES;DEFINES;OPTIONS" ${ARGN})
    if (NOT arg_CONFIG OR NOT arg_NAME)
        message(FATAL_ERROR "hyper_add_shaders: CONFIG and NAME are required")
    endif()

    get_filename_component(cfg_dir "${arg_CONFIG}" DIRECTORY)
    file(GLOB_RECURSE sources CONFIGURE_DEPENDS
        "${cfg_dir}/*.hlsl" "${cfg_dir}/*.hlsli" "${cfg_dir}/*.slang")

    set(manifest_name hotreload.txt)
    if (arg_OUT)
        set(out_base "${HYPER_BIN_DIR}/shaders/${arg_OUT}")
        set(manifest_name "hotreload-${arg_NAME}.txt")
    else()
        set(out_base "${HYPER_BIN_DIR}/shaders/${arg_NAME}")
    endif()
    set(watch_dirs "${cfg_dir}" ${HYPER_SHADER_INCLUDE_DIRS})

    set(common_args
        --config "${arg_CONFIG}"
        --binaryBlob --outputExt .bin
        -I "${DONUT_SHADER_INCLUDE_DIR}"
        --shaderModel 6_6
        --project "${arg_NAME}")
    foreach (dir IN LISTS HYPER_SHADER_INCLUDE_DIRS)
        list(APPEND common_args -I "${dir}")
    endforeach()
    foreach (dir IN LISTS arg_INCLUDES)
        list(APPEND common_args -I "${dir}")
    endforeach()
    foreach (def IN LISTS arg_DEFINES)
        list(APPEND common_args -D "${def}")
    endforeach()
    foreach (opt IN LISTS arg_OPTIONS)
        list(APPEND common_args -X "${opt}")
    endforeach()

    if (arg_SLANG)
        if (NOT EXISTS "${SHADERMAKE_SLANG_PATH}")
            message(FATAL_ERROR "hyper_add_shaders: SLANG needs SHADERMAKE_SLANG_PATH (e.g. the Vulkan SDK's slangc)")
        endif()
        set(vk_compiler --compiler "${SHADERMAKE_SLANG_PATH}" --slang)
        set(dx_compiler --compiler "${SHADERMAKE_SLANG_PATH}" --slang)
    else()
        set(vk_compiler --compiler "${SHADERMAKE_DXC_VK_PATH}")
        set(dx_compiler --compiler "${SHADERMAKE_DXC_PATH}")
    endif()

    set(platforms SPIRV)
    set(SPIRV_dir spirv)
    set(SPIRV_args ${common_args} ${vk_compiler}
        --out "${out_base}/spirv" --platform SPIRV
        -D SPIRV -D TARGET_VULKAN
        ${HYPER_VK_REGISTER_OFFSETS}
        --vulkanVersion 1.3
        # Mesh / task shaders: the cross-vendor extension, not NV's (DXC's default).
        --spirvExt SPV_EXT_mesh_shader)

    if (HYPER_WITH_DX12)
        list(APPEND platforms DXIL)
        set(DXIL_dir dxil)
        set(DXIL_args ${common_args} ${dx_compiler}
            --out "${out_base}/dxil" --platform DXIL
            -D TARGET_D3D12)
    endif()

    set(shader_target ${target}_shaders)
    if (TARGET ${shader_target})
        set(shader_target ${target}_${arg_NAME}_shaders)
    endif()
    set_source_files_properties(${sources} PROPERTIES VS_TOOL_OVERRIDE "None")
    add_custom_target(${shader_target} DEPENDS ShaderMake SOURCES ${sources} "${arg_CONFIG}")
    set_target_properties(${shader_target} PROPERTIES FOLDER "Shaders")

    foreach (platform IN LISTS platforms)
        # ShaderMake is incremental (it tracks includes), so running it every build is cheap.
        add_custom_command(TARGET ${shader_target} PRE_BUILD
            COMMAND ShaderMake ${${platform}_args}
            VERBATIM)

        set(manifest "")
        foreach (dir IN LISTS watch_dirs)
            string(APPEND manifest "watch ${dir}\n")
        endforeach()
        string(APPEND manifest "arg $<TARGET_FILE:ShaderMake>\n")
        foreach (a IN LISTS ${platform}_args)
            string(APPEND manifest "arg ${a}\n")
        endforeach()
        file(GENERATE OUTPUT "${out_base}/${${platform}_dir}/${manifest_name}" CONTENT "${manifest}")
    endforeach()

    add_dependencies(${target} ${shader_target})
endfunction()
