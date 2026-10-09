# hyper's feature modules: each a library an app links when it uses the feature
# (cmake/hyper_module.cmake). DEPENDS lists the modules one is built on.

# Shader headers are shared: any shader set can include any module's (headers cost nothing).
file(GLOB module_shader_includes LIST_DIRECTORIES true "${CMAKE_CURRENT_SOURCE_DIR}/*/shaders/include")
set(HYPER_SHADER_INCLUDE_DIRS ${HYPER_SHADER_INCLUDE_DIRS} ${module_shader_includes} CACHE INTERNAL "")

hyper_add_module(post)                              # TAA, bloom, tone mapping, overlays
hyper_add_module(debugview)                         # G-buffer and lighting debug views
hyper_add_module(gpu                                # GPU-driven G-buffer and depth: culling, meshlets, cluster LOD
    PRIVATE_LIBS meshoptimizer)
target_include_directories(hyper_gpu PRIVATE ${meshoptimizer_SOURCE_DIR}/demo)   # clusterlod.h (MIT)
hyper_add_module(ao)                                # XeGTAO ambient occlusion
hyper_add_module(atmosphere)                        # physically based sky and aerial perspective
hyper_add_module(environment)                       # IBL probe from a sky capture or an HDR image
hyper_add_module(shadows DEPENDS gpu)               # cascaded sun shadows
hyper_add_module(ssr)                               # screen-space reflections
hyper_add_module(traced)                            # traced bounce light and reflections, the app's hit shading
hyper_add_module(rtlighting)                        # how ray hits are lit: sun, environment, a probe volume
hyper_add_module(rtshadows)                         # ray-traced sun shadows
hyper_add_module(rtreflections DEPENDS rtlighting)  # ray-traced reflections
hyper_add_module(ddgi DEPENDS rtlighting)           # probe-volume GI
hyper_add_module(reference DEPENDS rtlighting)      # ground-truth path tracer
hyper_add_module(noise)                             # baked 3D noise volumes
hyper_add_module(fog DEPENDS noise rtlighting)      # froxel volumetric fog
hyper_add_module(volumes DEPENDS noise)             # ray-marched media and clouds
hyper_add_module(ocean)                             # FFT ocean: cascaded spectra, foam, a clipmap surface
hyper_add_module(atmospheres)                       # planets' atmospheres from their make-up (Hillaire's tables, Mie)
hyper_add_module(planet)                            # planets' surfaces: a cube-sphere quadtree from one CPU/GPU height function
