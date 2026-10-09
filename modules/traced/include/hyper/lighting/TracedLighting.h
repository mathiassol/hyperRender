#pragma once

#include <hyper/platform.h>

#include <donut/core/math/math.h>
#include <nvrhi/nvrhi.h>

namespace hyper
{
    struct Context;
    class Scene;
    class ComputePasses;

    struct TracedLightingSettings
    {
        uint32_t giRays = 1;            // bounce rays per half-res pixel
        float giClamp = 8.f;            // luminance one pixel's bounce light may carry (fireflies)
        float reflectionClamp = 40.f;   // the same for one reflection sample
        float tolerance = 0.02f;        // m: the depth difference history and filters accept on
                                        // top of a few % of the distance (a street's 2 cm, a macro
                                        // shot's half a millimetre)
    };

    // Ray-traced bounce light (half resolution) and reflections, with their temporal and
    // spatial filters (hyper/traced_lighting.hlsli). What a ray sees, and which surfaces
    // reflect, is the app's: it compiles that file in its own shader set with its hooks, and
    // passes the shader's name here. The dispatches go through the app's ComputePasses.
    //
    //     TracedLighting traced(ctx, passes);
    //     traced.CreatePipelines("app/traced_cs", scene.BindlessLayout());
    //     nvrhi::ITexture* gi = traced.Gi(cmd, { depth, normals, appTexture });
    //     nvrhi::ITexture* reflection = traced.Reflections(cmd, { depth, normals, appTexture });
    class TracedLighting
    {
    public:
        TracedLighting(Context& ctx, ComputePasses& passes);

        TracedLightingSettings settings;

        void CreatePipelines(const char* shader, nvrhi::IBindingLayout* bindless);
        void BuildUI();

        struct Inputs
        {
            nvrhi::ITexture* depth = nullptr;     // reverse-Z, full resolution
            nvrhi::ITexture* normals = nullptr;   // xyz normal, w roughness
            nvrhi::ITexture* app = nullptr;       // bound at t2 for the app's hooks
        };

        // Half resolution: rgb bounce radiance (multiply by albedo), a distance to the camera.
        nvrhi::ITexture* Gi(nvrhi::ICommandList* cmd, const Inputs& in);
        // Full resolution: rgb reflected radiance (multiply by the specular term), a hit distance.
        nvrhi::ITexture* Reflections(nvrhi::ICommandList* cmd, const Inputs& in);

    private:
        Context& m_Ctx;
        ComputePasses& m_Passes;
        nvrhi::ComputePipelineHandle m_GiTrace, m_GiTemporal, m_GiFilter, m_ReflTrace, m_ReflTemporal, m_ReflFilter;
        dm::uint2 m_Size = dm::uint2::zero();
        nvrhi::TextureHandle m_GiRaw, m_GiHistory[2], m_GiA, m_GiB;
        nvrhi::TextureHandle m_ReflRaw, m_ReflHistory[2], m_Reflection;
        uint32_t m_GiIndex = 0, m_ReflIndex = 0;
        bool m_GiValid = false, m_ReflValid = false;

        void EnsureTargets(nvrhi::ITexture* depth);
        struct Constants;
        Constants Make(uint32_t step) const;
    };
}
