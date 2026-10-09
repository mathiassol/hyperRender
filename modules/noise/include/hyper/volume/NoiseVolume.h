#pragma once

#include <hyper/platform.h>

#include <nvrhi/nvrhi.h>

namespace hyper
{
    struct Context;

    enum class NoiseType
    {
        Gradient,       // R16F, [-1, 1]
        PerlinWorley,   // RGBA8: R Perlin-Worley, GBA Worley fbm at 2x / 4x / 8x frequency
        Worley,         // RGBA8: RGB Worley fbm at 1x / 2x / 4x frequency
    };

    // Bakes tileable 3D noise textures on the GPU (modules/noise/shaders/volume/noise_bake_cs,
    // functions in hyper/noise.hlsli). Sample them with a wrap sampler; `period`
    // lattice cells span the volume once.
    //
    //     hyper::NoiseVolume baker(ctx);
    //     auto shape = baker.Bake(cmd, hyper::NoiseType::PerlinWorley, 128, 4, "clouds/shape");
    class NoiseVolume
    {
    public:
        explicit NoiseVolume(Context& ctx);

        nvrhi::TextureHandle Bake(nvrhi::ICommandList* cmd, NoiseType type, uint32_t size, uint32_t period, const char* name);
        // Re-bakes into an existing texture (after a shader reload, say).
        void BakeInto(nvrhi::ICommandList* cmd, nvrhi::ITexture* texture, NoiseType type, uint32_t period);
        void CreatePipelines();

    private:
        bool m_Used = false;   // pipelines are built on first use
        Context& m_Ctx;
        nvrhi::BindingLayoutHandle m_Layout;
        nvrhi::ComputePipelineHandle m_Pipelines[3];
        nvrhi::TextureHandle m_DummyScalar, m_DummyColor;   // for the unused output slot
    };
}
