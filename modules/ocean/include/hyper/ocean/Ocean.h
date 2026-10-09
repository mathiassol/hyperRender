#pragma once

#include <hyper/platform.h>

#include <donut/core/math/math.h>
#include <nvrhi/nvrhi.h>

#include <memory>
#include <unordered_map>
#include <vector>

namespace donut::engine { class PlanarView; }

namespace hyper
{
    struct Context;
    class GBuffer;

    struct OceanSettings
    {
        bool enabled = true;
        // The waves: a JONSWAP spectrum of the wind's sea, plus a swell from elsewhere.
        float windSpeed = 9.f;                // m/s, 10 m above the sea
        float windDirectionDeg = 20.f;        // where the wind blows towards, from +x towards +z
        float fetchKm = 150.f;                // how far the wind has blown over open water
        float swell = 0.4f;                   // 0..1, long waves from a storm far away
        float swellDirectionDeg = 60.f;
        float swellPeriod = 11.f;             // seconds
        float choppiness = 1.f;               // sharper crests, flatter troughs
        uint32_t seed = 1;
        float timeScale = 1.f;
        // Foam where crests fold over (the surface's Jacobian).
        float foam = 1.f;
        float foamDecay = 2.5f;               // seconds a patch of foam takes to fade
        // The look.
        dm::float3 waterAlbedo = dm::float3(0.004f, 0.016f, 0.022f);
        dm::float3 scatterColor = dm::float3(0.03f, 0.16f, 0.14f);
        float scatter = 0.5f;
        float roughness = 0.035f;             // a calm surface's, below what the waves resolve
        float detail = 1.f;                   // > 1: denser surface mesh
        float seaLevel = 0.f;
    };

    // An FFT ocean (Tessendorf) of three cascades: a spectrum built on the CPU from the
    // wind, evolved and transformed on the GPU each frame into displacement, slopes and
    // foam, drawn as a camera-centred clipmap mesh into the standard G-buffer. Shadows,
    // reflections, AO and GI then light it like any other surface; its own part is the
    // light scattered through thin crests (written as emissive) and the foam.
    //
    //     ocean.Update(cmd, ctx.time);                                  // once a frame
    //     ocean.RenderGBuffer(cmd, view, previous, gbuffer, light);     // after the scene's G-buffer
    //     float y = ocean.HeightAt(xz, ctx.time);                       // buoyancy, on the CPU
    class Ocean
    {
    public:
        explicit Ocean(Context& ctx);
        ~Ocean();

        OceanSettings settings;

        // Steps the waves to `time` (seconds; ctx.time keeps captures repeatable).
        void Update(nvrhi::ICommandList* cmd, float time);

        struct Light
        {
            dm::float3 sunDirection = dm::float3(0.f, 1.f, 0.f);   // towards the sun
            dm::float3 sunRadiance = dm::float3(1.f);              // colour x irradiance
        };
        // Depth, material and motion vectors (with the G-buffer's stencil bit) where the
        // water is in front of what's already drawn.
        void RenderGBuffer(nvrhi::ICommandList* cmd, const donut::engine::PlanarView& view,
                           const donut::engine::PlanarView& previous, GBuffer& gbuffer, const Light& light);

        // The water's height under world xz at `time`, on the CPU: the waves longer than
        // minWavelength, exactly as the GPU has them (horizontal displacement included).
        // What floats on it follows these; shorter waves barely move a hull.
        [[nodiscard]] float HeightAt(dm::float2 xz, float time, float minWavelength = 12.f) const;

        void CreatePipelines();
        void BuildUI();

    private:
        bool m_Used = false;   // resources and pipelines are made on first use
        Context& m_Ctx;

        nvrhi::BufferHandle m_SimConstants, m_SurfaceConstants, m_Indices;
        nvrhi::BindingLayoutHandle m_SimLayout, m_SurfaceLayout;
        nvrhi::ComputePipelineHandle m_Evolve, m_Fft, m_Assemble, m_Mip;
        nvrhi::ShaderHandle m_SurfaceVs, m_SurfacePs;
        std::unordered_map<size_t, nvrhi::GraphicsPipelineHandle> m_SurfacePipelines;   // by framebuffer
        nvrhi::SamplerHandle m_Sampler;

        nvrhi::TextureHandle m_Spectrum;          // per cascade: h0(k), conj(h0(-k))
        nvrhi::TextureHandle m_FftA, m_FftB;      // two complex fields each, transformed in place
        nvrhi::TextureHandle m_Displacement[2];   // this frame's and last frame's: xyz, foam
        nvrhi::TextureHandle m_Slopes;            // dy/dx, dy/dz and their squares, for filtering
        nvrhi::BindingSetHandle m_SimSets[2];
        std::vector<nvrhi::BindingSetHandle> m_MipSets[2];
        nvrhi::BindingSetHandle m_SurfaceSets[2];
        uint32_t m_Current = 0;
        bool m_Updated = false;       // the textures hold a frame
        float m_WaveHeight = 0.f;     // a typical crest, metres
        float m_LastTime = 0.f;

        // The long waves of cascade 0, for HeightAt.
        struct Wave
        {
            float kx, kz, k, omega;
            float h0re, h0im, hmre, hmim;   // h0(k) and conj(h0(-k))
        };
        std::vector<Wave> m_LongWaves;
        OceanSettings m_Built;
        bool m_SpectrumValid = false;

        void CreateResources(nvrhi::ICommandList* cmd);
        void BuildSpectrum(nvrhi::ICommandList* cmd);
        nvrhi::IGraphicsPipeline* SurfacePipeline(nvrhi::IFramebuffer* fb);
    };
}
