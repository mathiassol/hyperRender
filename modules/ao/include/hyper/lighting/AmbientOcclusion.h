#pragma once

#include <hyper/platform.h>

#include <nvrhi/nvrhi.h>

namespace donut::engine { class PlanarView; }

namespace hyper
{
    struct Context;

    struct AmbientOcclusionSettings
    {
        bool enabled = true;
        int quality = 2;            // 0 low, 1 medium, 2 high, 3 ultra
        float radius = 0.5f;        // world units
        float falloff = 0.615f;
        float power = 2.2f;         // final value power: higher = darker contacts
        int denoisePasses = 2;      // 1..3
        // XeGTAO's thin occluder compensation, 0..0.7: an occluder is taken to be only so
        // thick, so a thin one (a leaf, a blade, a needle) doesn't hide the whole hemisphere
        // behind it. 0 is the plain heightfield, whose foliage comes out speckled black.
        float thinOccluders = 0.5f;
        // Multi-bounce (Jimenez et al. 2016): light bounces between bright surfaces into
        // their creases, so their occlusion is lighter. Needs the albedo (Render's albedo).
        bool multiBounce = true;
    };

    // Ground-truth ambient occlusion with XeGTAO (Intel, MIT, vendored in
    // modules/ao/thirdparty/xegtao). Reads reverse-Z depth and world normals (and the
    // diffuse albedo, for multi-bounce), writes an R8_UNORM visibility term (1 =
    // unoccluded) that Donut's deferred lighting multiplies into indirect light.
    class AmbientOcclusion
    {
    public:
        explicit AmbientOcclusion(Context& ctx);

        AmbientOcclusionSettings settings;

        // Returns the AO texture, or null when disabled.
        // albedo: the G-buffer's diffuse albedo, for multi-bounce (null: single bounce).
        nvrhi::ITexture* Render(nvrhi::ICommandList* cmd, const donut::engine::PlanarView& view,
                                nvrhi::ITexture* depth, nvrhi::ITexture* normals, nvrhi::ITexture* albedo = nullptr);
        void CreatePipelines();
        void BuildUI();

    private:
        bool m_Used = false;   // pipelines are built on first use
        Context& m_Ctx;
        nvrhi::BindingLayoutHandle m_PrefilterLayout, m_MainLayout, m_DenoiseLayout, m_ResolveLayout;
        nvrhi::ComputePipelineHandle m_Prefilter, m_Main, m_Denoise, m_DenoiseLast, m_Resolve;
        nvrhi::BufferHandle m_Constants;
        nvrhi::TextureHandle m_WorkingDepth;
        nvrhi::TextureHandle m_Term[2];
        nvrhi::TextureHandle m_Edges;
        nvrhi::TextureHandle m_Output;
        uint32_t m_Width = 0, m_Height = 0;
    };
}
