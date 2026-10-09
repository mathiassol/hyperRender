// The atmospheres over a lit HDR image (camera-relative, exposed): along each pixel's view
// ray, up to the surface it sees (or out to space), the light the air scatters towards
// the camera is added and the light from behind is dimmed by the air's transmittance.
// Several atmospheres: the farthest first.

#include <hyper/common.hlsli>
#include <donut/shaders/view_cb.h>
#include <hyper/planet_atmosphere.hlsli>

struct CompositeConstants
{
    PlanarViewConstants view;
    uint count;
    uint3 pad;
    PlanetAtmosphereConstants atmospheres[PLANET_ATMOSPHERES];
};
ConstantBuffer<CompositeConstants> g_C : register(b0);
Texture2D<float> t_Depth : register(t0);
Texture2DArray<float4> t_Transmittance : register(t1);
Texture2DArray<float4> t_MultiScatter : register(t2);
Texture2DArray<float4> t_Phase : register(t3);
SamplerState s_Clamp : register(s0);
VK_IMAGE_FORMAT("rgba16f") RWTexture2D<float4> u_Hdr : register(u0);

[numthreads(8, 8, 1)]
void main(uint2 pixel : SV_DispatchThreadID)
{
    if (any(pixel >= uint2(g_C.view.viewportSize)))
        return;
    // The view ray from the camera (the origin), through the pixel's centre.
    const float2 uv = (float2(pixel) + 0.5) * g_C.view.viewportSizeInv;
    const float4 clip = float4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.5, 1.0);
    const float4 w = mul(clip, g_C.view.matClipToWorld);
    const float3 dir = normalize(w.xyz / w.w);
    const float depth = t_Depth[pixel];
    float tMax = 3.0e38;
    if (depth > 0.0)
    {
        const float4 sc = float4(clip.xy, depth, 1.0);
        const float4 sw = mul(sc, g_C.view.matClipToWorld);
        tMax = length(sw.xyz / sw.w);
    }

    float3 color = u_Hdr[pixel].rgb;
    // Farthest first, so the nearer one's air lies over the farther's.
    uint order[PLANET_ATMOSPHERES] = { 0, 1, 2, 3 };
    for (uint i = 0; i < g_C.count; ++i)
        for (uint j = i + 1; j < g_C.count; ++j)
            if (dot(g_C.atmospheres[order[j]].center, g_C.atmospheres[order[j]].center) > dot(g_C.atmospheres[order[i]].center, g_C.atmospheres[order[i]].center))
            {
                const uint t = order[i];
                order[i] = order[j];
                order[j] = t;
            }
    for (uint k = 0; k < g_C.count; ++k)
    {
        const PlanetAtmosphereConstants a = g_C.atmospheres[order[k]];
        float3 transmittance;
        const float3 light = AtmosphereAlongRay(a, 0.0, dir, tMax, 32, transmittance, t_Transmittance, t_MultiScatter, t_Phase, s_Clamp);
        color = color * transmittance + light;
    }
    u_Hdr[pixel] = float4(min(color, 60000.0), 0.0);
}
