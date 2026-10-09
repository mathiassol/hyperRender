// dst = base + overlay, with the overlay resampled to base's size.
// PostProcess uses it to add layers that must skip TAA (e.g. analytic point
// lights that would smear under history reprojection).

#include <hyper/common.hlsli>

struct AddConstants
{
    float2 size;
    float2 pad;
};

DECLARE_PUSH_CONSTANTS(AddConstants, g_Add, 0, 0);
Texture2D<float4> t_Base : register(t0);
Texture2D<float4> t_Overlay : register(t1);
SamplerState s_Linear : register(s0);
VK_IMAGE_FORMAT("rgba16f") RWTexture2D<float4> u_Dest : register(u0);

[numthreads(8, 8, 1)]
void main(uint2 id : SV_DispatchThreadID)
{
    if (any(float2(id) >= g_Add.size))
        return;
    float2 uv = (float2(id) + 0.5) / g_Add.size;
    u_Dest[id] = float4(t_Base[id].rgb + t_Overlay.SampleLevel(s_Linear, uv, 0).rgb, 1.0);
}
