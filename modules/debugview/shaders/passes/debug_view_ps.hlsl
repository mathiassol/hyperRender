// Shows one texture channel set as-is: the value lands on screen unchanged
// (converted to linear first when the target is sRGB).

#include <hyper/common.hlsli>
#include <hyper/color.hlsli>

struct DebugConstants
{
    uint channel;       // 0 colour, 1 red, 2 alpha, 3 normal
    uint outputLinear;
    float scale;
    float pad;
};

DECLARE_PUSH_CONSTANTS(DebugConstants, g_Debug, 0, 0);
Texture2D<float4> t_Source : register(t0);
SamplerState s_Point : register(s0);

void main(in float4 pos : SV_Position, in float2 uv : UV, out float4 o_Color : SV_Target)
{
    const float4 v = t_Source.SampleLevel(s_Point, uv, 0);
    float3 display;
    if (g_Debug.channel == 1)
        display = saturate(v.r * g_Debug.scale);
    else if (g_Debug.channel == 2)
        display = saturate(v.a * g_Debug.scale);
    else if (g_Debug.channel == 3)
        display = saturate(v.xyz * 0.5 + 0.5);
    else
    {
        const float3 c = max(v.rgb * g_Debug.scale, 0.0);
        display = LinearToSrgb(c / (1.0 + c));
    }
    o_Color = float4(g_Debug.outputLinear != 0 ? SrgbToLinear(display) : display, 1.0);
}
