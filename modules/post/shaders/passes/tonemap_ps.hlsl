// HDR -> display: bloom composite, exposure (manual EV and/or Donut's
// histogram auto exposure), tonemapper, dither.

#include <hyper/common.hlsli>
#include <hyper/color.hlsli>

#define TONEMAP_AUTO_EXPOSURE   1u
#define TONEMAP_DITHER          2u
#define TONEMAP_ENCODE_SRGB     4u   // write sRGB-encoded values (a UNORM target)

struct TonemapConstants
{
    float exposureScale;    // exp2(EV)
    float bloomIntensity;   // 0 = no bloom
    float bloomNorm;        // 1 / bloom levels
    uint op;
    uint flags;
    uint frameIndex;
    float grain;            // film grain strength, 0 = off
    float power;            // the look (Tonemap.h): 1 = none
    float saturation;
    float pad0, pad1, pad2;
};

DECLARE_PUSH_CONSTANTS(TonemapConstants, g_Tonemap, 0, 0);
Texture2D<float4> t_Hdr : register(t0);
Texture2D<float4> t_Bloom : register(t1);
Buffer<uint> t_Exposure : register(t2);
SamplerState s_Linear : register(s0);

// Triangular-ish noise in [0,1), different each frame, so 8-bit output doesn't band.
float Hash(uint2 p, uint frame)
{
    uint h = p.x * 1973u + p.y * 9277u + frame * 26699u;
    h = (h ^ (h >> 13)) * 0x5bd1e995u;
    h ^= h >> 15;
    return float(h & 0xffffu) / 65536.0;
}

void main(in float4 pos : SV_Position, in float2 uv : UV, out float4 o_Color : SV_Target)
{
    float3 c = t_Hdr.SampleLevel(s_Linear, uv, 0).rgb;

    if (g_Tonemap.bloomIntensity > 0.0)
    {
        float3 bloom = t_Bloom.SampleLevel(s_Linear, uv, 0).rgb * g_Tonemap.bloomNorm;
        c = lerp(c, bloom, g_Tonemap.bloomIntensity);
    }

    float exposure = g_Tonemap.exposureScale;
    if (g_Tonemap.flags & TONEMAP_AUTO_EXPOSURE)
    {
        // Donut's exposure pass stores the adapted scene luminance; map it to middle grey.
        float adapted = asfloat(t_Exposure[0]);
        if (adapted > 0.0)
            exposure *= 0.18 / adapted;
    }

    float3 display = Tonemap(c * exposure, g_Tonemap.op);
    if (g_Tonemap.power != 1.0 || g_Tonemap.saturation != 1.0)
    {
        // The look, on the curve's output under a 2.2 gamma.
        float3 v = pow(saturate(display), 1.0 / 2.2);
        v = pow(v, g_Tonemap.power);
        const float luma = dot(v, float3(0.2126, 0.7152, 0.0722));
        v = saturate(luma + (v - luma) * g_Tonemap.saturation);
        display = pow(v, 2.2);
    }
    float3 encoded = LinearToSrgb(display);

    if (g_Tonemap.flags & TONEMAP_DITHER)
    {
        uint2 p = uint2(pos.xy);
        float noise = Hash(p, g_Tonemap.frameIndex) - Hash(p + 7919u, g_Tonemap.frameIndex);
        encoded += noise / 255.0;
    }
    if (g_Tonemap.grain > 0.0)
    {
        // Luminance-shaped, per-frame grain: strongest in the mid-tones, like film.
        uint2 p = uint2(pos.xy);
        float n = (Hash(p, g_Tonemap.frameIndex * 3u + 1u) + Hash(p + 3571u, g_Tonemap.frameIndex * 3u + 2u) - 1.0);
        float luma = dot(encoded, float3(0.299, 0.587, 0.114));
        encoded += n * g_Tonemap.grain * 0.08 * (0.25 + luma * (1.0 - luma) * 3.0);
    }
    encoded = saturate(encoded);

    o_Color.rgb = (g_Tonemap.flags & TONEMAP_ENCODE_SRGB) ? encoded : SrgbToLinear(encoded);
    o_Color.a = 1.0;
}
