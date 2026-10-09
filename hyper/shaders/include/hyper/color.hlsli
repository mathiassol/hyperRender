#ifndef HYPER_COLOR_HLSLI
#define HYPER_COLOR_HLSLI

// Colour helpers. All inputs are linear Rec.709 unless noted.

float Luminance(float3 c)
{
    return dot(c, float3(0.2126, 0.7152, 0.0722));
}

float3 LinearToSrgb(float3 c)
{
    c = max(c, 0.0);
    return select(c <= 0.0031308, c * 12.92, 1.055 * pow(c, 1.0 / 2.4) - 0.055);
}

float3 SrgbToLinear(float3 c)
{
    return select(c <= 0.04045, c / 12.92, pow((c + 0.055) / 1.055, 2.4));
}

// --- Tonemappers: scene-referred linear in, display-referred linear [0,1] out ---

// AgX, minimal polynomial fit (Benjamin Wrensch, after Troy Sobotka's AgX).
float3 AgxContrast(float3 x)
{
    float3 x2 = x * x;
    float3 x4 = x2 * x2;
    return 15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x + 0.4298 * x2 + 0.1191 * x - 0.00232;
}

float3 TonemapAgx(float3 c)
{
    const float3x3 inset = float3x3(
        0.842479062253094,  0.0423282422610123, 0.0423756549057051,
        0.0784335999999992, 0.878468636469772,  0.0784336,
        0.0792237451477643, 0.0791661274605434, 0.879142973793104);
    const float3x3 outset = float3x3(
         1.19687900512017,  -0.0528968517574562, -0.0529716355144438,
        -0.0980208811401368, 1.15190312990417,   -0.0980434501171241,
        -0.0990297440797205, -0.0989611768448433, 1.15107367264116);
    const float minEv = -12.47393;
    const float maxEv = 4.026069;

    c = mul(max(c, 1e-10), inset);
    c = clamp(log2(c), minEv, maxEv);
    c = (c - minEv) / (maxEv - minEv);
    c = AgxContrast(c);
    c = mul(c, outset);
    return pow(saturate(c), 2.2);
}

// ACES filmic, Stephen Hill's fit of the RRT+ODT.
float3 TonemapAcesFitted(float3 c)
{
    const float3x3 inputMat = float3x3(
        0.59719, 0.07600, 0.02840,
        0.35458, 0.90834, 0.13383,
        0.04823, 0.01566, 0.83777);
    const float3x3 outputMat = float3x3(
         1.60475, -0.10208, -0.00327,
        -0.53108,  1.10813, -0.07276,
        -0.07367, -0.00605,  1.07602);

    c = mul(c, inputMat);
    float3 a = c * (c + 0.0245786) - 0.000090537;
    float3 b = c * (0.983729 * c + 0.4329510) + 0.238081;
    c = a / b;
    return saturate(mul(c, outputMat));
}

// Khronos PBR Neutral: keeps base colours intact up to a point, then desaturates highlights.
float3 TonemapKhronosNeutral(float3 c)
{
    const float startCompression = 0.8 - 0.04;
    const float desaturation = 0.15;

    float x = min(c.r, min(c.g, c.b));
    float offset = x < 0.08 ? x - 6.25 * x * x : 0.04;
    c -= offset;

    float peak = max(c.r, max(c.g, c.b));
    if (peak < startCompression)
        return c;

    const float d = 1.0 - startCompression;
    float newPeak = 1.0 - d * d / (peak + d - startCompression);
    c *= newPeak / peak;

    float g = 1.0 - 1.0 / (desaturation * (peak - newPeak) + 1.0);
    return lerp(c, newPeak, g);
}

float3 TonemapReinhard(float3 c)
{
    return c / (1.0 + Luminance(c));
}

#define HYPER_TONEMAP_AGX       0
#define HYPER_TONEMAP_ACES      1
#define HYPER_TONEMAP_NEUTRAL   2
#define HYPER_TONEMAP_REINHARD  3
#define HYPER_TONEMAP_CLAMP     4

float3 Tonemap(float3 c, uint op)
{
    switch (op)
    {
    case HYPER_TONEMAP_AGX:      return TonemapAgx(c);
    case HYPER_TONEMAP_ACES:     return TonemapAcesFitted(c);
    case HYPER_TONEMAP_NEUTRAL:  return TonemapKhronosNeutral(c);
    case HYPER_TONEMAP_REINHARD: return TonemapReinhard(c);
    default:                     return saturate(c);
    }
}

#endif
