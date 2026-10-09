// XeGTAO (Intel, MIT; modules/ao/thirdparty/xegtao) wrapped for hyper's G-buffer:
//   prefilter   reverse-Z depth -> viewspace depth, 5 mips
//   main        GTAO (slices/steps by quality), world normals from the G-buffer
//   denoise     edge-aware spatial filter; the last pass applies the final scale
//   resolve     packed uint term -> R8_UNORM, which Donut's deferred lighting reads

#define XE_GTAO_USE_HALF_FLOAT_PRECISION 0
// The settings from the constants, not XeGTAO's compiled-in defaults (with those, thin
// occluder compensation and the other tuning did nothing).
#define XE_GTAO_USE_DEFAULT_CONSTANTS 0
#define XE_GTAO_FP32_DEPTHS
#define VA_SATURATE saturate

#include <hyper/common.hlsli>
#include "../../thirdparty/xegtao/XeGTAO.h"

ConstantBuffer<GTAOConstants> g_GTAOConsts : register(b0);

struct GtaoExtra
{
    float4 worldToView0;   // rows of the world->view rotation
    float4 worldToView1;
    float4 worldToView2;
    uint qualitySlices;
    uint qualitySteps;
    uint2 size;
    uint multiBounce;      // 1: brighten by the albedo (resolve)
    uint3 pad;
};
DECLARE_PUSH_CONSTANTS(GtaoExtra, g_Extra, 1, 0);

#include "../../thirdparty/xegtao/XeGTAO.hlsli"

SamplerState s_PointClamp : register(s0);

// R2 sequence over XeGTAO.h's 64x64 Hilbert curve (as in XeGTAO's sample).
lpfloat2 SpatioTemporalNoise(uint2 pixel, uint temporalIndex)
{
    uint index = HilbertIndex(pixel.x % 64, pixel.y % 64) + 288 * (temporalIndex % 64);
    return lpfloat2(frac(0.5 + index * float2(0.75487766624669276005, 0.5698402909980532659114)));
}

// ---- prefilter

Texture2D<float> t_RawDepth : register(t0);
RWTexture2D<lpfloat> u_Depth0 : register(u0);
RWTexture2D<lpfloat> u_Depth1 : register(u1);
RWTexture2D<lpfloat> u_Depth2 : register(u2);
RWTexture2D<lpfloat> u_Depth3 : register(u3);
RWTexture2D<lpfloat> u_Depth4 : register(u4);

[numthreads(8, 8, 1)]
void prefilter(uint2 dispatchThreadID : SV_DispatchThreadID, uint2 groupThreadID : SV_GroupThreadID)
{
    XeGTAO_PrefilterDepths16x16(dispatchThreadID, groupThreadID, g_GTAOConsts, t_RawDepth, s_PointClamp,
        u_Depth0, u_Depth1, u_Depth2, u_Depth3, u_Depth4);
}

// ---- main

Texture2D<lpfloat> t_WorkingDepth : register(t0);
Texture2D<float4> t_Normals : register(t1);
RWTexture2D<uint> u_AOTerm : register(u0);
RWTexture2D<unorm float> u_Edges : register(u1);

[numthreads(XE_GTAO_NUMTHREADS_X, XE_GTAO_NUMTHREADS_Y, 1)]
void main(uint2 pixel : SV_DispatchThreadID)
{
    float3 n = normalize(t_Normals[pixel].xyz);   // world space
    float3 viewNormal = float3(dot(g_Extra.worldToView0.xyz, n), dot(g_Extra.worldToView1.xyz, n), dot(g_Extra.worldToView2.xyz, n));
    XeGTAO_MainPass(pixel, lpfloat(g_Extra.qualitySlices), lpfloat(g_Extra.qualitySteps),
        SpatioTemporalNoise(pixel, g_GTAOConsts.NoiseIndex), (lpfloat3)viewNormal, g_GTAOConsts,
        t_WorkingDepth, s_PointClamp, u_AOTerm, u_Edges);
}

// ---- denoise (two horizontal pixels per thread)

Texture2D<uint> t_AOTerm : register(t0);
Texture2D<lpfloat> t_Edges : register(t1);
RWTexture2D<uint> u_OutAOTerm : register(u0);

[numthreads(XE_GTAO_NUMTHREADS_X, XE_GTAO_NUMTHREADS_Y, 1)]
void denoise(uint2 id : SV_DispatchThreadID)
{
    XeGTAO_Denoise(id * uint2(2, 1), g_GTAOConsts, t_AOTerm, t_Edges, s_PointClamp, u_OutAOTerm, false);
}

[numthreads(XE_GTAO_NUMTHREADS_X, XE_GTAO_NUMTHREADS_Y, 1)]
void denoise_last(uint2 id : SV_DispatchThreadID)
{
    XeGTAO_Denoise(id * uint2(2, 1), g_GTAOConsts, t_AOTerm, t_Edges, s_PointClamp, u_OutAOTerm, true);
}

// ---- resolve

RWTexture2D<unorm float> u_AO : register(u1);
Texture2D<float4> t_Albedo : register(t2);

// Multi-bounce (Jimenez et al. 2016, "Practical Real-Time Strategies for Accurate Indirect
// Occlusion"): the visibility a surface of this albedo really sees once light bounces
// between it and its occluders, fitted to path-traced references.
float MultiBounce(float visibility, float albedo)
{
    const float a = 2.0404 * albedo - 0.3324, b = -4.7951 * albedo + 0.6417, c = 2.7552 * albedo + 0.6903;
    return max(visibility, ((visibility * a + b) * visibility + c) * visibility);
}

[numthreads(8, 8, 1)]
void resolve(uint2 id : SV_DispatchThreadID)
{
    if (any(id >= g_Extra.size))
        return;
    float visibility = saturate(float(t_AOTerm[id]) / 255.0);
    if (g_Extra.multiBounce != 0)
        visibility = MultiBounce(visibility, dot(t_Albedo[id].rgb, float3(0.2126, 0.7152, 0.0722)));
    u_AO[id] = saturate(visibility);
}
