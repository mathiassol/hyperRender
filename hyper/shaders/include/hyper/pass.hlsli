#ifndef HYPER_PASS_HLSLI
#define HYPER_PASS_HLSLI

// The bindings of a hyper::ComputePasses pass, and what every such pass needs from them.
//
//   b0  g_View        this frame's view (PlanarViewConstants)
//   b1  g_ViewPrev    the previous frame's
//   b2  the app's own constants
//   b3  push constants: each shader declares its own struct (at most 32 bytes)
//   t0..t9, u0..u3    the textures one dispatch passes
//   t20..t39          the app's shared resources
//   t40..t44, s40     the scene for rays (hyper/rt.hlsli), when the passes trace
//   s0  linear clamp, s1 linear wrap

#include <hyper/common.hlsli>
#include <hyper/color.hlsli>
#include <donut/shaders/view_cb.h>

ConstantBuffer<PlanarViewConstants> g_View : register(b0);
ConstantBuffer<PlanarViewConstants> g_ViewPrev : register(b1);
SamplerState s_Linear : register(s0);
SamplerState s_LinearWrap : register(s1);

float3 CameraPosition()
{
    return g_View.matViewToWorld[3].xyz;
}

// The world position a pixel's depth (reverse-Z) stands for.
float3 WorldFromDepth(float2 pixel, float depth)
{
    const float2 uv = (pixel + 0.5) * g_View.viewportSizeInv;
    const float4 clip = float4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    const float4 w = mul(clip, g_View.matClipToWorld);
    return w.xyz / w.w;
}

// Where a world point lands in a view, in pixels (centres at +0.5).
float2 PixelFromWorld(float3 p, PlanarViewConstants view)
{
    const float4 clip = mul(float4(p, 1.0), view.matWorldToClip);
    const float2 ndc = clip.xy / clip.w;
    return float2(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5) * view.viewportSize;
}

#endif
