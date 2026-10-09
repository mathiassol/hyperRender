#ifndef HYPER_VIEW_HLSLI
#define HYPER_VIEW_HLSLI

// Camera helpers over Donut's PlanarViewConstants (filled on the C++ side by
// hyper::Camera::GetView + PlanarView::FillPlanarViewConstants).

#include <hyper/common.hlsli>
#include <donut/shaders/view_cb.h>

struct Ray
{
    float3 origin;
    float3 direction;
};

// Primary ray through the centre of a pixel (window coordinates, 0,0 top-left).
Ray CameraRay(PlanarViewConstants view, float2 pixel)
{
    float2 clip = (pixel + 0.5) * view.windowToClipScale + view.windowToClipBias;
    // Reverse-Z: clip z = 1 is the near plane, which stays finite with an infinite far plane.
    float4 nearPoint = mul(float4(clip, 1.0, 1.0), view.matClipToWorld);
    nearPoint.xyz /= nearPoint.w;

    Ray ray;
    ray.origin = view.cameraDirectionOrPosition.xyz;
    ray.direction = normalize(nearPoint.xyz - ray.origin);
    return ray;
}

#endif
