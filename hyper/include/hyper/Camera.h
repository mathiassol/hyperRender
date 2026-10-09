#pragma once

#include <hyper/platform.h>

#include <donut/app/Camera.h>
#include <donut/core/math/math.h>
#include <donut/engine/View.h>

namespace hyper
{
    // The app's camera. hyper feeds it input (after ImGui and the app had
    // their turn) and animates it each frame; apps just read it.
    //
    // Controls (Donut's FirstPersonCamera): hold left mouse + drag to look,
    // WASD move, Q/E down/up, Shift faster, Ctrl slower.
    struct Camera
    {
        donut::app::FirstPersonCamera fly;
        float verticalFov = donut::math::radians(60.f);
        float zNear = 0.05f;
        bool enableInput = true;

        Camera()
        {
            fly.SetMoveSpeed(3.f);
            fly.LookAt(donut::math::float3(0.f, 1.5f, -5.f), donut::math::float3(0.f, 1.f, 0.f));
        }

        // Fills a planar view for a width x height viewport: reverse-Z,
        // infinite far plane, the layout Donut's passes expect.
        void GetView(donut::engine::PlanarView& view, uint32_t width, uint32_t height) const
        {
            const float aspect = float(width) / float(std::max(height, 1u));
            view.SetViewport(nvrhi::Viewport(float(width), float(height)));
            view.SetMatrices(fly.GetWorldToViewMatrix(),
                             donut::math::perspProjD3DStyleReverse(verticalFov, aspect, zNear));
            view.UpdateCache();
        }
    };
}
