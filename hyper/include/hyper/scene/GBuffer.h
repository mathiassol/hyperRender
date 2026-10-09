#pragma once

#include <hyper/platform.h>

#include <donut/render/GBuffer.h>

#include <memory>

namespace donut::engine { class FramebufferFactory; }

namespace hyper
{
    // The standard G-buffer every lighting module reads. It's Donut's (depth with
    // stencil, reverse-Z; albedo, specular, normals, emissive, object motion
    // vectors), plus the HDR target the frame is lit into.
    //
    //   Depth          D32S8: float reverse-Z, cleared to 0 (Donut's own choice where
    //                  D32S8 isn't supported)
    //   MotionVectors  RG16_FLOAT, pixels current -> previous; stencil bit 1 marks written pixels
    //   HdrColor       RGBA16_FLOAT, UAV + render target, linear scene-referred
    //   GBufferEmissive.w  the material ID + 1 when GpuDrivenRenderer drew the pixel, else 0
    class GBuffer : public donut::render::GBufferRenderTargets
    {
    public:
        static constexpr uint32_t MotionVectorStencilBit = 1;

        // How a pass that draws into the G-buffer tests and writes depth (reverse-Z: greater
        // or equal), and marks the pixels it drew in MotionVectorStencilBit, where
        // PostProcess's TAA takes their motion vectors instead of the camera's.
        [[nodiscard]] static nvrhi::DepthStencilState DepthStencil(bool reverseZ = true);

        nvrhi::TextureHandle HdrColor;
        // HdrColor + Depth: for passes drawn after lighting (sky, transparency).
        std::shared_ptr<donut::engine::FramebufferFactory> HdrFramebuffer;

        void Init(nvrhi::IDevice* device, dm::uint2 size, dm::uint sampleCount,
                  bool enableMotionVectors, bool useReverseProjection) override;
        void Clear(nvrhi::ICommandList* commandList) override;
    };
}
