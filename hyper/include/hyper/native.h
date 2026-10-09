#pragma once

// Level 0: raw Vulkan next to NVRHI.
//
// NVRHI tracks resource states and batches barriers. Before recording raw
// commands, tell it which states you need and flush:
//
//     cmd->setTextureState(tex, nvrhi::AllSubresources, nvrhi::ResourceStates::CopyDest);
//     cmd->commitBarriers();
//     {
//         hyper::vk::RawCommands raw(cmd);
//         vkCmdClearColorImage(raw.cmd, hyper::vk::Image(tex), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, ...);
//     }   // NVRHI forgets its bound pipeline/state here, so the next draw rebinds everything
//
// Vulkan entry points go through the vulkan.hpp default dispatcher that
// Donut initialises, so no vulkan-1.lib link is needed.

#include <hyper/platform.h>

#ifndef VULKAN_HPP_DISPATCH_LOADER_DYNAMIC
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#endif
#include <vulkan/vulkan.hpp>

#include <nvrhi/nvrhi.h>

namespace hyper::vk
{
    inline bool IsVulkan(nvrhi::IDevice* device)
    {
        return device->getGraphicsAPI() == nvrhi::GraphicsAPI::VULKAN;
    }

    inline VkInstance Instance(nvrhi::IDevice* d) { return d->getNativeObject(nvrhi::ObjectTypes::VK_Instance); }
    inline VkPhysicalDevice PhysicalDevice(nvrhi::IDevice* d) { return d->getNativeObject(nvrhi::ObjectTypes::VK_PhysicalDevice); }
    inline VkDevice Device(nvrhi::IDevice* d) { return d->getNativeObject(nvrhi::ObjectTypes::VK_Device); }
    inline VkImage Image(nvrhi::ITexture* t) { return t->getNativeObject(nvrhi::ObjectTypes::VK_Image); }
    inline VkBuffer Buffer(nvrhi::IBuffer* b) { return b->getNativeObject(nvrhi::ObjectTypes::VK_Buffer); }
    inline VkCommandBuffer CommandBuffer(nvrhi::ICommandList* c) { return c->getNativeObject(nvrhi::ObjectTypes::VK_CommandBuffer); }

    // The vulkan.hpp dispatcher, for calling any entry point Donut loaded.
    inline const auto& Dispatch() { return VULKAN_HPP_DEFAULT_DISPATCHER; }

    // Scope for raw commands on an open NVRHI command list. Flushes pending
    // barriers on entry; on exit drops NVRHI's cached bindings, which raw
    // commands may have invalidated.
    struct RawCommands
    {
        nvrhi::ICommandList* list;
        VkCommandBuffer cmd;

        explicit RawCommands(nvrhi::ICommandList* commandList)
            : list(commandList), cmd(CommandBuffer(commandList))
        {
            list->commitBarriers();
        }
        ~RawCommands() { list->clearState(); }

        RawCommands(const RawCommands&) = delete;
        RawCommands& operator=(const RawCommands&) = delete;
    };
}
