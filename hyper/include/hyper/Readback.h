#pragma once

#include <hyper/platform.h>

#include <nvrhi/nvrhi.h>

#include <cstdint>
#include <vector>

namespace hyper
{
    struct Context;

    // Reads a GPU buffer back without stalling the frame: Copy() records a copy into one of
    // a ring of staging buffers, and Latest() hands back the newest copy the GPU is certainly
    // done with. Donut keeps up to two frames in flight while one records, so that's a copy
    // from three frames ago or older.
    //
    //     hyper::Readback stats(ctx, 16, "app/stats");
    //     stats.Copy(cmd, counters);                       // when there's something to read
    //     if (const uint32_t* w = stats.Latest<uint32_t>())
    //         ...
    class Readback
    {
    public:
        Readback(Context& ctx, uint64_t bytes, const char* name);

        void Copy(nvrhi::ICommandList* cmd, nvrhi::IBuffer* source, uint64_t sourceOffset = 0);

        // The newest finished copy, null before there is one. Valid until the next call.
        template <typename T>
        [[nodiscard]] const T* Latest() { return static_cast<const T*>(LatestBytes()); }
        // The frame (Context::frameIndex) the data Latest() returned was copied in.
        [[nodiscard]] uint64_t LatestFrame() const { return m_DataFrame; }

    private:
        static constexpr uint32_t c_Slots = 4;
        static constexpr uint64_t c_None = ~0ull;

        Context& m_Ctx;
        uint64_t m_Bytes;
        nvrhi::BufferHandle m_Staging[c_Slots];
        uint64_t m_CopiedAt[c_Slots] = { c_None, c_None, c_None, c_None };
        uint32_t m_Next = 0;
        std::vector<uint8_t> m_Data;
        uint64_t m_DataFrame = c_None;

        const void* LatestBytes();
    };
}
