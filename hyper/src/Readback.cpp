#include <hyper/Readback.h>

#include <hyper/App.h>

#include <cstring>
#include <string>

namespace hyper
{
    Readback::Readback(Context& ctx, uint64_t bytes, const char* name) : m_Ctx(ctx), m_Bytes(bytes)
    {
        for (uint32_t i = 0; i < c_Slots; ++i)
            m_Staging[i] = ctx.device->createBuffer(nvrhi::BufferDesc()
                .setDebugName(std::string(name) + "-readback-" + std::to_string(i)).setByteSize(bytes)
                .setCpuAccess(nvrhi::CpuAccessMode::Read).setInitialState(nvrhi::ResourceStates::CopyDest).setKeepInitialState(true));
    }

    void Readback::Copy(nvrhi::ICommandList* cmd, nvrhi::IBuffer* source, uint64_t sourceOffset)
    {
        cmd->copyBuffer(m_Staging[m_Next], 0, source, sourceOffset, m_Bytes);
        m_CopiedAt[m_Next] = m_Ctx.frameIndex;
        m_Next = (m_Next + 1) % c_Slots;
    }

    const void* Readback::LatestBytes()
    {
        // Two frames in flight while one records: a copy three frames old is done.
        constexpr uint64_t c_Latency = 3;
        uint32_t best = c_Slots;
        for (uint32_t i = 0; i < c_Slots; ++i)
        {
            const uint64_t at = m_CopiedAt[i];
            if (at != c_None && at + c_Latency <= m_Ctx.frameIndex && (best == c_Slots || at > m_CopiedAt[best]))
                best = i;
        }
        if (best != c_Slots && (m_DataFrame == c_None || m_CopiedAt[best] > m_DataFrame))
        {
            if (const void* p = m_Ctx.device->mapBuffer(m_Staging[best], nvrhi::CpuAccessMode::Read))
            {
                m_Data.assign(static_cast<const uint8_t*>(p), static_cast<const uint8_t*>(p) + m_Bytes);
                m_Ctx.device->unmapBuffer(m_Staging[best]);
                m_DataFrame = m_CopiedAt[best];
            }
        }
        return m_DataFrame == c_None ? nullptr : m_Data.data();
    }
}
