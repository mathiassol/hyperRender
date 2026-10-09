// Word copy between raw buffers. GpuScene gathers Donut's mesh buffers with
// it: their permanent state allows shader reads but not copy-source use.

#include <hyper/common.hlsli>

struct CopyConstants
{
    uint srcOffset;     // bytes, multiple of 4
    uint dstOffset;
    uint words;
    uint pad;
};

DECLARE_PUSH_CONSTANTS(CopyConstants, g_Copy, 0, 0);
ByteAddressBuffer t_Source : register(t0);
RWByteAddressBuffer u_Destination : register(u0);

[numthreads(64, 1, 1)]
void main(uint id : SV_DispatchThreadID)
{
    if (id >= g_Copy.words)
        return;
    u_Destination.Store(g_Copy.dstOffset + id * 4u, t_Source.Load(g_Copy.srcOffset + id * 4u));
}
