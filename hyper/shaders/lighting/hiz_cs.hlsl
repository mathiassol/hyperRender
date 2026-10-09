// Hi-Z depth pyramid from a reverse-Z depth buffer. Each texel holds
//   x = nearest depth in its footprint (max, since reverse-Z puts near at 1)
//   y = farthest depth (min)
// SSR traces against x; occlusion culling tests against y.

#include <hyper/common.hlsli>

struct HizConstants
{
    uint2 dstSize;
    uint2 srcSize;
};

DECLARE_PUSH_CONSTANTS(HizConstants, g_Hiz, 0, 0);
Texture2D<float> t_Depth : register(t0);
Texture2D<float2> t_Source : register(t1);
VK_IMAGE_FORMAT("rg32f") RWTexture2D<float2> u_Dest : register(u0);

[numthreads(8, 8, 1)]
void copy(uint2 id : SV_DispatchThreadID)
{
    if (any(id >= g_Hiz.dstSize))
        return;
    float d = t_Depth[id];
    u_Dest[id] = float2(d, d);
}

[numthreads(8, 8, 1)]
void reduce(uint2 id : SV_DispatchThreadID)
{
    if (any(id >= g_Hiz.dstSize))
        return;

    // A texel covers 2x2 source texels; 3 wide/tall where the source size is odd,
    // so the last row/column isn't lost.
    uint2 base = id * 2;
    uint2 extent = uint2(g_Hiz.srcSize.x & 1 ? 3 : 2, g_Hiz.srcSize.y & 1 ? 3 : 2);
    float nearest = 0.0;
    float farthest = 1.0;
    for (uint y = 0; y < extent.y; ++y)
    for (uint x = 0; x < extent.x; ++x)
    {
        uint2 p = min(base + uint2(x, y), g_Hiz.srcSize - 1);
        float2 s = t_Source[p];
        nearest = max(nearest, s.x);
        farthest = min(farthest, s.y);
    }
    u_Dest[id] = float2(nearest, farthest);
}
