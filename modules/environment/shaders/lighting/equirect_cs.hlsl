// Equirectangular (latitude-longitude) HDR image -> cubemap mip 0, all faces.

#include <hyper/common.hlsli>
#include <hyper/cubemap.hlsli>

struct EquirectConstants
{
    uint size;
    float intensity;
    float rotation;
    float pad;
};

DECLARE_PUSH_CONSTANTS(EquirectConstants, g_Equirect, 0, 0);
Texture2D<float4> t_Source : register(t0);
SamplerState s_Linear : register(s0);
VK_IMAGE_FORMAT("rgba16f") RWTexture2DArray<float4> u_Cube : register(u0);

[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= g_Equirect.size || id.y >= g_Equirect.size)
        return;
    float3 d = CubeDirection(id.z, (float2(id.xy) + 0.5) / float(g_Equirect.size));
    float azimuth = atan2(d.x, -d.z) + g_Equirect.rotation;
    float2 uv = float2(azimuth / HYPER_TWO_PI + 0.5, acos(clamp(d.y, -1.0, 1.0)) / HYPER_PI);
    float3 c = t_Source.SampleLevel(s_Linear, uv, 0).rgb * g_Equirect.intensity;
    u_Cube[id] = float4(min(c, 65000.0), 1.0);
}
