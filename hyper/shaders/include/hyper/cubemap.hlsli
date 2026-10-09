#ifndef HYPER_CUBEMAP_HLSLI
#define HYPER_CUBEMAP_HLSLI

// Direction through texel centre uv of cube face 'face' (D3D/Vulkan face layout).
float3 CubeDirection(uint face, float2 uv)
{
    float u = uv.x * 2.0 - 1.0;
    float v = uv.y * 2.0 - 1.0;
    switch (face)
    {
    case 0:  return normalize(float3( 1.0, -v, -u));
    case 1:  return normalize(float3(-1.0, -v,  u));
    case 2:  return normalize(float3( u,  1.0,  v));
    case 3:  return normalize(float3( u, -1.0, -v));
    case 4:  return normalize(float3( u, -v,  1.0));
    default: return normalize(float3(-u, -v, -1.0));
    }
}

#endif
