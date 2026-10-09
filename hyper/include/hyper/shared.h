#pragma once

// For a header C++ and HLSL share (a constant buffer's struct): HLSL's type names, from
// donut::math. C++ includes this, then the shared header; HLSL includes the shared header
// as it is. Lay the struct out in float4-sized rows, as HLSL packs it.

#include <donut/core/math/math.h>

using donut::math::float2;
using donut::math::float3;
using donut::math::float4;
using donut::math::float4x4;
using donut::math::int2;
using donut::math::int4;
using donut::math::uint;
using donut::math::uint2;
using donut::math::uint3;
using donut::math::uint4;
