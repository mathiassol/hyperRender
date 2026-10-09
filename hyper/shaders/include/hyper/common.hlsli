#ifndef HYPER_COMMON_HLSLI
#define HYPER_COMMON_HLSLI

// Shared by every hyper and app shader. Include as <hyper/common.hlsli>.

// Matrices are row-major with row vectors, as in donut::math: mul(v, M).
#pragma pack_matrix(row_major)

#include <donut/shaders/binding_helpers.hlsli>

static const float HYPER_PI = 3.14159265358979323846;
static const float HYPER_TWO_PI = 6.28318530717958647692;

#endif
