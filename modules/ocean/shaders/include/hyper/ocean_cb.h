#ifndef HYPER_OCEAN_CB_H
#define HYPER_OCEAN_CB_H

// The ocean's constants, shared by Ocean.cpp and its shaders (C++ includes hyper/shared.h
// first). The layout is in float4 rows.

#include <donut/shaders/view_cb.h>

#define OCEAN_N 256          // FFT size per cascade
#define OCEAN_LOG_N 8
#define OCEAN_MIPS 9         // 256 .. 1
#define OCEAN_CASCADES 3
#define OCEAN_GRID 128       // cells along a clipmap level's side
#define OCEAN_MAX_LEVELS 16

// The simulation (ocean_cs.hlsl): evolve, FFT, assemble, mips.
struct OceanSimConstants
{
    float4 patchSize;        // metres a cascade's texture spans; w unused
    float time;              // seconds
    float choppiness;        // horizontal displacement scale
    float foamKeep;          // fraction of last frame's foam kept
    float foamThreshold;     // Jacobian below which the surface folds into foam
    float foamGain;
    float3 pad;
};

// One clipmap level: a square of OCEAN_GRID x OCEAN_GRID cells around the camera.
struct OceanLevel
{
    float2 center;           // world xz, a multiple of 2 x cellSize
    float cellSize;
    float halfExtent;        // OCEAN_GRID / 2 x cellSize
};

// The surface (surface.hlsl).
struct OceanSurfaceConstants
{
    PlanarViewConstants view;
    PlanarViewConstants viewPrev;
    OceanLevel levels[OCEAN_MAX_LEVELS];
    float4 patchSize;        // per cascade
    float4 invPatchSize;
    float3 cameraPos;
    float earthRadius;       // metres; the surface curves away with distance
    float3 sunDirection;     // towards the sun
    float pad1;
    float3 sunRadiance;      // colour x irradiance
    float roughness;         // the smallest, of a calm surface
    float3 waterAlbedo;      // the body of the water, lit like a diffuse surface
    float foamAmount;
    float3 scatterColor;     // light through thin crests
    float scatterStrength;
    float3 foamAlbedo;
    float waveHeight;        // a typical crest height, metres
    float3 pad0;
    float seaLevel;
    uint levelCount;         // drawn, finest first (a high camera skips the finest)
    float morphStart;        // where in a level vertices start to morph to the next one (0..1)
    float2 pad;
};

#endif
