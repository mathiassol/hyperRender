#ifndef HYPER_PLANET_TERRAIN_H
#define HYPER_PLANET_TERRAIN_H

// A planet's surface as one function of the direction from its centre (the planet module):
// its height and what it is made of. C++ (after <hyper/shared.h>)
// and HLSL both compile this file, so the CPU walks on exactly the ground the GPU draws.
// Everything is in the planet's own axes (it turns with the planet), lengths in metres.
//
// Where a body's elevation is measured (a map: 1440 x 720, 0.25 degrees, rows from the
// north pole, columns east from the prime meridian), the ground is the map, and only the
// detail and craters finer than its cells are generated. The includer says how to read a
// map's samples: TERRAIN_MAP_LOAD(slice, x, y), metres (none: no maps).
//
// Patches: a few places can have a finer map of their own, over a latitude-longitude box
// (TerrainPatch): there the ground is the patch, with only what's finer than its samples
// generated, faded into the global map over its border. Patches nest: each, in its
// order, is faded over what's there already, so a fine one sits in a coarser one round it. The includer says how to read
// them: TERRAIN_PATCH(i) gives a TerrainPatch, TERRAIN_PATCH_LOAD(i) a sample (none: no
// patches).
//
// Plain float arithmetic only, in the same order in both languages: integer hashes for the
// lattice, no intrinsics that differ. Floats limit the detail: a planet's finest wavelength
// is at least radius / 2^17 (about 50 m on Earth), where the lattice is still exact.

#define TERRAIN_GRID 65              // vertices along a tile's edge (64 quads)
#define TERRAIN_GROUP 8
#define TERRAIN_MAX_BODIES 8
#define TERRAIN_MATERIAL_WATER 1u    // the vertex is under the sea
#define TERRAIN_MATERIAL_LAVA 2u
#define TERRAIN_MAP_W 1440
#define TERRAIN_MAP_H 720
#ifdef __cplusplus
// C++ reads them from the planet module (PlanetTerrain.cpp), in every file alike: these
// functions are inline, so they must be the same wherever they're compiled.
struct TerrainPatch;
namespace hyper::terrain
{
    float MapLoad(int slice, int x, int y);
    const TerrainPatch& Patch(unsigned i);
    float PatchLoad(unsigned i);
}
#undef TERRAIN_MAP_LOAD
#define TERRAIN_MAP_LOAD(slice, x, y) ::hyper::terrain::MapLoad(slice, x, y)
#define TERRAIN_PATCH(i) ::hyper::terrain::Patch(i)
#define TERRAIN_PATCH_LOAD(i) ::hyper::terrain::PatchLoad(i)
#endif
#ifndef TERRAIN_MAP_LOAD
#define TERRAIN_MAP_LOAD(slice, x, y) 0.0f
#endif
#define TERRAIN_MAX_PATCHES 16
#define TERRAIN_PATCH_SAMPLES (16u * 1024u * 1024u)   // all the patches' samples together, at most

// A finer map over part of a body's ground: width x height samples, rows from the north,
// columns going east, each a point (not a cell's average), metres above the datum. After
// them, its coarser levels: each a tent-filtered half of the one before (((width - 1) >> l)
// + 1 by ((height - 1) >> l) + 1, sample k on the finest's sample k 2^l), while both are
// at least 4: what a coarse tile reads, so its few vertices don't alias the fine ground.
struct TerrainPatch
{
    float3 origin;           // the first (north-west) sample's direction (unit, the planet's axes)
    float cell;              // metres between its samples (the larger way): detail is generated below
    float dLat;              // radians from one row to the next (going south)
    float dLon;              //          one column to the next (going east)
    uint width;
    uint height;
    uint offset;             // its first sample among all the patches'
    float pad0, pad1, pad2;
};

struct PlanetTerrainConstants
{
    float radius;            // the sea's (or the mean ground's) radius
    float scale;             // metres per unit of the shape (set so its range is the body's)
    float offset;            // metres subtracted: the sea, or the mean ground, is height 0
    uint seed;

    float minWavelength;     // the finest detail, metres
    float mountains;         // how much of the relief is ridges, 0..1
    float craters;           // the share of each crater size's cells that hold a crater, 0..1
    float maria;             // the share of the surface that is dark lowland (lunar maria, Martian basalt)

    float3 brightAlbedo;     // highlands, dust, bare rock
    float ocean;             // 1: water fills everything below height 0

    float3 darkAlbedo;       // maria, basalt
    float iceTemperature;    // K: colder ground is ice or snow (0: never)

    float3 iceAlbedo;
    float meltTemperature;   // K: hotter ground is lava (0: never)

    float3 sandAlbedo;       // deserts
    float vegetation;        // 1: plants where it's warm and not dry

    float3 plantAlbedo;
    float meanTemperature;   // K

    float3 substellar;       // a tidally locked body's hot point (unit, planet axes); else 0
    float substellar4;       // its temperature^4, K^4

    float redistribution;    // the share of the absorbed heat that reaches the night side
    float insolationP2;      // the annual-mean insolation's P2(sin latitude) coefficient (from the obliquity)
    float heatTransport;     // 0..1: how much of the latitude contrast air and sea even out
    float lapseRate;         // K per metre up

    float3 lavaAlbedo;
    float craterGravity;     // m/s^2: where simple craters turn complex

    float map;               // > 0: the measured elevation map's slice + 1
    float detailStart;       // m: with a map, detail is generated below this wavelength
    float detailAmplitude;   // m: that detail's rms at 1 km wavelength
    float craterMaxRadius;   // m: with a map, only craters smaller than its cells are generated

    float mariaBelow;        // with a map, < 0: the dark lowlands are where it's below this height (m)
    uint patchFirst;         // its patches: TERRAIN_PATCH(patchFirst) on, patchCount of them
    uint patchCount;
    float patchSpacing;      // m: how far apart the samples asking are; patches are read at the level for it (0: the finest)
};

#ifdef __cplusplus
// (<cmath> comes with <hyper/shared.h>: the shader build scans every #include, even here)
#define TERRAIN_FN inline
#define TERRAIN_OUT(t) t&
inline float terrain_floor(float x) { return std::floor(x); }
inline float terrain_abs(float x) { return std::fabs(x); }
inline float terrain_sqrt(float x) { return std::sqrt(x); }
inline float terrain_pow(float x, float y) { return std::pow(x, y); }
inline float terrain_min(float a, float b) { return a < b ? a : b; }
inline float terrain_max(float a, float b) { return a > b ? a : b; }
inline float terrain_saturate(float x) { return x < 0.f ? 0.f : (x > 1.f ? 1.f : x); }
inline float3 terrain_lerp(float3 a, float3 b, float t) { return a + (b - a) * t; }
inline float terrain_asin(float x) { return std::asin(x); }
inline float terrain_atan2(float y, float x) { return std::atan2(y, x); }
inline float terrain_log2(float x) { return std::log2(x); }
#else
#define TERRAIN_FN
#define TERRAIN_OUT(t) out t
float terrain_floor(float x) { return floor(x); }
float terrain_abs(float x) { return abs(x); }
float terrain_sqrt(float x) { return sqrt(x); }
float terrain_pow(float x, float y) { return pow(x, y); }
float terrain_min(float a, float b) { return a < b ? a : b; }
float terrain_max(float a, float b) { return a > b ? a : b; }
float terrain_saturate(float x) { return x < 0.0 ? 0.0 : (x > 1.0 ? 1.0 : x); }
float3 terrain_lerp(float3 a, float3 b, float t) { return a + (b - a) * t; }
float terrain_asin(float x) { return asin(x); }
float terrain_atan2(float y, float x) { return atan2(y, x); }
float terrain_log2(float x) { return log2(x); }
#endif

TERRAIN_FN float terrain_smoothstep(float a, float b, float x)
{
    const float t = terrain_saturate((x - a) / (b - a));
    return t * t * (3.0f - 2.0f * t);
}

// lowbias32 (Chris Wellons): a full-avalanche integer hash.
TERRAIN_FN uint terrain_hash(uint x)
{
    x ^= x >> 16u;
    x *= 0x7feb352du;
    x ^= x >> 15u;
    x *= 0x846ca68bu;
    x ^= x >> 16u;
    return x;
}

TERRAIN_FN uint terrain_hash3(int x, int y, int z, uint seed)
{
    return terrain_hash(uint(x) * 0x8da6b343u ^ terrain_hash(uint(y) * 0xd8163841u ^ terrain_hash(uint(z) * 0xcb1ab31fu ^ seed)));
}

TERRAIN_FN float terrain_unit(uint h) { return float(h & 0xffffu) * (1.0f / 65535.0f); }   // 0..1

// The gradient at a lattice point dotted with the offset from it.
TERRAIN_FN float terrain_corner(int x, int y, int z, uint seed, float dx, float dy, float dz)
{
    const uint h = terrain_hash3(x, y, z, seed);
    const float gx = float(h & 1023u) * (2.0f / 1023.0f) - 1.0f;
    const float gy = float((h >> 10u) & 1023u) * (2.0f / 1023.0f) - 1.0f;
    const float gz = float((h >> 20u) & 1023u) * (2.0f / 1023.0f) - 1.0f;
    return gx * dx + gy * dy + gz * dz;
}

TERRAIN_FN float terrain_fade(float t) { return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f); }

// Gradient noise (Perlin's, random gradients), about -1..1.
TERRAIN_FN float terrain_noise(float3 p, uint seed)
{
    const float fx = terrain_floor(p.x), fy = terrain_floor(p.y), fz = terrain_floor(p.z);
    const int ix = int(fx), iy = int(fy), iz = int(fz);
    const float x = p.x - fx, y = p.y - fy, z = p.z - fz;
    const float u = terrain_fade(x), v = terrain_fade(y), w = terrain_fade(z);
    const float c000 = terrain_corner(ix, iy, iz, seed, x, y, z);
    const float c100 = terrain_corner(ix + 1, iy, iz, seed, x - 1.0f, y, z);
    const float c010 = terrain_corner(ix, iy + 1, iz, seed, x, y - 1.0f, z);
    const float c110 = terrain_corner(ix + 1, iy + 1, iz, seed, x - 1.0f, y - 1.0f, z);
    const float c001 = terrain_corner(ix, iy, iz + 1, seed, x, y, z - 1.0f);
    const float c101 = terrain_corner(ix + 1, iy, iz + 1, seed, x - 1.0f, y, z - 1.0f);
    const float c011 = terrain_corner(ix, iy + 1, iz + 1, seed, x, y - 1.0f, z - 1.0f);
    const float c111 = terrain_corner(ix + 1, iy + 1, iz + 1, seed, x - 1.0f, y - 1.0f, z - 1.0f);
    const float x00 = c000 + (c100 - c000) * u, x10 = c010 + (c110 - c010) * u;
    const float x01 = c001 + (c101 - c001) * u, x11 = c011 + (c111 - c011) * u;
    const float y0 = x00 + (x10 - x00) * v, y1 = x01 + (x11 - x01) * v;
    return (y0 + (y1 - y0) * w) * 1.6f;
}

// Fractal noise: octaves doubling in frequency, each `gain` as strong.
TERRAIN_FN float terrain_fbm(float3 p, uint seed, int octaves, float gain)
{
    float sum = 0.0f, amp = 1.0f, norm = 0.0f;
    for (int i = 0; i < octaves; ++i)
    {
        sum += terrain_noise(p, seed + uint(i) * 101u) * amp;
        norm += amp;
        amp *= gain;
        p = p * 2.0f;
    }
    return sum / norm;
}

// Catmull-Rom weights: through the samples, smooth across the cells.
TERRAIN_FN float terrain_cubic(float a, float b, float c, float d, float t)
{
    return b + 0.5f * t * (c - a + t * (2.0f * a - 5.0f * b + 4.0f * c - d + t * (3.0f * (b - c) + d - a)));
}

TERRAIN_FN float terrain_mapRow(int slice, int y, int x0, float fx)
{
    const int w = TERRAIN_MAP_W;
    const int xa = (x0 - 1 + w) % w, xb = x0 % w, xc = (x0 + 1) % w, xd = (x0 + 2) % w;
    return terrain_cubic(TERRAIN_MAP_LOAD(slice, xa, y), TERRAIN_MAP_LOAD(slice, xb, y), TERRAIN_MAP_LOAD(slice, xc, y),
                         TERRAIN_MAP_LOAD(slice, xd, y), fx);
}

// The measured elevation under a direction, metres.
TERRAIN_FN float terrain_mapHeight(PlanetTerrainConstants c, float3 dir)
{
    const float pi = 3.14159265f;
    const float lat = terrain_asin(dir.z < -1.0f ? -1.0f : (dir.z > 1.0f ? 1.0f : dir.z));
    float u = terrain_atan2(dir.y, dir.x) * (0.5f / pi);
    if (u < 0.0f)
        u += 1.0f;
    const float x = u * float(TERRAIN_MAP_W) - 0.5f, y = (0.5f - lat / pi) * float(TERRAIN_MAP_H) - 0.5f;
    const float fx0 = terrain_floor(x), fy0 = terrain_floor(y);
    const int x0 = (int(fx0) + TERRAIN_MAP_W) % TERRAIN_MAP_W, y0 = int(fy0);
    const float fx = x - fx0, fy = y - fy0;
    const int slice = int(c.map) - 1;
    const int h = TERRAIN_MAP_H - 1;
    const int ya = y0 - 1 < 0 ? 0 : (y0 - 1 > h ? h : y0 - 1), yb = y0 < 0 ? 0 : (y0 > h ? h : y0);
    const int yc = y0 + 1 < 0 ? 0 : (y0 + 1 > h ? h : y0 + 1), yd = y0 + 2 < 0 ? 0 : (y0 + 2 > h ? h : y0 + 2);
    return terrain_cubic(terrain_mapRow(slice, ya, x0, fx), terrain_mapRow(slice, yb, x0, fx), terrain_mapRow(slice, yc, x0, fx),
                         terrain_mapRow(slice, yd, x0, fx), fy);
}

// Where the dark lowlands are, 0..1 (maria, basalt plains).
TERRAIN_FN float terrain_maria(PlanetTerrainConstants c, float3 dir)
{
    if (c.map > 0.0f && c.mariaBelow < 0.0f)
        return 1.0f - terrain_smoothstep(c.mariaBelow - 400.0f, c.mariaBelow + 400.0f, terrain_mapHeight(c, dir));
    if (c.maria <= 0.0f)
        return 0.0f;
    const float n = terrain_fbm(dir * 1.7f, c.seed ^ 0x5bd1e995u, 4, 0.55f);
    // The threshold that covers about `maria` of the sphere (the noise is roughly normal, sigma 0.3).
    const float threshold = 0.3f * (1.0f - 2.0f * c.maria) * 1.2f;
    return terrain_smoothstep(threshold - 0.04f, threshold + 0.04f, n);
}

// Craters: in cells of each size (doubling), a share of them hold one, at a random place,
// of a random size. Simple bowls 0.2 as deep as wide, shallower once gravity makes them
// complex (past about 15 km on the Moon, scaling as 1 / g); a raised rim; ejecta fading out.
TERRAIN_FN float terrain_craters(PlanetTerrainConstants c, float3 dir)
{
    if (c.craters <= 0.0f)
        return 0.0f;
    float h = 0.0f;
    const float transition = 15000.0f * 1.62f / terrain_max(c.craterGravity, 0.01f);
    for (int k = 0; k < 14; ++k)
    {
        const float f = 4.0f * terrain_pow(2.0f, float(k));
        // Done once they're too small to draw (and with a map, begun once they're under its cells).
        if (0.2f / f * c.radius < 2.0f * c.minWavelength)
            break;
        if (c.map > 0.0f && 0.06f / f * c.radius > c.craterMaxRadius)
            continue;
        const float3 p = dir * f;
        const float bx = terrain_floor(p.x - 0.5f), by = terrain_floor(p.y - 0.5f), bz = terrain_floor(p.z - 0.5f);
        for (int n = 0; n < 8; ++n)
        {
            const int cx = int(bx) + (n & 1), cy = int(by) + ((n >> 1) & 1), cz = int(bz) + ((n >> 2) & 1);
            const uint s = terrain_hash3(cx, cy, cz, c.seed ^ (0x27d4eb2du + uint(k)));
            if (terrain_unit(s) > c.craters)
                continue;
            // Its centre, somewhere in the cell, on the sphere.
            float3 centre = float3(float(cx) + terrain_unit(terrain_hash(s ^ 1u)), float(cy) + terrain_unit(terrain_hash(s ^ 2u)),
                                   float(cz) + terrain_unit(terrain_hash(s ^ 3u))) * (1.0f / f);
            const float len = terrain_sqrt(dot(centre, centre));
            if (len < 1e-6f)
                continue;
            centre = centre * (1.0f / len);
            // Radius in unit-sphere units: at most 0.2 of a cell, so its ejecta (to 2.4 radii)
            // stay within the half cell the 2 x 2 x 2 neighbourhood is sure to cover.
            const float rUnit = (0.06f + 0.14f * terrain_unit(terrain_hash(s ^ 4u))) / f;
            if (c.map > 0.0f && rUnit * c.radius > c.craterMaxRadius)
                continue;
            const float3 d = dir - centre;
            const float x = terrain_sqrt(dot(d, d)) / rUnit;
            if (x >= 2.4f)
                continue;
            const float diameter = 2.0f * rUnit * c.radius;
            const float shallow = terrain_min(1.0f, terrain_pow(transition / diameter, 0.6f));
            const float depth = 0.2f * diameter * shallow, rim = 0.04f * diameter * shallow;
            // The ejecta fall off as x^-3, brought to 0 at the edge so nothing steps there.
            h += x < 1.0f ? depth * (x * x - 1.0f) + rim : rim * (1.0f / (x * x * x) - 0.0723f) * (1.0f / 0.9277f);
        }
    }
    return h;
}

// The ground's shape without craters, in shape units (scaled by c.scale): continents,
// ridged mountains on them, and detail down to the finest wavelength, its amplitude
// falling as wavelength^0.8 (about a natural terrain's spectrum).
TERRAIN_FN float terrain_shape(PlanetTerrainConstants c, float3 dir)
{
    const float continents = terrain_fbm(dir * 1.3f, c.seed, 5, 0.5f);
    float ridges = 0.0f;
    if (c.mountains > 0.0f)
    {
        float amp = 0.5f, f = 3.0f, weight = 1.0f;
        for (int i = 0; i < 6; ++i)
        {
            float n = 1.0f - terrain_abs(terrain_noise(dir * f, c.seed ^ (0x9e3779b9u + uint(i))));
            n = n * n * weight;
            weight = terrain_saturate(n * 2.0f);
            ridges += n * amp;
            amp *= 0.5f;
            f *= 2.0f;
        }
        ridges *= terrain_smoothstep(-0.05f, 0.35f, continents);
    }
    float detail = 0.0f;
    {
        const float top = c.radius / terrain_max(c.minWavelength, 1.0f);
        float f = 24.0f, amp = 0.12f;
        for (int i = 0; i < 18 && f < top; ++i)
        {
            detail += terrain_noise(dir * f, c.seed ^ (0x85ebca6bu + uint(i))) * amp;
            f *= 2.0f;
            amp *= 0.574f;   // 2^-0.8
        }
    }
    // The lowlands are flooded and smooth.
    const float lowland = terrain_maria(c, dir);
    return continents + ridges * c.mountains - lowland * 0.35f + detail * (1.0f - 0.7f * lowland);
}

// With a map: what's finer than `start` (its cells), amplitude falling as wavelength^0.8
// from detailAmplitude at 1 km; the lowlands smoother.
TERRAIN_FN float terrain_mapDetail(PlanetTerrainConstants c, float3 dir, float start)
{
    const float top = c.radius / terrain_max(c.minWavelength, 1.0f);
    float f = c.radius / terrain_max(start, 1.0f);
    float amp = c.detailAmplitude * terrain_pow(start / 1000.0f, 0.8f);
    float detail = 0.0f;
    for (int i = 0; i < 20 && f < top; ++i)
    {
        detail += terrain_noise(dir * f, c.seed ^ (0x85ebca6bu + uint(i))) * amp;
        f *= 2.0f;
        amp *= 0.574f;   // 2^-0.8
    }
    return detail * (1.0f - 0.7f * terrain_maria(c, dir));
}

#if defined(TERRAIN_PATCH) && !defined(__cplusplus)
// A patch from the three uint4s it's stored as (an HLSL includer's TERRAIN_PATCH uses this).
TerrainPatch terrain_patchFrom(uint4 a, uint4 b, uint4 c)
{
    TerrainPatch p;
    p.origin = asfloat(a.xyz);
    p.cell = asfloat(a.w);
    p.dLat = asfloat(b.x);
    p.dLon = asfloat(b.y);
    p.width = b.z;
    p.height = b.w;
    p.offset = c.x;
    p.pad0 = p.pad1 = p.pad2 = 0.0;
    return p;
}
#endif

#ifdef TERRAIN_PATCH
// Cubic B-spline weights: smooth (C2) and never overshooting, a little blurred. A patch is
// read this way: through its samples exactly (Catmull-Rom), a surface model's trees and
// noise were bumps at every sample.
TERRAIN_FN float terrain_bspline(float a, float b, float c, float d, float t)
{
    const float t2 = t * t, t3 = t2 * t, u = 1.0f - t;
    return (a * u * u * u + b * (3.0f * t3 - 6.0f * t2 + 4.0f) + c * (-3.0f * t3 + 3.0f * t2 + 3.0f * t + 1.0f) + d * t3) * (1.0f / 6.0f);
}

// How many levels a patch has (TerrainPatch).
TERRAIN_FN int terrain_patchLevels(TerrainPatch p)
{
    int n = 1;
    while (n < 16 && ((p.width - 1u) >> uint(n)) >= 3u && ((p.height - 1u) >> uint(n)) >= 3u)
        ++n;
    return n;
}

TERRAIN_FN float terrain_patchRow(uint row, int w1, int x0, float fx)
{
    const int xa = x0 - 1 < 0 ? 0 : x0 - 1, xb = x0, xc = x0 + 1 > w1 ? w1 : x0 + 1, xd = x0 + 2 > w1 ? w1 : x0 + 2;
    return terrain_bspline(TERRAIN_PATCH_LOAD(row + uint(xa)), TERRAIN_PATCH_LOAD(row + uint(xb)), TERRAIN_PATCH_LOAD(row + uint(xc)),
                           TERRAIN_PATCH_LOAD(row + uint(xd)), fx);
}

// A level of a patch at (x, y) in its finest samples, a B-spline across them.
TERRAIN_FN float terrain_patchLevel(TerrainPatch p, int level, float x, float y)
{
    uint offset = p.offset, w = p.width, h = p.height;
    for (int l = 0; l < level; ++l)
    {
        offset += w * h;
        w = (w - 1u) / 2u + 1u;
        h = (h - 1u) / 2u + 1u;
    }
    const float s = 1.0f / float(1u << uint(level));
    x *= s;
    y *= s;
    const int w1 = int(w) - 1, h1 = int(h) - 1;
    const float fx0 = terrain_floor(x), fy0 = terrain_floor(y);
    const int x0 = int(fx0) > w1 - 1 ? w1 - 1 : int(fx0);
    const int y0 = int(fy0) > h1 - 1 ? h1 - 1 : int(fy0);
    const float fx = x - float(x0), fy = y - float(y0);
    const int ya = y0 - 1 < 0 ? 0 : y0 - 1, yc = y0 + 1 > h1 ? h1 : y0 + 1, yd = y0 + 2 > h1 ? h1 : y0 + 2;
    return terrain_bspline(terrain_patchRow(offset + uint(ya) * w, w1, x0, fx), terrain_patchRow(offset + uint(y0) * w, w1, x0, fx),
                           terrain_patchRow(offset + uint(yc) * w, w1, x0, fx), terrain_patchRow(offset + uint(yd) * w, w1, x0, fx), fy);
}

// The latitude and longitude of r + d against r's (radians), from the small quantities
// alone: d small, its every product with r computed apart, so nothing is lost to the
// floats' rounding of the whole angles.
TERRAIN_FN float2 terrain_angles(float3 r, float3 d)
{
    const float3 q = r + d;
    const float rho0 = terrain_sqrt(r.x * r.x + r.y * r.y), rho = terrain_sqrt(q.x * q.x + q.y * q.y);
    const float dRho = (2.0f * (r.x * d.x + r.y * d.y) + d.x * d.x + d.y * d.y) / terrain_max(rho + rho0, 1e-30f);
    return float2(terrain_atan2(d.z * rho0 - r.z * dRho, q.z * r.z + rho * rho0), terrain_atan2(r.x * d.y - r.y * d.x, r.x * q.x + r.y * q.y));
}

// The ground under dir0 + delta with the body's patches over it (h without them), each read
// at the level for c.patchSpacing, between the two levels nearest it, with what's finer
// than its samples generated. Where in a patch is measured from its own origin: a float
// latitude or direction is only good to about 20-40 cm on Earth, and on a steep 30 m wall
// that was a spike at every vertex. dir0 - origin is exact (two floats close together),
// delta (a vertex from its tile's centre) is kept apart, and so the angles between come
// out to millimetres. (dir0's own rounding is the same for a tile's every vertex.)
TERRAIN_FN float terrain_patches(PlanetTerrainConstants c, float3 dir0, float3 delta, float h)
{
    if (c.patchCount == 0u)
        return h;
    const float3 dir = dir0 + delta;
    for (uint i = 0u; i < c.patchCount; ++i)
    {
        const TerrainPatch p = TERRAIN_PATCH(c.patchFirst + i);
        const float2 a = terrain_angles(p.origin, (dir0 - p.origin) + delta);
        const float x = a.y / p.dLon, y = -a.x / p.dLat;
        const float w1 = float(p.width - 1u), h1 = float(p.height - 1u);
        if (x < 0.0f || y < 0.0f || x > w1 || y > h1)
            continue;
        // The border: a twentieth of the patch, at least 16 samples.
        const float border = terrain_max(16.0f, 0.05f * terrain_min(w1, h1));
        const float edge = terrain_min(terrain_min(x, w1 - x), terrain_min(y, h1 - y));
        const float weight = terrain_smoothstep(0.0f, border, edge);
        const float top = float(terrain_patchLevels(p) - 1);
        float level = c.patchSpacing > p.cell ? terrain_log2(c.patchSpacing / p.cell) : 0.0f;
        level = level > top ? top : level;
        const float l0 = terrain_floor(level);
        float ph = terrain_patchLevel(p, int(l0), x, y);
        if (level > l0)
            ph += (terrain_patchLevel(p, int(l0) + 1, x, y) - ph) * (level - l0);
        h += (ph + terrain_mapDetail(c, dir, 2.0f * p.cell) - h) * weight;
    }
    return h;
}
#endif

// The height above the sea (or the mean ground, or the map's datum), metres, under the
// direction dir0 + delta: delta small (a vertex from its tile's centre), kept apart so the
// patches can be read precisely (terrain_patches).
TERRAIN_FN float TerrainHeightSplit(PlanetTerrainConstants c, float3 dir0, float3 delta)
{
    const float3 dir = dir0 + delta;
    if (c.map > 0.0f)
    {
        float h = terrain_mapHeight(c, dir) + terrain_mapDetail(c, dir, c.detailStart);
#ifdef TERRAIN_PATCH
        h = terrain_patches(c, dir0, delta, h);
#endif
        return h + terrain_craters(c, dir) - c.offset;
    }
    return terrain_shape(c, dir) * c.scale + terrain_craters(c, dir) - c.offset;
}

// The height above the sea (or the mean ground, or the map's datum), metres.
TERRAIN_FN float TerrainHeight(PlanetTerrainConstants c, float3 dir)
{
    return TerrainHeightSplit(c, dir, dir * 0.0f);
}

// The ground's temperature (K): a tidally locked body's radiative equilibrium about its
// hot point; else the annual-mean insolation by latitude (P2 in sin latitude, from the
// obliquity), its contrast evened out by the air and sea; colder with height.
TERRAIN_FN float TerrainTemperature(PlanetTerrainConstants c, float3 dir, float height)
{
    float t;
    if (c.substellar4 > 0.0f)
    {
        const float mu = terrain_max(dot(dir, c.substellar), 0.0f);
        t = terrain_sqrt(terrain_sqrt(c.substellar4 * ((1.0f - c.redistribution) * mu + c.redistribution * 0.25f)));
    }
    else
    {
        const float s = dir.z;
        const float p2 = 0.5f * (3.0f * s * s - 1.0f);
        const float insolation = terrain_max(1.0f + c.insolationP2 * p2, 0.0f);
        t = c.meanTemperature * terrain_pow((1.0f - c.heatTransport) * insolation + c.heatTransport, 0.25f);
    }
    return t - c.lapseRate * terrain_max(height, 0.0f);
}

// What the ground is: its albedo (linear) in xyz, and in w the water's depth over it
// (0 on land). up: the ground's normal dotted with the vertical (1 on the flat). spacing:
// how far apart the samples are, in radii; detail finer than they can show is left out
// (1: none).
TERRAIN_FN float4 TerrainMaterial(PlanetTerrainConstants c, float3 dir, float height, float up, float spacing)
{
    const float t = TerrainTemperature(c, dir, height);
    if (c.meltTemperature > 0.0f && t > c.meltTemperature)
        return float4(c.lavaAlbedo, 0.0f);
    // A patchiness at every scale, so no surface is one flat colour.
    const float patches = terrain_fbm(dir * 40.0f, c.seed ^ 0x3c6ef372u, 4, 0.5f);
    const float vary = 1.0f + 0.25f * patches;
    float3 ground = terrain_lerp(c.brightAlbedo, c.darkAlbedo, terrain_maria(c, dir)) * vary;
    if (c.vegetation > 0.0f)
    {
        // Deserts under the subtropics' sinking air (15-35 degrees), patchy, and nowhere
        // else; plants where it's warm enough and not dry; bare rock where it's cold or steep.
        const float lat = terrain_abs(dir.z);
        const float subtropics = terrain_smoothstep(0.17f, 0.30f, lat) * (1.0f - terrain_smoothstep(0.50f, 0.62f, lat));
        const float dry = terrain_saturate(subtropics * 1.4f + patches * (0.25f + 0.55f * subtropics) - 0.3f);
        // Boreal forest grows where the annual mean is as low as about -7 C, thinning out
        // below (to nothing at -14 C).
        const float warm = terrain_smoothstep(259.0f, 266.0f, t) * (1.0f - terrain_smoothstep(305.0f, 315.0f, t));
        // Woods and open fields, from a few kilometres to ten metres across (on Earth): a
        // canopy darker than the mean, grass and crops lighter and yellower, their mean the
        // mean. Each scale only where the samples are close enough to show it.
        // (Scaled by the octaves' rms, all of them, so each scale's share doesn't change as
        // the finer ones fade in.)
        float fields = 0.0f, amp = 1.0f, f = 2500.0f;
        for (int i = 0; i < 9; ++i)
        {
            const float shown = 1.0f - terrain_smoothstep(0.15f, 0.5f, spacing * f);
            if (shown <= 0.0f)
                break;
            fields += terrain_noise(dir * f, c.seed ^ (0x9e3779b9u + uint(i) * 101u)) * amp * shown;
            amp *= 0.85f;
            f *= 2.0f;
        }
        fields *= 0.45f;   // 1 / sqrt(the sum of 0.85^2i over the nine)
        const float3 tint = terrain_lerp(float3(0.7f, 0.76f, 0.78f), float3(1.35f, 1.25f, 1.05f), terrain_saturate(0.5f + fields));
        const float3 living = terrain_lerp(c.plantAlbedo * tint * vary, c.sandAlbedo * vary, dry);
        ground = terrain_lerp(ground, living, warm);
    }
    // Steep ground sheds dust, sand and snow: bare rock.
    const float steep = 1.0f - terrain_smoothstep(0.75f, 0.9f, up);
    ground = terrain_lerp(ground, c.brightAlbedo * 0.85f, steep);
    if (c.ocean > 0.0f && height < 0.0f)
    {
        // The sea: frozen where it's cold enough, else water over the seabed.
        if (c.iceTemperature > 0.0f && t < c.iceTemperature + 5.0f)
            return float4(c.iceAlbedo, 0.0f);
        return float4(c.sandAlbedo * vary, -height);
    }
    if (c.iceTemperature > 0.0f)
    {
        const float ice = (1.0f - terrain_smoothstep(c.iceTemperature - 4.0f, c.iceTemperature + 4.0f, t)) * (1.0f - steep);
        ground = terrain_lerp(ground, c.iceAlbedo, ice);
    }
    return float4(ground, 0.0f);
}

#endif
