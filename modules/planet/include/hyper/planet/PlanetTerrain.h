#pragma once

#include <hyper/platform.h>

#include <donut/core/math/math.h>
#include <nvrhi/nvrhi.h>

#include <hyper/shared.h>

// (The shared height function reads the maps and patches through hyper::terrain, which
// planet_terrain.h declares.)
#include <hyper/planet_tiles_cb.h>

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace donut::engine { class PlanarView; }

namespace hyper
{
    struct Context;

    // Calibrating a body's terrain on the CPU, with the same function the GPU draws.
    namespace terrain
    {
        // Sets c.scale so the ground spans `range` metres (its 0.05% to 99.95% heights), and
        // c.offset so height 0 is the sea's level for `ocean` (the share of the surface under
        // water), or the mean ground when there's none.
        void Calibrate(PlanetTerrainConstants& c, double range, double ocean);

        // What the ground averages, over the sphere: its albedo (water counted as
        // `waterAlbedo`), and the share of it that is water.
        struct Average
        {
            dm::float3 albedo = dm::float3(0.f);
            float water = 0.f;
            float minHeight = 0.f, maxHeight = 0.f;
        };
        Average Sample(const PlanetTerrainConstants& c, const dm::float3& waterAlbedo, int samples = 20000);
    }

    // Planets' surfaces (the planet module): each a
    // cube-sphere quadtree of tiles, made on the GPU as the camera comes near, from one
    // height and material function (hyper/planet_terrain.h) the CPU can evaluate too.
    //
    //     int id = terrain.Add(constants);              // at load (terrain::Calibrate first)
    //     float h = terrain.HeightAt(id, dir);          // the ground under a direction, CPU
    //     terrain.Render(cmd, view, gbuffer, ...);      // each frame: choose, make, draw
    //
    // Tiles are made coarse first, at most settings.tilesPerFrame a frame (the
    // budget): until a tile's children are made, it's drawn instead of them.
    // Positions are camera-relative from doubles, so the ground is exact a metre away on
    // an Earth-sized planet. The pool's tiles are reused, least recently drawn first.
    class PlanetTerrain
    {
    public:
        struct Settings
        {
            uint32_t tilesPerFrame = 24;    // the budget
            float pixelsPerQuad = 6.f;     // split a tile when its quads are bigger than this
            int maxLevel = 19;
        };
        Settings settings;

        PlanetTerrain(Context& ctx, uint32_t poolTiles = 1536);
        ~PlanetTerrain();

        int Add(const PlanetTerrainConstants& c);
        // A measured elevation map (TERRAIN_MAP_W x TERRAIN_MAP_H metres, planet_terrain.h's
        // layout); a constants' `map` is the returned slice + 1. Before Add of the bodies on it.
        static int AddMap(std::vector<float> heights);
        // A finer map over a latitude-longitude box (planet_terrain.h's TerrainPatch): width x
        // height metres, rows from lat0 going south by dLat, columns from lon0 going east by
        // dLon (degrees), on a body of that radius. A body's patches are consecutive: its
        // constants' patchFirst is the first one's index, patchCount how many. Before Add of
        // the body. -1 when there's no room (TERRAIN_MAX_PATCHES, TERRAIN_PATCH_SAMPLES).
        static int AddPatch(double lat0, double lon0, double dLat, double dLon, uint32_t width, uint32_t height, double radius,
                            std::vector<float> heights);
        [[nodiscard]] const PlanetTerrainConstants& Constants(int body) const { return m_Bodies[size_t(body)].c; }

        // The ground's height above the sea (or the mean), metres, at a unit direction in the
        // planet's axes. Over the sea, the sea's surface (0) when `sea` is set.
        [[nodiscard]] float HeightAt(int body, const dm::double3& dir, bool sea = true) const;

        // A body this frame, seen from the camera (at the origin, the world's axes).
        struct Placement
        {
            int body = -1;
            dm::double3 centre;                  // camera-relative
            dm::double3 axes[3];                 // the planet's x, y, z in the world
            dm::double3 prevCentre;              // last frame's, in this frame's camera space
            dm::double3 prevAxes[3];
            float lightBody = 0.f;               // written to the G-buffer's emissive.w
        };

        // Chooses each body's tiles for this view, makes what the budget allows, and draws
        // them into the G-buffer's framebuffer (depth and stencil as hyper's GBuffer).
        // view / prevView: the app's view constant buffers (b0 / b1).
        void Render(nvrhi::ICommandList* cmd, const donut::engine::PlanarView& view, nvrhi::IFramebuffer* gbuffer,
                    nvrhi::IBuffer* viewConstants, nvrhi::IBuffer* prevViewConstants, const std::vector<Placement>& bodies);

        struct Stats
        {
            uint32_t drawn = 0, made = 0, waiting = 0, resident = 0;
            uint32_t visited = 0, belowHorizon = 0, outside = 0, missing = 0;   // the walk's nodes
            uint32_t deepest = 0;     // the finest level drawn
        };
        [[nodiscard]] const Stats& LastStats() const { return m_Stats; }

        void CreatePipelines();

        // The elevation maps (planet_terrain.h's layout, a slice a map), for shaders that walk
        // the ground themselves (hyper/planet_shadow.hlsli). Made on first call.
        nvrhi::ITexture* MapsTexture();
        // The patches (TERRAIN_MAX_PATCHES of them, each three uint4s) and their samples (floats),
        // for those shaders too: TERRAIN_PATCH(i) and TERRAIN_PATCH_LOAD(i) read them.
        nvrhi::IBuffer* PatchesBuffer();
        nvrhi::IBuffer* PatchSamplesBuffer();

    private:
        struct Body
        {
            PlanetTerrainConstants c;
            float minHeight = 0.f, maxHeight = 0.f;
        };
        struct Node
        {
            int slot = -1;
            uint64_t used = 0;
        };
        struct Request
        {
            uint64_t key;
            int level;
            float pixels;
        };

        Context& m_Ctx;
        uint32_t m_PoolTiles;
        std::vector<Body> m_Bodies;
        std::unordered_map<uint64_t, Node> m_Nodes;
        // Each tile's heights (lowest, highest), from a few samples of the function plus a
        // margin for what's between them: its bounds for culling and for its distance.
        std::unordered_map<uint64_t, std::pair<float, float>> m_Heights;
        std::vector<uint64_t> m_SlotKey;     // which node holds each slot (0: free)
        std::vector<int> m_Free;
        uint64_t m_Frame = 0;
        Stats m_Stats;

        bool m_Used = false;
        nvrhi::BufferHandle m_Vertices, m_Draws, m_BodyBuffer, m_Jobs, m_Indices;
        nvrhi::TextureHandle m_Maps;
        size_t m_MapsUploaded = 0;
        nvrhi::BufferHandle m_Patches, m_PatchSamples;
        size_t m_PatchesUploaded = 0;
        nvrhi::BindingLayoutHandle m_GenerateLayout, m_DrawLayout;
        nvrhi::ComputePipelineHandle m_Generate;
        nvrhi::ShaderHandle m_Vs, m_Ps;
        nvrhi::GraphicsPipelineHandle m_Pipeline;
        nvrhi::FramebufferInfoEx m_PipelineFb;

        void CreateResources();
        int Slot(uint64_t key);
    };
}
