#include <hyper/planet/PlanetTerrain.h>

#include <hyper/App.h>
#include <hyper/Profiler.h>
#include <hyper/scene/GBuffer.h>

#include <donut/core/log.h>
#include <donut/engine/BindingCache.h>
#include <donut/engine/ShaderFactory.h>
#include <donut/engine/View.h>
#include <nvrhi/utils.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <unordered_set>

namespace hyper
{
    namespace
    {
        constexpr uint32_t c_Grid = TERRAIN_GRID;
        constexpr uint32_t c_Row = c_Grid + 2;   // with the skirt
        constexpr uint32_t c_Indices = (c_Row - 1) * (c_Row - 1) * 6;
        constexpr uint32_t c_MaxJobs = 64;   // tiles made in a frame, at most

        // A node's key: body (3 bits), face (3), level (5), x and y (21 each), and a bit so
        // no key is 0.
        uint64_t Key(uint32_t body, uint32_t face, uint32_t level, uint32_t x, uint32_t y)
        {
            return (1ull << 63) | (uint64_t(body) << 50) | (uint64_t(face) << 47) | (uint64_t(level) << 42)
                 | (uint64_t(x) << 21) | uint64_t(y);
        }
        uint32_t KeyBody(uint64_t k) { return uint32_t(k >> 50) & 7u; }
        uint32_t KeyFace(uint64_t k) { return uint32_t(k >> 47) & 7u; }
        uint32_t KeyLevel(uint64_t k) { return uint32_t(k >> 42) & 31u; }
        uint32_t KeyX(uint64_t k) { return uint32_t(k >> 21) & 0x1fffffu; }
        uint32_t KeyY(uint64_t k) { return uint32_t(k) & 0x1fffffu; }

        dm::double3 D(const float3& v) { return dm::double3(v.x, v.y, v.z); }

        // A tile's centre on its face and its half size.
        void TileSquare(uint32_t level, uint32_t x, uint32_t y, double& u, double& v, double& half)
        {
            half = 1.0 / double(1u << level);
            u = -1.0 + (2.0 * x + 1.0) * half;
            v = -1.0 + (2.0 * y + 1.0) * half;
        }

        dm::double3 CubeDirection(uint32_t face, double u, double v)
        {
            return dm::normalize(D(terrain_faceNormal(face)) + D(terrain_faceU(face)) * u + D(terrain_faceV(face)) * v);
        }

        // Fibonacci directions on the sphere.
        float3 Fibonacci(int i, int n)
        {
            const double z = 1.0 - (i + 0.5) * 2.0 / n, r = std::sqrt(std::max(0.0, 1.0 - z * z));
            const double phi = i * 2.399963229728653;
            return float3(float(r * std::cos(phi)), float(r * std::sin(phi)), float(z));
        }
    }

    namespace terrain
    {
        namespace
        {
            std::vector<std::vector<float>>& Maps()
            {
                static std::vector<std::vector<float>> maps;
                return maps;
            }
            std::vector<TerrainPatch>& PatchList()
            {
                static std::vector<TerrainPatch> patches;
                return patches;
            }
            std::vector<float>& PatchData()
            {
                static std::vector<float> samples;
                return samples;
            }
        }

        static_assert(sizeof(TerrainPatch) == 48, "TerrainPatch is three uint4s on the GPU");
        const TerrainPatch& Patch(unsigned i) { return PatchList()[i]; }
        float PatchLoad(unsigned i) { return PatchData()[i]; }

        float MapLoad(int slice, int x, int y)
        {
            const auto& maps = Maps();
            if (slice < 0 || size_t(slice) >= maps.size())
                return 0.f;
            return maps[size_t(slice)][size_t(y) * TERRAIN_MAP_W + size_t(x)];
        }

        void Calibrate(PlanetTerrainConstants& c, double range, double ocean)
        {
            if (c.map > 0.f)
            {
                // Measured: its datum is the sea (or the areoid, or the reference sphere).
                c.scale = 1.f;
                c.offset = 0.f;
                return;
            }
            constexpr int n = 20000;
            std::vector<float> shape(n), craters(n), h(n);
            PlanetTerrainConstants raw = c;
            raw.scale = 1.f;
            raw.offset = 0.f;
            for (int i = 0; i < n; ++i)
            {
                const float3 d = Fibonacci(i, n);
                shape[i] = terrain_shape(raw, d);
                craters[i] = terrain_craters(raw, d);
            }
            auto spread = [&](float scale)
            {
                for (int i = 0; i < n; ++i)
                    h[i] = shape[i] * scale + craters[i];
                std::vector<float> s = h;
                std::sort(s.begin(), s.end());
                return double(s[size_t(n * 0.9995)] - s[size_t(n * 0.0005)]);
            };
            // The craters' depths are in metres, the rest scales: a few rounds to the range.
            std::vector<float> s0 = shape;
            std::sort(s0.begin(), s0.end());
            float scale = float(range / std::max(double(s0[size_t(n * 0.9995)] - s0[size_t(n * 0.0005)]), 1e-6));
            for (int round = 0; round < 6; ++round)
            {
                const double now = spread(scale);
                if (now <= 0.0)
                    break;
                scale = float(std::max(0.0, double(scale) * range / now));
            }
            spread(scale);
            c.scale = scale;
            std::vector<float> s = h;
            std::sort(s.begin(), s.end());
            if (ocean > 0.0)
                c.offset = s[size_t(std::clamp(ocean, 0.0, 0.9999) * n)];
            else
            {
                double mean = 0.0;
                for (float v : h)
                    mean += v;
                c.offset = float(mean / n);
            }
        }

        Average Sample(const PlanetTerrainConstants& c, const dm::float3& waterAlbedo, int samples)
        {
            Average a;
            a.minHeight = 1e30f;
            a.maxHeight = -1e30f;
            dm::double3 sum(0.0);
            int water = 0;
            for (int i = 0; i < samples; ++i)
            {
                const float3 d = Fibonacci(i, samples);
                const float h = TerrainHeight(c, d);
                a.minHeight = std::min(a.minHeight, h);
                a.maxHeight = std::max(a.maxHeight, h);
                const float4 m = TerrainMaterial(c, d, h, 1.f, 1.f);
                if (m.w > 0.f)
                {
                    sum += dm::double3(waterAlbedo.x, waterAlbedo.y, waterAlbedo.z);
                    ++water;
                }
                else
                    sum += dm::double3(m.x, m.y, m.z);
            }
            sum /= double(samples);
            a.albedo = dm::float3(float(sum.x), float(sum.y), float(sum.z));
            a.water = float(water) / float(samples);
            return a;
        }
    }

    PlanetTerrain::PlanetTerrain(Context& ctx, uint32_t poolTiles) : m_Ctx(ctx), m_PoolTiles(poolTiles) { }

    int PlanetTerrain::AddMap(std::vector<float> heights)
    {
        auto& maps = terrain::Maps();
        if (heights.size() != size_t(TERRAIN_MAP_W) * TERRAIN_MAP_H || maps.size() >= TERRAIN_MAX_BODIES)
            return -1;
        maps.push_back(std::move(heights));
        return int(maps.size()) - 1;
    }
    int PlanetTerrain::AddPatch(double lat0, double lon0, double dLat, double dLon, uint32_t width, uint32_t height, double radius,
                                std::vector<float> heights)
    {
        auto& patches = terrain::PatchList();
        auto& samples = terrain::PatchData();
        if (width < 4 || height < 4 || heights.size() != size_t(width) * height || patches.size() >= TERRAIN_MAX_PATCHES)
            return -1;
        constexpr double degree = 3.14159265358979323846 / 180.0;
        TerrainPatch p;
        std::memset(&p, 0, sizeof(p));
        const double la = lat0 * degree, lo = lon0 * degree;
        p.origin = float3(float(std::cos(la) * std::cos(lo)), float(std::cos(la) * std::sin(lo)), float(std::sin(la)));
        p.dLat = float(dLat * degree);
        p.dLon = float(dLon * degree);
        p.width = width;
        p.height = height;
        p.offset = uint32_t(samples.size());
        const double middle = (lat0 - 0.5 * dLat * (height - 1)) * degree;
        p.cell = float(radius * std::max(dLat, dLon * std::cos(middle)) * degree);
        // Its coarser levels (planet_terrain.h): each sample k the tent [1 2 1] / 4 round
        // the finer level's sample 2k, both ways.
        size_t total = heights.size();
        for (uint32_t w = width, h = height, l = 1; int(l) < terrain_patchLevels(p); ++l)
        {
            w = (w - 1) / 2 + 1;
            h = (h - 1) / 2 + 1;
            total += size_t(w) * h;
        }
        if (samples.size() + total > TERRAIN_PATCH_SAMPLES)
            return -1;
        samples.insert(samples.end(), heights.begin(), heights.end());
        std::vector<float> level = std::move(heights);
        for (uint32_t w = width, h = height, l = 1; int(l) < terrain_patchLevels(p); ++l)
        {
            const uint32_t w2 = (w - 1) / 2 + 1, h2 = (h - 1) / 2 + 1;
            auto at = [&](int x, int y) { return level[size_t(std::clamp(y, 0, int(h) - 1)) * w + size_t(std::clamp(x, 0, int(w) - 1))]; };
            std::vector<float> next(size_t(w2) * h2);
            for (uint32_t y = 0; y < h2; ++y)
                for (uint32_t x = 0; x < w2; ++x)
                {
                    const int cx = int(2 * x), cy = int(2 * y);
                    float sum = 0.f;
                    for (int dy = -1; dy <= 1; ++dy)
                        for (int dx = -1; dx <= 1; ++dx)
                            sum += at(cx + dx, cy + dy) * float((2 - std::abs(dx)) * (2 - std::abs(dy)));
                    next[size_t(y) * w2 + x] = sum / 16.f;
                }
            samples.insert(samples.end(), next.begin(), next.end());
            level = std::move(next);
            w = w2;
            h = h2;
        }
        patches.push_back(p);
        return int(patches.size()) - 1;
    }

    PlanetTerrain::~PlanetTerrain() = default;

    int PlanetTerrain::Add(const PlanetTerrainConstants& c)
    {
        if (m_Bodies.size() >= TERRAIN_MAX_BODIES)
            return -1;
        Body b;
        b.c = c;
        terrain::Average a = terrain::Sample(c, dm::float3(0.f), 4000);
        // Its patches' heights too: too small for the samples to find, and the highest
        // ground there is (Everest's).
        for (uint32_t i = 0; i < c.patchCount; ++i)
        {
            const TerrainPatch& p = terrain::PatchList()[c.patchFirst + i];
            const auto range = std::minmax_element(terrain::PatchData().begin() + p.offset,
                                                   terrain::PatchData().begin() + p.offset + size_t(p.width) * p.height);
            a.minHeight = std::min(a.minHeight, *range.first - c.offset);
            a.maxHeight = std::max(a.maxHeight, *range.second - c.offset);
        }
        // A margin over what the samples found, for the peaks between them.
        const float margin = 0.1f * (a.maxHeight - a.minHeight) + 10.f;
        b.minHeight = a.minHeight - margin;
        b.maxHeight = a.maxHeight + margin;
        if (c.ocean > 0.f)
            b.minHeight = std::min(b.minHeight, 0.f);
        m_Bodies.push_back(b);
        return int(m_Bodies.size()) - 1;
    }

    float PlanetTerrain::HeightAt(int body, const dm::double3& dir, bool sea) const
    {
        const PlanetTerrainConstants& c = m_Bodies[size_t(body)].c;
        // The direction as a float and the rest of it, apart (TerrainHeightSplit).
        const dm::double3 n = dm::normalize(dir);
        const float3 n0(float(n.x), float(n.y), float(n.z));
        const float3 rest(float(n.x - double(n0.x)), float(n.y - double(n0.y)), float(n.z - double(n0.z)));
        const float h = TerrainHeightSplit(c, n0, rest);
        return sea && c.ocean > 0.f ? std::max(h, 0.f) : h;
    }

    void PlanetTerrain::CreateResources()
    {
        nvrhi::IDevice* device = m_Ctx.device;
        m_Vertices = device->createBuffer(nvrhi::BufferDesc().setDebugName("hyper/planet-vertices")
            .setByteSize(uint64_t(m_PoolTiles) * c_Grid * c_Grid * 16).setStructStride(16).setCanHaveUAVs(true)
            .setInitialState(nvrhi::ResourceStates::ShaderResource).setKeepInitialState(true));
        m_Draws = device->createBuffer(nvrhi::BufferDesc().setDebugName("hyper/planet-draws")
            .setByteSize(TERRAIN_MAX_DRAWS * sizeof(TerrainDraw)).setStructStride(sizeof(TerrainDraw))
            .setInitialState(nvrhi::ResourceStates::ShaderResource).setKeepInitialState(true));
        m_BodyBuffer = device->createBuffer(nvrhi::BufferDesc().setDebugName("hyper/planet-bodies")
            .setByteSize(TERRAIN_MAX_BODIES * sizeof(TerrainBody)).setStructStride(sizeof(TerrainBody))
            .setInitialState(nvrhi::ResourceStates::ShaderResource).setKeepInitialState(true));
        m_Jobs = device->createBuffer(nvrhi::BufferDesc().setDebugName("hyper/planet-jobs")
            .setByteSize(c_MaxJobs * sizeof(TerrainGenerateConstants)).setStructStride(sizeof(TerrainGenerateConstants))
            .setInitialState(nvrhi::ResourceStates::ShaderResource).setKeepInitialState(true));
        m_Indices = device->createBuffer(nvrhi::BufferDesc().setDebugName("hyper/planet-indices")
            .setByteSize(c_Indices * 4).setIsIndexBuffer(true)
            .setInitialState(nvrhi::ResourceStates::IndexBuffer).setKeepInitialState(true));
        MapsTexture();
        PatchesBuffer();
        PatchSamplesBuffer();
        m_GenerateLayout = device->createBindingLayout(nvrhi::BindingLayoutDesc().setVisibility(nvrhi::ShaderType::Compute)
            .addItem(nvrhi::BindingLayoutItem::Texture_SRV(0))
            .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_SRV(1))
            .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_SRV(2))
            .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_SRV(3))
            .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_UAV(0)));
        m_DrawLayout = device->createBindingLayout(nvrhi::BindingLayoutDesc()
            .setVisibility(nvrhi::ShaderType::Vertex | nvrhi::ShaderType::Pixel)
            .addItem(nvrhi::BindingLayoutItem::VolatileConstantBuffer(0))
            .addItem(nvrhi::BindingLayoutItem::VolatileConstantBuffer(1))
            .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_SRV(0))
            .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_SRV(1))
            .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_SRV(2)));
        m_SlotKey.assign(m_PoolTiles, 0);
        m_Free.clear();
        for (int i = int(m_PoolTiles) - 1; i >= 0; --i)
            m_Free.push_back(i);
    }

    nvrhi::ITexture* PlanetTerrain::MapsTexture()
    {
        if (!m_Maps)
        {
            m_Maps = m_Ctx.device->createTexture(nvrhi::TextureDesc().setDebugName("hyper/planet-maps")
                .setDimension(nvrhi::TextureDimension::Texture2DArray).setWidth(TERRAIN_MAP_W).setHeight(TERRAIN_MAP_H)
                .setArraySize(TERRAIN_MAX_BODIES).setFormat(nvrhi::Format::R32_FLOAT)
                .setInitialState(nvrhi::ResourceStates::ShaderResource).setKeepInitialState(true));
            m_MapsUploaded = 0;
        }
        return m_Maps;
    }

    nvrhi::IBuffer* PlanetTerrain::PatchesBuffer()
    {
        if (!m_Patches)
        {
            m_Patches = m_Ctx.device->createBuffer(nvrhi::BufferDesc().setDebugName("hyper/planet-patches")
                .setByteSize(TERRAIN_MAX_PATCHES * sizeof(TerrainPatch)).setStructStride(16)   // three uint4s each
                .setInitialState(nvrhi::ResourceStates::ShaderResource).setKeepInitialState(true));
            m_PatchesUploaded = 0;
        }
        return m_Patches;
    }

    nvrhi::IBuffer* PlanetTerrain::PatchSamplesBuffer()
    {
        if (!m_PatchSamples)
        {
            // Sized for what's been added (at least a sample): patches come before the first frame.
            const size_t n = std::max<size_t>(terrain::PatchData().size(), 1);
            m_PatchSamples = m_Ctx.device->createBuffer(nvrhi::BufferDesc().setDebugName("hyper/planet-patch-samples")
                .setByteSize(n * 4).setStructStride(4)
                .setInitialState(nvrhi::ResourceStates::ShaderResource).setKeepInitialState(true));
            m_PatchesUploaded = 0;
        }
        return m_PatchSamples;
    }

    void PlanetTerrain::CreatePipelines()
    {
        if (!m_Used)
            return;   // built on first use
        if (auto cs = m_Ctx.shaders->CreateShader("hyper/planet/generate_cs", "main", nullptr, nvrhi::ShaderType::Compute))
            m_Generate = m_Ctx.device->createComputePipeline(nvrhi::ComputePipelineDesc().setComputeShader(cs).addBindingLayout(m_GenerateLayout));
        auto vs = m_Ctx.shaders->CreateShader("hyper/planet/terrain", "vs", nullptr, nvrhi::ShaderType::Vertex);
        auto ps = m_Ctx.shaders->CreateShader("hyper/planet/terrain", "ps", nullptr, nvrhi::ShaderType::Pixel);
        if (vs && ps)
        {
            m_Vs = vs;
            m_Ps = ps;
            m_Pipeline = nullptr;   // made for the framebuffer
        }
        // New shaders may compute a different ground: make every tile again.
        for (auto& [key, node] : m_Nodes)
            node.slot = -1;
        m_Nodes.clear();
        m_Heights.clear();
        m_SlotKey.assign(m_PoolTiles, 0);
        m_Free.clear();
        for (int i = int(m_PoolTiles) - 1; i >= 0; --i)
            m_Free.push_back(i);
    }

    int PlanetTerrain::Slot(uint64_t key)
    {
        int slot = -1;
        if (!m_Free.empty())
        {
            slot = m_Free.back();
            m_Free.pop_back();
        }
        else
        {
            // The least recently drawn tile not used this frame, never a face's root.
            uint64_t oldest = ~0ull;
            for (uint32_t s = 0; s < m_PoolTiles; ++s)
            {
                const uint64_t k = m_SlotKey[s];
                if (!k || KeyLevel(k) == 0)
                    continue;
                const Node& n = m_Nodes[k];
                if (n.used < m_Frame && n.used < oldest)
                {
                    oldest = n.used;
                    slot = int(s);
                }
            }
            if (slot < 0)
                return -1;
            m_Nodes.erase(m_SlotKey[size_t(slot)]);
        }
        m_SlotKey[size_t(slot)] = key;
        m_Nodes[key].slot = slot;
        return slot;
    }

    void PlanetTerrain::Render(nvrhi::ICommandList* cmd, const donut::engine::PlanarView& view, nvrhi::IFramebuffer* gbuffer,
                               nvrhi::IBuffer* viewConstants, nvrhi::IBuffer* prevViewConstants,
                               const std::vector<Placement>& bodies)
    {
        m_Stats = {};
        if (bodies.empty() || m_Bodies.empty())
            return;
        if (!m_Used)
        {
            m_Used = true;
            CreateResources();
            CreatePipelines();
            std::vector<uint32_t> indices;
            indices.reserve(c_Indices);
            for (uint32_t j = 0; j + 1 < c_Row; ++j)
                for (uint32_t i = 0; i + 1 < c_Row; ++i)
                {
                    const uint32_t a = j * c_Row + i, b = a + 1, c = a + c_Row, d = c + 1;
                    indices.insert(indices.end(), { a, c, b, b, c, d });
                }
            cmd->writeBuffer(m_Indices, indices.data(), indices.size() * 4);
        }
        if (!m_Generate || !m_Vs || !m_Ps)
            return;
        for (const auto& maps = terrain::Maps(); m_MapsUploaded < maps.size(); ++m_MapsUploaded)
            cmd->writeTexture(m_Maps, uint32_t(m_MapsUploaded), 0, maps[m_MapsUploaded].data(), TERRAIN_MAP_W * 4);
        if (const auto& patches = terrain::PatchList(); m_PatchesUploaded < patches.size())
        {
            const auto& samples = terrain::PatchData();
            if (m_PatchSamples->getDesc().byteSize < samples.size() * 4)
                donut::log::error("planet: patches were added after their buffer was made; they're left out on the GPU");
            else
            {
                cmd->writeBuffer(m_Patches, patches.data(), patches.size() * sizeof(TerrainPatch));
                cmd->writeBuffer(m_PatchSamples, samples.data(), samples.size() * 4);
            }
            m_PatchesUploaded = patches.size();
        }
        ++m_Frame;
        auto scope = m_Ctx.profiler->Measure(cmd, "terrain");

        // What the camera sees: the frustum, and how many pixels a radian is.
        const dm::frustum frustum = view.GetViewFrustum();
        const nvrhi::Viewport vp = view.GetViewportState().viewports[0];
        const double pixelsPerRadian = 0.5 * double(vp.height()) * double(view.GetProjectionMatrix(false)[1][1]);

        std::vector<TerrainDraw> draws;
        std::vector<TerrainBody> gpuBodies;
        std::vector<Request> requests;
        std::unordered_set<uint64_t> requested;
        auto request = [&](uint64_t key, int level, float pixels)
        {
            if (requested.insert(key).second)
                requests.push_back({ key, level, pixels });
        };

        for (size_t bi = 0; bi < bodies.size() && bi < TERRAIN_MAX_BODIES; ++bi)
        {
            const Placement& p = bodies[bi];
            const Body& body = m_Bodies[size_t(p.body)];
            const double radius = body.c.radius;
            TerrainBody gb;
            auto axis = [](const dm::double3& a, float w) { return float4(float(a.x), float(a.y), float(a.z), w); };
            gb.axisX = axis(p.axes[0], float(radius));
            gb.axisY = axis(p.axes[1], p.lightBody);
            gb.axisZ = axis(p.axes[2], 0.f);
            gb.prevAxisX = axis(p.prevAxes[0], 0.f);
            gb.prevAxisY = axis(p.prevAxes[1], 0.f);
            gb.prevAxisZ = axis(p.prevAxes[2], 0.f);
            gpuBodies.push_back(gb);

            // The camera in the planet's axes.
            const dm::double3 cam(-dm::dot(p.centre, p.axes[0]), -dm::dot(p.centre, p.axes[1]), -dm::dot(p.centre, p.axes[2]));
            const double camDist = dm::length(cam);
            const double lowest = radius + body.minHeight, highest = radius + body.maxHeight;
            const double horizon = camDist > lowest ? std::acos(std::clamp(lowest / camDist, -1.0, 1.0))
                                                     + std::acos(std::clamp(lowest / highest, -1.0, 1.0)) : 1e9;
            const dm::double3 camDir = camDist > 0.0 ? cam / camDist : dm::double3(0.0, 0.0, 1.0);
            auto world = [&](const dm::double3& local, const dm::double3* axes, const dm::double3& centre)
            {
                return centre + axes[0] * local.x + axes[1] * local.y + axes[2] * local.z;
            };

            // Walks a face's quadtree: draws what's made at the detail the view wants, asks
            // for what isn't.
            std::function<void(uint32_t, uint32_t, uint32_t, uint32_t)> walk = [&](uint32_t face, uint32_t level, uint32_t x, uint32_t y)
            {
                const uint64_t key = Key(uint32_t(p.body), face, level, x, y);
                ++m_Stats.visited;
                double u, v, half;
                TileSquare(level, x, y, u, v, half);
                const dm::double3 dir0 = CubeDirection(face, u, v);
                double spread = 0.0;   // the angle from the centre to the farthest corner
                for (int k = 0; k < 4; ++k)
                    spread = std::max(spread, std::acos(std::clamp(dm::dot(dir0,
                        CubeDirection(face, u + ((k & 1) ? half : -half), v + ((k & 2) ? half : -half))), -1.0, 1.0)));
                // Below the horizon.
                const double away = std::acos(std::clamp(dm::dot(camDir, dir0), -1.0, 1.0));
                if (away - spread > horizon)
                {
                    ++m_Stats.belowHorizon;
                    return;
                }
                // Its heights: the centre and corners sampled, and a margin for what lies
                // between them (the finer relief, about 5% of the tile's size).
                const double edge = radius * spread * 1.41421356;
                auto bounds = m_Heights.find(key);
                if (bounds == m_Heights.end())
                {
                    float lo = HeightAt(p.body, dir0), hi = lo;
                    for (int k = 0; k < 4; ++k)
                    {
                        const float h = HeightAt(p.body, CubeDirection(face, u + ((k & 1) ? half : -half), v + ((k & 2) ? half : -half)));
                        lo = std::min(lo, h);
                        hi = std::max(hi, h);
                    }
                    const float margin = float(std::min(0.05 * edge + 30.0, double(body.maxHeight - body.minHeight)));
                    if (m_Heights.size() > 200000)
                        m_Heights.clear();
                    bounds = m_Heights.emplace(key, std::make_pair(std::max(lo - margin, body.minHeight),
                                                                    std::min(hi + margin, body.maxHeight))).first;
                }
                const double lowH = bounds->second.first, highH = bounds->second.second;
                // Outside the frustum: a sphere round it, at its middle height.
                const double midH = 0.5 * (lowH + highH);
                const dm::double3 centre = world(dir0 * (radius + midH), p.axes, p.centre);
                const double bound = (radius + highH) * spread + radius * (1.0 - std::cos(spread)) + 0.5 * (highH - lowH);
                for (const dm::plane& pl : frustum.planes)
                {
                    const double len = std::sqrt(double(dm::dot(pl.normal, pl.normal)));
                    if (len < 1e-6)
                        continue;
                    const double side = (pl.normal.x * centre.x + pl.normal.y * centre.y + pl.normal.z * centre.z - pl.distance) / len;
                    if (side > bound)
                    {
                        ++m_Stats.outside;
                        return;
                    }
                }
                const double distance = std::max(dm::length(centre) - bound, 1.0);
                const float pixels = float(edge / (c_Grid - 1) / distance * pixelsPerRadian);
                auto found = m_Nodes.find(key);
                const bool ready = found != m_Nodes.end() && found->second.slot >= 0;
                if (ready)
                    found->second.used = m_Frame;
                const bool split = pixels > settings.pixelsPerQuad && int(level) < settings.maxLevel;
                if (split)
                {
                    bool children = true;
                    for (uint32_t k = 0; k < 4; ++k)
                    {
                        auto c = m_Nodes.find(Key(uint32_t(p.body), face, level + 1, x * 2 + (k & 1), y * 2 + (k >> 1)));
                        if (c != m_Nodes.end() && c->second.slot >= 0)
                            c->second.used = m_Frame;   // kept while its siblings are made, or they'd evict it
                        else
                        {
                            children = false;
                            request(Key(uint32_t(p.body), face, level + 1, x * 2 + (k & 1), y * 2 + (k >> 1)), int(level) + 1, pixels * 0.5f);
                        }
                    }
                    if (children)
                    {
                        for (uint32_t k = 0; k < 4; ++k)
                            walk(face, level + 1, x * 2 + (k & 1), y * 2 + (k >> 1));
                        return;
                    }
                }
                if (!ready)
                {
                    ++m_Stats.missing;
                    request(key, int(level), pixels);
                    return;
                }
                if (draws.size() >= TERRAIN_MAX_DRAWS)
                    return;
                m_Stats.deepest = std::max(m_Stats.deepest, level);
                TerrainDraw d;
                std::memset(&d, 0, sizeof(d));
                const dm::double3 prev = world(dir0 * radius, p.prevAxes, p.prevCentre);
                const dm::double3 base = world(dir0 * radius, p.axes, p.centre);   // the shader adds the heights
                d.centre = float3(float(base.x), float(base.y), float(base.z));
                d.prevCentre = float3(float(prev.x), float(prev.y), float(prev.z));
                d.slot = uint32_t(found->second.slot);
                d.packed = face | (uint32_t(bi) << 3) | (level << 6);
                d.uv = float2(float(u), float(v));
                d.halfSize = float(half);
                d.skirt = float(std::max(edge * 0.03, 2.0));
                draws.push_back(d);
            };
            for (uint32_t face = 0; face < 6; ++face)
                walk(face, 0, 0, 0);
        }

        // Make what was asked for, coarse first, within the budget.
        std::sort(requests.begin(), requests.end(), [](const Request& a, const Request& b)
        {
            return a.level != b.level ? a.level < b.level : a.pixels > b.pixels;
        });
        nvrhi::IBindingSet* generateSet = m_Ctx.bindingCache->GetOrCreateBindingSet(nvrhi::BindingSetDesc()
            .addItem(nvrhi::BindingSetItem::Texture_SRV(0, m_Maps))
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(1, m_Jobs))
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(2, m_Patches))
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(3, m_PatchSamples))
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(0, m_Vertices)), m_GenerateLayout);
        uint32_t made = 0;
        std::vector<TerrainGenerateConstants> jobs;
        for (const Request& r : requests)
        {
            if (made >= std::min(settings.tilesPerFrame, c_MaxJobs))
                break;
            const int slot = Slot(r.key);
            if (slot < 0)
                break;   // the pool is full of tiles in use
            TerrainGenerateConstants g;
            std::memset(&g, 0, sizeof(g));
            g.terrain = m_Bodies[KeyBody(r.key)].c;
            double u, v, half;
            TileSquare(KeyLevel(r.key), KeyX(r.key), KeyY(r.key), u, v, half);
            g.centre = float2(float(u), float(v));
            g.halfSize = float(half);
            // (generate_cs's spacing, in metres: a face spans about 1.6 radii.)
            g.terrain.patchSpacing = float(1.6 * half / (c_Grid - 1) * g.terrain.radius);
            g.face = KeyFace(r.key);
            g.slot = uint32_t(slot);
            jobs.push_back(g);
            m_Nodes[r.key].used = m_Frame;
            ++made;
        }
        if (!jobs.empty())
        {
            // One dispatch for them all: a layer of groups a tile.
            cmd->writeBuffer(m_Jobs, jobs.data(), jobs.size() * sizeof(TerrainGenerateConstants));
            nvrhi::ComputeState state;
            state.pipeline = m_Generate;
            state.bindings = { generateSet };
            cmd->setComputeState(state);
            const uint32_t groups = (c_Grid + TERRAIN_GROUP - 1) / TERRAIN_GROUP;
            cmd->dispatch(groups, groups, uint32_t(jobs.size()));
        }

        m_Stats.drawn = uint32_t(draws.size());
        m_Stats.made = made;
        m_Stats.waiting = uint32_t(requests.size()) - made;
        m_Stats.resident = m_PoolTiles - uint32_t(m_Free.size());
        if (draws.empty())
            return;

        cmd->writeBuffer(m_Draws, draws.data(), draws.size() * sizeof(TerrainDraw));
        cmd->writeBuffer(m_BodyBuffer, gpuBodies.data(), gpuBodies.size() * sizeof(TerrainBody));
        if (!m_Pipeline || m_PipelineFb != gbuffer->getFramebufferInfo())
        {
            nvrhi::GraphicsPipelineDesc d;
            d.setPrimType(nvrhi::PrimitiveType::TriangleList).setVertexShader(m_Vs).setPixelShader(m_Ps).addBindingLayout(m_DrawLayout);
            d.renderState.rasterState.setCullNone();
            d.renderState.depthStencilState = GBuffer::DepthStencil();
            m_Pipeline = m_Ctx.device->createGraphicsPipeline(d, gbuffer->getFramebufferInfo());
            m_PipelineFb = gbuffer->getFramebufferInfo();
        }
        nvrhi::IBindingSet* set = m_Ctx.bindingCache->GetOrCreateBindingSet(nvrhi::BindingSetDesc()
            .addItem(nvrhi::BindingSetItem::ConstantBuffer(0, viewConstants))
            .addItem(nvrhi::BindingSetItem::ConstantBuffer(1, prevViewConstants))
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(0, m_Draws))
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(1, m_BodyBuffer))
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(2, m_Vertices)), m_DrawLayout);
        nvrhi::GraphicsState state;
        state.pipeline = m_Pipeline;
        state.framebuffer = gbuffer;
        state.bindings = { set };
        state.indexBuffer = nvrhi::IndexBufferBinding().setBuffer(m_Indices).setFormat(nvrhi::Format::R32_UINT);
        state.viewport.addViewportAndScissorRect(vp);
        cmd->setGraphicsState(state);
        cmd->drawIndexed(nvrhi::DrawArguments().setVertexCount(c_Indices).setInstanceCount(uint32_t(draws.size())));
    }
}
