#include <hyper/ocean/Ocean.h>

#include <hyper/App.h>
#include <hyper/scene/GBuffer.h>

#include <donut/engine/FramebufferFactory.h>
#include <donut/engine/ShaderFactory.h>
#include <donut/engine/View.h>
#include <nvrhi/utils.h>

#include <imgui.h>

#include <algorithm>
#include <cmath>

using namespace donut;
#include <hyper/shared.h>
#include <hyper/ocean_cb.h>

namespace hyper
{
    namespace
    {
        constexpr float c_Gravity = 9.81f;
        constexpr float c_Pi = 3.14159265358979f;
        constexpr float c_EarthRadius = 6.371e6f;
        // Patch sizes in metres, not multiples of each other so their tiling doesn't line up.
        constexpr float c_Patch[OCEAN_CASCADES] = { 400.f, 61.f, 9.7f };
        constexpr float c_BaseCell = 0.25f;   // the finest clipmap cell at detail 1, metres

        // Where one cascade's band of wavenumbers hands over to the next: six of the next
        // cascade's wavelengths per its patch, well under each cascade's Nyquist limit.
        float BandStart(uint32_t cascade)
        {
            return cascade == 0 ? 0.f : 6.f * 2.f * c_Pi / c_Patch[cascade];
        }

        // JONSWAP (Hasselmann et al. 1973): energy by angular frequency.
        float Jonswap(float omega, float alpha, float peak, float gamma)
        {
            if (omega <= 1e-4f)
                return 0.f;
            const float sigma = omega <= peak ? 0.07f : 0.09f;
            const float r = std::exp(-(omega - peak) * (omega - peak) / (2.f * sigma * sigma * peak * peak));
            return alpha * c_Gravity * c_Gravity / std::pow(omega, 5.f)
                * std::exp(-1.25f * std::pow(peak / omega, 4.f)) * std::pow(gamma, r);
        }

        // Donelan-Banner spreading, normalised over -pi..pi; `narrow` > 1 for a swell.
        float Spreading(float theta, float omega, float peak, float narrow)
        {
            const float ratio = omega / peak;
            float beta;
            if (ratio < 0.95f)
                beta = 2.61f * std::pow(ratio, 1.3f);
            else if (ratio < 1.6f)
                beta = 2.28f * std::pow(ratio, -1.3f);
            else
                beta = std::pow(10.f, -0.4f + 0.8393f * std::exp(-0.567f * std::log(ratio * ratio)));
            beta *= narrow;
            const float ch = std::cosh(std::min(beta * theta, 40.f));
            return beta / (2.f * std::tanh(beta * c_Pi)) / (ch * ch);
        }

        float WrapAngle(float a)
        {
            a = std::fmod(a + c_Pi, 2.f * c_Pi);
            return (a < 0.f ? a + 2.f * c_Pi : a) - c_Pi;
        }

        // Repeatable Gaussian pairs: PCG hashes into Box-Muller.
        uint32_t Pcg(uint32_t v)
        {
            const uint32_t state = v * 747796405u + 2891336453u;
            const uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
            return (word >> 22u) ^ word;
        }
        std::pair<float, float> Gaussian(uint32_t key)
        {
            const float u1 = (float(Pcg(key) >> 8) + 1.f) / 16777217.f;
            const float u2 = float(Pcg(key ^ 0x9e3779b9u) >> 8) / 16777216.f;
            const float r = std::sqrt(-2.f * std::log(u1));
            return { r * std::cos(2.f * c_Pi * u2), r * std::sin(2.f * c_Pi * u2) };
        }

        bool SameSpectrum(const OceanSettings& a, const OceanSettings& b)
        {
            return a.windSpeed == b.windSpeed && a.windDirectionDeg == b.windDirectionDeg && a.fetchKm == b.fetchKm
                && a.swell == b.swell && a.swellDirectionDeg == b.swellDirectionDeg && a.swellPeriod == b.swellPeriod
                && a.seed == b.seed;
        }
    }

    Ocean::Ocean(Context& ctx) : m_Ctx(ctx) { }
    Ocean::~Ocean() = default;

    void Ocean::CreateResources(nvrhi::ICommandList* cmd)
    {
        nvrhi::IDevice* device = m_Ctx.device;
        m_SimConstants = device->createBuffer(
            nvrhi::utils::CreateVolatileConstantBufferDesc(sizeof(OceanSimConstants), "hyper/ocean-sim", 4));
        m_SurfaceConstants = device->createBuffer(
            nvrhi::utils::CreateVolatileConstantBufferDesc(sizeof(OceanSurfaceConstants), "hyper/ocean-surface", 4));

        nvrhi::BindingLayoutDesc sim;
        sim.setVisibility(nvrhi::ShaderType::Compute)
            .addItem(nvrhi::BindingLayoutItem::VolatileConstantBuffer(0))
            .addItem(nvrhi::BindingLayoutItem::PushConstants(1, 8));
        for (uint32_t i = 0; i < 4; ++i)
            sim.addItem(nvrhi::BindingLayoutItem::Texture_SRV(i)).addItem(nvrhi::BindingLayoutItem::Texture_UAV(i));
        m_SimLayout = device->createBindingLayout(sim);
        m_SurfaceLayout = device->createBindingLayout(nvrhi::BindingLayoutDesc()
            .setVisibility(nvrhi::ShaderType::Vertex | nvrhi::ShaderType::Pixel)
            .addItem(nvrhi::BindingLayoutItem::VolatileConstantBuffer(0))
            .addItem(nvrhi::BindingLayoutItem::Texture_SRV(0))
            .addItem(nvrhi::BindingLayoutItem::Texture_SRV(1))
            .addItem(nvrhi::BindingLayoutItem::Texture_SRV(2))
            .addItem(nvrhi::BindingLayoutItem::Sampler(0)));
        m_Sampler = device->createSampler(nvrhi::SamplerDesc().setAllFilters(true).setAllAddressModes(nvrhi::SamplerAddressMode::Wrap)
            .setMaxAnisotropy(16.f));

        auto array = [&](const char* name, nvrhi::Format format, uint32_t mips)
        {
            return device->createTexture(nvrhi::TextureDesc().setDebugName(name)
                .setDimension(nvrhi::TextureDimension::Texture2DArray)
                .setWidth(OCEAN_N).setHeight(OCEAN_N).setArraySize(OCEAN_CASCADES).setMipLevels(mips)
                .setFormat(format).setIsUAV(true)
                .setInitialState(nvrhi::ResourceStates::ShaderResource).setKeepInitialState(true));
        };
        m_Spectrum = array("hyper/ocean-spectrum", nvrhi::Format::RGBA32_FLOAT, 1);
        m_FftA = array("hyper/ocean-fft-a", nvrhi::Format::RGBA32_FLOAT, 1);
        m_FftB = array("hyper/ocean-fft-b", nvrhi::Format::RGBA32_FLOAT, 1);
        m_Displacement[0] = array("hyper/ocean-displacement-0", nvrhi::Format::RGBA16_FLOAT, OCEAN_MIPS);
        m_Displacement[1] = array("hyper/ocean-displacement-1", nvrhi::Format::RGBA16_FLOAT, OCEAN_MIPS);
        m_Slopes = array("hyper/ocean-slopes", nvrhi::Format::RGBA16_FLOAT, OCEAN_MIPS);

        const auto mip = [](uint32_t m) { return nvrhi::TextureSubresourceSet(m, 1, 0, OCEAN_CASCADES); };
        for (uint32_t cur = 0; cur < 2; ++cur)
        {
            nvrhi::ITexture* disp = m_Displacement[cur];
            nvrhi::ITexture* prev = m_Displacement[cur ^ 1];
            auto set = [&](nvrhi::ITexture* mipDisp, nvrhi::ITexture* mipSlopes, uint32_t m)
            {
                nvrhi::BindingSetDesc d;
                d.addItem(nvrhi::BindingSetItem::ConstantBuffer(0, m_SimConstants))
                    .addItem(nvrhi::BindingSetItem::PushConstants(1, 8))
                    .addItem(nvrhi::BindingSetItem::Texture_SRV(0, m_Spectrum))
                    .addItem(nvrhi::BindingSetItem::Texture_SRV(1, prev, nvrhi::Format::UNKNOWN, mip(0)))
                    .addItem(m ? nvrhi::BindingSetItem::Texture_SRV(2, mipDisp, nvrhi::Format::UNKNOWN, mip(m - 1))
                               : nvrhi::BindingSetItem::Texture_SRV(2, m_Spectrum))
                    .addItem(m ? nvrhi::BindingSetItem::Texture_SRV(3, mipSlopes, nvrhi::Format::UNKNOWN, mip(m - 1))
                               : nvrhi::BindingSetItem::Texture_SRV(3, m_Spectrum))
                    .addItem(nvrhi::BindingSetItem::Texture_UAV(0, m_FftA))
                    .addItem(nvrhi::BindingSetItem::Texture_UAV(1, m_FftB))
                    .addItem(nvrhi::BindingSetItem::Texture_UAV(2, disp, nvrhi::Format::UNKNOWN, mip(m)))
                    .addItem(nvrhi::BindingSetItem::Texture_UAV(3, m_Slopes, nvrhi::Format::UNKNOWN, mip(m)));
                return device->createBindingSet(d, m_SimLayout);
            };
            m_SimSets[cur] = set(nullptr, nullptr, 0);
            m_MipSets[cur].clear();
            for (uint32_t m = 1; m < OCEAN_MIPS; ++m)
                m_MipSets[cur].push_back(set(disp, m_Slopes, m));

            m_SurfaceSets[cur] = device->createBindingSet(nvrhi::BindingSetDesc()
                .addItem(nvrhi::BindingSetItem::ConstantBuffer(0, m_SurfaceConstants))
                .addItem(nvrhi::BindingSetItem::Texture_SRV(0, disp))
                .addItem(nvrhi::BindingSetItem::Texture_SRV(1, prev))
                .addItem(nvrhi::BindingSetItem::Texture_SRV(2, m_Slopes))
                .addItem(nvrhi::BindingSetItem::Sampler(0, m_Sampler)), m_SurfaceLayout);
        }

        // One level's grid, its diagonals alternating so the triangles don't all lean one way.
        std::vector<uint32_t> indices;
        indices.reserve(OCEAN_GRID * OCEAN_GRID * 6);
        constexpr uint32_t row = OCEAN_GRID + 1;
        for (uint32_t y = 0; y < OCEAN_GRID; ++y)
            for (uint32_t x = 0; x < OCEAN_GRID; ++x)
            {
                const uint32_t a = y * row + x, b = a + 1, c = a + row, d = c + 1;
                if ((x + y) & 1)
                    indices.insert(indices.end(), { a, c, b, b, c, d });
                else
                    indices.insert(indices.end(), { a, c, d, a, d, b });
            }
        m_Indices = device->createBuffer(nvrhi::BufferDesc().setDebugName("hyper/ocean-indices")
            .setByteSize(indices.size() * 4).setIsIndexBuffer(true)
            .setInitialState(nvrhi::ResourceStates::IndexBuffer).setKeepInitialState(true));
        cmd->writeBuffer(m_Indices, indices.data(), indices.size() * 4);
        for (auto& t : m_Displacement)
            cmd->clearTextureFloat(t, nvrhi::AllSubresources, nvrhi::Color(0.f));
        cmd->clearTextureFloat(m_Slopes, nvrhi::AllSubresources, nvrhi::Color(0.f));
    }

    void Ocean::CreatePipelines()
    {
        if (!m_Used)
            return;   // built on first use
        auto make = [&](const char* entry) -> nvrhi::ComputePipelineHandle
        {
            auto shader = m_Ctx.shaders->CreateShader("hyper/ocean/ocean_cs", entry, nullptr, nvrhi::ShaderType::Compute);
            return shader ? m_Ctx.device->createComputePipeline(
                nvrhi::ComputePipelineDesc().setComputeShader(shader).addBindingLayout(m_SimLayout)) : nullptr;
        };
        auto evolve = make("evolve");
        auto fft = make("fft");
        auto assemble = make("assemble");
        auto mip = make("mip");
        auto vs = m_Ctx.shaders->CreateShader("hyper/ocean/surface", "vs", nullptr, nvrhi::ShaderType::Vertex);
        auto ps = m_Ctx.shaders->CreateShader("hyper/ocean/surface", "ps", nullptr, nvrhi::ShaderType::Pixel);
        if (evolve && fft && assemble && mip && vs && ps)
        {
            m_Evolve = evolve;
            m_Fft = fft;
            m_Assemble = assemble;
            m_Mip = mip;
            m_SurfaceVs = vs;
            m_SurfacePs = ps;
            m_SurfacePipelines.clear();
        }
    }

    // The initial spectrum h0(k) of every cascade (Tessendorf): Gaussian amplitudes whose
    // variance is the wind sea's and the swell's energy at k.
    void Ocean::BuildSpectrum(nvrhi::ICommandList* cmd)
    {
        const OceanSettings& s = settings;
        const float fetch = std::max(s.fetchKm, 0.1f) * 1000.f;
        const float wind = std::max(s.windSpeed, 0.1f);
        const float alpha = 0.076f * std::pow(wind * wind / (fetch * c_Gravity), 0.22f);
        const float peak = 22.f * std::pow(c_Gravity * c_Gravity / (wind * fetch), 1.f / 3.f);
        const float windAngle = s.windDirectionDeg * c_Pi / 180.f;
        const float swellPeak = 2.f * c_Pi / std::max(s.swellPeriod, 1.f);
        const float swellAlpha = 0.0081f * 0.25f * s.swell * s.swell;
        const float swellAngle = s.swellDirectionDeg * c_Pi / 180.f;
        constexpr float c_ShortCut = 0.01f;   // metres: ripples shorter than a few cm fade

        std::vector<float> h0(OCEAN_N * OCEAN_N * 4);
        std::vector<float> data(OCEAN_N * OCEAN_N * 4);
        m_LongWaves.clear();
        double variance = 0.0;
        for (uint32_t c = 0; c < OCEAN_CASCADES; ++c)
        {
            const float dk = 2.f * c_Pi / c_Patch[c];
            const float kMin = BandStart(c);
            const float kMax = c + 1 < OCEAN_CASCADES ? BandStart(c + 1) : 1e9f;
            for (uint32_t y = 0; y < OCEAN_N; ++y)
                for (uint32_t x = 0; x < OCEAN_N; ++x)
                {
                    const float kx = (float(x) - OCEAN_N / 2) * dk, kz = (float(y) - OCEAN_N / 2) * dk;
                    const float k = std::sqrt(kx * kx + kz * kz);
                    float amplitude = 0.f;
                    if (k > 1e-6f && k >= kMin && k < kMax)
                    {
                        const float omega = std::sqrt(c_Gravity * k);
                        const float dOmegaDk = c_Gravity / (2.f * omega);
                        const float theta = std::atan2(kz, kx);
                        const float sea = Jonswap(omega, alpha, peak, 3.3f) * Spreading(WrapAngle(theta - windAngle), omega, peak, 1.f);
                        const float swell = swellAlpha > 0.f
                            ? Jonswap(omega, swellAlpha, swellPeak, 3.3f) * Spreading(WrapAngle(theta - swellAngle), omega, swellPeak, 4.f)
                            : 0.f;
                        const float spectrum = (sea + swell) * dOmegaDk / k * std::exp(-k * k * c_ShortCut * c_ShortCut);
                        amplitude = 0.5f * dk * std::sqrt(spectrum);
                        variance += 2.0 * double(spectrum) * dk * dk * 0.5;
                    }
                    const auto [gr, gi] = Gaussian(s.seed * 0x632be5abu ^ (c * OCEAN_N * OCEAN_N + y * OCEAN_N + x));
                    float* h = &h0[(y * OCEAN_N + x) * 4];
                    h[0] = gr * amplitude;
                    h[1] = gi * amplitude;
                }
            // conj(h0(-k)): the mirrored texel (index N - i, wrapping N to 0).
            for (uint32_t y = 0; y < OCEAN_N; ++y)
                for (uint32_t x = 0; x < OCEAN_N; ++x)
                {
                    const float* m = &h0[(((OCEAN_N - y) % OCEAN_N) * OCEAN_N + (OCEAN_N - x) % OCEAN_N) * 4];
                    float* d = &data[(y * OCEAN_N + x) * 4];
                    const float* h = &h0[(y * OCEAN_N + x) * 4];
                    d[0] = h[0];
                    d[1] = h[1];
                    d[2] = m[0];
                    d[3] = -m[1];
                    if (c == 0)
                    {
                        const float kx = (float(x) - OCEAN_N / 2) * dk, kz = (float(y) - OCEAN_N / 2) * dk;
                        const float k = std::sqrt(kx * kx + kz * kz);
                        if (k > 1e-6f && k < BandStart(1) && std::abs(d[0]) + std::abs(d[1]) + std::abs(d[2]) + std::abs(d[3]) > 1e-5f)
                            m_LongWaves.push_back({ kx, kz, k, std::sqrt(c_Gravity * k), d[0], d[1], d[2], d[3] });
                    }
                }
            cmd->writeTexture(m_Spectrum, c, 0, data.data(), OCEAN_N * 16);
        }
        // Significant wave height is 4 standard deviations; a typical crest is half of it.
        m_WaveHeight = 2.f * float(std::sqrt(variance));
        m_Built = settings;
        m_SpectrumValid = true;
    }

    void Ocean::Update(nvrhi::ICommandList* cmd, float time)
    {
        if (!settings.enabled)
            return;
        if (!m_Used)
        {
            m_Used = true;
            CreateResources(cmd);
            CreatePipelines();
        }
        if (!m_Evolve)
            return;
        auto scope = m_Ctx.profiler->Measure(cmd, "ocean sim");
        if (!m_SpectrumValid || !SameSpectrum(m_Built, settings))
            BuildSpectrum(cmd);

        const float dt = std::clamp(time - m_LastTime, 0.f, 0.25f);
        m_LastTime = time;
        m_Updated = true;
        m_Current ^= 1;

        OceanSimConstants c = {};
        c.patchSize = float4(c_Patch[0], c_Patch[1], c_Patch[2], 0.f);
        c.time = time * settings.timeScale;
        c.choppiness = settings.choppiness;
        c.foamKeep = std::exp(-dt * std::max(settings.timeScale, 0.f) / std::max(settings.foamDecay, 0.01f));
        c.foamThreshold = 0.85f;
        c.foamGain = 2.f * settings.foam;
        cmd->writeBuffer(m_SimConstants, &c, sizeof(c));

        struct Push { uint32_t columns, size; };
        auto run = [&](nvrhi::IComputePipeline* pipeline, nvrhi::IBindingSet* set, Push push, uint32_t x, uint32_t y)
        {
            nvrhi::ComputeState state;
            state.pipeline = pipeline;
            state.bindings = { set };
            // NVRHI places barriers only when the bound sets change; the passes below share
            // one, and each reads what the one before wrote.
            cmd->setResourceStatesForBindingSet(set);
            cmd->setComputeState(state);
            cmd->setPushConstants(&push, sizeof(push));
            cmd->dispatch(x, y, OCEAN_CASCADES);
        };
        nvrhi::IBindingSet* set = m_SimSets[m_Current];
        run(m_Evolve, set, { 0, 0 }, OCEAN_N / 8, OCEAN_N / 8);
        run(m_Fft, set, { 0, 0 }, OCEAN_N, 1);
        run(m_Fft, set, { 1, 0 }, OCEAN_N, 1);
        run(m_Assemble, set, { 0, 0 }, OCEAN_N / 8, OCEAN_N / 8);
        for (uint32_t m = 1; m < OCEAN_MIPS; ++m)
        {
            const uint32_t size = OCEAN_N >> m;
            run(m_Mip, m_MipSets[m_Current][m - 1], { 0, size }, (size + 7) / 8, (size + 7) / 8);
        }
    }

    nvrhi::IGraphicsPipeline* Ocean::SurfacePipeline(nvrhi::IFramebuffer* fb)
    {
        const size_t key = std::hash<nvrhi::FramebufferInfo>()(fb->getFramebufferInfo());
        auto& pipeline = m_SurfacePipelines[key];
        if (pipeline)
            return pipeline;
        nvrhi::GraphicsPipelineDesc desc;
        desc.setPrimType(nvrhi::PrimitiveType::TriangleList)
            .setVertexShader(m_SurfaceVs).setPixelShader(m_SurfacePs)
            .addBindingLayout(m_SurfaceLayout);
        desc.renderState.rasterState.setCullNone();
        desc.renderState.depthStencilState = GBuffer::DepthStencil();
        pipeline = m_Ctx.device->createGraphicsPipeline(desc, fb->getFramebufferInfo());
        return pipeline;
    }

    void Ocean::RenderGBuffer(nvrhi::ICommandList* cmd, const engine::PlanarView& view, const engine::PlanarView& previous,
                              GBuffer& gbuffer, const Light& light)
    {
        if (!settings.enabled || !m_Updated || !m_SurfaceVs)
            return;
        auto scope = m_Ctx.profiler->Measure(cmd, "ocean");

        OceanSurfaceConstants c = {};
        view.FillPlanarViewConstants(c.view);
        previous.FillPlanarViewConstants(c.viewPrev);
        const float3 camera = float3(view.GetViewOrigin());
        const float height = std::max(std::abs(camera.y - settings.seaLevel), 0.1f);

        // Levels from the first whose cells are about a pixel or more under the camera, out
        // past the horizon (and at least 20 km).
        const float base = c_BaseCell / std::clamp(settings.detail, 0.25f, 4.f);
        const int first = std::max(0, int(std::floor(std::log2(height / (OCEAN_GRID * base)))));
        const float horizon = std::sqrt(2.f * c_EarthRadius * height + height * height);
        const float reach = std::max(1.2f * horizon, 20000.f);
        uint32_t count = 0;
        for (int l = first; count < OCEAN_MAX_LEVELS; ++l)
        {
            const float cell = base * std::exp2(float(l));
            const float snap = 2.f * cell;
            OceanLevel& level = c.levels[count++];
            level.center = float2(std::round(camera.x / snap) * snap, std::round(camera.z / snap) * snap);
            level.cellSize = cell;
            level.halfExtent = 0.5f * OCEAN_GRID * cell;
            if (level.halfExtent >= reach)
                break;
        }
        c.levelCount = count;
        c.morphStart = 0.7f;
        for (uint32_t i = 0; i < OCEAN_CASCADES; ++i)
        {
            c.patchSize[i] = c_Patch[i];
            c.invPatchSize[i] = 1.f / c_Patch[i];
        }
        c.cameraPos = camera;
        c.earthRadius = c_EarthRadius;
        c.sunDirection = dm::normalize(light.sunDirection);
        c.sunRadiance = light.sunRadiance;
        c.roughness = settings.roughness;
        c.waterAlbedo = settings.waterAlbedo;
        c.foamAmount = settings.foam;
        c.scatterColor = settings.scatterColor;
        c.scatterStrength = settings.scatter;
        c.foamAlbedo = float3(0.85f, 0.88f, 0.9f);
        c.waveHeight = std::max(m_WaveHeight, 0.05f);
        c.seaLevel = settings.seaLevel;
        cmd->writeBuffer(m_SurfaceConstants, &c, sizeof(c));

        nvrhi::IFramebuffer* fb = gbuffer.GBufferFramebuffer->GetFramebuffer(view);
        nvrhi::GraphicsState state;
        state.pipeline = SurfacePipeline(fb);
        state.framebuffer = fb;
        state.bindings = { m_SurfaceSets[m_Current] };
        state.indexBuffer = { m_Indices, nvrhi::Format::R32_UINT, 0 };
        state.viewport.addViewportAndScissorRect(view.GetViewportState().viewports[0]);
        cmd->setGraphicsState(state);
        cmd->drawIndexed(nvrhi::DrawArguments().setVertexCount(OCEAN_GRID * OCEAN_GRID * 6).setInstanceCount(count));
    }

    float Ocean::HeightAt(dm::float2 xz, float time, float minWavelength) const
    {
        if (m_LongWaves.empty())
            return settings.seaLevel;
        const float kMax = 2.f * c_Pi / std::max(minWavelength, 1.f);
        const float t = time * settings.timeScale;
        const float lambda = settings.choppiness;
        // Displacement at an undisplaced point: (dx, height, dz).
        auto at = [&](float px, float pz)
        {
            float dx = 0.f, dy = 0.f, dz = 0.f;
            for (const Wave& w : m_LongWaves)
            {
                if (w.k >= kMax)
                    continue;
                const float st = std::sin(w.omega * t), ct = std::cos(w.omega * t);
                // h = h0 e^(iwt) + conj(h0(-k)) e^(-iwt), then times e^(i k.x).
                const float hr = w.h0re * ct - w.h0im * st + w.hmre * ct + w.hmim * st;
                const float hi = w.h0re * st + w.h0im * ct - w.hmre * st + w.hmim * ct;
                const float phase = w.kx * px + w.kz * pz;
                const float sp = std::sin(phase), cp = std::cos(phase);
                const float vr = hr * cp - hi * sp, vi = hr * sp + hi * cp;
                dy += vr;
                // i kn v: real part -kn vi.
                dx -= w.kx / w.k * vi;
                dz -= w.kz / w.k * vi;
            }
            return float3(dx * lambda, dy, dz * lambda);
        };
        // The point whose displacement lands on xz.
        float px = xz.x, pz = xz.y;
        for (int i = 0; i < 3; ++i)
        {
            const float3 d = at(px, pz);
            px = xz.x - d.x;
            pz = xz.y - d.z;
        }
        return settings.seaLevel + at(px, pz).y;
    }

    void Ocean::BuildUI()
    {
        OceanSettings& s = settings;
        ImGui::Checkbox("ocean", &s.enabled);
        if (!s.enabled)
            return;
        ImGui::SliderFloat("wind (m/s)", &s.windSpeed, 0.5f, 25.f);
        ImGui::SliderFloat("wind direction", &s.windDirectionDeg, -180.f, 180.f);
        ImGui::SliderFloat("fetch (km)", &s.fetchKm, 1.f, 1000.f, "%.0f", ImGuiSliderFlags_Logarithmic);
        ImGui::SliderFloat("swell", &s.swell, 0.f, 1.f);
        ImGui::SliderFloat("swell direction", &s.swellDirectionDeg, -180.f, 180.f);
        ImGui::SliderFloat("swell period (s)", &s.swellPeriod, 4.f, 20.f);
        ImGui::SliderFloat("choppiness", &s.choppiness, 0.f, 2.f);
        int seed = int(s.seed);
        if (ImGui::InputInt("seed", &seed))
            s.seed = uint32_t(std::max(seed, 0));
        ImGui::SliderFloat("time scale", &s.timeScale, 0.f, 2.f);
        ImGui::SliderFloat("foam", &s.foam, 0.f, 3.f);
        ImGui::SliderFloat("foam decay (s)", &s.foamDecay, 0.1f, 10.f);
        ImGui::ColorEdit3("water", &s.waterAlbedo.x);
        ImGui::ColorEdit3("scatter colour", &s.scatterColor.x);
        ImGui::SliderFloat("scatter", &s.scatter, 0.f, 4.f);
        ImGui::SliderFloat("roughness", &s.roughness, 0.01f, 0.3f);
        ImGui::SliderFloat("mesh detail", &s.detail, 0.25f, 4.f);
        ImGui::Text("crest height %.2f m, %zu long waves", m_WaveHeight, m_LongWaves.size());
    }
}
