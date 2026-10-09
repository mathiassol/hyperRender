#pragma once

#include <hyper/platform.h>

#include <donut/core/math/math.h>
#include <nvrhi/nvrhi.h>

#include <map>
#include <string>
#include <vector>

namespace donut::engine { class PlanarView; }

namespace hyper
{
    struct Context;

    // An atmosphere's physics, from what it is made of (derived, not
    // tuned). Per channel at 612, 549 and 465 nm.
    namespace atmosphere
    {
        // Rayleigh scattering of a gas mixture at the ground, per metre: each gas's
        // refractivity at STP and King factor (Bates 1984, Sneep & Ubachs 2005), at the
        // number density P / kT. Gases it doesn't know are left out (and reported).
        dm::double3 Rayleigh(const std::map<std::string, double>& fractions, double pressure, double temperature,
                             std::vector<std::string>* unknown = nullptr);
        // The same at any three wavelengths (metres), for comparing with published values.
        dm::double3 RayleighAt(const std::map<std::string, double>& fractions, double pressure, double temperature,
                               const dm::double3& wavelengths);

        // An aerosol from its particles (Mie theory, Bohren & Huffman's BHMIE, over a
        // Hansen-Travis gamma distribution of effective radius and variance): extinction
        // and scattering at the ground for a given optical depth at 550 nm and scale height,
        // the asymmetry parameter, and the phase function itself (its diffraction peak is
        // what makes a sunset's glow and a dusty sky's aureole).
        struct Aerosol
        {
            dm::double3 extinction, scattering, g;
            std::vector<float> phase;   // ATMO_PHASE x RGB, per steradian, at theta = pi (k / (ATMO_PHASE - 1))^2
        };
        Aerosol Mie(double effectiveRadius, double effectiveVariance, double realIndex, const dm::double3& imaginaryIndex,
                    double opticalDepth550, double scaleHeight);

        // Ozone at its peak, per metre, from its column (Dobson units) spread as a tent.
        dm::double3 Ozone(double columnDu, double width);
    }

    // An atmosphere as the tables and shaders use it (planet_atmosphere_cb.h), in metres.
    struct AtmosphereParameters
    {
        double bottom = 6.371e6, top = 6.471e6;
        dm::double3 rayleigh = dm::double3(0.0);
        double rayleighH = 8000.0;
        dm::double3 mieScattering = dm::double3(0.0), mieExtinction = dm::double3(0.0), mieG = dm::double3(0.8);
        double mieH = 1200.0;
        dm::double3 absorption = dm::double3(0.0);
        double ozonePeak = 25000.0, ozoneWidth = 0.0;
        dm::double3 groundAlbedo = dm::double3(0.3);
        std::vector<float> miePhase;   // Aerosol::phase; empty: Cornette-Shanks from mieG
    };

    // Planets' atmospheres (the atmospheres module): Hillaire 2020's tables, built on the
    // CPU per atmosphere (transmittance, multiple scattering, the sky's irradiance on the
    // ground), uploaded into one slice each, and composited over a lit image: the light
    // the air scatters towards the camera, and what it lets through, along every view ray,
    // from space, at the limb or from the ground.
    //
    //     int slice = atmospheres.Add(parameters);                      // at load: about 50 ms each
    //     atmospheres.Prepare(cmd);                                     // once a frame, first
    //     atmospheres.Render(cmd, view, hdr, depth, constants, count);  // after lighting
    // Shaders that light surfaces under an atmosphere bind Transmittance() and Irradiance()
    // with hyper/planet_atmosphere.hlsli.
    class PlanetAtmospheres
    {
    public:
        explicit PlanetAtmospheres(Context& ctx);
        ~PlanetAtmospheres();

        static constexpr int c_Max = 4;

        // Builds its tables; the slice, or -1 when full.
        int Add(const AtmosphereParameters& p);
        // New parameters for a slice: its tables rebuilt (and uploaded on the next Prepare).
        void Replace(int slice, const AtmosphereParameters& p);
        [[nodiscard]] int Count() const { return int(m_Atmospheres.size()); }
        [[nodiscard]] const AtmosphereParameters& Parameters(int slice) const { return m_Atmospheres[size_t(slice)].p; }

        // The CPU's copy of the tables, for the camera's exposure: sunlight's transmittance
        // to a point, and the sky's irradiance on a level surface there (per lux of sunlight).
        [[nodiscard]] dm::double3 Transmittance(int slice, double r, double mu) const;
        [[nodiscard]] dm::double3 Irradiance(int slice, double r, double muS) const;
        // The light the air scatters towards `origin` along `dir` (per lux of sunlight from
        // `towardsSun`), the planet at the origin, as the composite shader marches it; and the
        // ray's transmittance and where it meets the ground (-1: it doesn't).
        [[nodiscard]] dm::double3 Radiance(int slice, const dm::double3& origin, const dm::double3& dir,
                                           const dm::double3& towardsSun, dm::double3* transmittance = nullptr,
                                           double* ground = nullptr) const;
        // The planet's geometric albedo with its air over a ground of this albedo (the disk at
        // full phase, as a fraction of a white Lambert disk's brightness). The ground is
        // Lambert, or Lommel-Seeliger (regolith) scaled so that `groundAlbedo` is 1.5 x its
        // own geometric albedo, as for Lambert.
        [[nodiscard]] dm::double3 GeometricAlbedo(int slice, const dm::double3& groundAlbedo, bool regolith = false) const;
        // The ground albedo under the air that gives the planet a measured geometric albedo
        // (its air brightens the disk and dims the ground), the tables rebuilt for it. A
        // measured albedo is the planet's with its air: the ground's alone is derived.
        dm::double3 FitGroundAlbedo(int slice, const dm::double3& geometricAlbedo, bool regolith = false);

        // Once a frame, before anything samples the tables: makes them on first use, uploads
        // new ones.
        void Prepare(nvrhi::ICommandList* cmd);

        // The light the air adds to a lit HDR image (UAV, exposed) and what it lets through,
        // for each atmosphere given (planet_atmosphere_cb.h's PlanetAtmosphereConstants, raw).
        void Render(nvrhi::ICommandList* cmd, const donut::engine::PlanarView& view, nvrhi::ITexture* hdr,
                    nvrhi::ITexture* depth, const void* constants, uint32_t count);

        [[nodiscard]] nvrhi::ITexture* TransmittanceTexture() const { return m_Transmittance; }
        [[nodiscard]] nvrhi::ITexture* IrradianceTexture() const { return m_Irradiance; }
        [[nodiscard]] nvrhi::ITexture* MultiScatterTexture() const { return m_MultiScatter; }
        [[nodiscard]] nvrhi::ITexture* PhaseTexture() const { return m_Phase; }
        [[nodiscard]] nvrhi::ISampler* Sampler() const { return m_Sampler; }

        void CreatePipelines();

    private:
        struct Atmosphere
        {
            AtmosphereParameters p;
            std::vector<float> transmittance, multiScatter, irradiance;   // RGBA, row-major
            bool uploaded = false;
        };

        bool m_Used = false;   // the GPU side is made on first use
        Context& m_Ctx;
        std::vector<Atmosphere> m_Atmospheres;
        nvrhi::TextureHandle m_Transmittance, m_MultiScatter, m_Irradiance, m_Phase;
        nvrhi::SamplerHandle m_Sampler;
        nvrhi::BufferHandle m_Constants;
        nvrhi::BindingLayoutHandle m_Layout;
        nvrhi::ComputePipelineHandle m_Composite;

        void CreateResources();
        static Atmosphere Build(const AtmosphereParameters& p);
        [[nodiscard]] dm::double3 MultiScatter(int slice, double r, double muS) const;
    };
}
