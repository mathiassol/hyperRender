#include <hyper/atmospheres/PlanetAtmospheres.h>

#include <hyper/App.h>

#include <donut/engine/ShaderFactory.h>
#include <donut/engine/View.h>
#include <nvrhi/utils.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstring>

#include <hyper/shared.h>
#include <donut/shaders/view_cb.h>
#include <hyper/planet_atmosphere_cb.h>

namespace hyper
{
    namespace
    {
        constexpr double c_Pi = 3.14159265358979323846;
        constexpr double c_Boltzmann = 1.380649e-23;
        constexpr double c_Loschmidt = 2.6867811e25;   // molecules / m^3 at 273.15 K and 101,325 Pa
        const dm::double3 c_Wavelengths(612e-9, 549e-9, 465e-9);

        // Refractivity (n - 1) at STP near 550 nm and King correction factor, by gas.
        struct Gas
        {
            const char* name;
            double refractivity;
            double king;
        };
        const Gas c_Gases[] = {
            { "N2", 2.98e-4, 1.034 }, { "O2", 2.71e-4, 1.096 }, { "Ar", 2.81e-4, 1.0 }, { "CO2", 4.49e-4, 1.15 },
            { "H2", 1.39e-4, 1.0 }, { "He", 3.5e-5, 1.0 }, { "CH4", 4.41e-4, 1.0 }, { "H2O", 2.56e-4, 1.0 },
        };

        // A molecule's Rayleigh cross-section, m^2: 24 pi^3 / (lambda^4 N_s^2) ((n^2 - 1) / (n^2 + 2))^2 F_K.
        double RayleighCrossSection(const Gas& g, double lambda)
        {
            const double n2 = (1.0 + g.refractivity) * (1.0 + g.refractivity);
            const double f = (n2 - 1.0) / (n2 + 2.0);
            return 24.0 * c_Pi * c_Pi * c_Pi / (std::pow(lambda, 4.0) * c_Loschmidt * c_Loschmidt) * f * f * g.king;
        }

        // Bohren & Huffman's BHMIE (Absorption and Scattering of Light by Small Particles,
        // appendix A): extinction and scattering efficiencies and the asymmetry parameter of a
        // sphere of size parameter x and complex index m (absorbing: positive imaginary part).
        // mus / s11: optionally, the scattering matrix's S11 = (|S1|^2 + |S2|^2) / 2 at each
        // cosine (the phase function, unnormalised: dC_sca / dOmega = S11 / k^2).
        void Bhmie(double x, std::complex<double> m, double& qext, double& qsca, double& g,
                   const std::vector<double>* mus = nullptr, std::vector<double>* s11 = nullptr)
        {
            const size_t angles = mus ? mus->size() : 0;
            std::vector<std::complex<double>> s1(angles), s2(angles);
            std::vector<double> pi0(angles, 0.0), pi1(angles, 1.0);
            const std::complex<double> y = m * x;
            const int nstop = int(x + 4.0 * std::cbrt(x) + 2.0);
            const int nmx = int(std::max(double(nstop), std::abs(y))) + 15;
            std::vector<std::complex<double>> d(size_t(nmx) + 1);
            d[size_t(nmx)] = 0.0;
            for (int n = nmx; n >= 1; --n)
            {
                const double en = n;
                d[size_t(n) - 1] = en / y - 1.0 / (d[size_t(n)] + en / y);
            }
            double psi0 = std::cos(x), psi1 = std::sin(x), chi0 = -std::sin(x), chi1 = std::cos(x);
            std::complex<double> xi1(psi1, -chi1), an1, bn1;
            double sca = 0.0, ext = 0.0, gsum = 0.0;
            for (int n = 1; n <= nstop; ++n)
            {
                const double en = n;
                const double psi = (2.0 * en - 1.0) * psi1 / x - psi0;
                const double chi = (2.0 * en - 1.0) * chi1 / x - chi0;
                const std::complex<double> xi(psi, -chi);
                const std::complex<double> da = d[size_t(n)] / m + en / x, db = m * d[size_t(n)] + en / x;
                const std::complex<double> an = (da * psi - psi1) / (da * xi - xi1);
                const std::complex<double> bn = (db * psi - psi1) / (db * xi - xi1);
                sca += (2.0 * en + 1.0) * (std::norm(an) + std::norm(bn));
                ext += (2.0 * en + 1.0) * (an.real() + bn.real());
                const double fn = (2.0 * en + 1.0) / (en * (en + 1.0));
                for (size_t j = 0; j < angles; ++j)
                {
                    // The angular functions pi_n and tau_n, by their recurrences.
                    const double mu = (*mus)[j];
                    const double tau = en * mu * pi1[j] - (en + 1.0) * pi0[j];
                    s1[j] += fn * (an * pi1[j] + bn * tau);
                    s2[j] += fn * (an * tau + bn * pi1[j]);
                    const double next = ((2.0 * en + 1.0) * mu * pi1[j] - (en + 1.0) * pi0[j]) / en;
                    pi0[j] = pi1[j];
                    pi1[j] = next;
                }
                gsum += (2.0 * en + 1.0) / (en * (en + 1.0)) * (an.real() * bn.real() + an.imag() * bn.imag());
                if (n > 1)
                    gsum += (en - 1.0) * (en + 1.0) / en
                        * (an1.real() * an.real() + an1.imag() * an.imag() + bn1.real() * bn.real() + bn1.imag() * bn.imag());
                psi0 = psi1;
                psi1 = psi;
                chi0 = chi1;
                chi1 = chi;
                xi1 = std::complex<double>(psi1, -chi1);
                an1 = an;
                bn1 = bn;
            }
            g = sca > 0.0 ? 2.0 * gsum / sca : 0.0;
            qsca = sca * 2.0 / (x * x);
            qext = ext * 2.0 / (x * x);
            if (s11)
            {
                s11->resize(angles);
                for (size_t j = 0; j < angles; ++j)
                    (*s11)[j] = 0.5 * (std::norm(s1[j]) + std::norm(s2[j]));
            }
        }

        // The phase table's angles: theta = pi (k / (n - 1))^2, close together forward.
        std::vector<double> PhaseCosines()
        {
            std::vector<double> mus(ATMO_PHASE);
            for (int k = 0; k < ATMO_PHASE; ++k)
            {
                const double u = double(k) / (ATMO_PHASE - 1);
                mus[size_t(k)] = std::cos(c_Pi * u * u);
            }
            return mus;
        }

        // Cross-sections averaged over a Hansen-Travis gamma distribution of radii:
        // n(r) ~ r^((1 - 3v) / v) exp(-r / (a v)), a the effective radius, v the variance.
        // phase: optionally, the distribution's phase function per steradian at PhaseCosines().
        void MieAverage(double a, double v, double lambda, std::complex<double> m, double& cext, double& csca, double& g,
                        std::vector<double>* phase = nullptr)
        {
            constexpr int n = 160;
            const double lo = std::log(a * 0.02), hi = std::log(a * 12.0), step = (hi - lo) / n;
            double wsum = 0.0, ext = 0.0, sca = 0.0, gsca = 0.0;
            const std::vector<double> mus = phase ? PhaseCosines() : std::vector<double>();
            std::vector<double> s11, sum(mus.size(), 0.0);
            const double k = 2.0 * c_Pi / lambda;
            for (int i = 0; i < n; ++i)
            {
                const double r = std::exp(lo + (i + 0.5) * step);
                const double w = std::exp((1.0 - 3.0 * v) / v * std::log(r / a) - r / (a * v)) * r * step;   // n(r) dr
                double qe, qs, gg;
                Bhmie(2.0 * c_Pi * r / lambda, m, qe, qs, gg, phase ? &mus : nullptr, phase ? &s11 : nullptr);
                const double area = c_Pi * r * r;
                wsum += w;
                ext += w * area * qe;
                sca += w * area * qs;
                gsca += w * area * qs * gg;
                for (size_t j = 0; j < s11.size(); ++j)
                    sum[j] += w * s11[j] / (k * k);   // dC_sca / dOmega
            }
            if (phase)
            {
                phase->resize(mus.size());
                for (size_t j = 0; j < mus.size(); ++j)
                    (*phase)[j] = sca > 0.0 ? sum[j] / sca : 0.0;   // per steradian: integrates to 1
            }
            cext = ext / wsum;
            csca = sca / wsum;
            g = sca > 0.0 ? gsca / sca : 0.0;
        }

        double Height(double r, double mu, double d, double bottom)
        {
            return std::sqrt(d * d + 2.0 * r * mu * d + r * r) - bottom;
        }

        dm::double3 Extinction(const AtmosphereParameters& p, double h)
        {
            h = std::max(h, 0.0);
            const double ozone = p.ozoneWidth > 0.0 ? std::max(0.0, 1.0 - std::abs(h - p.ozonePeak) / (0.5 * p.ozoneWidth)) : 0.0;
            return p.rayleigh * std::exp(-h / p.rayleighH) + p.mieExtinction * std::exp(-h / p.mieH) + p.absorption * ozone;
        }

        dm::double3 Scattering(const AtmosphereParameters& p, double h, dm::double3* rayleigh = nullptr, dm::double3* mie = nullptr)
        {
            h = std::max(h, 0.0);
            const dm::double3 r = p.rayleigh * std::exp(-h / p.rayleighH), m = p.mieScattering * std::exp(-h / p.mieH);
            if (rayleigh)
                *rayleigh = r;
            if (mie)
                *mie = m;
            return r + m;
        }

        // The distance from radius r along mu to the top of the atmosphere.
        double DistanceToTop(const AtmosphereParameters& p, double r, double mu)
        {
            const double disc = r * r * (mu * mu - 1.0) + p.top * p.top;
            return std::max(-r * mu + std::sqrt(std::max(disc, 0.0)), 0.0);
        }
        // To the ground, or -1 if it's missed.
        double DistanceToGround(const AtmosphereParameters& p, double r, double mu)
        {
            const double disc = r * r * (mu * mu - 1.0) + p.bottom * p.bottom;
            if (mu >= 0.0 || disc < 0.0)
                return -1.0;
            return std::max(-r * mu - std::sqrt(disc), 0.0);
        }

        // Bilinear lookup in an RGBA table at texel-centred uv.
        dm::double3 Sample(const std::vector<float>& t, int w, int h, double u, double v)
        {
            const double x = std::clamp(u * w - 0.5, 0.0, double(w - 1)), y = std::clamp(v * h - 0.5, 0.0, double(h - 1));
            const int x0 = int(x), y0 = int(y), x1 = std::min(x0 + 1, w - 1), y1 = std::min(y0 + 1, h - 1);
            const double fx = x - x0, fy = y - y0;
            auto at = [&](int xx, int yy)
            {
                const float* c = &t[(size_t(yy) * w + xx) * 4];
                return dm::double3(c[0], c[1], c[2]);
            };
            return (at(x0, y0) * (1 - fx) + at(x1, y0) * fx) * (1 - fy) + (at(x0, y1) * (1 - fx) + at(x1, y1) * fx) * fy;
        }

        // planet_atmosphere.hlsli's TransmittanceUv, in doubles.
        void TransmittanceUv(const AtmosphereParameters& p, double r, double mu, double& u, double& v)
        {
            const double H = std::sqrt(std::max(p.top * p.top - p.bottom * p.bottom, 0.0));
            const double rho = std::sqrt(std::max(r * r - p.bottom * p.bottom, 0.0));
            const double d = DistanceToTop(p, r, mu);
            const double dMin = p.top - r, dMax = rho + H;
            const double xMu = (d - dMin) / std::max(dMax - dMin, 1e-3), xR = rho / std::max(H, 1e-3);
            u = 0.5 / ATMO_TRANSMITTANCE_W + xMu * (1.0 - 1.0 / ATMO_TRANSMITTANCE_W);
            v = 0.5 / ATMO_TRANSMITTANCE_H + xR * (1.0 - 1.0 / ATMO_TRANSMITTANCE_H);
        }

        // The aerosol's phase function at a cosine: its Mie table, else Cornette-Shanks.
        dm::double3 MiePhaseAt(const AtmosphereParameters& p, double c)
        {
            dm::double3 pm;
            if (p.miePhase.size() == size_t(ATMO_PHASE) * 3)
            {
                const double theta = std::acos(std::clamp(c, -1.0, 1.0));
                const double x = std::sqrt(theta / c_Pi) * (ATMO_PHASE - 1);
                const int k0 = std::min(int(x), ATMO_PHASE - 2);
                const double f = x - k0;
                for (int ch = 0; ch < 3; ++ch)
                    pm[ch] = p.miePhase[size_t(k0) * 3 + size_t(ch)] * (1.0 - f) + p.miePhase[size_t(k0 + 1) * 3 + size_t(ch)] * f;
                return pm;
            }
            for (int ch = 0; ch < 3; ++ch)
            {
                const double g = p.mieG[ch], g2 = g * g;
                pm[ch] = 3.0 / (8.0 * c_Pi) * (1.0 - g2) * (1.0 + c * c) / ((2.0 + g2) * std::pow(std::max(1.0 + g2 - 2.0 * g * c, 1e-4), 1.5));
            }
            return pm;
        }

        // Fibonacci directions on the sphere.
        dm::double3 Sphere(int i, int n)
        {
            const double z = 1.0 - (i + 0.5) * 2.0 / n, r = std::sqrt(std::max(0.0, 1.0 - z * z));
            const double phi = i * 2.399963229728653;
            return dm::double3(r * std::cos(phi), r * std::sin(phi), z);
        }
    }

    namespace atmosphere
    {
        dm::double3 RayleighAt(const std::map<std::string, double>& fractions, double pressure, double temperature,
                               const dm::double3& wavelengths)
        {
            const double number = pressure / (c_Boltzmann * temperature);
            dm::double3 beta(0.0);
            for (const auto& [name, fraction] : fractions)
                for (const Gas& g : c_Gases)
                    if (name == g.name)
                        for (int c = 0; c < 3; ++c)
                            beta[c] += fraction * number * RayleighCrossSection(g, wavelengths[c]);
            return beta;
        }

        dm::double3 Rayleigh(const std::map<std::string, double>& fractions, double pressure, double temperature,
                             std::vector<std::string>* unknown)
        {
            if (unknown)
                for (const auto& [name, fraction] : fractions)
                    if (std::none_of(std::begin(c_Gases), std::end(c_Gases), [&](const Gas& g) { return name == g.name; }))
                        unknown->push_back(name);
            return RayleighAt(fractions, pressure, temperature, c_Wavelengths);
        }

        Aerosol Mie(double a, double v, double realIndex, const dm::double3& imaginary, double tau550, double scaleHeight)
        {
            Aerosol out;
            double cext550, csca550, g550;
            MieAverage(a, v, 550e-9, std::complex<double>(realIndex, imaginary.y), cext550, csca550, g550);
            const double norm = tau550 / (cext550 * scaleHeight);   // particles per m^3 at the ground
            out.phase.assign(size_t(ATMO_PHASE) * 3, 0.f);
            for (int c = 0; c < 3; ++c)
            {
                double cext, csca, g;
                std::vector<double> phase;
                MieAverage(a, v, c_Wavelengths[c], std::complex<double>(realIndex, imaginary[c]), cext, csca, g, &phase);
                out.extinction[c] = cext * norm;
                out.scattering[c] = csca * norm;
                out.g[c] = g;
                for (int k = 0; k < ATMO_PHASE; ++k)
                    out.phase[size_t(k) * 3 + size_t(c)] = float(phase[size_t(k)]);
            }
            return out;
        }

        dm::double3 Ozone(double columnDu, double width)
        {
            // Approximate cross-sections at the channels' wavelengths (Serdyuchenko et al. 2014,
            // 293 K): the Chappuis band, strongest near 600 nm.
            const dm::double3 sigma(4.8e-25, 3.3e-25, 0.7e-25);   // m^2
            const double column = columnDu * 2.6867e20;          // molecules / m^2
            return sigma * (width > 0.0 ? 2.0 * column / width : 0.0);   // the tent's peak density
        }
    }

    PlanetAtmospheres::PlanetAtmospheres(Context& ctx) : m_Ctx(ctx) { }
    PlanetAtmospheres::~PlanetAtmospheres() = default;

    int PlanetAtmospheres::Add(const AtmosphereParameters& p)
    {
        if (int(m_Atmospheres.size()) >= c_Max)
            return -1;
        m_Atmospheres.push_back(Build(p));
        return int(m_Atmospheres.size()) - 1;
    }

    void PlanetAtmospheres::Replace(int slice, const AtmosphereParameters& p)
    {
        m_Atmospheres[size_t(slice)] = Build(p);
    }

    PlanetAtmospheres::Atmosphere PlanetAtmospheres::Build(const AtmosphereParameters& p)
    {
        Atmosphere a;
        a.p = p;

        // Transmittance to the top, by (mu, r).
        a.transmittance.resize(ATMO_TRANSMITTANCE_W * ATMO_TRANSMITTANCE_H * 4);
        const double H = std::sqrt(p.top * p.top - p.bottom * p.bottom);
        for (int j = 0; j < ATMO_TRANSMITTANCE_H; ++j)
            for (int i = 0; i < ATMO_TRANSMITTANCE_W; ++i)
            {
                const double xMu = (i + 0.5 - 0.5) / (ATMO_TRANSMITTANCE_W - 1), xR = (j + 0.5 - 0.5) / (ATMO_TRANSMITTANCE_H - 1);
                const double rho = xR * H, r = std::sqrt(rho * rho + p.bottom * p.bottom);
                const double dMin = p.top - r, dMax = rho + H, d = dMin + xMu * (dMax - dMin);
                const double mu = d <= 0.0 ? 1.0 : std::clamp((H * H - rho * rho - d * d) / (2.0 * r * d), -1.0, 1.0);
                dm::double3 depth(0.0);
                constexpr int steps = 64;
                for (int s = 0; s < steps; ++s)
                    depth += Extinction(p, Height(r, mu, (s + 0.5) * d / steps, p.bottom)) * (d / steps);
                float* t = &a.transmittance[(size_t(j) * ATMO_TRANSMITTANCE_W + i) * 4];
                for (int c = 0; c < 3; ++c)
                    t[c] = float(std::exp(-depth[c]));
                t[3] = 1.f;
            }
        auto transmittance = [&](double r, double mu)
        {
            if (DistanceToGround(p, r, mu) >= 0.0)
                return dm::double3(0.0);
            double u, v;
            TransmittanceUv(p, r, mu, u, v);
            return Sample(a.transmittance, ATMO_TRANSMITTANCE_W, ATMO_TRANSMITTANCE_H, u, v);
        };

        // Multiple scattering (Hillaire 2020, section 5.5), by (mu_s, r): second-order light
        // and the fraction f_ms re-scattered, over every direction with an isotropic phase,
        // summed as the geometric series L2 / (1 - f_ms).
        a.multiScatter.resize(ATMO_MULTISCATTER * ATMO_MULTISCATTER * 4);
        constexpr int dirs = 64, steps = 20;
        for (int j = 0; j < ATMO_MULTISCATTER; ++j)
            for (int i = 0; i < ATMO_MULTISCATTER; ++i)
            {
                const double muS = -1.0 + 2.0 * i / (ATMO_MULTISCATTER - 1);
                const double r = p.bottom + (p.top - p.bottom) * std::clamp(double(j) / (ATMO_MULTISCATTER - 1), 1e-4, 1.0 - 1e-4);
                const dm::double3 sun(std::sqrt(std::max(0.0, 1.0 - muS * muS)), 0.0, muS);
                dm::double3 l2(0.0), fms(0.0);
                for (int k = 0; k < dirs; ++k)
                {
                    const dm::double3 w = Sphere(k, dirs);
                    const double mu = w.z;
                    const double ground = DistanceToGround(p, r, mu);
                    const double tMax = ground >= 0.0 ? ground : DistanceToTop(p, r, mu);
                    dm::double3 tr(1.0), lw(0.0), fw(0.0);
                    const double dt = tMax / steps;
                    for (int s = 0; s < steps; ++s)
                    {
                        const double t = (s + 0.5) * dt;
                        const dm::double3 pos = dm::double3(0.0, 0.0, r) + w * t;
                        const double rr = dm::length(pos);
                        const double ms = dm::dot(pos / rr, sun);
                        const dm::double3 ext = Extinction(p, rr - p.bottom), sca = Scattering(p, rr - p.bottom);
                        const dm::double3 st(std::exp(-ext.x * dt), std::exp(-ext.y * dt), std::exp(-ext.z * dt));
                        const dm::double3 integral = dm::double3((1.0 - st.x) / std::max(ext.x, 1e-12), (1.0 - st.y) / std::max(ext.y, 1e-12),
                                                                 (1.0 - st.z) / std::max(ext.z, 1e-12));
                        lw += tr * sca * transmittance(rr, ms) * (1.0 / (4.0 * c_Pi)) * integral;
                        fw += tr * sca * integral;
                        tr *= st;
                    }
                    if (ground >= 0.0)
                    {
                        const dm::double3 gp = dm::double3(0.0, 0.0, r) + w * ground;
                        const double gm = dm::dot(gp / dm::length(gp), sun);
                        lw += tr * transmittance(p.bottom, gm) * std::max(gm, 0.0) * p.groundAlbedo / c_Pi;
                    }
                    // The mean over the sphere: the isotropic phase's 1 / 4 pi times 4 pi / dirs.
                    l2 += lw * (1.0 / double(dirs));
                    fms += fw * (1.0 / double(dirs));
                }
                float* t = &a.multiScatter[(size_t(j) * ATMO_MULTISCATTER + i) * 4];
                for (int c = 0; c < 3; ++c)
                    t[c] = float(l2[c] / std::max(1.0 - fms[c], 1e-3));
                t[3] = 1.f;
            }
        auto multi = [&](double r, double muS)
        {
            const double u = 0.5 / ATMO_MULTISCATTER + std::clamp(muS * 0.5 + 0.5, 0.0, 1.0) * (1.0 - 1.0 / ATMO_MULTISCATTER);
            const double v = 0.5 / ATMO_MULTISCATTER + std::clamp((r - p.bottom) / (p.top - p.bottom), 0.0, 1.0) * (1.0 - 1.0 / ATMO_MULTISCATTER);
            return Sample(a.multiScatter, ATMO_MULTISCATTER, ATMO_MULTISCATTER, u, v);
        };

        // The sky's irradiance on a level surface, by (mu_s, r): the upper hemisphere's
        // in-scattered light (single from the Sun, multiple from the table), cosine-weighted.
        a.irradiance.resize(ATMO_IRRADIANCE_W * ATMO_IRRADIANCE_H * 4);
        constexpr int rings = 8, segments = 16, isteps = 16;
        for (int j = 0; j < ATMO_IRRADIANCE_H; ++j)
            for (int i = 0; i < ATMO_IRRADIANCE_W; ++i)
            {
                const double muS = -1.0 + 2.0 * i / (ATMO_IRRADIANCE_W - 1);
                const double r = p.bottom + (p.top - p.bottom) * std::clamp(double(j) / (ATMO_IRRADIANCE_H - 1), 1e-4, 1.0 - 1e-4);
                const dm::double3 sun(std::sqrt(std::max(0.0, 1.0 - muS * muS)), 0.0, muS);
                dm::double3 e(0.0);
                for (int ri = 0; ri < rings; ++ri)
                    for (int si = 0; si < segments; ++si)
                    {
                        // Cosine-weighted: E = pi x the mean radiance.
                        const double u1 = (ri + 0.5) / rings, u2 = (si + 0.5) / segments;
                        const double sinT = std::sqrt(u1), cosT = std::sqrt(1.0 - u1), phi = 2.0 * c_Pi * u2;
                        const dm::double3 w(sinT * std::cos(phi), sinT * std::sin(phi), cosT);
                        const double c = dm::dot(w, sun);
                        const double tMax = DistanceToTop(p, r, cosT);
                        dm::double3 tr(1.0), lw(0.0);
                        const double dt = tMax / isteps;
                        for (int s = 0; s < isteps; ++s)
                        {
                            const double t = (s + 0.5) * dt;
                            const dm::double3 pos = dm::double3(0.0, 0.0, r) + w * t;
                            const double rr = dm::length(pos);
                            const double ms = dm::dot(pos / rr, sun);
                            dm::double3 ray, mie;
                            const dm::double3 sca = Scattering(p, rr - p.bottom, &ray, &mie);
                            const dm::double3 ext = Extinction(p, rr - p.bottom);
                            const double pr = 3.0 / (16.0 * c_Pi) * (1.0 + c * c);
                            const dm::double3 pm = MiePhaseAt(p, c);
                            const dm::double3 sc = (ray * pr + mie * pm) * transmittance(rr, ms) + sca * multi(rr, ms);
                            const dm::double3 st(std::exp(-ext.x * dt), std::exp(-ext.y * dt), std::exp(-ext.z * dt));
                            lw += tr * sc * dm::double3((1.0 - st.x) / std::max(ext.x, 1e-12), (1.0 - st.y) / std::max(ext.y, 1e-12),
                                                        (1.0 - st.z) / std::max(ext.z, 1e-12));
                            tr *= st;
                        }
                        e += lw * (c_Pi / (rings * segments));
                    }
                float* t = &a.irradiance[(size_t(j) * ATMO_IRRADIANCE_W + i) * 4];
                for (int c = 0; c < 3; ++c)
                    t[c] = float(e[c]);
                t[3] = 1.f;
            }
        return a;
    }

    dm::double3 PlanetAtmospheres::MultiScatter(int slice, double r, double muS) const
    {
        const AtmosphereParameters& p = m_Atmospheres[size_t(slice)].p;
        const double u = 0.5 / ATMO_MULTISCATTER + std::clamp(muS * 0.5 + 0.5, 0.0, 1.0) * (1.0 - 1.0 / ATMO_MULTISCATTER);
        const double v = 0.5 / ATMO_MULTISCATTER + std::clamp((r - p.bottom) / (p.top - p.bottom), 0.0, 1.0) * (1.0 - 1.0 / ATMO_MULTISCATTER);
        return Sample(m_Atmospheres[size_t(slice)].multiScatter, ATMO_MULTISCATTER, ATMO_MULTISCATTER, u, v);
    }

    dm::double3 PlanetAtmospheres::Radiance(int slice, const dm::double3& origin, const dm::double3& dir,
                                            const dm::double3& towardsSun, dm::double3* transmittance, double* ground) const
    {
        const AtmosphereParameters& p = m_Atmospheres[size_t(slice)].p;
        if (transmittance)
            *transmittance = dm::double3(1.0);
        if (ground)
            *ground = -1.0;
        // The span inside the shell, and the ground (planet_atmosphere.hlsli's AtmosphereSpan, GroundDistance).
        const double b = dm::dot(origin, dir);
        const dm::double3 q = origin - dir * b;
        const double q2 = dm::dot(q, q);
        if (q2 >= p.top * p.top)
            return dm::double3(0.0);
        const double h = std::sqrt(p.top * p.top - q2);
        const double tEnter = std::max(-b - h, 0.0);
        double tExit = -b + h;
        if (q2 < p.bottom * p.bottom)
        {
            const double tg = -b - std::sqrt(p.bottom * p.bottom - q2);
            if (tg > 0.0)
            {
                tExit = std::min(tExit, tg);
                if (ground)
                    *ground = tg;
            }
        }
        if (tExit <= tEnter)
            return dm::double3(0.0);
        constexpr int steps = 64;
        const double dt = (tExit - tEnter) / steps;
        const double c = dm::dot(dir, towardsSun);
        const double pr = 3.0 / (16.0 * c_Pi) * (1.0 + c * c);
        const dm::double3 pm = MiePhaseAt(p, c);
        dm::double3 tr(1.0), light(0.0);
        for (int s = 0; s < steps; ++s)
        {
            const dm::double3 pos = origin + dir * (tEnter + (s + 0.5) * dt);
            const double r = dm::length(pos);
            const double muS = dm::dot(pos / r, towardsSun);
            dm::double3 ray, mie;
            const dm::double3 sca = Scattering(p, r - p.bottom, &ray, &mie);
            const dm::double3 ext = Extinction(p, r - p.bottom);
            const dm::double3 sc = (ray * pr + mie * pm) * Transmittance(slice, r, muS) + sca * MultiScatter(slice, r, muS);
            const dm::double3 st(std::exp(-ext.x * dt), std::exp(-ext.y * dt), std::exp(-ext.z * dt));
            light += tr * sc * dm::double3((1.0 - st.x) / std::max(ext.x, 1e-12), (1.0 - st.y) / std::max(ext.y, 1e-12),
                                           (1.0 - st.z) / std::max(ext.z, 1e-12));
            tr *= st;
        }
        if (transmittance)
            *transmittance = tr;
        return light;
    }

    dm::double3 PlanetAtmospheres::GeometricAlbedo(int slice, const dm::double3& groundAlbedo, bool regolith) const
    {
        // The disk at full phase, from far away: parallel rays at impact parameters spread
        // evenly in area out to the top of the air, the Sun behind the viewer.
        const AtmosphereParameters& p = m_Atmospheres[size_t(slice)].p;
        const dm::double3 dir(0.0, 0.0, -1.0), sun(0.0, 0.0, 1.0);
        constexpr int rings = 256;
        dm::double3 sum(0.0);
        for (int i = 0; i < rings; ++i)
        {
            const double b = p.top * std::sqrt((i + 0.5) / rings);
            const dm::double3 origin(b, 0.0, 2.0 * p.top);
            dm::double3 t;
            double ground;
            dm::double3 l = Radiance(slice, origin, dir, sun, &t, &ground);
            if (ground > 0.0)
            {
                // The ground, lit by the Sun through the air and by the sky.
                const dm::double3 n = dm::normalize(origin + dir * ground);
                const double mu0 = dm::dot(n, sun), mu = std::max(-dm::dot(n, dir), 1e-3);
                const double direct = regolith ? (2.0 / 1.5) * mu0 / (mu0 + mu) : mu0;   // x albedo / pi
                l += t * groundAlbedo / c_Pi * (Transmittance(slice, p.bottom, mu0) * direct + Irradiance(slice, p.bottom, mu0));
            }
            sum += l;
        }
        // p = pi x the mean radiance over the solid disk (radius `bottom`), per lux.
        return sum * (c_Pi * p.top * p.top / (rings * p.bottom * p.bottom));
    }

    dm::double3 PlanetAtmospheres::FitGroundAlbedo(int slice, const dm::double3& geometricAlbedo, bool regolith)
    {
        // Linear in the ground's albedo for fixed tables; the tables' multiple scattering
        // depends on it a little, so twice round.
        AtmosphereParameters p = m_Atmospheres[size_t(slice)].p;
        for (int pass = 0; pass < 3; ++pass)
        {
            const dm::double3 air = GeometricAlbedo(slice, dm::double3(0.0), regolith);
            const dm::double3 slope = GeometricAlbedo(slice, dm::double3(1.0), regolith) - air;
            for (int c = 0; c < 3; ++c)
                p.groundAlbedo[c] = std::clamp((geometricAlbedo[c] - air[c]) / std::max(slope[c], 1e-6), 0.0, 1.0);
            Replace(slice, p);
        }
        return p.groundAlbedo;
    }

    dm::double3 PlanetAtmospheres::Transmittance(int slice, double r, double mu) const
    {
        const AtmosphereParameters& p = m_Atmospheres[size_t(slice)].p;
        if (r >= p.top)
        {
            // Outside: from where the ray enters (if it does at all).
            const double disc = r * r * (mu * mu - 1.0) + p.top * p.top;
            if (mu >= 0.0 || disc < 0.0)
                return dm::double3(1.0);
            const double t = -r * mu - std::sqrt(disc);
            const dm::double3 entry = dm::double3(std::sqrt(std::max(0.0, 1.0 - mu * mu)) * t, 0.0, r + mu * t);
            const double re = dm::length(entry);
            return Transmittance(slice, re * (1.0 - 1e-9), dm::dot(entry / re, dm::double3(std::sqrt(std::max(0.0, 1.0 - mu * mu)), 0.0, mu)));
        }
        if (DistanceToGround(p, r, mu) >= 0.0)
            return dm::double3(0.0);
        double u, v;
        TransmittanceUv(p, std::max(r, p.bottom), mu, u, v);
        return Sample(m_Atmospheres[size_t(slice)].transmittance, ATMO_TRANSMITTANCE_W, ATMO_TRANSMITTANCE_H, u, v);
    }

    dm::double3 PlanetAtmospheres::Irradiance(int slice, double r, double muS) const
    {
        const AtmosphereParameters& p = m_Atmospheres[size_t(slice)].p;
        const double u = 0.5 / ATMO_IRRADIANCE_W + std::clamp(muS * 0.5 + 0.5, 0.0, 1.0) * (1.0 - 1.0 / ATMO_IRRADIANCE_W);
        const double v = 0.5 / ATMO_IRRADIANCE_H + std::clamp((r - p.bottom) / (p.top - p.bottom), 0.0, 1.0) * (1.0 - 1.0 / ATMO_IRRADIANCE_H);
        return Sample(m_Atmospheres[size_t(slice)].irradiance, ATMO_IRRADIANCE_W, ATMO_IRRADIANCE_H, u, v);
    }

    void PlanetAtmospheres::CreateResources()
    {
        nvrhi::IDevice* device = m_Ctx.device;
        auto array = [&](const char* name, uint32_t w, uint32_t h)
        {
            return device->createTexture(nvrhi::TextureDesc().setDebugName(name)
                .setDimension(nvrhi::TextureDimension::Texture2DArray).setWidth(w).setHeight(h).setArraySize(c_Max)
                .setFormat(nvrhi::Format::RGBA32_FLOAT)
                .setInitialState(nvrhi::ResourceStates::ShaderResource).setKeepInitialState(true));
        };
        m_Transmittance = array("hyper/atmospheres-transmittance", ATMO_TRANSMITTANCE_W, ATMO_TRANSMITTANCE_H);
        m_MultiScatter = array("hyper/atmospheres-multiscatter", ATMO_MULTISCATTER, ATMO_MULTISCATTER);
        m_Irradiance = array("hyper/atmospheres-irradiance", ATMO_IRRADIANCE_W, ATMO_IRRADIANCE_H);
        m_Phase = array("hyper/atmospheres-phase", ATMO_PHASE, 1);
        m_Sampler = device->createSampler(nvrhi::SamplerDesc().setAllFilters(true).setAllAddressModes(nvrhi::SamplerAddressMode::Clamp));
        m_Constants = device->createBuffer(nvrhi::utils::CreateVolatileConstantBufferDesc(
            sizeof(PlanarViewConstants) + 16 + sizeof(PlanetAtmosphereConstants) * c_Max, "hyper/atmospheres", 4));
        m_Layout = device->createBindingLayout(nvrhi::BindingLayoutDesc().setVisibility(nvrhi::ShaderType::Compute)
            .addItem(nvrhi::BindingLayoutItem::VolatileConstantBuffer(0))
            .addItem(nvrhi::BindingLayoutItem::Texture_SRV(0))
            .addItem(nvrhi::BindingLayoutItem::Texture_SRV(1))
            .addItem(nvrhi::BindingLayoutItem::Texture_SRV(2))
            .addItem(nvrhi::BindingLayoutItem::Texture_SRV(3))
            .addItem(nvrhi::BindingLayoutItem::Sampler(0))
            .addItem(nvrhi::BindingLayoutItem::Texture_UAV(0)));
    }

    void PlanetAtmospheres::CreatePipelines()
    {
        if (!m_Used)
            return;   // built on first use
        auto shader = m_Ctx.shaders->CreateShader("hyper/atmospheres/composite_cs", "main", nullptr, nvrhi::ShaderType::Compute);
        if (shader)
            m_Composite = m_Ctx.device->createComputePipeline(nvrhi::ComputePipelineDesc().setComputeShader(shader).addBindingLayout(m_Layout));
    }

    void PlanetAtmospheres::Prepare(nvrhi::ICommandList* cmd)
    {
        if (!m_Used)
        {
            m_Used = true;
            CreateResources();
            CreatePipelines();
        }
        for (size_t i = 0; i < m_Atmospheres.size(); ++i)
        {
            Atmosphere& a = m_Atmospheres[i];
            if (a.uploaded)
                continue;
            cmd->writeTexture(m_Transmittance, uint32_t(i), 0, a.transmittance.data(), ATMO_TRANSMITTANCE_W * 16);
            cmd->writeTexture(m_MultiScatter, uint32_t(i), 0, a.multiScatter.data(), ATMO_MULTISCATTER * 16);
            cmd->writeTexture(m_Irradiance, uint32_t(i), 0, a.irradiance.data(), ATMO_IRRADIANCE_W * 16);
            std::vector<float> phase(size_t(ATMO_PHASE) * 4, 0.f);
            if (a.p.miePhase.size() == size_t(ATMO_PHASE) * 3)
                for (int k = 0; k < ATMO_PHASE; ++k)
                    for (int c = 0; c < 3; ++c)
                        phase[size_t(k) * 4 + size_t(c)] = a.p.miePhase[size_t(k) * 3 + size_t(c)];
            cmd->writeTexture(m_Phase, uint32_t(i), 0, phase.data(), ATMO_PHASE * 16);
            a.uploaded = true;
        }
    }

    void PlanetAtmospheres::Render(nvrhi::ICommandList* cmd, const donut::engine::PlanarView& view, nvrhi::ITexture* hdr,
                                   nvrhi::ITexture* depth, const void* constants, uint32_t count)
    {
        if (!m_Used || !m_Composite || count == 0)
            return;
        auto scope = m_Ctx.profiler->Measure(cmd, "atmospheres");
        struct
        {
            PlanarViewConstants view;
            uint32_t count, pad[3];
            PlanetAtmosphereConstants atmospheres[c_Max];
        } c;
        std::memset(&c, 0, sizeof(c));
        view.FillPlanarViewConstants(c.view);
        c.count = std::min<uint32_t>(count, c_Max);
        std::memcpy(c.atmospheres, constants, sizeof(PlanetAtmosphereConstants) * c.count);
        cmd->writeBuffer(m_Constants, &c, sizeof(c));
        nvrhi::IBindingSet* set = m_Ctx.bindingCache->GetOrCreateBindingSet(nvrhi::BindingSetDesc()
            .addItem(nvrhi::BindingSetItem::ConstantBuffer(0, m_Constants))
            .addItem(nvrhi::BindingSetItem::Texture_SRV(0, depth))
            .addItem(nvrhi::BindingSetItem::Texture_SRV(1, m_Transmittance))
            .addItem(nvrhi::BindingSetItem::Texture_SRV(2, m_MultiScatter))
            .addItem(nvrhi::BindingSetItem::Texture_SRV(3, m_Phase))
            .addItem(nvrhi::BindingSetItem::Sampler(0, m_Sampler))
            .addItem(nvrhi::BindingSetItem::Texture_UAV(0, hdr)), m_Layout);
        nvrhi::ComputeState state;
        state.pipeline = m_Composite;
        state.bindings = { set };
        cmd->setComputeState(state);
        const auto& vp = view.GetViewportState().viewports[0];
        cmd->dispatch((uint32_t(vp.width()) + 7) / 8, (uint32_t(vp.height()) + 7) / 8);
    }
}
