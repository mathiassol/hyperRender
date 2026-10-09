// Reference path tracer: ground truth for the renderer's lighting
// model, to check the real-time passes against. Per pixel and sample, a camera
// ray through a random point of the pixel, then a path with the full material
// BSDF: Lambert diffuse + GGX specular with height-correlated Smith, as Donut's
// lighting shades them. Light comes from the sun (a disc of its angular size,
// sampled directly and weighted against BSDF samples with the power heuristic),
// the environment cube for rays that escape (captured without the sun disc),
// and emissive surfaces hit by BSDF samples. Russian roulette after 3 bounces.
// Hits use normal maps and the finest texture mips.
//
// u_Accumulation holds the running mean over sampleIndex earlier samples.

#include <hyper/rt_lighting.hlsli>
#include <hyper/sampling.hlsli>
#include <hyper/view.hlsli>

struct ReferenceConstants
{
    PlanarViewConstants view;       // without TAA jitter
    uint2 size;
    uint sampleIndex;               // samples accumulated before this dispatch
    uint samplesPerFrame;
    uint maxBounces;                // indirect bounces; 0 = direct light only
    float clampLuminance;           // per-sample firefly clamp; 0 = off (unbiased)
    float sunCosMax;                // cos of the sun's angular radius
    float sunPdf;                   // 1 / solid angle of the sun
};

ConstantBuffer<ReferenceConstants> g_Ref : register(b0);
VK_IMAGE_FORMAT("rgba32f") RWTexture2D<float4> u_Accumulation : register(u0);

float Luminance(float3 c)
{
    return dot(c, float3(0.2126, 0.7152, 0.0722));
}

// Chance of sampling the specular lobe: its Fresnel-weighted share at this view.
float SpecularProbability(HitSurface s, float nv)
{
    const float3 f = s.specularF0 + (1.0 - s.specularF0) * pow(1.0 - saturate(nv), 5.0);
    const float spec = Luminance(f), diff = Luminance(s.diffuseAlbedo);
    return clamp(spec / max(spec + diff, 1e-4), 0.1, 0.9);
}

// BSDF value (no cosine) for v -> l, and the pdf of SampleBsdf producing l.
float3 EvalBsdf(HitSurface s, float3 v, float3 l, float pSpec, out float pdf)
{
    pdf = 0.0;
    const float3 n = s.normal;
    const float nl = dot(n, l), nv = dot(n, v);
    if (nl <= 0.0 || nv <= 0.0)
        return 0.0;
    const float3 h = normalize(v + l);
    const float nh = saturate(dot(n, h)), vh = saturate(dot(v, h));
    const float alpha = max(s.roughness * s.roughness, 1e-3);
    const float a2 = alpha * alpha;

    const float d = a2 / (HYPER_PI * pow(nh * nh * (a2 - 1.0) + 1.0, 2.0));
    const float lambdaV = nl * sqrt(a2 + (1.0 - a2) * nv * nv);
    const float lambdaL = nv * sqrt(a2 + (1.0 - a2) * nl * nl);
    const float g2 = 2.0 * nl * nv / (lambdaV + lambdaL);
    const float3 f = s.specularF0 + (1.0 - s.specularF0) * pow(1.0 - vh, 5.0);
    const float3 specular = f * (d * g2 / (4.0 * nl * nv));

    const float g1v = 2.0 * nv / (nv + sqrt(a2 + (1.0 - a2) * nv * nv));
    const float pdfSpecular = g1v * d / (4.0 * nv);     // VNDF: D_v(h) / (4 v.h)
    const float pdfDiffuse = nl / HYPER_PI;
    pdf = pSpec * pdfSpecular + (1.0 - pSpec) * pdfDiffuse;
    return s.diffuseAlbedo / HYPER_PI + specular;
}

float3 SampleBsdf(HitSurface s, float3 v, float pSpec, inout Rng rng)
{
    if (rng.Next() < pSpec)
    {
        const float alpha = max(s.roughness * s.roughness, 1e-3);
        return reflect(-v, SampleGgxHalfVector(s.normal, v, alpha, float2(rng.Next(), rng.Next())));
    }
    return SampleCosine(s.normal, rng);
}

float PowerHeuristic(float a, float b)
{
    return a * a / max(a * a + b * b, 1e-20);
}

float3 TracePath(Ray ray, inout Rng rng)
{
    const float3 sunRadiance = g_Gi.sunRadiance * g_Ref.sunPdf;   // irradiance / solid angle
    float3 radiance = 0.0;
    float3 throughput = 1.0;
    float3 origin = ray.origin;
    float3 dir = ray.direction;
    float lastPdf = 0.0;   // pdf of the BSDF sample that made dir; 0 for the camera ray
    uint passes = 0;

    [loop]
    for (uint bounce = 0; bounce <= g_Ref.maxBounces; ++bounce)
    {
        const RtHit hit = TraceClosest(origin, dir, 1e7);
        if (!hit.hit)
        {
            radiance += throughput * MissRadiance(dir, 0.0);
            if (dot(dir, g_Gi.sunDir) >= g_Ref.sunCosMax)
                radiance += throughput * sunRadiance * (lastPdf > 0.0 ? PowerHeuristic(lastPdf, g_Ref.sunPdf) : 1.0);
            break;
        }
        const HitSurface s = GetHitSurface(hit, dir, bounce == 0 ? 0.0 : 1.0, true);
        if (!s.frontFace)
        {
            // Camera rays pass through the backs of single-sided surfaces, as the
            // raster passes cull them (a camera inside a column still sees the room).
            // Later segments stop there: closed geometry is dark inside.
            if (bounce > 0 || ++passes > 8)
                break;
            origin = OffsetRayOrigin(s.position, -s.geometricNormal);
            --bounce;
            continue;
        }
        radiance += throughput * s.emissive;

        const float3 v = -dir;
        const float pSpec = SpecularProbability(s, dot(s.normal, v));
        const float3 next = OffsetRayOrigin(s.position, s.geometricNormal);
        const bool last = bounce == g_Ref.maxBounces;

        // The sun, sampled directly. On the last vertex no BSDF sample can find it, so no MIS.
        if (any(g_Gi.sunRadiance > 0.0))
        {
            const float3 l = SampleCone(g_Gi.sunDir, g_Ref.sunCosMax, float2(rng.Next(), rng.Next()));
            float pdf;
            const float3 f = EvalBsdf(s, v, l, pSpec, pdf);
            if (pdf > 0.0 && dot(s.geometricNormal, l) > 0.0 && TraceVisible(next, l, 1e7))
                radiance += throughput * f * dot(s.normal, l) * g_Gi.sunRadiance * (last ? 1.0 : PowerHeuristic(g_Ref.sunPdf, pdf));
        }
        if (last)
            break;

        const float3 l = SampleBsdf(s, v, pSpec, rng);
        float pdf;
        const float3 f = EvalBsdf(s, v, l, pSpec, pdf);
        if (pdf <= 0.0 || dot(s.geometricNormal, l) <= 0.0)
            break;
        throughput *= f * dot(s.normal, l) / pdf;
        lastPdf = pdf;

        if (bounce >= 3)
        {
            const float survive = min(max(throughput.x, max(throughput.y, throughput.z)), 0.95);
            if (rng.Next() >= survive)
                break;
            throughput /= survive;
        }
        origin = next;
        dir = l;
    }
    return radiance;
}

[numthreads(8, 8, 1)]
void main(uint2 id : SV_DispatchThreadID)
{
    if (any(id >= g_Ref.size))
        return;

    float3 sum = 0.0;
    for (uint i = 0; i < g_Ref.samplesPerFrame; ++i)
    {
        Rng rng = MakeRngHashed(id, g_Ref.sampleIndex + i);
        const float2 jitter = float2(rng.Next(), rng.Next());
        float3 l = TracePath(CameraRay(g_Ref.view, float2(id) + jitter - 0.5), rng);
        if (any(isnan(l)) || any(isinf(l)))
            l = 0.0;
        if (g_Ref.clampLuminance > 0.0)
            l *= min(1.0, g_Ref.clampLuminance / max(Luminance(l), 1e-6));
        sum += l;
    }

    const float n = float(g_Ref.sampleIndex);
    const float3 previous = g_Ref.sampleIndex > 0 ? u_Accumulation[id].rgb : 0.0;
    u_Accumulation[id] = float4((previous * n + sum) / (n + float(g_Ref.samplesPerFrame)), 1.0);
}
