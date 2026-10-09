// Screen-space reflections, three passes:
//   trace      one GGX-importance-sampled ray per pixel, marched through the Hi-Z
//              pyramid (reverse-Z, nearest depth in .x); hit -> lit colour + confidence
//   temporal   reprojected accumulation with neighbourhood clamping (resolves the
//              sampling noise the way TAA does)
//   composite  after lighting: blends from the IBL probe's specular towards SSR by
//              confidence, so reflections aren't counted twice (Donut's deferred
//              pass already added the probe's specular)

#include <hyper/common.hlsli>
#include <hyper/color.hlsli>
#include <donut/shaders/view_cb.h>

struct SsrConstants
{
    PlanarViewConstants view;
    float2 size;
    float2 invSize;
    uint hizMips;
    uint maxSteps;
    uint frameIndex;
    uint hasProbe;
    float maxRoughness;
    float thickness;
    float intensity;
    float historyWeight;
    float probeMips;
    float specularScale;
    float zNear;
    uint historyValid;
};

ConstantBuffer<SsrConstants> g_Ssr : register(b0);
Texture2D<float4> t0_ : register(t0);
Texture2D<float4> t1_ : register(t1);
Texture2D<float4> t2_ : register(t2);
Texture2D<float4> t3_ : register(t3);
TextureCubeArray<float4> t_ProbeSpecular : register(t4);
Texture2D<float4> t_SpecularOcclusion : register(t5);   // .a; white when there is none
SamplerState s_Linear : register(s0);
SamplerState s_Point : register(s1);
VK_IMAGE_FORMAT("rgba16f") RWTexture2D<float4> u_Out : register(u0);

// ------------------------------------------------------------------ helpers

float Hash(uint3 v)
{
    v = v * 1664525u + 1013904223u;
    v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
    v ^= v >> 16u;
    v.x += v.y * v.z;
    return float(v.x) / 4294967295.0;
}

float3 WorldPosition(float2 pixel, float depth)
{
    float2 clip = pixel * g_Ssr.view.windowToClipScale + g_Ssr.view.windowToClipBias;
    float4 p = mul(float4(clip, depth, 1.0), g_Ssr.view.matClipToWorld);
    return p.xyz / p.w;
}

// Heitz 2018, sampling the distribution of visible normals. v is in tangent space (z = normal).
float3 SampleGgxVndf(float3 v, float alpha, float2 u)
{
    float3 vh = normalize(float3(alpha * v.x, alpha * v.y, v.z));
    float lensq = vh.x * vh.x + vh.y * vh.y;
    float3 t1 = lensq > 0.0 ? float3(-vh.y, vh.x, 0.0) * rsqrt(lensq) : float3(1.0, 0.0, 0.0);
    float3 t2 = cross(vh, t1);
    float r = sqrt(u.x);
    float phi = HYPER_TWO_PI * u.y;
    float p1 = r * cos(phi);
    float p2 = r * sin(phi);
    float s = 0.5 * (1.0 + vh.z);
    p2 = (1.0 - s) * sqrt(1.0 - p1 * p1) + s * p2;
    float3 nh = p1 * t1 + p2 * t2 + sqrt(max(0.0, 1.0 - p1 * p1 - p2 * p2)) * vh;
    return normalize(float3(alpha * nh.x, alpha * nh.y, max(0.0, nh.z)));
}

// Karis' analytic fit of the split-sum environment BRDF.
float2 EnvBrdfApprox(float roughness, float nDotV)
{
    const float4 c0 = float4(-1.0, -0.0275, -0.572, 0.022);
    const float4 c1 = float4(1.0, 0.0425, 1.04, -0.04);
    float4 r = roughness * c0 + c1;
    float a004 = min(r.x * r.x, exp2(-9.28 * nDotV)) * r.x + r.y;
    return float2(-1.04, 1.04) * a004 + r.zw;
}

// ------------------------------------------------------------------ trace
// t0 depth, t1 normals (xyz) + roughness (w), t2 Hi-Z (RG), t3 lit HDR colour

bool TraceHiZ(float3 origin, float3 dir, out float2 hitUv)
{
    hitUv = 0;
    const uint maxLevel = min(g_Ssr.hizMips - 1, 7u);
    const float pixelLength = length(dir.xy * g_Ssr.size);
    if (pixelLength < 1e-3)
        return false;
    const float eps = 0.05 / pixelLength;     // ~1/20 pixel, to step over cell edges
    float t = 1.5 / pixelLength;              // start 1.5 pixels out, past the surface itself
    uint level = 0;

    [loop]
    for (uint i = 0; i < g_Ssr.maxSteps; ++i)
    {
        float3 p = origin + dir * t;
        if (t > 1.0 || any(p.xy < 0.0) || any(p.xy > 1.0) || p.z <= 0.0)
            return false;

        const float2 mipSize = max(floor(g_Ssr.size / float(1u << level)), 1.0);
        const float2 cell = floor(p.xy * mipSize);
        const float cellNearest = t2_.Load(int3(cell, level)).x;

        // Where the ray leaves this cell.
        const float2 bound = (cell + select(dir.xy > 0.0, 1.0, 0.0)) / mipSize;
        const float2 tBound = select(abs(dir.xy) > 1e-8, (bound - origin.xy) / dir.xy, 1e9);
        const float tExit = min(tBound.x, tBound.y);

        if (p.z > cellNearest)
        {
            // In front of everything in the cell: step to where the ray reaches that depth, or out.
            const float tDepth = dir.z < 0.0 ? (cellNearest - origin.z) / dir.z : 1e9;
            if (tDepth < tExit)
            {
                t = max(tDepth, t);
                if (level == 0)
                {
                    hitUv = (origin + dir * t).xy;
                    return true;
                }
                --level;
            }
            else
            {
                t = tExit + eps;
                level = min(level + 1, maxLevel);
            }
        }
        else if (level > 0)
        {
            --level;   // behind the nearest surface of a coarse cell: look closer
        }
        else
        {
            // Behind the surface at full resolution: a hit if within thickness, else we went behind a thin object.
            const float sceneLinear = g_Ssr.zNear / max(cellNearest, 1e-7);
            const float rayLinear = g_Ssr.zNear / max(p.z, 1e-7);
            if (rayLinear - sceneLinear < g_Ssr.thickness * (1.0 + 0.05 * sceneLinear))
            {
                hitUv = p.xy;
                return true;
            }
            t = tExit + eps;
        }
    }
    return false;
}

[numthreads(8, 8, 1)]
void trace(uint2 id : SV_DispatchThreadID)
{
    if (any(float2(id) >= g_Ssr.size))
        return;

    const float depth = t0_[id].x;
    const float4 nr = t1_[id];
    const float roughness = nr.w;
    if (depth <= 0.0 || roughness > g_Ssr.maxRoughness)
    {
        u_Out[id] = 0;
        return;
    }

    const float2 pixel = float2(id) + 0.5;
    const float3 position = WorldPosition(pixel, depth);
    const float3 n = normalize(nr.xyz);
    const float3 incident = normalize(position - g_Ssr.view.cameraDirectionOrPosition.xyz);

    // Sample a microfacet normal, reflect about it.
    const float3 up = abs(n.y) < 0.999 ? float3(0, 1, 0) : float3(1, 0, 0);
    const float3 tx = normalize(cross(up, n));
    const float3 ty = cross(n, tx);
    const float3 toEye = -incident;
    const float3 vLocal = float3(dot(toEye, tx), dot(toEye, ty), dot(toEye, n));
    const float2 xi = float2(Hash(uint3(id, g_Ssr.frameIndex)), Hash(uint3(id.yx, g_Ssr.frameIndex + 7919u)));
    const float alpha = max(roughness * roughness, 1e-3);
    const float3 hLocal = SampleGgxVndf(vLocal, alpha, xi);
    const float3 h = normalize(hLocal.x * tx + hLocal.y * ty + hLocal.z * n);
    float3 r = reflect(incident, h);
    if (dot(r, n) <= 0.0)
        r = reflect(incident, n);

    // Ray end point, kept in front of the camera.
    const float3 start = position + n * 0.02;
    float rayLength = 200.0;
    const float3 viewStart = mul(float4(start, 1.0), g_Ssr.view.matWorldToView).xyz;
    const float3 viewDir = mul(float4(r, 0.0), g_Ssr.view.matWorldToView).xyz;
    if (viewDir.z < 0.0)
        rayLength = min(rayLength, (viewStart.z - g_Ssr.zNear * 2.0) / -viewDir.z);
    const float3 end = start + r * max(rayLength, 0.01);

    float4 c0 = mul(float4(start, 1.0), g_Ssr.view.matWorldToClip);
    float4 c1 = mul(float4(end, 1.0), g_Ssr.view.matWorldToClip);
    c0.xyz /= c0.w;
    c1.xyz /= c1.w;
    const float3 o = float3((c0.xy * g_Ssr.view.clipToWindowScale + g_Ssr.view.clipToWindowBias) * g_Ssr.invSize, c0.z);
    const float3 e = float3((c1.xy * g_Ssr.view.clipToWindowScale + g_Ssr.view.clipToWindowBias) * g_Ssr.invSize, c1.z);

    float2 hitUv;
    float4 result = 0;
    if (TraceHiZ(o, e - o, hitUv))
    {
        const float3 colour = t3_.SampleLevel(s_Linear, hitUv, 0).rgb;
        const float2 edge = abs(hitUv * 2.0 - 1.0);
        float confidence = 1.0 - smoothstep(0.8, 1.0, max(edge.x, edge.y));
        confidence *= 1.0 - smoothstep(g_Ssr.maxRoughness * 0.6, g_Ssr.maxRoughness, roughness);
        confidence *= saturate(1.0 - dot(r, toEye) * 1.5);   // rays heading back at the camera see back faces
        result = float4(min(colour, 1000.0), confidence);
    }
    u_Out[id] = result;
}

// ------------------------------------------------------------------ temporal
// t0 this frame's trace, t1 history, t2 motion vectors, t3 depth

[numthreads(8, 8, 1)]
void temporal(uint2 id : SV_DispatchThreadID)
{
    if (any(float2(id) >= g_Ssr.size))
        return;

    const float4 current = t0_[id];
    if (t3_[id].x <= 0.0)
    {
        u_Out[id] = 0;
        return;
    }

    float4 lo = current, hi = current;
    [unroll]
    for (int y = -1; y <= 1; ++y)
    [unroll]
    for (int x = -1; x <= 1; ++x)
    {
        const float4 s = t0_[clamp(int2(id) + int2(x, y), int2(0, 0), int2(g_Ssr.size) - 1)];
        lo = min(lo, s);
        hi = max(hi, s);
    }

    const float2 prevUv = (float2(id) + 0.5 + t2_[id].xy) * g_Ssr.invSize;
    const bool valid = g_Ssr.historyValid != 0 && all(prevUv >= 0.0) && all(prevUv <= 1.0);
    const float4 history = clamp(t1_.SampleLevel(s_Linear, prevUv, 0), lo, hi);
    u_Out[id] = valid ? lerp(history, current, g_Ssr.historyWeight) : current;
}

// ------------------------------------------------------------------ composite
// t0 resolved reflections, t1 depth, t2 normals + roughness, t3 specular F0 + occlusion; u0 lit HDR

[numthreads(8, 8, 1)]
void composite(uint2 id : SV_DispatchThreadID)
{
    if (any(float2(id) >= g_Ssr.size))
        return;

    const float4 reflection = t0_[id];
    const float depth = t1_[id].x;
    if (depth <= 0.0 || reflection.a <= 0.0)
        return;

    const float4 nr = t2_[id];
    const float4 specular = t3_[id];
    const float3 n = normalize(nr.xyz);
    const float3 position = WorldPosition(float2(id) + 0.5, depth);
    const float3 incident = normalize(position - g_Ssr.view.cameraDirectionOrPosition.xyz);
    const float nDotV = saturate(-dot(n, incident));

    const float2 ab = EnvBrdfApprox(nr.w, nDotV);
    const float3 brdf = specular.rgb * ab.x + ab.y;

    // What the deferred pass already added for this reflection: the probe's specular.
    float3 probe = 0;
    if (g_Ssr.hasProbe != 0)
    {
        const float3 r = reflect(incident, n);
        probe = t_ProbeSpecular.SampleLevel(s_Linear, float4(r, 0.0), sqrt(saturate(nr.w)) * (g_Ssr.probeMips - 1.0)).rgb
              * g_Ssr.specularScale * t_SpecularOcclusion[id].a;
    }

    const float weight = reflection.a * g_Ssr.intensity;
    const float4 lit = u_Out[id];
    u_Out[id] = float4(max(lit.rgb + weight * (reflection.rgb - probe) * brdf * specular.a, 0.0), lit.a);
}
