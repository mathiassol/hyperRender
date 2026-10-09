// A sphere on a checkerboard under a moving sun, traced analytically per pixel.
// Writes linear HDR colour and reverse-Z depth for hyper's post stack.
// Edit and save while the app runs: it recompiles and reloads.

#include <hyper/view.hlsli>

struct Constants
{
    float time;
    float sunSpeed;
    float2 pad;
};

ConstantBuffer<PlanarViewConstants> g_View : register(b0);
DECLARE_PUSH_CONSTANTS(Constants, g_Const, 1, 0);
VK_IMAGE_FORMAT("rgba16f") RWTexture2D<float4> u_Color : register(u0);
VK_IMAGE_FORMAT("r32f") RWTexture2D<float> u_Depth : register(u1);

static const float3 c_SphereCenter = float3(0.0, 1.0, 0.0);
static const float c_SphereRadius = 1.0;

float IntersectSphere(Ray ray)
{
    float3 oc = ray.origin - c_SphereCenter;
    float b = dot(oc, ray.direction);
    float c = dot(oc, oc) - c_SphereRadius * c_SphereRadius;
    float h = b * b - c;
    if (h < 0.0)
        return -1.0;
    float t = -b - sqrt(h);
    return t > 0.0 ? t : -1.0;
}

float IntersectGround(Ray ray)
{
    return ray.direction.y < 0.0 ? -ray.origin.y / ray.direction.y : -1.0;
}

float3 Sky(float3 dir, float3 sunDir)
{
    float3 sky = lerp(float3(0.55, 0.65, 0.8), float3(0.15, 0.3, 0.65), saturate(dir.y));
    float sun = pow(saturate(dot(dir, sunDir)), 2000.0) * 2000.0;
    return sky + sun;
}

float ReverseZDepth(float3 worldPos)
{
    float4 clip = mul(float4(worldPos, 1.0), g_View.matWorldToClip);
    return clip.z / clip.w;
}

[numthreads(8, 8, 1)]
void main(uint2 id : SV_DispatchThreadID)
{
    if (any(float2(id) >= g_View.viewportSize))
        return;

    float angle = g_Const.time * g_Const.sunSpeed;
    float3 sunDir = normalize(float3(cos(angle), 0.6, sin(angle)));
    const float3 sunColor = float3(4.0, 3.6, 3.0);

    Ray ray = CameraRay(g_View, float2(id));
    float tSphere = IntersectSphere(ray);
    float tGround = IntersectGround(ray);

    float3 color;
    float depth = 0.0;   // reverse-Z: 0 is infinitely far
    if (tSphere > 0.0 && (tGround < 0.0 || tSphere < tGround))
    {
        float3 p = ray.origin + ray.direction * tSphere;
        float3 n = normalize(p - c_SphereCenter);
        float diffuse = saturate(dot(n, sunDir));
        float3 albedo = float3(0.9, 0.35, 0.15);
        float3 h = normalize(sunDir - ray.direction);
        float specular = pow(saturate(dot(n, h)), 200.0) * 20.0;
        color = albedo * (diffuse * sunColor + 0.3 * Sky(n, sunDir)) + specular * sunColor * diffuse;
        depth = ReverseZDepth(p);
    }
    else if (tGround > 0.0)
    {
        float3 p = ray.origin + ray.direction * tGround;
        float checker = ((int(floor(p.x)) + int(floor(p.z))) & 1) ? 0.8 : 0.35;

        Ray shadowRay = { p + float3(0, 1e-3, 0), sunDir };
        float lit = IntersectSphere(shadowRay) > 0.0 ? 0.0 : 1.0;
        float fog = exp(-0.02 * tGround);
        float3 ground = checker * (lit * sunDir.y * sunColor + 0.3 * float3(0.55, 0.65, 0.8));
        color = lerp(Sky(ray.direction, sunDir), ground, fog);
        depth = ReverseZDepth(p);
    }
    else
    {
        color = Sky(ray.direction, sunDir);
    }

    u_Color[id] = float4(color, 1.0);
    u_Depth[id] = depth;
}
