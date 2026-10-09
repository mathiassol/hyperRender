#include <hyper/lighting/TracedLighting.h>

#include <hyper/App.h>
#include <hyper/passes/ComputePasses.h>

#include <imgui.h>

namespace hyper
{
    // hyper/traced_lighting.hlsli's TracedConstants.
    struct TracedLighting::Constants
    {
        uint32_t size[2];
        uint32_t frameIndex;
        uint32_t step;
        uint32_t rays;
        float giClamp;
        float reflectionClamp;
        float tolerance;
    };

    TracedLighting::TracedLighting(Context& ctx, ComputePasses& passes) : m_Ctx(ctx), m_Passes(passes) { }

    void TracedLighting::CreatePipelines(const char* shader, nvrhi::IBindingLayout* bindless)
    {
        m_Passes.Pipeline(m_GiTrace, shader, "gi_trace", bindless);
        m_Passes.Pipeline(m_GiTemporal, shader, "gi_temporal", bindless);
        m_Passes.Pipeline(m_GiFilter, shader, "gi_filter", bindless);
        m_Passes.Pipeline(m_ReflTrace, shader, "reflect_trace", bindless);
        m_Passes.Pipeline(m_ReflTemporal, shader, "reflect_temporal", bindless);
        m_Passes.Pipeline(m_ReflFilter, shader, "reflect_filter", bindless);
    }

    void TracedLighting::EnsureTargets(nvrhi::ITexture* depth)
    {
        const auto& d = depth->getDesc();
        if (m_GiRaw && m_Size.x == d.width && m_Size.y == d.height)
            return;
        m_Size = dm::uint2(d.width, d.height);
        auto tex = [&](const char* name, uint32_t div)
        {
            return m_Ctx.device->createTexture(nvrhi::TextureDesc().setDebugName(name)
                .setWidth((d.width + div - 1) / div).setHeight((d.height + div - 1) / div)
                .setFormat(nvrhi::Format::RGBA16_FLOAT).setIsUAV(true)
                .setInitialState(nvrhi::ResourceStates::ShaderResource).setKeepInitialState(true));
        };
        m_GiRaw = tex("hyper/gi-raw", 2);
        m_GiHistory[0] = tex("hyper/gi-0", 2);
        m_GiHistory[1] = tex("hyper/gi-1", 2);
        m_GiA = tex("hyper/gi-a", 2);
        m_GiB = tex("hyper/gi-b", 2);
        m_ReflRaw = tex("hyper/reflection-raw", 1);
        m_ReflHistory[0] = tex("hyper/reflection-0", 1);
        m_ReflHistory[1] = tex("hyper/reflection-1", 1);
        m_Reflection = tex("hyper/reflection", 1);
        m_GiValid = m_ReflValid = false;
        m_Passes.ClearSets();
    }

    TracedLighting::Constants TracedLighting::Make(uint32_t step) const
    {
        static_assert(sizeof(Constants) <= ComputePasses::c_PushBytes);
        return { { m_Size.x, m_Size.y }, uint32_t(m_Ctx.frameIndex), step, settings.giRays,
                 settings.giClamp, settings.reflectionClamp, settings.tolerance };
    }

    nvrhi::ITexture* TracedLighting::Gi(nvrhi::ICommandList* cmd, const Inputs& in)
    {
        EnsureTargets(in.depth);
        auto t = m_Ctx.profiler->Measure(cmd, "gi");
        const uint32_t w = (m_Size.x + 1) / 2, h = (m_Size.y + 1) / 2;
        Constants c = Make(0);
        m_Passes.Dispatch(cmd, m_GiTrace, { in.depth, in.normals, in.app }, { m_GiRaw }, w, h, &c, sizeof(c));
        nvrhi::ITexture* history = m_GiHistory[m_GiIndex];
        nvrhi::ITexture* accumulated = m_GiHistory[m_GiIndex ^ 1];
        c.step = m_GiValid ? 1u : 0u;
        m_Passes.Dispatch(cmd, m_GiTemporal, { in.depth, in.normals, in.app, m_GiRaw, history }, { accumulated }, w, h, &c, sizeof(c));
        m_GiIndex ^= 1;
        m_GiValid = true;
        // Three a-trous passes, steps 1, 2, 4.
        nvrhi::ITexture* src = accumulated;
        nvrhi::ITexture* dst = m_GiA;
        for (uint32_t step = 1; step <= 4; step *= 2)
        {
            c.step = step;
            m_Passes.Dispatch(cmd, m_GiFilter, { in.depth, in.normals, in.app, src }, { dst }, w, h, &c, sizeof(c));
            src = dst;
            dst = dst == m_GiA.Get() ? m_GiB.Get() : m_GiA.Get();
        }
        return src;
    }

    nvrhi::ITexture* TracedLighting::Reflections(nvrhi::ICommandList* cmd, const Inputs& in)
    {
        EnsureTargets(in.depth);
        auto t = m_Ctx.profiler->Measure(cmd, "reflect");
        Constants c = Make(0);
        m_Passes.Dispatch(cmd, m_ReflTrace, { in.depth, in.normals, in.app }, { m_ReflRaw }, m_Size.x, m_Size.y, &c, sizeof(c));
        nvrhi::ITexture* history = m_ReflHistory[m_ReflIndex];
        nvrhi::ITexture* accumulated = m_ReflHistory[m_ReflIndex ^ 1];
        c.step = m_ReflValid ? 1u : 0u;
        m_Passes.Dispatch(cmd, m_ReflTemporal, { in.depth, in.normals, in.app, m_ReflRaw, history }, { accumulated }, m_Size.x, m_Size.y, &c, sizeof(c));
        m_Passes.Dispatch(cmd, m_ReflFilter, { in.depth, in.normals, in.app, accumulated }, { m_Reflection }, m_Size.x, m_Size.y, &c, sizeof(c));
        m_ReflIndex ^= 1;
        m_ReflValid = true;
        return m_Reflection;
    }

    void TracedLighting::BuildUI()
    {
        int rays = int(settings.giRays);
        if (ImGui::SliderInt("bounce rays per pixel", &rays, 1, 8))
            settings.giRays = uint32_t(rays);
        ImGui::SliderFloat("bounce clamp", &settings.giClamp, 0.5f, 64.f, "%.1f", ImGuiSliderFlags_Logarithmic);
        ImGui::SliderFloat("reflection clamp", &settings.reflectionClamp, 0.5f, 256.f, "%.1f", ImGuiSliderFlags_Logarithmic);
        ImGui::SliderFloat("depth tolerance m", &settings.tolerance, 0.0001f, 0.1f, "%.4f", ImGuiSliderFlags_Logarithmic);
    }
}
