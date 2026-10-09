#include <hyper/Profiler.h>

#include <imgui.h>

#include <algorithm>
#include <cstdio>

namespace hyper
{
    namespace
    {
        // Scopes not measured for this many frames are hidden.
        constexpr uint64_t c_HideAfterFrames = 120;
        // A scope whose slow frames (90th percentile) take this much longer than its
        // median is reported unsteady: on a GPU shared with another process timings come
        // in two modes, and one reading is either.
        constexpr float c_Unsteady = 1.5f;

    }

    void Profiler::Window::Add(float value)
    {
        samples[next] = value;
        next = (next + 1) % Size;
        count = std::min(count + 1, Size);
    }

    float Profiler::Window::Percentile(float p) const
    {
        if (count == 0)
            return 0.f;
        std::array<float, Size> sorted;
        std::copy(samples.begin(), samples.begin() + count, sorted.begin());
        const uint32_t k = std::min(uint32_t(p * float(count)), count - 1);
        std::nth_element(sorted.begin(), sorted.begin() + k, sorted.begin() + count);
        return sorted[k];
    }

    Profiler::Scope::Scope(Profiler* profiler, size_t entry, nvrhi::ICommandList* cmd, nvrhi::ITimerQuery* query)
        : m_Profiler(profiler), m_Entry(entry), m_Cmd(cmd), m_Query(query), m_Start(std::chrono::steady_clock::now())
    {
    }

    Profiler::Scope::Scope(Scope&& other) noexcept
        : m_Profiler(other.m_Profiler), m_Entry(other.m_Entry), m_Cmd(other.m_Cmd), m_Query(other.m_Query), m_Start(other.m_Start)
    {
        other.m_Profiler = nullptr;
    }

    Profiler::Scope::~Scope()
    {
        if (!m_Profiler)
            return;
        const double cpuMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - m_Start).count();
        m_Profiler->EndScope(m_Entry, m_Cmd, m_Query, cpuMs);
    }

    Profiler::Scope Profiler::Measure(nvrhi::ICommandList* cmd, const char* name)
    {
        auto [it, inserted] = m_Index.try_emplace(name, m_Entries.size());
        if (inserted)
            m_Entries.push_back(Entry{ name });

        Entry& e = m_Entries[it->second];
        e.depth = m_Depth++;
        e.order = m_Order++;
        if (e.framesUsed == 0 || e.lastUsedFrame != m_Frame)
            ++e.framesUsed;
        e.lastUsedFrame = m_Frame;

        nvrhi::TimerQueryHandle query;
        if (!e.free.empty())
        {
            query = e.free.back();
            e.free.pop_back();
        }
        else
        {
            query = m_Device->createTimerQuery();
        }
        e.pending.push_back({ m_Frame, query });

        cmd->beginMarker(name);
        cmd->beginTimerQuery(query);
        return Scope(this, it->second, cmd, query);
    }

    void Profiler::EndScope(size_t entry, nvrhi::ICommandList* cmd, nvrhi::ITimerQuery* query, double cpuMs)
    {
        cmd->endTimerQuery(query);
        cmd->endMarker();
        m_Entries[entry].cpuThisFrame += cpuMs;
        --m_Depth;
    }

    void Profiler::BeginFrame()
    {
        for (Entry& e : m_Entries)
        {
            // CPU: the frame that just ended is fully recorded.
            if (e.lastUsedFrame == m_Frame)
            {
                e.cpu.Add(float(e.cpuThisFrame));
            }
            e.cpuThisFrame = 0.0;

            // GPU: consume whole frames, oldest first, once all their queries resolved.
            // BeginFrame runs before anything is recorded, so every pending frame is complete.
            while (!e.pending.empty())
            {
                const uint64_t frame = e.pending.front().frame;
                size_t n = 0;
                bool ready = true;
                while (n < e.pending.size() && e.pending[n].frame == frame)
                {
                    if (!m_Device->pollTimerQuery(e.pending[n].query))
                    {
                        ready = false;
                        break;
                    }
                    ++n;
                }
                if (!ready)
                    break;

                float seconds = 0.f;
                for (size_t i = 0; i < n; ++i)
                {
                    auto& q = e.pending.front().query;
                    seconds += m_Device->getTimerQueryTime(q);
                    m_Device->resetTimerQuery(q);
                    e.free.push_back(q);
                    e.pending.pop_front();
                }
                e.gpu.Add(seconds * 1e3f);
            }
        }
        ++m_Frame;
        m_Depth = 0;
        m_Order = 0;
    }

    std::vector<const Profiler::Entry*> Profiler::Ordered() const
    {
        std::vector<const Entry*> entries;
        for (const Entry& e : m_Entries)
            entries.push_back(&e);
        std::stable_sort(entries.begin(), entries.end(), [](const Entry* a, const Entry* b) { return a->order < b->order; });
        return entries;
    }

    void Profiler::BuildUI() const
    {
        if (m_Entries.empty())
            return;

        if (!ImGui::BeginTable("hyper_profiler", 3, ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_RowBg))
            return;

        ImGui::TableSetupColumn("scope");
        ImGui::TableSetupColumn("GPU ms");
        ImGui::TableSetupColumn("CPU ms");
        ImGui::TableHeadersRow();

        for (const Entry* entry : Ordered())
        {
            const Entry& e = *entry;
            if (e.lastUsedFrame + c_HideAfterFrames < m_Frame)
                continue;
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::Indent(float(e.depth) * 12.f + 0.001f);
            ImGui::TextUnformatted(e.name.c_str());
            ImGui::Unindent(float(e.depth) * 12.f + 0.001f);
            ImGui::TableNextColumn();
            ImGui::Text("%.3f", e.gpu.Median());
            ImGui::TableNextColumn();
            ImGui::Text("%.3f", e.cpu.Median());
        }
        ImGui::EndTable();
    }

    std::string Profiler::Report() const
    {
        // Medians, with what a median hides: work that ran once (a capture of the sky, a
        // build), work that stopped, and timings too unsteady to read one number from.
        std::string out;
        char line[256];
        bool unsteady = false;
        for (const Entry* e : Ordered())
        {
            const float gpu = e->gpu.Median();
            std::string note;
            if (e->framesUsed == 1)
                note = "  (once)";
            else if (e->lastUsedFrame + 2 < m_Frame)
                note = "  (not in the last " + std::to_string(m_Frame - e->lastUsedFrame - 1) + " frames)";
            else if (e->gpu.count >= 8 && gpu > 0.05f && e->gpu.Percentile(0.9f) > c_Unsteady * gpu)
            {
                note = "  (unsteady)";
                unsteady = true;
            }
            std::snprintf(line, sizeof(line), "%*s%-24s gpu %7.3f ms  cpu %7.3f ms%s\n",
                          e->depth * 2, "", e->name.c_str(), gpu, e->cpu.Median(), note.c_str());
            out += line;
        }
        if (unsteady)
            out += "unsteady: the slow frames took 1.5x the median or more; if nothing in the frame varies, "
                   "something else is using the GPU. Measure again.\n";
        return out;
    }

    float Profiler::GpuMs(const char* name) const
    {
        auto it = m_Index.find(name);
        return it == m_Index.end() ? 0.f : m_Entries[it->second].gpu.Median();
    }
}
