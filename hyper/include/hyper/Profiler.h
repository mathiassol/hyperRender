#pragma once

#include <hyper/platform.h>

#include <nvrhi/nvrhi.h>

#include <array>
#include <chrono>
#include <deque>
#include <string>
#include <unordered_map>
#include <vector>

namespace hyper
{
    // GPU + CPU timings per named scope, shown in the hyper overlay.
    //
    //     {
    //         auto t = ctx.profiler->Measure(cmd, "bloom");
    //         ... record commands ...
    //     }
    //
    // A scope is also a debug marker, so it shows up in Nsight / RenderDoc.
    // GPU times arrive a few frames late. Displayed values are the median of
    // the last 64 frames, so one-off spikes (loading, pipeline creation) drop out.
    class Profiler
    {
    public:
        class Scope
        {
        public:
            Scope(Scope&& other) noexcept;
            ~Scope();
            Scope(const Scope&) = delete;
            Scope& operator=(const Scope&) = delete;
            Scope& operator=(Scope&&) = delete;

        private:
            friend class Profiler;
            Scope(Profiler* profiler, size_t entry, nvrhi::ICommandList* cmd, nvrhi::ITimerQuery* query);

            Profiler* m_Profiler;
            size_t m_Entry;
            nvrhi::ICommandList* m_Cmd;
            nvrhi::ITimerQuery* m_Query;
            std::chrono::steady_clock::time_point m_Start;
        };

        explicit Profiler(nvrhi::IDevice* device) : m_Device(device) { }

        [[nodiscard]] Scope Measure(nvrhi::ICommandList* cmd, const char* name);

        // Called by hyper at the start of every frame: collects finished queries.
        void BeginFrame();

        // Draws the timing table into the current ImGui window.
        void BuildUI() const;

        // Median GPU milliseconds for a scope over recent frames, or 0 if never measured.
        [[nodiscard]] float GpuMs(const char* name) const;

        // The timing table as text, one scope per line (logged by --capture).
        [[nodiscard]] std::string Report() const;

    private:
        struct Window
        {
            static constexpr uint32_t Size = 64;
            std::array<float, Size> samples{};
            uint32_t count = 0;
            uint32_t next = 0;
            void Add(float value);
            [[nodiscard]] float Percentile(float p) const;   // of the samples held, 0..1
            [[nodiscard]] float Median() const { return Percentile(0.5f); }
        };

        struct PendingQuery
        {
            uint64_t frame;
            nvrhi::TimerQueryHandle query;
        };

        struct Entry
        {
            std::string name;
            int depth = 0;
            uint32_t order = 0;          // position in the last frame that used it
            Window gpu;
            Window cpu;
            double cpuThisFrame = 0.0;
            uint64_t lastUsedFrame = 0;
            uint64_t framesUsed = 0;
            std::vector<nvrhi::TimerQueryHandle> free;
            std::deque<PendingQuery> pending;
        };

        nvrhi::IDevice* m_Device;
        std::vector<Entry> m_Entries;
        std::unordered_map<std::string, size_t> m_Index;
        uint64_t m_Frame = 0;
        int m_Depth = 0;
        uint32_t m_Order = 0;

        // Entries in the order they ran (first-seen order breaks when a pass starts late).
        [[nodiscard]] std::vector<const Entry*> Ordered() const;
        void EndScope(size_t entry, nvrhi::ICommandList* cmd, nvrhi::ITimerQuery* query, double cpuMs);
    };
}
