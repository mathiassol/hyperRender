#pragma once

#include <hyper/platform.h>

#include <atomic>
#include <filesystem>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace hyper
{
    // Watches shader sources and recompiles them in the background with the
    // same ShaderMake command the build uses (bin/shaders/<name>/<api>/hotreload.txt).
    // hyper::Run drives it; apps only see App::ShadersReloaded().
    class ShaderHotReload
    {
    public:
        ~ShaderHotReload();

        // Registers a shader set by its manifest. Missing manifests are ignored.
        void AddManifest(const std::filesystem::path& manifest);

        // Call once per frame. Returns true on the frame a rebuild succeeded,
        // i.e. when shaders should be recreated.
        bool Poll();

        void RequestRebuild() { m_ForceRebuild = true; }

        [[nodiscard]] bool IsEnabled() const { return !m_Sets.empty(); }
        [[nodiscard]] bool IsBusy() const { return m_Busy; }
        // Compiler output of the last failed rebuild; empty after a success.
        [[nodiscard]] std::string LastError() const;

    private:
        struct ShaderSet
        {
            std::vector<std::filesystem::path> watchDirs;
            std::vector<std::string> command;
        };

        std::vector<ShaderSet> m_Sets;
        std::filesystem::file_time_type m_Stamp{};
        size_t m_FileCount = 0;
        double m_NextScan = 0.0;
        bool m_HaveStamp = false;
        bool m_ForceRebuild = false;

        std::thread m_Worker;
        std::atomic<bool> m_Busy{ false };
        std::atomic<bool> m_Finished{ false };
        bool m_Succeeded = false;
        mutable std::mutex m_ErrorMutex;
        std::string m_LastError;

        bool SourcesChanged();
        void StartRebuild();
    };
}
