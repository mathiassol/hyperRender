#include <hyper/ShaderHotReload.h>

#include "Process.h"

#include <donut/core/log.h>

#include <GLFW/glfw3.h>

#include <fstream>

using namespace donut;
namespace fs = std::filesystem;

namespace hyper
{
    namespace
    {
        constexpr double c_ScanIntervalSeconds = 0.25;

        bool IsShaderSource(const fs::path& p)
        {
            const auto ext = p.extension();
            return ext == ".hlsl" || ext == ".hlsli" || ext == ".slang" || ext == ".cfg" || ext == ".h";
        }
    }

    ShaderHotReload::~ShaderHotReload()
    {
        if (m_Worker.joinable())
            m_Worker.join();
    }

    void ShaderHotReload::AddManifest(const fs::path& manifest)
    {
        std::ifstream in(manifest);
        if (!in)
            return;

        ShaderSet set;
        std::string line;
        while (std::getline(in, line))
        {
            if (line.rfind("watch ", 0) == 0)
                set.watchDirs.emplace_back(line.substr(6));
            else if (line.rfind("arg ", 0) == 0)
                set.command.push_back(line.substr(4));
        }

        if (!set.command.empty())
            m_Sets.push_back(std::move(set));
    }

    bool ShaderHotReload::SourcesChanged()
    {
        fs::file_time_type newest{};
        size_t count = 0;
        std::error_code ec;

        for (const auto& set : m_Sets)
        {
            for (const auto& dir : set.watchDirs)
            {
                for (auto it = fs::recursive_directory_iterator(dir, ec); !ec && it != fs::recursive_directory_iterator(); it.increment(ec))
                {
                    if (!it->is_regular_file(ec) || !IsShaderSource(it->path()))
                        continue;
                    newest = std::max(newest, it->last_write_time(ec));
                    ++count;
                }
            }
        }

        // Count catches added and deleted files, which don't move the newest timestamp.
        const bool changed = m_HaveStamp && (newest != m_Stamp || count != m_FileCount);
        m_Stamp = newest;
        m_FileCount = count;
        m_HaveStamp = true;
        return changed;
    }

    void ShaderHotReload::StartRebuild()
    {
        if (m_Worker.joinable())
            m_Worker.join();

        m_Busy = true;
        m_Finished = false;
        log::info("hyper: shaders changed, recompiling...");

        m_Worker = std::thread([this, sets = m_Sets]
        {
            bool ok = true;
            std::string errors;
            for (const auto& set : sets)
            {
                ProcessResult r = RunProcess(set.command);
                if (r.exitCode != 0)
                {
                    ok = false;
                    errors += r.output;
                }
            }

            {
                std::lock_guard<std::mutex> lock(m_ErrorMutex);
                m_LastError = errors;
            }
            m_Succeeded = ok;
            m_Finished = true;
            m_Busy = false;
        });
    }

    bool ShaderHotReload::Poll()
    {
        if (m_Sets.empty())
            return false;

        if (m_Finished.exchange(false))
        {
            if (m_Succeeded)
            {
                log::info("hyper: shaders reloaded");
                return true;
            }
            log::warning("hyper: shader compilation failed, keeping the previous shaders\n%s", LastError().c_str());
        }

        const double now = glfwGetTime();
        if (m_Busy || now < m_NextScan)
            return false;
        m_NextScan = now + c_ScanIntervalSeconds;

        if (SourcesChanged() || m_ForceRebuild)
        {
            m_ForceRebuild = false;
            StartRebuild();
        }
        return false;
    }

    std::string ShaderHotReload::LastError() const
    {
        std::lock_guard<std::mutex> lock(m_ErrorMutex);
        return m_LastError;
    }
}
