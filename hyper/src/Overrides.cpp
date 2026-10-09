#include <hyper/Overrides.h>

#include <donut/core/log.h>

#include <cstdlib>

namespace hyper
{
    Overrides::Overrides(const std::vector<std::string>& args)
    {
        for (size_t i = 0; i + 1 < args.size(); ++i)
        {
            if (args[i] != "--set")
                continue;
            const std::string& kv = args[++i];
            const size_t eq = kv.find('=');
            if (eq == std::string::npos || eq == 0)
            {
                donut::log::warning("hyper: ignoring --set %s (expected name=value)", kv.c_str());
                continue;
            }
            m_Entries.push_back({ kv.substr(0, eq), kv.substr(eq + 1) });
        }
    }

    const Overrides::Entry* Overrides::Find(const char* name) const
    {
        // Last one wins, like repeated command-line flags usually do.
        for (auto it = m_Entries.rbegin(); it != m_Entries.rend(); ++it)
        {
            if (it->name == name)
            {
                for (const Entry& e : m_Entries)
                    if (e.name == name)
                        e.used = true;
                return &*it;
            }
        }
        return nullptr;
    }

    bool Overrides::Apply(const char* name, float& value)
    {
        const Entry* e = Find(name);
        if (!e)
            return false;
        value = float(std::atof(e->value.c_str()));
        return true;
    }

    bool Overrides::Apply(const char* name, int& value)
    {
        const Entry* e = Find(name);
        if (!e)
            return false;
        value = std::atoi(e->value.c_str());
        return true;
    }

    bool Overrides::Apply(const char* name, bool& value)
    {
        const Entry* e = Find(name);
        if (!e)
            return false;
        value = e->value == "1" || e->value == "true" || e->value == "on";
        return true;
    }

    void Overrides::WarnUnused() const
    {
        for (const Entry& e : m_Entries)
            if (!e.used)
                donut::log::warning("hyper: unknown --set %s", e.name.c_str());
    }
}
