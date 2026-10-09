#pragma once

#include <hyper/platform.h>

#include <string>
#include <vector>

namespace hyper
{
    // Command-line overrides for settings, for captures and experiments:
    //
    //     app.exe --set exposure=-1 --set bloom=0 --capture out.png
    //
    //     hyper::Overrides o(ctx.args);
    //     o.Apply("exposure", post.tonemap.exposureEV);
    //     o.Apply("bloom", post.bloom);
    //     o.WarnUnused();   // logs every --set nobody asked for (typos)
    class Overrides
    {
    public:
        explicit Overrides(const std::vector<std::string>& args);

        // Each returns true if the value was overridden.
        bool Apply(const char* name, float& value);
        bool Apply(const char* name, int& value);
        bool Apply(const char* name, bool& value);   // 1/0, true/false, on/off

        void WarnUnused() const;

    private:
        struct Entry
        {
            std::string name;
            std::string value;
            mutable bool used = false;
        };
        std::vector<Entry> m_Entries;

        const Entry* Find(const char* name) const;
    };
}
