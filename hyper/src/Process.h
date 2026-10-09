#pragma once

#include <string>
#include <vector>

namespace hyper
{
    struct ProcessResult
    {
        int exitCode = -1;
        std::string output;   // stdout and stderr, interleaved
    };

    // Runs args[0] with the remaining args, waits, and captures its output.
    // No shell is involved, so arguments need no escaping by the caller.
    ProcessResult RunProcess(const std::vector<std::string>& args);
}
