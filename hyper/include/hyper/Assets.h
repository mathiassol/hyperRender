#pragma once

#include <hyper/platform.h>

#include <filesystem>

namespace hyper
{
    // The repository's assets/ folder, found by walking up from the executable
    // (bin/ lives in build/<preset>/bin/). Empty if there is none.
    std::filesystem::path AssetsDirectory();

    // assets/<relative>, or an absolute path unchanged.
    std::filesystem::path AssetPath(const std::filesystem::path& relative);
}
