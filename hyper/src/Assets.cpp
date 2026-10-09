#include <hyper/Assets.h>

#include <donut/app/ApplicationBase.h>

namespace hyper
{
    std::filesystem::path AssetsDirectory()
    {
        static const std::filesystem::path found = []
        {
            std::error_code ec;
            for (auto dir = donut::app::GetDirectoryWithExecutable(); !dir.empty(); dir = dir.parent_path())
            {
                if (std::filesystem::is_directory(dir / "assets", ec))
                    return dir / "assets";
                if (dir == dir.parent_path())
                    break;
            }
            return std::filesystem::path();
        }();
        return found;
    }

    std::filesystem::path AssetPath(const std::filesystem::path& relative)
    {
        if (relative.is_absolute())
            return relative;
        return AssetsDirectory() / relative;
    }
}
