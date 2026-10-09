# XeGTAO

Intel's XeGTAO (MIT, see LICENSE), from https://github.com/GameTechDev/XeGTAO
(archived; `Source/Rendering/Shaders/XeGTAO.h` and `XeGTAO.hlsli`).

Local changes, both marked `// hyper:`:

- `XeGTAO.h`: `#include <cmath>` removed; C++ includers include it first.
- `XeGTAO.hlsli`: the debug-viz `#include "vaShared.hlsl"` removed.

ShaderMake's dependency scanner follows every `#include` regardless of
`#ifdef`, and fails on files it can't find.
