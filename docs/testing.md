# Testing

hyperRender is tested by images. Every app can render a fixed number of frames to a PNG
and exit; the regression suite does that for a list of cases and compares each image with
a reference. A case fails if its app crashes, logs an error or a Vulkan validation
message, or its image drifts.

## One capture

```powershell
build\vs2026\bin\scene.exe --capture out.png --frames 300
```

- Renders 300 frames at a fixed 1/60 s step (`ctx.time` and `ctx.deltaTime` follow it), so
  the result doesn't depend on how fast the machine is.
- Saves the last frame without the UI, prints every profiler scope with its GPU and CPU
  time, and exits.
- Exits with code 2 if anything logged an error or a validation message. Run the Debug
  build (`scene-debug.exe`) to have the validation layers check everything.
- Scenes load synchronously, so frame N is always the same frame. `--async-load` loads the
  way the interactive app does (the scene first, one frame without textures, then every
  texture), in the same order every run: it tests streaming.

## The regression suite

```powershell
pwsh tools/capture_all.ps1                     # every case, release build
pwsh tools/capture_all.ps1 -Only 'scene*'      # some of them
pwsh tools/capture_all.ps1 -Suffix -debug      # the Debug build, with validation
pwsh tools/capture_all.ps1 -Update             # write the references from this run
```

- The cases are in `tools/captures.txt`: a name, the app, and its arguments. An app with no
  case gets one with no arguments.
- Each image is compared with its 4× downscaled reference in `tests/reference/`. A case
  passes with a mean difference up to 0.5 and no pixel over 24 (out of 255). Runs differ a
  little (dither, GPU atomics): a mean of about 0.3 is normal.
- Images and logs land in `build/captures/`.
- The scene cases need `pwsh tools/fetch_assets.ps1`, `scene-field-lod` needs
  `python tools/make_stress_scene.py field`, and the ray-traced cases a GPU with ray
  queries.
- The comparison needs Python with `numpy` and `pillow`.

**References depend on the GPU and the driver.** The ones in the repository come from an
NVIDIA RTX 3070 Ti. On another GPU, or after a driver update, run with `-Update` once,
look at the images, and use those as your baseline.

**After an intended change**, look at the new images in `build/captures/` before running
`-Update`: the suite can't tell an improvement from a regression.

## Adding a case

Add a line to `tools/captures.txt`:

```
myapp-fast     myapp  --set speed=4
```

then `pwsh tools/capture_all.ps1 -Only 'myapp*' -Update` to write its reference, and look
at it. Settings that come from `--set` (see `hyper::Overrides`) are what make an app
testable this way: give every setting worth testing a name.

## Comparing images by hand

`tools/png_stats.py` compares two images (`--diff`), measures a region (`--rect`) or crops
one at full resolution (`--crop`).

## Timing

- Use the release build, and `--no-vsync` interactively (captures turn vsync off).
- Every capture prints its profiler scopes. Repeat a run until two agree: a GPU shared with
  other work gives slow, bimodal numbers (`nvidia-smi` shows what else is running).
- GPU timings with vsync on include the wait, charged to the first pass that draws.
