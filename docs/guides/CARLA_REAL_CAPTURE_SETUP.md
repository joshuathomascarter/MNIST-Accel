# CARLA Real Capture Setup

This guide is the supported path for getting a real CARLA screenshot or short clip to replace the generated competition visuals.

## Why This Was Not Installed On The Current Machine

The current development host is not a supported CARLA simulator target.

Current host facts:

- macOS 26.4
- Apple Silicon `arm64`
- Apple M1
- `8 GB` memory

Official CARLA quick-start support is for:

- Windows 10 or 11 x64
- Ubuntu 20.04 or 22.04 x86_64

The official docs also recommend a dedicated GPU roughly equivalent to an NVIDIA 2070 or better with at least `8 GB` VRAM, plus about `20 GB` for the packaged install or about `130 GB` if building from source.

That means this Mac is the wrong place to install the full simulator.

## Recommended Supported Setup

Use one of these:

1. Ubuntu 22.04 x86_64 workstation with NVIDIA GPU and at least `8 GB` VRAM.
2. Windows 11 x64 workstation with a dedicated GPU and enough disk for the packaged release.

For the fastest path to real screenshots or clips, use the packaged CARLA release rather than building from source.

## Package Install Path

1. Download the packaged CARLA release from the official release/download page.
2. Extract it into a working directory such as `~/carla` on Ubuntu or a local folder on Windows.
3. Install the matching CARLA Python client library from the wheel inside the package, or use the released PyPI package only if the version matches the server package.
4. Install the example requirements and Pillow in the Python environment that will run the capture script.

On Ubuntu, the usual layout looks like this:

```bash
cd ~/carla
./CarlaUE4.sh
```

If you want to try an off-screen server on Linux, CARLA documentation also shows `-RenderOffScreen` in some workflows.

## Python Environment For Capture

Inside the CARLA package or your chosen Python environment:

```bash
python3 -m pip install --upgrade pip
python3 -m pip install pillow numpy
cd PythonAPI/examples
python3 -m pip install -r requirements.txt
```

If you are using the wheel that ships with the CARLA package, install it from the package's `PythonAPI/carla/dist` directory so the client matches the server version.

## Real Screenshot And Clip Capture

Once the CARLA server is running, use this repo script:

```bash
python3 tools/carla_capture_visuals.py \
  --host 127.0.0.1 \
  --port 2000 \
  --traffic-port 8000 \
  --town Town05 \
  --output-dir docs/figs/carla_real_capture \
  --still-name carla_town05_real.png \
  --clip-name carla_town05_real.gif \
  --frames 90 \
  --warmup 30 \
  --vehicles 18
```

Outputs:

- real PNG still image
- real GIF clip from the simulator camera attached to the ego vehicle

## How To Use These Assets In The Deck

Replace the generated placeholders with the real outputs:

- use the real still on the title slide
- optionally use one frame from the clip or the GIF itself in the appendix/demo section
- keep the trace-driven NoC results on the right side of the slide so the visual and the architectural claim stay connected

## Practical Notes

- If you want a quick, stable demo, capture the still and clip ahead of time instead of relying on a live simulator on stage.
- Keep Town05 and the same scene framing every time so the competition visuals remain reproducible.
- The current repo already contains generated fallback visuals for the unsupported macOS development environment; the script in this guide is the upgrade path to replace them with real CARLA captures.