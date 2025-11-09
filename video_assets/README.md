# ACCEL-v1 Video Production Assets

Complete package for creating a 5-minute technical demonstration video.

## Quick Reference

- **Script:** `video_script.md` - Full voiceover script with timing
- **Production Guide:** `PRODUCTION_GUIDE.md` - Step-by-step editing instructions
- **Graphics:** `output/*.png` - 7 static visualizations ready for overlay
- **Animation:** `output/animation_frames/` - 8-frame systolic array dataflow
- **Recording Script:** `record_demos.sh` - Automated terminal demo recordings

## Generated Assets

All static graphics have been generated and saved to `output/`:

1. `accuracy_comparison.png` - FP32 vs INT8 accuracy (98.9% → 98.7%)
2. `key_metrics.png` - Hardware specifications (2×2 PEs, INT8, etc.)
3. `systolic_array_static.png` - PE array diagram with dataflow
4. `quantization_process.png` - FP32→INT8 transformation visualization
5. `uart_protocol.png` - 7-byte packet structure
6. `module_breakdown.png` - Hardware module composition pie chart
7. `video_timeline.png` - 5-minute video segment structure

Animation frames (8 PNGs) are in `output/animation_frames/`.

## Usage

### 1. Generate Animation Video (Optional)

Compile animation frames into MP4:
```bash
cd /workspaces/ACCEL-v1/video_assets
ffmpeg -framerate 2 -i output/animation_frames/frame_%03d.png \
       -c:v libx264 -pix_fmt yuv420p \
       output/systolic_array_animation.mp4
```

### 2. Record Terminal Demos

Run the guided recording script:
```bash
./record_demos.sh
```

This will walk you through recording 6 terminal demonstrations:
- Python unit tests
- MNIST quantization results  
- Golden model verification
- Matrix tiling examples
- Code structure overview
- Documentation files

### 3. Record Voiceover

Use `video_script.md` to record voiceover:
- 5-minute script with 8 segments
- Technical but accessible language
- Timestamps for video editing sync

### 4. Edit Video

Follow `PRODUCTION_GUIDE.md` for complete editing workflow:
- Import all assets
- Sync voiceover with visuals
- Add graphics overlays at specified timestamps
- Insert screen recordings
- Export as 1080p MP4

## File Structure

```
video_assets/
├── README.md                    (this file)
├── PRODUCTION_GUIDE.md          (detailed editing guide)
├── video_script.md              (5-min voiceover script)
├── generate_visuals.py          (script that generated graphics)
├── create_animation.py          (script that generated animation)
├── record_demos.sh              (terminal recording guide)
└── output/
    ├── accuracy_comparison.png
    ├── key_metrics.png
    ├── systolic_array_static.png
    ├── quantization_process.png
    ├── uart_protocol.png
    ├── module_breakdown.png
    ├── video_timeline.png
    └── animation_frames/
        ├── frame_000.png
        ├── frame_001.png
        └── ... (8 frames total)
```

## Timeline

Estimated production time: **5-6 hours**
- Screen recordings: 30 min
- Voiceover: 1 hour
- Video editing: 3-4 hours
- Review: 1 hour

## Target Platforms

- **YouTube:** Full 5-minute technical deep-dive
- **LinkedIn:** Professional presentation for recruiters
- **Reddit:** Honest technical demonstration (r/FPGA, r/MachineLearning)
- **Twitter/X:** 2-minute highlight reel

## Notes

All assets emphasize technical accuracy and honest project status. Some features are conceptual/future work, which is clearly stated in the script. Graphics use professional color scheme and are optimized for 1080p video overlay.

**Start with PRODUCTION_GUIDE.md for complete instructions.**
