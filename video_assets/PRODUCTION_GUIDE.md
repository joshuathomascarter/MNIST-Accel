# ACCEL-v1 Video Production Guide

Complete guide for producing the 5-minute technical demonstration video.

## Quick Start

All assets have been generated and are ready for production:

```bash
cd /workspaces/ACCEL-v1/video_assets
```

## Generated Assets

### Static Graphics (`output/`)

1. **accuracy_comparison.png** - FP32 vs INT8 accuracy chart
   - Use at: 0:30-0:45 (Model Performance section)
   - Shows 98.9% → 98.7% accuracy retention

2. **key_metrics.png** - Architecture specifications
   - Use at: 1:00-1:30 (Hardware Architecture section)
   - Key specs: 2×2 PEs, INT8, 7-byte packets

3. **systolic_array_static.png** - PE array diagram
   - Use at: 1:30-2:00 (Systolic Array explanation)
   - Shows 2×2 grid with dataflow arrows

4. **quantization_process.png** - FP32→INT8 transformation
   - Use at: 2:00-2:30 (Quantization section)
   - Side-by-side distribution histograms

5. **uart_protocol.png** - Packet structure diagram
   - Use at: 2:30-3:00 (Communication Protocol section)
   - 7-byte packet format visualization

6. **module_breakdown.png** - Hardware composition
   - Use at: 3:00-3:30 (Integration section)
   - Pie chart showing module percentages

7. **video_timeline.png** - Video structure overview
   - Use for: Planning and editing reference
   - Shows 5-minute segment breakdown

### Animation (`output/animation_frames/`)

8. **Systolic array animation** (8 frames)
   - Use at: 1:30-2:00 (with static diagram)
   - Shows data flowing through PE array
   - 2 FPS playback recommended

To compile animation into video:
```bash
cd /workspaces/ACCEL-v1/video_assets
ffmpeg -framerate 2 -i output/animation_frames/frame_%03d.png \
       -c:v libx264 -pix_fmt yuv420p \
       output/systolic_array_animation.mp4
```

## Screen Recordings

Use the provided script to record terminal demonstrations:

```bash
cd /workspaces/ACCEL-v1/video_assets
./record_demos.sh
```

This will guide you through recording:
1. Python unit tests (pytest output)
2. MNIST quantization results
3. Golden model verification
4. Matrix tiling demonstration
5. Code structure overview
6. Documentation overview

**Recording Tips:**
- Use OBS Studio or similar screen recorder
- Set resolution to 1920×1080
- Record terminal with dark theme
- Increase terminal font size (16-18pt)
- Trim dead time in post-production

## Video Timeline (5 minutes)

### Segment 1: Introduction (0:00-0:30)
**Visuals:** Title card → GitHub repository view
**Script:** Opening hook and project overview
**Overlay:** None

### Segment 2: Model & Performance (0:30-1:00)
**Visuals:** accuracy_comparison.png
**Script:** MNIST CNN, quantization results
**Overlay:** Accuracy metrics (98.7% INT8)

### Segment 3: Hardware Architecture (1:00-2:00)
**Visuals:** 
- key_metrics.png (1:00-1:30)
- systolic_array_static.png + animation (1:30-2:00)
**Script:** Systolic array explanation, dataflow
**Overlay:** Architecture specifications

### Segment 4: Quantization Deep-Dive (2:00-2:30)
**Visuals:** quantization_process.png
**Script:** Post-training quantization, scale factors
**Overlay:** Recording of quantization code

### Segment 5: Communication Protocol (2:30-3:00)
**Visuals:** uart_protocol.png
**Script:** UART packet structure, command format
**Overlay:** Terminal showing UART driver code

### Segment 6: Integration & Testing (3:00-4:00)
**Visuals:** 
- module_breakdown.png (3:00-3:30)
- Test recordings (3:30-4:00)
**Script:** Verilog modules, verification strategy
**Overlay:** pytest output, golden model tests

### Segment 7: Results & Future Work (4:00-4:45)
**Visuals:** Code structure recording
**Script:** What works, what's conceptual, next steps
**Overlay:** GitHub repository statistics

### Segment 8: Call to Action (4:45-5:00)
**Visuals:** GitHub/LinkedIn/contact info
**Script:** Hiring pitch, collaboration invitation
**Overlay:** Contact information

## Voiceover Recording

Use the script provided in `video_script.md`:
- Record in quiet environment
- Use USB microphone (Blue Yeti or similar)
- Speak clearly at moderate pace
- Leave pauses at segment transitions
- Record multiple takes of difficult sections

**Editing Tips:**
- Remove breaths and filler words
- Normalize audio to -3dB
- Add subtle background music (non-copyright)
- Sync voiceover with visual transitions

## Video Editing Workflow

### Software Recommendations
- **DaVinci Resolve** (free, professional-grade)
- **Kdenlive** (open-source, Linux-friendly)
- **Adobe Premiere** (if available)

### Editing Steps

1. **Import Assets**
   - All PNG graphics from `output/`
   - Animation frames or compiled MP4
   - Screen recordings from `recordings/`
   - Voiceover audio

2. **Timeline Assembly**
   - Follow 8-segment structure above
   - Place voiceover on audio track
   - Layer graphics over voiceover
   - Add screen recordings with transitions

3. **Transitions**
   - Use simple crossfades (0.5s duration)
   - Avoid distracting wipes/effects
   - Keep focus on technical content

4. **Text Overlays**
   - Add title cards for each segment
   - Overlay key metrics on relevant sections
   - Use clean sans-serif font (Arial/Helvetica)
   - White text with black stroke for readability

5. **Background Music** (optional)
   - Low-volume instrumental (~10% mix)
   - Avoid vocals or distracting melodies
   - Free music: YouTube Audio Library, Free Music Archive

6. **Color Grading**
   - Keep natural colors for code/terminal
   - Slight contrast boost for graphics
   - Consistent look across all segments

7. **Export Settings**
   - Format: MP4 (H.264)
   - Resolution: 1920×1080
   - Frame rate: 30fps
   - Bitrate: 10-15 Mbps
   - Audio: AAC, 192 kbps

## Publishing Checklist

### Before Upload
- [ ] Watch full video for errors/typos
- [ ] Check audio levels (no clipping)
- [ ] Verify all graphics are visible/readable
- [ ] Ensure contact info is correct
- [ ] Test video playback on mobile device

### YouTube Upload
- Title: "ACCEL-v1: INT8 Systolic Array Accelerator | FPGA ML Inference [Technical Deep-Dive]"
- Description: (Include GitHub link, timestamps, contact info)
- Tags: FPGA, machine learning, systolic array, INT8, quantization, Verilog, hardware design
- Thumbnail: Use key_metrics.png with bold title overlay

### Social Media
- **Twitter/X**: 2-minute cut focusing on results + GitHub link
- **LinkedIn**: Full 5-minute video with professional intro
- **Reddit** (r/FPGA, r/MachineLearning):
  - Post as link + detailed comment
  - Emphasize open-source + honest limitations
  - Engage with comments professionally

### GitHub Repository
- Add video link to README.md
- Create `VIDEO.md` with timestamps and explanations
- Update PROJECT_COMPLETION_SUMMARY.md with video info

## File Checklist

Generated and ready:
- [x] accuracy_comparison.png
- [x] key_metrics.png
- [x] systolic_array_static.png
- [x] quantization_process.png
- [x] uart_protocol.png
- [x] module_breakdown.png
- [x] video_timeline.png
- [x] Animation frames (8 PNGs)
- [x] video_script.md
- [x] record_demos.sh
- [x] create_animation.py
- [x] generate_visuals.py

User must create:
- [ ] Screen recordings (via record_demos.sh)
- [ ] Voiceover audio (using video_script.md)
- [ ] Compiled animation MP4 (via ffmpeg)
- [ ] Final edited video
- [ ] YouTube thumbnail (edit key_metrics.png)

## Contact Information to Include

Add at video end (4:45-5:00):
```
GitHub: github.com/[your-username]/ACCEL-v1
LinkedIn: linkedin.com/in/[your-profile]
Email: [your-email]

"Open to hardware engineering roles and collaboration"
```

## Timeline Estimate

- Screen recordings: 30 minutes
- Voiceover recording: 1 hour
- Video editing: 3-4 hours
- Review and adjustments: 1 hour
- **Total: 5-6 hours production time**

## Notes

- All graphics use professional color scheme matching project branding
- Animation shows actual 2×2 matrix multiply computation
- Screen recordings demonstrate real working code
- Script emphasizes honesty about project status (some features conceptual)
- Video targets technical hiring managers and fellow engineers

Good luck with production! The assets are solid and ready to go.
