# Official VideoSeal Baseline Reproduction

## Official Repository

https://github.com/facebookresearch/videoseal

## Goal

Run the official VideoSeal streaming inference pipeline before modifying the model configuration or conducting watermark-overwriting experiments.

## Environment

- Device: Apple Silicon MacBook Pro
- Architecture: arm64
- Python: 3.10.20
- PyTorch: 2.4.0
- MPS available: True
- FFmpeg and ffprobe: installed through conda-forge

## Official Command Used

```bash
python inference_streaming.py --input assets/videos/1.mp4 --output_dir outputs/

```
