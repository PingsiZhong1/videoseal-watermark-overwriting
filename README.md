# VideoSeal Watermark Overwriting Study

This project reproduces the VideoSeal video watermarking pipeline and examines whether a later watermark weakens or replaces an earlier one at the decoding level.

## Base Implementation

Official repository: https://github.com/facebookresearch/videoseal

The experiment uses the legacy `videoseal_0.0` checkpoint with a 96-bit payload.

## Environment

- Chip: Apple M2 Pro
- Architecture: arm64
- Python: 3.10.20
- PyTorch: 2.4.0
- FFmpeg / ffprobe installed through conda-forge

## Reproduction

I first ran the official streaming inference pipeline:

```bash
python inference_streaming.py --input assets/videos/1.mp4 --output_dir outputs/
```

To reproduce the 96-bit setting, I changed the model loading line from:

```python
videoseal.load("videoseal")
```

to:

```python
videoseal.load("videoseal_0.0")
```

The resulting message file contained 96 bits.

## Overwriting Experiment

`scripts/overwrite_experiment_v0.py` keeps the official embedding and detection functions unchanged, but adds a controlled sequential-watermark experiment.

The script uses two fixed, bitwise complementary messages:

- A: first watermark
- B: second watermark
- Every decoded bit matches either A or B, allowing the final output to be compared directly with the earlier and later messages.

It evaluates:

- `A_only`: embed A once
- `B_only`: embed B once
- `A_transcode`: embed A, then apply H.264 transcoding
- `A_then_A`: embed A twice
- `A_then_B`: embed A, then embed B
- `B_then_A`: embed B, then embed A

Run the experiment with:

```bash
conda activate videoseal

python scripts/overwrite_experiment_v0.py \
  --input ~/Projects/videoseal/assets/videos/1.mp4 \
  --output_dir results/overwrite_v0 \
  --official_repo ~/Projects/videoseal \
  --seed 2025 \
  --crf 23
```

## Preliminary Result

The baseline and control conditions recovered the intended watermark completely:

- `A_only`: 96 / 96 bits matched A
- `B_only`: 96 / 96 bits matched B
- `A_transcode`: 96 / 96 bits matched A
- `A_then_A`: 96 / 96 bits matched A

When different watermarks were embedded sequentially, decoding shifted toward the later watermark:

- `A_then_B`: 70 / 96 bits matched later watermark B
- `B_then_A`: 72 / 96 bits matched later watermark A

These results suggest that a later, different watermark can partially replace an earlier watermark at the decoding level. The earlier watermark remains partially recoverable, so this does not demonstrate complete removal.

Generated MP4 files are kept locally and ignored by Git. The repository stores scripts, fixed messages, logs, decoded bit strings, and CSV results.