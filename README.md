# VideoSeal Watermark Overwriting Study

This project reproduces the VideoSeal video watermarking pipeline and tests whether a later watermark interferes with an earlier one.

## Base Implementation

Official repository: https://github.com/facebookresearch/videoseal

The experiment uses the legacy `videoseal_0.0` checkpoint with a 96-bit payload, corresponding to the 2024 VideoSeal paper.

## Environment

- Chip: Apple M2 Pro
- Architecture: arm64
- Python: 3.10.20
- PyTorch: 2.4.0
- FFmpeg / ffprobe installed through conda-forge

## Setup

Clone the official repository separately:

```bash
git clone https://github.com/facebookresearch/videoseal.git
cd videoseal
```

Create the environment and install dependencies:

```bash
conda create -n videoseal python=3.10 -y
conda activate videoseal

conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 -c pytorch -y
python -m pip install -r requirements.txt
conda install -c conda-forge ffmpeg -y
```

## Reproduction

I first ran the official streaming inference pipeline:

```bash
python inference_streaming.py --input assets/videos/1.mp4 --output_dir outputs/
```

The default script loads the current VideoSeal model. To match the paper setting, I copied the script and changed:

```python
videoseal.load("videoseal")
```

to:

```python
videoseal.load("videoseal_0.0")
```

The resulting message file contained 96 bits.

## Overwriting Experiment

`scripts/overwrite_experiment_v0.py` uses two fixed messages:

- A: first watermark
- B: second watermark

It compares four conditions:

- A only
- B only
- A followed by H.264 transcoding
- A followed by B

Run the experiment from this repository:

```bash
conda activate videoseal

python scripts/overwrite_experiment_v0.py \
  --input ~/Projects/videoseal/assets/videos/1.mp4 \
  --output_dir results/overwrite_v0
```

The script saves the fixed messages and result CSV in `results/overwrite_v0/`. Generated MP4 files are kept locally and ignored by Git.

## Preliminary Result

Watermark A was recovered at 100% after the first embedding and remained at 100% after the H.264 transcode control.

After watermark B was embedded into the A-watermarked video, recovery of A fell to 58.33%, while B was recovered at 83.33%.

This suggests substantial watermark interference or partial overwriting rather than loss caused by re-encoding alone.