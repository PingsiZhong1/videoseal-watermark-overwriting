# Reproduction environment

The Week 2 runs were performed in the existing `videoseal` conda
environment. No dependency installation or upgrade is part of this work.

| Component | Value |
|---|---|
| Host | Apple M2 Pro, macOS, arm64 |
| Conda environment | `videoseal` |
| Python | 3.10.20 |
| PyTorch | 2.4.0 |
| Device used by VideoSeal runs | MPS when available; CPU fallback for diagnostics |
| Official repository | External `facebookresearch/videoseal` checkout |
| Python executable | Python from the `videoseal` conda environment |
| FFmpeg | FFmpeg from the `videoseal` conda environment |
| ffprobe | ffprobe from the `videoseal` conda environment |
| Input video | Official `assets/videos/1.mp4` example |

The scripts accept an explicit `--device` argument. Runs should record the
actual selected device in their JSON output; `auto` is recommended when the
same command must work on a host where MPS is unavailable.

The official VideoSeal repository is kept outside this project and is not
modified by the training scripts. The `masks={"kind":"none"}` behavior is
already present in the official augmenter implementation, so no local official
repo patch is required.

The PixelSeal reproduction enables `PYTORCH_ENABLE_MPS_FALLBACK=1` before
importing PyTorch. In PyTorch 2.4, this lets the unsupported antialiased
bilinear-resize operation run on CPU while the remaining supported operations
continue to use MPS.
