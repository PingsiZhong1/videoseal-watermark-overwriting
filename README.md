# VideoSeal Watermark Overwriting Study

This project reproduces the VideoSeal watermarking pipeline and explores whether sequential watermark embedding weakens earlier watermarks.

Official implementation: https://github.com/facebookresearch/videoseal

The experiments use the legacy `videoseal_0.0` checkpoint with a 96-bit payload.

## Environment

- Chip: Apple M2 Pro
- Architecture: arm64
- Python: 3.10.20
- PyTorch: 2.4.0
- FFmpeg / ffprobe installed through conda-forge

## 1. Reproduction

I first ran the official streaming inference pipeline and verified that the 96-bit legacy checkpoint could embed and recover a watermark.

```bash
python inference_streaming.py --input assets/videos/1.mp4 --output_dir outputs/
```

For the legacy paper setting, the model was loaded with:

```python
videoseal.load("videoseal_0.0")
```

## 2. Watermark Overwriting Experiment

`scripts/overwrite_experiment_v0.py` tests controlled sequential embedding with two fixed complementary messages.

Conditions:

- `A_only`
- `B_only`
- `A_transcode`
- `A_then_A`
- `A_then_B`
- `B_then_A`

Key result:

- H.264 transcoding alone preserved watermark A.
- Embedding A twice also preserved A.
- After `A_then_B`, 70 of 96 decoded bits matched later watermark B.
- After `B_then_A`, 72 of 96 decoded bits matched later watermark A.

This provides preliminary evidence that a later, different watermark can dominate the final decoded message.

Results: `results/overwrite_v0/`

## 3. Chained Watermarking Prototypes

### Direct chain token

`scripts/chain.py` stores:

```text
current ID | parent ID | step
```

It recovered `A -> B`, but after `A -> B -> C`, one parent-ID error prevented chain verification.

`scripts/chain2.py` added more redundancy for the parent field and a simple consistency check. The three-step parent-ID error remained.

### Pointer-based chain

`scripts/chain3.py` uses one full 96-bit codeword for the latest record ID. The parent relationship is stored in an external registry.

```text
Recovered latest record ID
→ lookup registry
→ reconstruct A -> B -> C
```

In the current pilot run, `A`, `A -> B`, `A -> B -> C`, and `A -> B -> C -> H.264 transcode` were classified correctly.

This is a provenance reconstruction prototype. It does not preserve A, B, and C as independently decodable watermarks in the final video.

Results: `results/chain3/`

## 4. Multi-Slot Training Prototype

`scripts/multi.py` is a model-level prototype for direct multi-watermark recovery.

The 96-bit payload is divided into three ordered 32-bit slots:

```text
Slot 1: A
Slot 2: B
Slot 3: C
```

The training sequence is:

```text
original -> A
A-watermarked video -> A + B
A + B-watermarked video -> A + B + C
```

The local training path, backpropagation, and checkpoint saving ran successfully.

A 20-step run on one video did not yet recover all active slots reliably. This is an implementation prototype, not evidence that multi-watermark coexistence has been solved.

Results: `results/multi_20/` and `results/multi_20_test/`

## Repository Structure

```text
scripts/
  overwrite_experiment_v0.py
  chain.py
  chain2.py
  chain3.py
  multi.py

logs/
  overwrite_v0.txt
  chain.txt
  multi.txt

results/
  overwrite_v0/
  chain/
  chain2/
  chain3/
  multi_20/
```

Generated videos and training checkpoints are kept locally and ignored by Git.