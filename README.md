# VideoSeal Watermark Overwriting Study

Official implementation: https://github.com/facebookresearch/videoseal

Model: `videoseal_0.0` legacy checkpoint  
Payload: 96 bits

## Environment

- Apple M2 Pro, arm64
- Python 3.10.20
- PyTorch 2.4.0
- FFmpeg / ffprobe
- Official VideoSeal repository: `~/Projects/videoseal`

## Technical Route

```text
Official VideoSeal reproduction
→ sequential watermark overwriting experiment
→ direct chained-token experiments
→ pointer-based provenance reconstruction
→ slot-based multi-watermark training prototype
```

## 1. Official Reproduction

The official streaming inference pipeline was first reproduced with the VideoSeal repository.

```bash
python inference_streaming.py \
  --input assets/videos/1.mp4 \
  --output_dir outputs/
```

The legacy 96-bit checkpoint was loaded with:

```python
videoseal.load("videoseal_0.0")
```

The output message file contained 96 bits.

## 2. Sequential Watermark Overwriting

Script:

```text
scripts/overwrite_experiment_v0.py
```

Two fixed complementary 96-bit messages were used:

```text
A + B = 1 at every bit position
```

This makes every decoded bit attributable to either A or B.

Experimental conditions:

```text
A_only
B_only
A_transcode
A_then_A
A_then_B
B_then_A
```

Results:

```text
A_only:
A = 96 / 96
B = 0 / 96

B_only:
A = 0 / 96
B = 96 / 96

A_transcode:
A = 96 / 96
B = 0 / 96

A_then_A:
A = 96 / 96
B = 0 / 96

A_then_B:
A = 26 / 96
B = 70 / 96

B_then_A:
A = 72 / 96
B = 24 / 96
```

Interpretation:

```text
H.264 transcoding alone did not weaken the first watermark.
Embedding the same watermark twice also preserved it.
When a different second watermark was embedded, the decoded message shifted toward the later watermark.
```

Results:

```text
results/overwrite_v0/
```

## 3. Direct Chained-Token Experiments

### chain.py

Script:

```text
scripts/chain.py
```

Token format:

```text
current ID | parent ID | step
```

Each logical bit was repeated eight times to form a 96-bit payload.

Results:

```text
A:        recovered correctly
A -> B:   recovered correctly
A -> B -> C:
current ID = 3
parent ID = 0
step = 3
```

The expected parent ID for C was 2. The parent reference was decoded incorrectly, so the chain could not be verified.

### chain2.py

Script:

```text
scripts/chain2.py
```

Changes:

```text
current ID: 3 bits × 8 repeats
parent ID:  3 bits × 16 repeats
XOR check:  3 bits × 8 repeats
```

Results:

```text
A -> B -> C:
current ID = 3
parent ID = 0
```

Increasing redundancy for the parent field did not correct the three-step parent-ID error.

Results:

```text
results/chain/
results/chain2/
```

## 4. Pointer-Based Provenance Chain

Script:

```text
scripts/chain3.py
```

Instead of storing current ID, parent ID, and step separately inside the watermark, each record is represented by one complete 96-bit codeword.

```text
Record 1 = A
Record 2 = B
Record 3 = C
```

The parent relation is stored in an external registry:

```text
3 -> 2 -> 1
C -> B -> A
```

The final decoded codeword is classified as the closest record codeword, and the registry reconstructs the processing history.

Results:

```text
A:                classified as A
A -> B:           classified as B
A -> B -> C:      classified as C
A -> B -> C + H.264: classified as C
```

Recovered provenance chain:

```text
A -> B -> C
```

This prototype reconstructs provenance through the latest record ID and an external registry. It does not preserve A, B, and C as independently decodable watermarks in the final video.

Results:

```text
results/chain3/
```

## 5. Slot-Based Multi-Watermark Training Prototype

Script:

```text
scripts/multi.py
```

The 96-bit payload is divided into three ordered 32-bit slots:

```text
Slot 1: A
Slot 2: B
Slot 3: C
```

## 6. Week 2: Prior-Watermark Augmentation

### Research question

Can a fresh watermark B be embedded and decoded reliably when the input video already contains an earlier watermark A, while preserving acceptable visual quality? The target is the complete 96-bit VideoSeal payload.

### Method

`scripts/train_prior_watermark_aug_v0.py` freezes the original VideoSeal v0.0 embedder used to create A, freezes the detector, and trains only the fresh-B embedder. Training inputs contain A with probability 1.0 and use complementary random A/B messages rather than a fixed pair. The loss combines video-level signed-margin, hardest-bit, frame BCE, and a PSNR-budget term. `fresh_strength=2.0` is retained. Checkpoints are selected on 20 held-out random message pairs by complete-96-bit success first, then accuracy and PSNR.

### Final reported evaluation

The selected step-75 checkpoint was evaluated on a fixed 20-pair unseen-message bank (five sampled frames; B is the complement of A):

| Model | Complete 96/96 | Mean bit accuracy | Minimum bit accuracy | Mean PSNR vs A |
|---|---:|---:|---:|---:|
| Step 0 baseline | 2/20 (10.0%) | 96.824% | 91.67% | 47.298 dB |
| Step 75 checkpoint | 3/20 (15.0%) | 97.814% | 93.75% | 46.417 dB |

The separate checkpoint-selection bank used during the 50→100 resume run chose step 75 with 7/20 complete payloads, 98.283% mean accuracy, 91.67% minimum accuracy, and 46.387 dB mean PSNR. These are different message banks and must not be treated as the same test result.

### Honest conclusion

Prior-watermark augmentation improves average held-out bit accuracy and the complete-payload rate modestly, but does not yet achieve reliable 96/96 decoding on all unseen pairs. The PSNR decrease is measurable. This is a completed reproduction/prototype result, not evidence of a solved overwriting problem.

PixelSeal support is implemented in `scripts/pixelseal_overwrite_experiment.py`, but the official checkpoint download/validation is pending; no PixelSeal experimental result is claimed here. RivaGAN remains a planned baseline.

Detailed methods and limitations: [docs/week2_experiments.md](docs/week2_experiments.md). Environment: [docs/environment.md](docs/environment.md).

Training targets:

```text
Stage 1: [A, empty, empty]
Stage 2: [A, B, empty]
Stage 3: [A, B, C]
```

Training sequence:

```text
original video
→ embed [A, empty, empty]
→ embed [A, B, empty]
→ embed [A, B, C]
```

The local training pipeline completed forward propagation, backpropagation, optimisation, and checkpoint saving.

Configuration of the first run:

```text
Input: official example video
Training steps: 20
Frames per clip: 16
Resolution: 256 × 256
Device: Apple MPS
```

Results after 20 steps:

```text
A:   36 / 96
AB:  80 / 96
ABC: 84 / 96
```

For the `ABC` condition:

```text
Slot 1 presence: 0 / 1
Slot 2 presence: 1 / 1
Slot 3 presence: 1 / 1
```

The three active slots were not recovered reliably. The training code is operational, but the current one-video, 20-step configuration does not achieve stable multi-watermark coexistence.

Results:

```text
results/multi_20/
results/multi_20_test/
```

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

Generated videos and training checkpoints are kept locally and excluded from Git.
