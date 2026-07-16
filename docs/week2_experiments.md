# Week 2 experiments: prior-watermark augmentation

## Goal

The experiment tests whether a new 96-bit watermark B can be embedded into a
video that already contains a previous 96-bit watermark A. A successful method
must decode the complete B payload, not merely achieve a high average bit
accuracy, and should avoid severe visual degradation.

## Design

- Base model: official VideoSeal v0.0, 96-bit payload.
- A writer and detector: frozen.
- Trainable component: fresh-watermark embedder only.
- Prior augmentation: A is present with probability 1.0; conflict probability
  is 1.0.
- Messages: random A and complementary B (`B = 1 - A`); training uses random
  pairs (`fixed_training_pair=false`).
- Loss: signed video-level B margin + hardest-k-bit loss (`k=8`) + frame BCE +
  squared excess over the target-PSNR MSE budget.
- `fresh_strength=2.0`, target PSNR 40 dB, gradient clipping 1.0.
- Training uses 16 frames; evaluation uses 5 frames.

## Commands

Initial 50-step run:

```bash
/opt/anaconda3/envs/videoseal/bin/python scripts/train_prior_watermark_aug_v0.py \
  --input /Users/kayzhong/Projects/videoseal/assets/videos/1.mp4 \
  --official_repo /Users/kayzhong/Projects/videoseal \
  --output_dir results/prior_aug_three_changes_50 \
  --device auto --steps 50 --batch_size 4 --eval_batch_size 4 \
  --learning_rate 1e-6 --lambda_image 10 --prior_probability 1.0 \
  --max_frames 48 --sample_fps 3 --fresh_strength 2.0
```

Resume from step 50 through step 100:

```bash
/opt/anaconda3/envs/videoseal/bin/python scripts/train_prior_watermark_aug_v0.py \
  --input /Users/kayzhong/Projects/videoseal/assets/videos/1.mp4 \
  --official_repo /Users/kayzhong/Projects/videoseal \
  --output_dir results/prior_aug_three_changes_resume_50_to_100 \
  --resume_checkpoint results/prior_aug_three_changes_50/checkpoint_prior_aug_96bit.pth \
  --device auto --steps 100 --batch_size 4 --eval_batch_size 4 \
  --learning_rate 1e-6 --lambda_image 10 --prior_probability 1.0 \
  --max_frames 48 --sample_fps 3 --fresh_strength 2.0
```

Evaluation of the selected checkpoint on the same 20 unseen pairs:

```bash
/opt/anaconda3/envs/videoseal/bin/python scripts/train_prior_watermark_aug_v0.py \
  --input /Users/kayzhong/Projects/videoseal/assets/videos/1.mp4 \
  --official_repo /Users/kayzhong/Projects/videoseal \
  --evaluation_only_checkpoint results/prior_aug_three_changes_resume_50_to_100/checkpoint_best_prior_aug_96bit.pth \
  --output_dir results/prior_aug_step75_best_eval_5frames \
  --device auto --eval_message_pairs 20 --eval_frames 5 --fresh_strength 2.0
```

## Checkpoint selection

Selection is performed on 20 held-out random message pairs with seed
`2002029`. The primary score is the number of complete 96/96 payloads; ties are
broken by minimum/mean bit accuracy and PSNR. The resume run selected step 75.
Its selection-bank score was 7/20 complete, 98.283% mean accuracy, 91.67%
minimum accuracy, and 46.387 dB mean PSNR.

## Unseen-message test (20 pairs)

The independently saved evaluation at
`results/prior_aug_step75_best_eval_5frames/results.json` uses seed `1002029`,
five frames, and a fixed 20-pair bank not used for training:

| Model | Complete 96/96 | Mean accuracy | Minimum accuracy | Mean PSNR |
|---|---:|---:|---:|---:|
| Step 0 | 2/20 (10.0%) | 96.824% | 91.67% | 47.298 dB |
| Step 75 | 3/20 (15.0%) | 97.814% | 93.75% | 46.417 dB |

The complete-payload rate is therefore improved but remains low. The lower
post-training PSNR is a quality trade-off, not a hidden success.

## Reproducibility files

- `results/prior_aug_three_changes_resume_50_to_100/results.json`
- `results/prior_aug_three_changes_resume_50_to_100/training_log.csv`
- `results/prior_aug_step75_best_eval_5frames/results.json`
- `results/prior_aug_step75_best_eval_5frames/training_log.csv` (if present)

The large checkpoint files are intentionally excluded from git.

## Official augmenter dependency

The training script calls `videoseal.load("videoseal_0.0")` and uses the
official dummy augmenter. It does not require a local modification to
`official_repo/videoseal/augmentation/augmenter.py`: the official
`get_dummy_augmenter()` already passes `masks={"kind": "none"}`, and the
official `get_mask_embedder()` already maps that kind to `NoMaskEmbedder`.
Consequently no patch is stored under `patches/`.

## Limitations and next steps

The test uses one source video, five-frame logit averaging, and 20 message
pairs. It demonstrates a modest prototype improvement, not generalization to
arbitrary videos or payloads. Next steps are (1) larger held-out video/message
sets, (2) a controlled PSNR/accuracy sweep, and (3) independent baselines such
as PixelSeal and a RivaGAN-derived pretrained inference path. PixelSeal is only
documented as a completed script with official-checkpoint validation pending;
no PixelSeal result is reported here.
