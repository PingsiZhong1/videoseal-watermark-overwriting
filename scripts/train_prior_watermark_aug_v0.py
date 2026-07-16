from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
import torch.nn.functional as F


PAYLOAD_BITS = 96


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune the 96-bit VideoSeal v0.0 embedder using "
            "prior-watermarked inputs."
        )
    )

    parser.add_argument("--input", type=Path, required=True)

    parser.add_argument(
        "--official_repo",
        type=Path,
        default=Path.home() / "Projects" / "videoseal",
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/prior_aug_v0"),
    )

    parser.add_argument(
        "--device",
        choices=["auto", "mps", "cuda", "cpu"],
        default="auto",
    )

    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Number of consecutive frames in each training clip.",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=32,
        help="Number of consecutive evaluation frames processed as one clip.",
    )

    parser.add_argument("--learning_rate", type=float, default=1e-7)
    parser.add_argument(
        "--lambda_image",
        type=float,
        default=0.25,
        help=(
            "Weight of distortion above the --target_psnr budget. "
            "No image penalty is applied while PSNR is above the target."
        ),
    )
    parser.add_argument(
        "--target_psnr",
        type=float,
        default=40.0,
        help="Target minimum PSNR of x_AB relative to its x_A input.",
    )
    parser.add_argument(
        "--decode_margin",
        type=float,
        default=1.0,
        help="Required signed margin for every bit in the pooled video logits.",
    )
    parser.add_argument(
        "--frame_loss_weight",
        type=float,
        default=0.25,
        help="Auxiliary per-frame BCE weight; video-level loss remains primary.",
    )
    parser.add_argument(
        "--hard_bit_k",
        type=int,
        default=8,
        help="Number of lowest-margin video bits emphasized by the hard-bit loss.",
    )
    parser.add_argument(
        "--hard_bit_weight",
        type=float,
        default=1.0,
        help="Weight of the hardest-bit loss.",
    )
    parser.add_argument(
        "--message_pairs_per_step",
        type=int,
        default=4,
        help=(
            "Random A/B pairs accumulated before each optimizer step. "
            "Pairs are processed sequentially to limit memory use."
        ),
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Clip embedder gradient norm to prevent destructive updates.",
    )
    parser.add_argument(
        "--fresh_strength",
        type=float,
        default=2.0,
        help=(
            "Multiplier applied only to the new-B writer strength. "
            "The frozen old-A writer keeps the official strength."
        ),
    )
    parser.add_argument(
        "--fixed_training_pair",
        action="store_true",
        help=(
            "Train on the fixed evaluation A/B payload pair while keeping "
            "training and evaluation frames disjoint. Use only when the "
            "objective is reliable decoding of one designated fresh B."
        ),
    )
    parser.add_argument(
        "--evaluation_only_checkpoint",
        type=Path,
        default=None,
        help=(
            "Do not train. Compare step 0 with this trained checkpoint on "
            "held-out random message pairs."
        ),
    )
    parser.add_argument(
        "--resume_checkpoint",
        type=Path,
        default=None,
        help=(
            "Resume both model and optimizer state; --steps is the number "
            "of additional optimizer steps."
        ),
    )
    parser.add_argument(
        "--eval_message_pairs",
        type=int,
        default=20,
        help="Number of unseen complementary A/B pairs for evaluation (>=20).",
    )

    parser.add_argument(
        "--prior_probability",
        type=float,
        default=1.0,
        help="Probability that a training batch already contains watermark A.",
    )

    parser.add_argument(
        "--full_conflict_probability",
        type=float,
        default=1.0,
        help=(
            "For prior-watermarked batches, probability that B is the "
            "bitwise complement of A. Other batches use an independent B."
        ),
    )

    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--max_frames", type=int, default=96)
    parser.add_argument("--sample_fps", type=float, default=2.0)
    parser.add_argument(
        "--sample_start_time",
        type=float,
        default=0.0,
        help="Start offset in seconds used before FPS frame sampling.",
    )
    parser.add_argument("--eval_fraction", type=float, default=0.25)

    parser.add_argument("--log_every", type=int, default=5)
    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)

    return parser.parse_args()


def select_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")

        if torch.backends.mps.is_available():
            return torch.device("mps")

        return torch.device("cpu")

    device = torch.device(name)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")

    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable.")

    return device


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def require_binary(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"{name} was not found in PATH.")


def extract_frames(
    video_path: Path,
    image_size: int,
    max_frames: int,
    sample_fps: float,
    sample_start_time: float,
) -> torch.Tensor:
    """
    Extract frames with FFmpeg and resize them to image_size x image_size.

    Returns:
        Tensor with shape [frames, 3, H, W], range [0, 1].
    """

    filter_graph = (
        f"fps={sample_fps},"
        f"scale={image_size}:{image_size}:"
        "force_original_aspect_ratio=decrease,"
        f"pad={image_size}:{image_size}:(ow-iw)/2:(oh-ih)/2"
    )

    command = [
        "ffmpeg",
        "-v",
        "error",
        "-ss",
        str(sample_start_time),
        "-i",
        str(video_path),
        "-an",
        "-vf",
        filter_graph,
        "-frames:v",
        str(max_frames),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]

    process = subprocess.run(
        command,
        check=True,
        capture_output=True,
    )

    bytes_per_frame = image_size * image_size * 3

    if len(process.stdout) % bytes_per_frame != 0:
        raise RuntimeError("FFmpeg returned an incomplete frame.")

    frame_count = len(process.stdout) // bytes_per_frame

    if frame_count < 4:
        raise RuntimeError(
            f"Only {frame_count} frames were extracted. "
            "At least 4 frames are required."
        )

    frames = np.frombuffer(
        process.stdout,
        dtype=np.uint8,
    )

    frames = frames.reshape(
        frame_count,
        image_size,
        image_size,
        3,
    ).copy()

    frames = (
        torch.from_numpy(frames)
        .permute(0, 3, 1, 2)
        .float()
        .div(255.0)
    )

    return frames


def random_message(device: torch.device) -> torch.Tensor:
    return torch.randint(
        low=0,
        high=2,
        size=(1, PAYLOAD_BITS),
        device=device,
        dtype=torch.float32,
    )


def get_message_capacity(model: Any) -> int:
    message = model.get_random_msg(1)

    if message.ndim != 2:
        raise RuntimeError(
            f"Unexpected message shape: {message.shape}"
        )

    return int(message.shape[1])


def embed_frames(
    model: Any,
    frames: torch.Tensor,
    message: torch.Tensor,
    embedder: torch.nn.Module | None = None,
    blender: torch.nn.Module | None = None,
) -> torch.Tensor:
    """
    Differentiable watermark embedding.

    For the frozen prior writer, embedder and blender are supplied explicitly.
    For the trainable model, the current model modules are used.
    """

    if embedder is None:
        embedder = model.embedder

    if blender is None:
        blender = model.blender

    processing_size = int(model.img_size)

    if frames.shape[-2:] != (processing_size, processing_size):
        resized_frames = F.interpolate(
            frames,
            size=(processing_size, processing_size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
    else:
        resized_frames = frames

    step_size = int(model.step_size)
    key_frames = resized_frames[::step_size]
    key_messages = message.expand(
        key_frames.shape[0],
        -1,
    )

    if embedder.yuv:
        embedder_input = model.rgb2yuv(
            key_frames
        )[:, 0:1]
    else:
        embedder_input = key_frames

    key_watermarks = embedder(
        embedder_input,
        key_messages,
    )

    predicted_watermark = model._apply_video_mode(
        key_watermarks,
        frames.shape[0],
        step_size,
        model.video_mode,
    )

    if predicted_watermark.shape[-2:] != frames.shape[-2:]:
        predicted_watermark = F.interpolate(
            predicted_watermark,
            size=frames.shape[-2:],
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )

    # VideoSeal v0.0 normally has no JND attenuation.
    # This keeps compatibility in case attenuation exists.
    if model.attenuation is not None:
        heatmaps = model.attenuation.heatmaps(frames)
        predicted_watermark = (
            heatmaps * predicted_watermark
        )

    watermarked_frames = blender(
        frames,
        predicted_watermark,
    )

    return watermarked_frames.clamp(0.0, 1.0)


def detector_logits(
    model: Any,
    frames: torch.Tensor,
) -> torch.Tensor:
    """
    Run the detector directly.

    We do not call model.detect() because it uses inference mode and would
    stop gradients from reaching the trainable embedder.
    """

    processing_size = int(model.img_size)

    if frames.shape[-2:] != (processing_size, processing_size):
        frames = F.interpolate(
            frames,
            size=(processing_size, processing_size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )

    predictions = model.detector(frames)

    if predictions.ndim == 4:
        return predictions[:, 1:]

    if predictions.ndim == 2:
        return predictions[:, 1:]

    raise RuntimeError(
        f"Unsupported detector output shape: {predictions.shape}"
    )


def calculate_decoding_loss(
    bit_logits: torch.Tensor,
    message: torch.Tensor,
) -> torch.Tensor:
    target = message.expand(
        bit_logits.shape[0],
        -1,
    )

    if bit_logits.ndim == 4:
        target = target[:, :, None, None]
        target = target.expand_as(bit_logits)

    return F.binary_cross_entropy_with_logits(
        bit_logits,
        target,
    )


def calculate_video_decoding_loss(
    bit_logits: torch.Tensor,
    message: torch.Tensor,
    margin: float,
    frame_loss_weight: float,
    hard_bit_k: int,
    hard_bit_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Optimize the same video-pooled 96-bit decision used in evaluation."""

    frame_logits = pool_logits(bit_logits)
    video_logits = frame_logits.mean(dim=0, keepdim=True)
    signed_targets = 2.0 * message - 1.0

    # Softplus keeps gradients on every bit while concentrating them on bits
    # that are wrong or do not yet have the requested decoding margin.
    per_bit_video_loss = F.softplus(
        margin - signed_targets * video_logits
    )
    loss_video = per_bit_video_loss.mean()
    loss_hard_bits = per_bit_video_loss.topk(
        k=min(hard_bit_k, per_bit_video_loss.shape[1]),
        dim=1,
    ).values.mean()
    loss_frame = calculate_decoding_loss(
        frame_logits,
        message,
    )
    loss = (
        loss_video
        + hard_bit_weight * loss_hard_bits
        + frame_loss_weight * loss_frame
    )
    return loss, loss_video, loss_frame, loss_hard_bits


def pool_logits(bit_logits: torch.Tensor) -> torch.Tensor:
    if bit_logits.ndim == 4:
        return bit_logits.mean(dim=(2, 3))

    return bit_logits


def message_metrics_from_logits(
    frame_logits: torch.Tensor,
    target_message: torch.Tensor,
) -> dict[str, Any]:
    """
    Calculate frame-level and video-level aggregate decoding accuracy.
    """

    target_per_frame = (
        target_message
        .to(torch.int64)
        .expand(frame_logits.shape[0], -1)
    )

    decoded_per_frame = (
        frame_logits > 0
    ).to(torch.int64)

    frame_matches = decoded_per_frame.eq(
        target_per_frame
    )

    aggregate_logits = frame_logits.mean(
        dim=0,
        keepdim=True,
    )

    aggregate_decoded = (
        aggregate_logits > 0
    ).to(torch.int64)

    aggregate_target = target_message.to(
        torch.int64
    )

    aggregate_matches = aggregate_decoded.eq(
        aggregate_target
    )

    return {
        "frame_bit_accuracy_pct": round(
            100.0
            * frame_matches.float().mean().item(),
            2,
        ),
        "aggregate_correct_bits": int(
            aggregate_matches.sum().item()
        ),
        "aggregate_total_bits": PAYLOAD_BITS,
        "aggregate_bit_accuracy_pct": round(
            100.0
            * aggregate_matches.float().mean().item(),
            2,
        ),
        "complete_96bit": bool(
            aggregate_matches.all().item()
        ),
    }


def calculate_psnr(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
) -> float:
    mse = F.mse_loss(
        reconstructed,
        original,
    ).item()

    if mse == 0:
        return float("inf")

    return float(
        10.0 * np.log10(1.0 / mse)
    )


@torch.inference_mode()
def evaluate_a_then_b(
    model: Any,
    prior_embedder: torch.nn.Module,
    prior_blender: torch.nn.Module,
    eval_frames_cpu: torch.Tensor,
    device: torch.device,
    message_a: torch.Tensor,
    message_b: torch.Tensor,
    eval_batch_size: int,
) -> dict[str, Any]:
    """
    Evaluation design:

        x -> A
        x_A -> B
        decode x_A and x_AB

    B is the complement of A so every decoded bit can be attributed
    unambiguously to A or B.
    """


    a_only_logits_list = []
    b_only_logits_list = []
    final_logits_list = []

    mse_sum = 0.0
    frame_count = 0

    for start in range(
        0,
        eval_frames_cpu.shape[0],
        eval_batch_size,
    ):
        clean_frames = eval_frames_cpu[
            start:start + eval_batch_size
        ].to(device)

        a_only_frames = embed_frames(
            model=model,
            frames=clean_frames,
            message=message_a,
            embedder=prior_embedder,
            blender=prior_blender,
        )

        a_then_b_frames = embed_frames(
            model=model,
            frames=a_only_frames,
            message=message_b,
        )

        b_only_frames = embed_frames(
            model=model,
            frames=clean_frames,
            message=message_b,
        )

        a_only_logits = pool_logits(
            detector_logits(
                model,
                a_only_frames,
            )
        )

        final_logits = pool_logits(
            detector_logits(
                model,
                a_then_b_frames,
            )
        )

        b_only_logits = pool_logits(
            detector_logits(
                model,
                b_only_frames,
            )
        )

        a_only_logits_list.append(
            a_only_logits.cpu()
        )

        final_logits_list.append(
            final_logits.cpu()
        )

        b_only_logits_list.append(
            b_only_logits.cpu()
        )

        batch_mse = F.mse_loss(
            a_then_b_frames,
            a_only_frames,
            reduction="mean",
        ).item()

        mse_sum += (
            batch_mse * clean_frames.shape[0]
        )

        frame_count += clean_frames.shape[0]

    a_only_logits = torch.cat(
        a_only_logits_list,
        dim=0,
    )

    final_logits = torch.cat(
        final_logits_list,
        dim=0,
    )

    b_only_logits = torch.cat(
        b_only_logits_list,
        dim=0,
    )

    message_a_cpu = message_a.cpu()
    message_b_cpu = message_b.cpu()

    average_mse = mse_sum / frame_count

    if average_mse == 0:
        psnr_value = float("inf")
    else:
        psnr_value = float(
            10.0 * np.log10(
                1.0 / average_mse
            )
        )

    return {
        "payload_bits": PAYLOAD_BITS,

        "A_only_decoded_as_A":
            message_metrics_from_logits(
                a_only_logits,
                message_a_cpu,
            ),

        "B_only_decoded_as_B":
            message_metrics_from_logits(
                b_only_logits,
                message_b_cpu,
            ),

        "A_then_B_decoded_as_old_A":
            message_metrics_from_logits(
                final_logits,
                message_a_cpu,
            ),

        "A_then_B_decoded_as_fresh_B":
            message_metrics_from_logits(
                final_logits,
                message_b_cpu,
            ),

        "fresh_watermark_PSNR_vs_A_input":
            round(psnr_value, 3),
    }


@torch.inference_mode()
def evaluate_unseen_message_pairs(
    model: Any,
    prior_embedder: torch.nn.Module,
    prior_blender: torch.nn.Module,
    eval_frames_cpu: torch.Tensor,
    device: torch.device,
    message_pairs: list[tuple[torch.Tensor, torch.Tensor]],
    eval_batch_size: int,
) -> dict[str, Any]:
    """Evaluate complementary random pairs that never enter training."""

    pair_results = []

    for pair_index, (message_a_cpu, message_b_cpu) in enumerate(
        message_pairs,
        start=1,
    ):
        result = evaluate_a_then_b(
            model=model,
            prior_embedder=prior_embedder,
            prior_blender=prior_blender,
            eval_frames_cpu=eval_frames_cpu,
            device=device,
            message_a=message_a_cpu.to(device),
            message_b=message_b_cpu.to(device),
            eval_batch_size=eval_batch_size,
        )
        fresh_b = result["A_then_B_decoded_as_fresh_B"]
        pair_results.append({
            "pair": pair_index,
            "fresh_B_correct_bits": fresh_b["aggregate_correct_bits"],
            "fresh_B_bit_accuracy_pct": fresh_b["aggregate_bit_accuracy_pct"],
            "fresh_B_complete_96bit": fresh_b["complete_96bit"],
            "PSNR_vs_A_input": result["fresh_watermark_PSNR_vs_A_input"],
        })

    accuracies = [
        item["fresh_B_bit_accuracy_pct"]
        for item in pair_results
    ]
    psnr_values = [
        item["PSNR_vs_A_input"]
        for item in pair_results
    ]

    return {
        "message_pair_count": len(pair_results),
        "complete_96bit_successes": sum(
            int(item["fresh_B_complete_96bit"])
            for item in pair_results
        ),
        "complete_96bit_success_rate_pct": round(
            100.0
            * sum(
                int(item["fresh_B_complete_96bit"])
                for item in pair_results
            )
            / len(pair_results),
            2,
        ),
        "mean_bit_accuracy_pct": round(float(np.mean(accuracies)), 3),
        "minimum_bit_accuracy_pct": round(float(np.min(accuracies)), 3),
        "mean_PSNR_vs_A_input": round(float(np.mean(psnr_values)), 3),
        "per_pair": pair_results,
    }


def save_training_log(
    output_dir: Path,
    rows: list[dict[str, Any]],
) -> None:
    if not rows:
        return

    output_path = (
        output_dir / "training_log.csv"
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)


def evaluation_score(results: dict[str, Any]) -> tuple[int, float, float]:
    """Rank checkpoints by fresh-B decoding first, then visual quality."""

    fresh_metrics = results["A_then_B_decoded_as_fresh_B"]

    return (
        int(fresh_metrics["aggregate_correct_bits"]),
        float(fresh_metrics["frame_bit_accuracy_pct"]),
        float(results["fresh_watermark_PSNR_vs_A_input"]),
    )


def multi_pair_evaluation_score(
    results: dict[str, Any],
) -> tuple[int, float, float, float]:
    """Rank by complete payloads, worst pair, mean accuracy, then PSNR."""

    return (
        int(results["complete_96bit_successes"]),
        float(results["minimum_bit_accuracy_pct"]),
        float(results["mean_bit_accuracy_pct"]),
        float(results["mean_PSNR_vs_A_input"]),
    )


def save_checkpoint(
    path: Path,
    model: Any,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    step: int,
    evaluation: dict[str, Any],
) -> None:
    checkpoint = {
        "model": model.state_dict(),
        "embedder": model.embedder.state_dict(),
        "optimizer": optimizer.state_dict(),
        "payload_bits": PAYLOAD_BITS,
        "step": step,
        "steps_requested": args.steps,
        "prior_probability": args.prior_probability,
        "full_conflict_probability": args.full_conflict_probability,
        "target_psnr": args.target_psnr,
        "decode_margin": args.decode_margin,
        "frame_loss_weight": args.frame_loss_weight,
        "hard_bit_k": args.hard_bit_k,
        "hard_bit_weight": args.hard_bit_weight,
        "message_pairs_per_step": args.message_pairs_per_step,
        "max_grad_norm": args.max_grad_norm,
        "fresh_strength": args.fresh_strength,
        "fixed_training_pair": args.fixed_training_pair,
        "evaluation": evaluation,
    }

    torch.save(checkpoint, path)


def main() -> None:
    args = parse_args()

    if not args.input.is_file():
        raise SystemExit(
            f"Input video not found: {args.input}"
        )

    if not args.official_repo.is_dir():
        raise SystemExit(
            "Official VideoSeal repository not found: "
            f"{args.official_repo}"
        )

    if args.steps <= 0:
        raise SystemExit(
            "--steps must be positive."
        )

    if args.batch_size <= 0:
        raise SystemExit(
            "--batch_size must be positive."
        )

    if args.eval_batch_size <= 0:
        raise SystemExit(
            "--eval_batch_size must be positive."
        )

    if args.log_every <= 0:
        raise SystemExit(
            "--log_every must be positive."
        )

    if args.eval_every <= 0:
        raise SystemExit(
            "--eval_every must be positive."
        )

    if args.learning_rate <= 0:
        raise SystemExit(
            "--learning_rate must be positive."
        )

    if args.lambda_image < 0:
        raise SystemExit(
            "--lambda_image must be non-negative."
        )

    if args.target_psnr <= 0:
        raise SystemExit(
            "--target_psnr must be positive."
        )

    if args.decode_margin < 0:
        raise SystemExit(
            "--decode_margin must be non-negative."
        )

    if args.frame_loss_weight < 0:
        raise SystemExit(
            "--frame_loss_weight must be non-negative."
        )

    if not 1 <= args.hard_bit_k <= PAYLOAD_BITS:
        raise SystemExit(
            f"--hard_bit_k must be between 1 and {PAYLOAD_BITS}."
        )

    if args.hard_bit_weight < 0:
        raise SystemExit(
            "--hard_bit_weight must be non-negative."
        )

    if args.message_pairs_per_step <= 0:
        raise SystemExit(
            "--message_pairs_per_step must be positive."
        )

    if args.max_grad_norm <= 0:
        raise SystemExit(
            "--max_grad_norm must be positive."
        )

    if args.fresh_strength <= 0:
        raise SystemExit(
            "--fresh_strength must be positive."
        )

    if args.eval_message_pairs < 20:
        raise SystemExit(
            "--eval_message_pairs must be at least 20."
        )

    if (
        args.evaluation_only_checkpoint is not None
        and not args.evaluation_only_checkpoint.is_file()
    ):
        raise SystemExit(
            "Evaluation checkpoint not found: "
            f"{args.evaluation_only_checkpoint}"
        )

    if args.resume_checkpoint is not None and not args.resume_checkpoint.is_file():
        raise SystemExit(
            f"Resume checkpoint not found: {args.resume_checkpoint}"
        )

    if (
        args.evaluation_only_checkpoint is not None
        and args.resume_checkpoint is not None
    ):
        raise SystemExit(
            "Use only one of --evaluation_only_checkpoint and --resume_checkpoint."
        )

    if args.image_size <= 0:
        raise SystemExit(
            "--image_size must be positive."
        )

    if args.max_frames < 4:
        raise SystemExit(
            "--max_frames must be at least 4."
        )

    if args.sample_fps <= 0:
        raise SystemExit(
            "--sample_fps must be positive."
        )

    if args.sample_start_time < 0:
        raise SystemExit(
            "--sample_start_time must be non-negative."
        )

    if not 0.0 <= args.prior_probability <= 1.0:
        raise SystemExit(
            "--prior_probability must be between 0 and 1."
        )

    if not 0.0 <= args.full_conflict_probability <= 1.0:
        raise SystemExit(
            "--full_conflict_probability must be between 0 and 1."
        )

    if not 0.0 < args.eval_fraction < 1.0:
        raise SystemExit(
            "--eval_fraction must be between 0 and 1."
        )

    require_binary("ffmpeg")
    set_seed(args.seed)

    device = select_device(args.device)

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"Device: {device}")
    print("Extracting frames...")

    all_frames = extract_frames(
        video_path=args.input,
        image_size=args.image_size,
        max_frames=args.max_frames,
        sample_fps=args.sample_fps,
        sample_start_time=args.sample_start_time,
    )

    eval_count = max(
        1,
        int(
            round(
                all_frames.shape[0]
                * args.eval_fraction
            )
        ),
    )

    eval_count = min(
        eval_count,
        all_frames.shape[0] - 1,
    )

    eval_frames = all_frames[:eval_count]
    train_frames = all_frames[eval_count:]

    if train_frames.shape[0] < args.batch_size:
        raise SystemExit(
            "Not enough training frames for one consecutive clip: "
            f"have {train_frames.shape[0]}, need --batch_size={args.batch_size}. "
            "Increase --max_frames, increase --sample_fps, or reduce --batch_size."
        )

    print(
        f"Training frames: {train_frames.shape[0]}"
    )

    print(
        f"Evaluation frames: {eval_frames.shape[0]}"
    )

    sys.path.insert(
        0,
        str(args.official_repo.resolve()),
    )

    import videoseal

    print("Loading VideoSeal v0.0...")

    model = videoseal.load(
        "videoseal_0.0"
    ).to(device)

    capacity = get_message_capacity(model)

    if capacity != PAYLOAD_BITS:
        raise RuntimeError(
            f"Loaded model has {capacity} bits. "
            f"Expected {PAYLOAD_BITS} bits."
        )

    print(
        f"Confirmed payload: {capacity} bits"
    )

    # Frozen original embedder:
    # generates the previous watermark A.
    prior_embedder = copy.deepcopy(
        model.embedder
    ).to(device)

    prior_blender = copy.deepcopy(
        model.blender
    ).to(device)

    prior_embedder.eval()
    prior_blender.eval()

    prior_embedder.requires_grad_(False)
    prior_blender.requires_grad_(False)

    # Equal-strength complementary A/B residuals tend to cancel. Keep the
    # historical A writer unchanged and give the fresh B writer enough
    # residual budget to overwrite A while the PSNR budget limits distortion.
    model.blender.scaling_w *= args.fresh_strength

    if args.evaluation_only_checkpoint is not None:
        if args.fixed_training_pair:
            raise SystemExit(
                "--fixed_training_pair is forbidden in evaluation-only mode."
            )

        checkpoint = torch.load(
            args.evaluation_only_checkpoint,
            map_location="cpu",
        )

        if checkpoint.get("fixed_training_pair", False):
            raise SystemExit(
                "Refusing checkpoint trained with fixed_training_pair=true."
            )

        checkpoint_strength = float(
            checkpoint.get("fresh_strength", 1.0)
        )
        if not np.isclose(checkpoint_strength, args.fresh_strength):
            raise SystemExit(
                "Checkpoint fresh_strength does not match evaluation: "
                f"checkpoint={checkpoint_strength}, "
                f"requested={args.fresh_strength}."
            )

        # Generate on CPU with a separate deterministic generator. These
        # messages are created only in this evaluation-only branch and are
        # reused unchanged for step 0 and the trained checkpoint.
        evaluation_seed = args.seed + 1_000_003
        generator = torch.Generator(device="cpu")
        generator.manual_seed(evaluation_seed)
        message_pairs = []
        for _ in range(args.eval_message_pairs):
            message_a = torch.randint(
                0,
                2,
                (1, PAYLOAD_BITS),
                generator=generator,
                dtype=torch.float32,
            )
            message_pairs.append((message_a, 1.0 - message_a))

        model.requires_grad_(False)
        model.eval()
        print("\nEvaluating unseen pairs at step 0...")
        baseline_multi = evaluate_unseen_message_pairs(
            model=model,
            prior_embedder=prior_embedder,
            prior_blender=prior_blender,
            eval_frames_cpu=eval_frames,
            device=device,
            message_pairs=message_pairs,
            eval_batch_size=args.eval_batch_size,
        )

        model.load_state_dict(checkpoint["model"], strict=True)
        model.requires_grad_(False)
        model.eval()
        print("Evaluating the trained checkpoint on the same unseen pairs...")
        after_multi = evaluate_unseen_message_pairs(
            model=model,
            prior_embedder=prior_embedder,
            prior_blender=prior_blender,
            eval_frames_cpu=eval_frames,
            device=device,
            message_pairs=message_pairs,
            eval_batch_size=args.eval_batch_size,
        )

        evaluation_results = {
            "experiment": "unseen random 96-bit A-then-B evaluation",
            "training_performed": False,
            "payload_bits": PAYLOAD_BITS,
            "message_pair_design": "random A; B is the bitwise complement of A",
            "messages_used_in_training": False,
            "message_pair_count": args.eval_message_pairs,
            "evaluation_seed": evaluation_seed,
            "evaluation_frames": int(eval_frames.shape[0]),
            "sample_fps": args.sample_fps,
            "sample_start_time": args.sample_start_time,
            "fresh_strength": args.fresh_strength,
            "checkpoint": {
                "path": str(args.evaluation_only_checkpoint),
                "step": int(checkpoint.get("step", -1)),
                "fixed_training_pair": False,
            },
            "step_0_baseline": baseline_multi,
            "after_training_checkpoint": after_multi,
        }

        output_path = args.output_dir / "results.json"
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(evaluation_results, file, indent=2)

        print(json.dumps(evaluation_results, indent=2))
        print(f"\nSaved evaluation to: {output_path.resolve()}")
        return

    # Freeze the full candidate model first.
    model.requires_grad_(False)

    # Only fine-tune the new embedder.
    model.embedder.requires_grad_(True)
    model.embedder.train()

    # Detector remains frozen.
    model.detector.eval()

    optimizer = torch.optim.AdamW(
        model.embedder.parameters(),
        lr=args.learning_rate,
        weight_decay=0.0,
    )

    start_step = 0
    if args.resume_checkpoint is not None:
        resume_checkpoint = torch.load(
            args.resume_checkpoint,
            map_location="cpu",
        )
        if resume_checkpoint.get("fixed_training_pair", False):
            raise SystemExit(
                "Refusing to resume a fixed_training_pair checkpoint."
            )
        resume_strength = float(
            resume_checkpoint.get("fresh_strength", 1.0)
        )
        if not np.isclose(resume_strength, args.fresh_strength):
            raise SystemExit(
                "Resume checkpoint fresh_strength does not match: "
                f"checkpoint={resume_strength}, requested={args.fresh_strength}."
            )
        model.load_state_dict(resume_checkpoint["model"], strict=True)
        optimizer.load_state_dict(resume_checkpoint["optimizer"])
        start_step = int(resume_checkpoint["step"])
        print(
            f"Resumed model and optimizer from step {start_step}: "
            f"{args.resume_checkpoint}"
        )

    final_step = start_step + args.steps

    # Use exactly the same 96-bit messages before and after training.
    fixed_eval_message_a = random_message(device)
    fixed_eval_message_b = 1.0 - fixed_eval_message_a

    # This deterministic held-out bank is generated from an isolated CPU RNG
    # and is never sampled by the training loop. It is used only for fair
    # checkpoint selection across steps.
    selection_seed = args.seed + 2_000_003
    selection_generator = torch.Generator(device="cpu")
    selection_generator.manual_seed(selection_seed)
    selection_message_pairs = []
    for _ in range(args.eval_message_pairs):
        selection_a = torch.randint(
            0,
            2,
            (1, PAYLOAD_BITS),
            generator=selection_generator,
            dtype=torch.float32,
        )
        selection_message_pairs.append((selection_a, 1.0 - selection_a))

    print("\nRunning baseline evaluation...")

    model.embedder.eval()

    baseline_results = evaluate_a_then_b(
        model=model,
        prior_embedder=prior_embedder,
        prior_blender=prior_blender,
        eval_frames_cpu=eval_frames,
        device=device,
        message_a=fixed_eval_message_a,
        message_b=fixed_eval_message_b,
        eval_batch_size=args.eval_batch_size,
    )

    print(
        json.dumps(
            baseline_results,
            indent=2,
        )
    )

    print("\nRunning held-out multi-message baseline evaluation...")
    baseline_selection_results = evaluate_unseen_message_pairs(
        model=model,
        prior_embedder=prior_embedder,
        prior_blender=prior_blender,
        eval_frames_cpu=eval_frames,
        device=device,
        message_pairs=selection_message_pairs,
        eval_batch_size=args.eval_batch_size,
    )
    print(json.dumps({
        key: value
        for key, value in baseline_selection_results.items()
        if key != "per_pair"
    }, indent=2))

    evaluation_history = [
        {
            "step": start_step,
            "results": baseline_selection_results,
        }
    ]
    best_score = multi_pair_evaluation_score(
        baseline_selection_results
    )
    best_step = start_step
    best_checkpoint_path = (
        args.output_dir / "checkpoint_best_prior_aug_96bit.pth"
    )

    save_checkpoint(
        path=best_checkpoint_path,
        model=model,
        optimizer=optimizer,
        args=args,
        step=0,
        evaluation=baseline_selection_results,
    )

    model.embedder.train()

    training_rows = []

    print("\nStarting training...")

    for step in range(
        start_step + 1,
        final_step + 1,
    ):
        clip_start = int(torch.randint(
            low=0,
            high=train_frames.shape[0] - args.batch_size + 1,
            size=(1,),
        ).item())

        clean_batch = train_frames[
            clip_start:clip_start + args.batch_size
        ].to(device)

        optimizer.zero_grad(
            set_to_none=True
        )

        accumulated = {
            "loss_total": 0.0,
            "loss_decode": 0.0,
            "loss_decode_video": 0.0,
            "loss_decode_frame": 0.0,
            "loss_hard_bits": 0.0,
            "loss_image": 0.0,
            "image_mse": 0.0,
            "normalized_image_mse": 0.0,
            "PSNR": 0.0,
        }
        prior_pair_count = 0
        conflict_pair_count = 0
        pair_correct_bits = []
        pair_frame_accuracies = []
        pair_complete_count = 0

        for _ in range(args.message_pairs_per_step):
            use_prior_watermark = (
                random.random() < args.prior_probability
            )
            used_full_conflict = False

            if use_prior_watermark:
                prior_pair_count += 1
                if args.fixed_training_pair:
                    message_a = fixed_eval_message_a
                    message_b = fixed_eval_message_b
                    used_full_conflict = True
                else:
                    message_a = random_message(device)
                    used_full_conflict = (
                        random.random()
                        < args.full_conflict_probability
                    )
                    message_b = (
                        1.0 - message_a
                        if used_full_conflict
                        else random_message(device)
                    )

                conflict_pair_count += int(used_full_conflict)
                with torch.no_grad():
                    second_embedder_input = embed_frames(
                        model=model,
                        frames=clean_batch,
                        message=message_a,
                        embedder=prior_embedder,
                        blender=prior_blender,
                    ).detach()
            else:
                message_b = random_message(device)
                second_embedder_input = clean_batch

            # Match the inference path exactly and backpropagate each message
            # sequentially so multiple messages do not multiply peak memory.
            final_frames = embed_frames(
                model=model,
                frames=second_embedder_input,
                message=message_b,
            )
            bit_logits = detector_logits(model, final_frames)

            (
                loss_decode,
                loss_decode_video,
                loss_decode_frame,
                loss_hard_bits,
            ) = calculate_video_decoding_loss(
                bit_logits,
                message_b,
                margin=args.decode_margin,
                frame_loss_weight=args.frame_loss_weight,
                hard_bit_k=args.hard_bit_k,
                hard_bit_weight=args.hard_bit_weight,
            )

            image_mse = F.mse_loss(
                final_frames,
                second_embedder_input,
            )
            target_mse = 10.0 ** (-args.target_psnr / 10.0)
            normalized_mse = image_mse / target_mse
            loss_image = F.relu(normalized_mse - 1.0).square()
            total_loss = loss_decode + args.lambda_image * loss_image

            (total_loss / args.message_pairs_per_step).backward()

            pooled = pool_logits(bit_logits.detach())
            pair_metrics = message_metrics_from_logits(pooled, message_b)
            pair_correct_bits.append(pair_metrics["aggregate_correct_bits"])
            pair_frame_accuracies.append(
                pair_metrics["frame_bit_accuracy_pct"]
            )
            pair_complete_count += int(pair_metrics["complete_96bit"])

            accumulated["loss_total"] += float(total_loss.item())
            accumulated["loss_decode"] += float(loss_decode.item())
            accumulated["loss_decode_video"] += float(
                loss_decode_video.item()
            )
            accumulated["loss_decode_frame"] += float(
                loss_decode_frame.item()
            )
            accumulated["loss_hard_bits"] += float(
                loss_hard_bits.item()
            )
            accumulated["loss_image"] += float(loss_image.item())
            accumulated["image_mse"] += float(image_mse.item())
            accumulated["normalized_image_mse"] += float(
                normalized_mse.item()
            )
            accumulated["PSNR"] += calculate_psnr(
                second_embedder_input,
                final_frames.detach(),
            )

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.embedder.parameters(),
            max_norm=args.max_grad_norm,
        )
        optimizer.step()

        pair_count = args.message_pairs_per_step
        averaged = {
            key: value / pair_count
            for key, value in accumulated.items()
        }

        row = {
            "step": step,
            "message_pairs": pair_count,
            "prior_pairs": prior_pair_count,
            "full_conflict_pairs": conflict_pair_count,
            "loss_total": averaged["loss_total"],
            "loss_decode": averaged["loss_decode"],
            "loss_decode_video": averaged["loss_decode_video"],
            "loss_decode_frame": averaged["loss_decode_frame"],
            "loss_hard_bits": averaged["loss_hard_bits"],
            "loss_image": averaged["loss_image"],
            "image_mse": averaged["image_mse"],
            "normalized_image_mse": averaged["normalized_image_mse"],
            "batch_PSNR_vs_input": averaged["PSNR"],
            "gradient_norm_before_clip": float(grad_norm.item()),
            "fresh_B_mean_frame_accuracy_pct": float(
                np.mean(pair_frame_accuracies)
            ),
            "fresh_B_mean_correct_bits": float(
                np.mean(pair_correct_bits)
            ),
            "fresh_B_min_correct_bits": int(min(pair_correct_bits)),
            "fresh_B_complete_pairs": pair_complete_count,
        }

        training_rows.append(row)

        if (
            step == 1
            or step % args.log_every == 0
            or step == final_step
        ):
            print(
                f"step={step:04d} "
                f"pairs={pair_count} "
                f"prior={prior_pair_count} "
                f"conflict={conflict_pair_count} "
                f"loss={row['loss_total']:.6f} "
                f"B_mean_bits={row['fresh_B_mean_correct_bits']:.2f}/96 "
                f"B_min_bits={row['fresh_B_min_correct_bits']}/96"
            )

        if step % args.eval_every == 0 and step < final_step:
            model.embedder.eval()

            step_results = evaluate_unseen_message_pairs(
                model=model,
                prior_embedder=prior_embedder,
                prior_blender=prior_blender,
                eval_frames_cpu=eval_frames,
                device=device,
                message_pairs=selection_message_pairs,
                eval_batch_size=args.eval_batch_size,
            )

            evaluation_history.append(
                {
                    "step": step,
                    "results": step_results,
                }
            )
            step_score = multi_pair_evaluation_score(step_results)

            print(
                f"evaluation step={step:04d} "
                f"complete={step_score[0]}/{args.eval_message_pairs} "
                f"min_acc={step_score[1]:.2f}% "
                f"mean_acc={step_score[2]:.3f}% "
                f"PSNR={step_score[3]:.3f}"
            )

            if step_score > best_score:
                best_score = step_score
                best_step = step
                save_checkpoint(
                    path=best_checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    args=args,
                    step=step,
                    evaluation=step_results,
                )
                print(f"Saved new best checkpoint at step {step}.")

            model.embedder.train()

    print("\nRunning final evaluation...")

    model.embedder.eval()

    final_results = evaluate_a_then_b(
        model=model,
        prior_embedder=prior_embedder,
        prior_blender=prior_blender,
        eval_frames_cpu=eval_frames,
        device=device,
        message_a=fixed_eval_message_a,
        message_b=fixed_eval_message_b,
        eval_batch_size=args.eval_batch_size,
    )

    print(
        json.dumps(
            final_results,
            indent=2,
        )
    )

    print("\nRunning final held-out multi-message evaluation...")
    final_selection_results = evaluate_unseen_message_pairs(
        model=model,
        prior_embedder=prior_embedder,
        prior_blender=prior_blender,
        eval_frames_cpu=eval_frames,
        device=device,
        message_pairs=selection_message_pairs,
        eval_batch_size=args.eval_batch_size,
    )
    print(json.dumps({
        key: value
        for key, value in final_selection_results.items()
        if key != "per_pair"
    }, indent=2))

    evaluation_history.append(
        {
            "step": final_step,
            "results": final_selection_results,
        }
    )
    final_score = multi_pair_evaluation_score(
        final_selection_results
    )

    if final_score > best_score:
        best_score = final_score
        best_step = final_step
        save_checkpoint(
            path=best_checkpoint_path,
            model=model,
            optimizer=optimizer,
            args=args,
            step=final_step,
            evaluation=final_selection_results,
        )
        print(f"Saved new best checkpoint at step {final_step}.")

    results = {
        "experiment": (
            "96-bit prior-watermark "
            "augmentation training"
        ),

        "training_design": {
            "prior_model": (
                "frozen VideoSeal v0.0 embedder"
            ),

            "trainable_component": (
                "fresh-watermark embedder only"
            ),

            "detector": "frozen",

            "payload_bits": PAYLOAD_BITS,

            "prior_probability":
                args.prior_probability,

            "full_conflict_probability":
                args.full_conflict_probability,

            "loss": (
                "video-level signed-margin loss(fresh B) "
                "+ hard_bit_weight * hardest-k-bit loss "
                "+ frame_loss_weight * frame BCE(fresh B) "
                "+ lambda_image * squared excess over the "
                "target-PSNR MSE budget"
            ),

            "video_forward": (
                "differentiable VideoSeal temporal propagation; "
                "training matches the unaugmented overwrite evaluation path"
            ),

            "target_psnr": args.target_psnr,

            "decode_margin": args.decode_margin,

            "frame_loss_weight": args.frame_loss_weight,

            "hard_bit_k": args.hard_bit_k,

            "hard_bit_weight": args.hard_bit_weight,

            "message_pairs_per_step": args.message_pairs_per_step,

            "checkpoint_selection_message_pairs": args.eval_message_pairs,

            "checkpoint_selection_seed": selection_seed,

            "max_grad_norm": args.max_grad_norm,

            "fresh_strength": args.fresh_strength,

            "fixed_training_pair": args.fixed_training_pair,

            "resume_checkpoint": (
                str(args.resume_checkpoint)
                if args.resume_checkpoint is not None
                else None
            ),

            "start_step": start_step,

            "final_step": final_step,

            "training_frames":
                int(train_frames.shape[0]),

            "evaluation_frames":
                int(eval_frames.shape[0]),
        },

        "baseline_before_training":
            baseline_results,

        "after_training":
            final_results,

        "held_out_random_message_evaluation": {
            "message_pair_design": (
                "random A; B is the bitwise complement of A"
            ),
            "messages_used_in_training": False,
            "baseline_before_training": baseline_selection_results,
            "after_training": final_selection_results,
        },

        "best_checkpoint": {
            "step": best_step,
            "score": {
                "complete_96bit_successes": best_score[0],
                "minimum_bit_accuracy_pct": best_score[1],
                "mean_bit_accuracy_pct": best_score[2],
                "mean_PSNR": best_score[3],
            },
            "path": str(best_checkpoint_path),
        },

        "evaluation_history": evaluation_history,
    }

    with (
        args.output_dir / "results.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=2,
        )

    save_training_log(
        args.output_dir,
        training_rows,
    )

    save_checkpoint(
        path=(
            args.output_dir
            / "checkpoint_prior_aug_96bit.pth"
        ),
        model=model,
        optimizer=optimizer,
        args=args,
        step=final_step,
        evaluation=final_selection_results,
    )

    print(
        "\nSaved results to: "
        f"{args.output_dir.resolve()}"
    )


if __name__ == "__main__":
    main()
