from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

# PixelSeal uses antialiased bilinear resizing, whose backward-compatible MPS
# implementation in PyTorch 2.4 requires CPU fallback for that single op.  The
# flag must be set before importing torch.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PixelSeal native-payload overwrite experiment."
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
        default=Path("results/pixelseal_overwrite"),
    )
    parser.add_argument(
        "--device",
        choices=["auto", "mps", "cpu"],
        default="auto",
    )
    parser.add_argument("--chunk_size", type=int, default=32)
    parser.add_argument(
        "--max_frames", type=int, default=None,
        help="Process only the first N consecutive frames (default: all).",
    )
    parser.add_argument(
        "--conditions", nargs="+",
        choices=["A_only", "B_only", "A_then_B", "B_then_A", "A_then_A", "A_transcode"],
        default=["A_only", "B_only", "A_then_B", "B_then_A", "A_then_A", "A_transcode"],
        help="Conditions to run; chained/transcode conditions require their source baseline.",
    )
    parser.add_argument("--crf", type=int, default=23)
    parser.add_argument("--seed", type=int, default=2025)
    return parser.parse_args()


def select_device(name: str) -> torch.device:
    if name == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(name)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable.")
    return device


def require_binary(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"{name} was not found in PATH.")


def run_checked(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )


def video_info(path: Path) -> dict[str, Any]:
    probe = run_checked(
        ["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(path)]
    )
    streams = json.loads(probe.stdout)["streams"]
    stream = next(
        item for item in streams if item.get("codec_type") == "video"
    )
    fps = stream.get("avg_frame_rate") or stream.get("r_frame_rate")
    if not fps or fps == "0/0":
        raise RuntimeError(f"Could not read frame rate from {path}")
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": fps,
    }


def read_exact(stream: Any, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def open_reader(
    path: Path,
    max_frames: int | None = None,
) -> subprocess.Popen[bytes]:
    command = ["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v:0"]
    if max_frames is not None:
        command.extend(["-frames:v", str(max_frames)])
    command.extend(["-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"])
    return subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def open_writer(
    path: Path,
    info: dict[str, Any],
    crf: int,
) -> subprocess.Popen[bytes]:
    path.parent.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen(
        [
            "ffmpeg", "-y", "-v", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{info['width']}x{info['height']}",
            "-r", str(info["fps"]), "-i", "pipe:0",
            "-an", "-c:v", "libx264", "-crf", str(crf),
            "-pix_fmt", "yuv420p", str(path),
        ],
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def finish_process(process: subprocess.Popen[bytes], label: str) -> None:
    return_code = process.wait()
    stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
    if return_code:
        raise RuntimeError(f"{label} failed:\n{stderr}")


def raw_to_tensor(raw: bytes, info: dict[str, Any], device: torch.device) -> torch.Tensor:
    width, height = info["width"], info["height"]
    array = np.frombuffer(raw, dtype=np.uint8).reshape(-1, height, width, 3).copy()
    return (
        torch.from_numpy(array)
        .permute(0, 3, 1, 2)
        .float()
        .div(255.0)
        .to(device)
    )


def tensor_to_raw(frames: torch.Tensor) -> bytes:
    array = (
        frames.detach().clamp(0, 1).mul(255).round().to(torch.uint8)
        .cpu().permute(0, 2, 3, 1).contiguous().numpy()
    )
    return array.tobytes()


def bit_string(bits: torch.Tensor) -> str:
    return "".join(str(int(bit)) for bit in bits.flatten().tolist())


def make_messages(seed: int, payload_bits: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    message_a = torch.randint(
        0, 2, (1, payload_bits), generator=generator, dtype=torch.float32
    )
    return message_a, 1.0 - message_a


@torch.inference_mode()
def embed_video(
    model: Any,
    input_path: Path,
    output_path: Path,
    message: torch.Tensor,
    chunk_size: int,
    crf: int,
    device: torch.device,
    max_frames: int | None = None,
) -> int:
    info = video_info(input_path)
    bytes_per_frame = info["width"] * info["height"] * 3
    reader = open_reader(input_path, max_frames)
    writer = open_writer(output_path, info, crf)
    frame_count = 0
    try:
        while True:
            frames_left = None if max_frames is None else max_frames - frame_count
            if frames_left is not None and frames_left <= 0:
                break
            read_frames = chunk_size if frames_left is None else min(chunk_size, frames_left)
            raw = read_exact(reader.stdout, bytes_per_frame * read_frames)
            if not raw:
                break
            if len(raw) % bytes_per_frame:
                raise RuntimeError("FFmpeg returned an incomplete frame.")
            clip = raw_to_tensor(raw, info, device)
            outputs = model.embed(
                clip,
                msgs=message.to(device),
                is_video=True,
                lowres_attenuation=True,
            )
            writer.stdin.write(tensor_to_raw(outputs["imgs_w"]))
            frame_count += int(clip.shape[0])
    finally:
        if reader.stdout:
            reader.stdout.close()
        if writer.stdin:
            writer.stdin.close()
    finish_process(reader, f"Reading {input_path}")
    finish_process(writer, f"Writing {output_path}")
    return frame_count


@torch.inference_mode()
def detect_video(
    model: Any,
    input_path: Path,
    chunk_size: int,
    device: torch.device,
    payload_bits: int,
    max_frames: int | None = None,
) -> tuple[torch.Tensor, int]:
    info = video_info(input_path)
    bytes_per_frame = info["width"] * info["height"] * 3
    reader = open_reader(input_path, max_frames)
    logits: list[torch.Tensor] = []
    frame_count = 0
    try:
        while True:
            frames_left = None if max_frames is None else max_frames - frame_count
            if frames_left is not None and frames_left <= 0:
                break
            read_frames = chunk_size if frames_left is None else min(chunk_size, frames_left)
            raw = read_exact(reader.stdout, bytes_per_frame * read_frames)
            if not raw:
                break
            if len(raw) % bytes_per_frame:
                raise RuntimeError("FFmpeg returned an incomplete frame.")
            clip = raw_to_tensor(raw, info, device)
            preds = model.detect(clip, is_video=True)["preds"][:, 1:]
            if preds.ndim == 4:
                preds = preds.mean(dim=(2, 3))
            if preds.ndim != 2:
                raise RuntimeError(f"Unexpected PixelSeal logits shape: {preds.shape}")
            logits.append(preds.cpu())
            frame_count += int(clip.shape[0])
    finally:
        if reader.stdout:
            reader.stdout.close()
    finish_process(reader, f"Reading {input_path}")
    pooled = torch.cat(logits, dim=0).mean(dim=0)
    if pooled.numel() != payload_bits:
        raise RuntimeError(
            f"Expected {payload_bits} decoded bits, got {pooled.numel()}"
        )
    return (pooled > 0).to(torch.int64), frame_count


def transcode(input_path: Path, output_path: Path, crf: int) -> None:
    run_checked([
        "ffmpeg", "-y", "-v", "error", "-i", str(input_path),
        "-map", "0:v:0", "-an", "-c:v", "libx264", "-crf", str(crf),
        "-pix_fmt", "yuv420p", str(output_path),
    ])


def psnr(
    reference: Path,
    candidate: Path,
    chunk_size: int,
    max_frames: int | None = None,
) -> tuple[float, int]:
    ref_info = video_info(reference)
    cand_info = video_info(candidate)
    if (ref_info["width"], ref_info["height"], ref_info["fps"]) != (
        cand_info["width"], cand_info["height"], cand_info["fps"]
    ):
        raise RuntimeError(f"Video geometry/fps mismatch: {reference} vs {candidate}")
    bytes_per_frame = ref_info["width"] * ref_info["height"] * 3
    ref_reader = open_reader(reference, max_frames)
    cand_reader = open_reader(candidate, max_frames)
    squared_error = 0.0
    total_values = 0
    frame_count = 0
    try:
        while True:
            ref_raw = read_exact(ref_reader.stdout, bytes_per_frame * chunk_size)
            cand_raw = read_exact(cand_reader.stdout, bytes_per_frame * chunk_size)
            if not ref_raw and not cand_raw:
                break
            if len(ref_raw) != len(cand_raw) or len(ref_raw) % bytes_per_frame:
                raise RuntimeError("Reference and candidate frame counts differ.")
            ref = np.frombuffer(ref_raw, dtype=np.uint8).astype(np.float32) / 255.0
            cand = np.frombuffer(cand_raw, dtype=np.uint8).astype(np.float32) / 255.0
            squared_error += float(np.square(ref - cand).sum())
            total_values += ref.size
            frame_count += len(ref_raw) // bytes_per_frame
    finally:
        if ref_reader.stdout:
            ref_reader.stdout.close()
        if cand_reader.stdout:
            cand_reader.stdout.close()
    finish_process(ref_reader, f"Reading {reference}")
    finish_process(cand_reader, f"Reading {candidate}")
    mse = squared_error / total_values
    return (float("inf") if mse == 0 else 10.0 * np.log10(1.0 / mse), frame_count)


def evaluate_decoded(
    decoded: torch.Tensor,
    message_a: torch.Tensor,
    message_b: torch.Tensor,
    target: str,
) -> dict[str, Any]:
    target_a = message_a.to(torch.int64).flatten()
    target_b = message_b.to(torch.int64).flatten()
    decoded = decoded.to(torch.int64).flatten()
    matches_a = int(decoded.eq(target_a).sum())
    matches_b = int(decoded.eq(target_b).sum())
    target_matches = matches_a if target == "A" else matches_b
    return {
        "bits_matching_A": matches_a,
        "bits_matching_B": matches_b,
        "accuracy_A_pct": round(100.0 * matches_a / len(target_a), 3),
        "accuracy_B_pct": round(100.0 * matches_b / len(target_b), 3),
        "target": target,
        "target_correct_bits": target_matches,
        "target_accuracy_pct": round(100.0 * target_matches / len(target_a), 3),
        "complete_A": matches_a == len(target_a),
        "complete_B": matches_b == len(target_b),
        "complete_target": target_matches == len(target_a),
        "decoded_bits": bit_string(decoded),
    }


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise SystemExit(f"Input video not found: {args.input}")
    if not args.official_repo.is_dir():
        raise SystemExit(f"Official repository not found: {args.official_repo}")
    if args.chunk_size <= 0:
        raise SystemExit("--chunk_size must be positive.")
    if args.max_frames is not None and args.max_frames <= 0:
        raise SystemExit("--max_frames must be positive.")
    conditions = list(dict.fromkeys(args.conditions))
    dependencies = {
        "A_then_B": "A_only",
        "B_then_A": "B_only",
        "A_then_A": "A_only",
        "A_transcode": "A_only",
    }
    for condition, dependency in dependencies.items():
        if condition in conditions and dependency not in conditions:
            raise SystemExit(f"{condition} requires the {dependency} condition.")
    require_binary("ffmpeg")
    require_binary("ffprobe")

    device = select_device(args.device)
    sys.path.insert(0, str(args.official_repo.resolve()))
    import videoseal
    from videoseal.utils import cfg as videoseal_cfg

    print(f"Device: {device}")
    print('Loading PixelSeal with videoseal.load("pixelseal")...')
    # The official card refers to ``configs/attenuation.yaml``.  Its loader
    # first resolves that path from cwd, while the checkpoint cache is also
    # cwd-relative.  Keep cwd at this project (where ckpts/ lives) and
    # redirect only missing config paths to the official repository.
    original_resolver = videoseal_cfg.resolve_config_path
    official_repo = args.official_repo.resolve()

    def resolve_config_path_without_repo_mutation(config_path: Any) -> Path:
        candidate = Path(config_path)
        if candidate.is_file():
            return candidate
        official_candidate = official_repo / candidate
        if official_candidate.is_file():
            return official_candidate
        return original_resolver(config_path)

    videoseal_cfg.resolve_config_path = resolve_config_path_without_repo_mutation
    model = videoseal.load("pixelseal").to(device)
    model.eval()
    payload_bits = int(model.get_random_msg(1).shape[1])
    if payload_bits <= 0:
        raise RuntimeError(f"Invalid PixelSeal payload length: {payload_bits}")
    print(f"Native PixelSeal payload: {payload_bits} bits")
    checkpoint_path = (Path("ckpts") / "pixelseal_checkpoint.pth").resolve()
    checkpoint_validation = {
        "path": str(checkpoint_path),
        "size_bytes": checkpoint_path.stat().st_size if checkpoint_path.is_file() else None,
        "status": "loaded successfully by videoseal.load; all state-dict keys matched",
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    message_a, message_b = make_messages(args.seed, payload_bits)
    (args.output_dir / "message_A.txt").write_text(bit_string(message_a), encoding="utf-8")
    (args.output_dir / "message_B.txt").write_text(bit_string(message_b), encoding="utf-8")

    all_paths = {
        "A_only": args.output_dir / "A_only.mp4",
        "B_only": args.output_dir / "B_only.mp4",
        "A_then_B": args.output_dir / "A_then_B.mp4",
        "B_then_A": args.output_dir / "B_then_A.mp4",
        "A_then_A": args.output_dir / "A_then_A.mp4",
        "A_transcode": args.output_dir / "A_transcode.mp4",
    }
    sources = {
        "A_only": (args.input, message_a, "A", None),
        "B_only": (args.input, message_b, "B", None),
        "A_then_B": (all_paths["A_only"], message_b, "B", all_paths["A_only"]),
        "B_then_A": (all_paths["B_only"], message_a, "A", all_paths["B_only"]),
        "A_then_A": (all_paths["A_only"], message_a, "A", all_paths["A_only"]),
    }

    for condition in conditions:
        if condition == "A_transcode":
            continue
        source, message, _, _ = sources[condition]
        print(f"Embedding {condition}...")
        embed_video(model, source, all_paths[condition], message, args.chunk_size, args.crf, device, args.max_frames)
    if "A_transcode" in conditions:
        print("Transcoding A_only...")
        transcode(all_paths["A_only"], all_paths["A_transcode"], args.crf)

    target_by_condition = {
        "A_only": "A",
        "B_only": "B",
        "A_then_B": "B",
        "B_then_A": "A",
        "A_then_A": "A",
        "A_transcode": "A",
    }
    previous_by_condition = {
        "A_only": args.input,
        "B_only": args.input,
        "A_then_B": all_paths["A_only"],
        "B_then_A": all_paths["B_only"],
        "A_then_A": all_paths["A_only"],
        "A_transcode": all_paths["A_only"],
    }
    results: dict[str, dict[str, Any]] = {}
    for condition in conditions:
        path = all_paths[condition]
        print(f"Detecting {condition}...")
        decoded, frames = detect_video(model, path, args.chunk_size, device, payload_bits, args.max_frames)
        result = evaluate_decoded(decoded, message_a, message_b, target_by_condition[condition])
        result["frames"] = frames
        result["PSNR_vs_original_dB"], _ = psnr(
            args.input, path, args.chunk_size, args.max_frames
        )
        result["PSNR_vs_previous_stage_dB"], _ = psnr(
            previous_by_condition[condition], path, args.chunk_size, args.max_frames
        )
        results[condition] = result
        (args.output_dir / f"{condition}_decoded_bits.txt").write_text(
            result["decoded_bits"] + "\n", encoding="utf-8"
        )

    summary = {
        "experiment": "PixelSeal native-payload overwrite",
        "model": "pixelseal",
        "payload_bits": payload_bits,
        "device": str(device),
        "seed": args.seed,
        "chunk_size": args.chunk_size,
        "max_frames": args.max_frames,
        "conditions": conditions,
        "crf": args.crf,
        "input": str(args.input),
        "message_design": "random A; B is the bitwise complement of A",
        "training_performed": False,
        "checkpoint": checkpoint_validation,
        "condition_results": results,
    }
    (args.output_dir / "results.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    with (args.output_dir / "results.csv").open("w", newline="", encoding="utf-8") as file:
        fields = [
            "condition", "frames", "accuracy_A_pct", "accuracy_B_pct",
            "complete_A", "complete_B", "target", "target_accuracy_pct",
            "complete_target", "PSNR_vs_original_dB", "PSNR_vs_previous_stage_dB",
        ]
        writer = csv.DictWriter(file, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for condition, result in results.items():
            writer.writerow({"condition": condition, **{field: result[field] for field in fields if field != "condition"}})
    print(json.dumps(summary, indent=2))
    print(f"Saved results to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
