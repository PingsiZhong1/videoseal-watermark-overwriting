from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

PAYLOAD_BITS = 96


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Controlled sequential-watermark experiment for VideoSeal v0.0."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/overwrite_v0"),
    )
    parser.add_argument(
        "--official_repo",
        type=Path,
        default=Path.home() / "Projects" / "videoseal",
    )
    parser.add_argument("--chunk_size", type=int, default=16)
    parser.add_argument("--crf", type=int, default=23)
    parser.add_argument("--seed", type=int, default=2025)
    return parser.parse_args()


def require_binary(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"{name} was not found in PATH. Install FFmpeg first.")


def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=True, capture_output=True, text=True)


def video_info(path: Path) -> dict[str, Any]:
    probe = run_command(
        ["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(path)]
    )
    streams = json.loads(probe.stdout)["streams"]
    stream = next(
        (item for item in streams if item.get("codec_type") == "video"),
        None,
    )

    if stream is None:
        raise RuntimeError(f"No video stream found in {path}")

    fps = stream.get("avg_frame_rate") or stream.get("r_frame_rate")

    if not fps or fps == "0/0":
        raise RuntimeError(f"Could not read frame rate from {path}")

    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": fps,
    }


def read_up_to(stream: Any, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size

    while remaining > 0:
        chunk = stream.read(remaining)

        if not chunk:
            break

        chunks.append(chunk)
        remaining -= len(chunk)

    return b"".join(chunks)


def open_raw_reader(path: Path) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def open_raw_writer(
    path: Path,
    info: dict[str, Any],
    crf: int,
) -> subprocess.Popen[bytes]:
    path.parent.mkdir(parents=True, exist_ok=True)

    return subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{info['width']}x{info['height']}",
            "-r",
            str(info["fps"]),
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def check_process(process: subprocess.Popen[bytes], label: str) -> None:
    return_code = process.wait()

    stderr = ""
    if process.stderr:
        stderr = process.stderr.read().decode("utf-8", errors="replace")

    if return_code != 0:
        raise RuntimeError(f"{label} failed:\n{stderr}")


def raw_to_tensor(
    raw: bytes,
    info: dict[str, Any],
    device: torch.device,
) -> torch.Tensor:
    width = info["width"]
    height = info["height"]

    frames = np.frombuffer(raw, dtype=np.uint8)
    frames = frames.reshape(-1, height, width, 3).copy()

    return (
        torch.from_numpy(frames)
        .permute(0, 3, 1, 2)
        .float()
        .div(255.0)
        .to(device)
    )


def tensor_to_raw(video: torch.Tensor) -> bytes:
    frames = (
        video.detach()
        .clamp(0, 1)
        .mul(255)
        .round()
        .to(torch.uint8)
        .cpu()
        .permute(0, 2, 3, 1)
        .contiguous()
        .numpy()
    )

    return frames.tobytes()


def bit_string(bits: torch.Tensor) -> str:
    return "".join(str(int(bit)) for bit in bits.flatten().tolist())


def complementary_messages(
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)

    message_a = torch.randint(
        0,
        2,
        (1, PAYLOAD_BITS),
        generator=generator,
        dtype=torch.float32,
    )

    message_b = 1.0 - message_a

    return message_a.to(device), message_b.to(device)


@torch.inference_mode()
def embed_video(
    model: Any,
    input_path: Path,
    output_path: Path,
    message: torch.Tensor,
    chunk_size: int,
    crf: int,
    device: torch.device,
) -> None:
    info = video_info(input_path)
    bytes_per_frame = info["width"] * info["height"] * 3

    reader = open_raw_reader(input_path)
    writer = open_raw_writer(output_path, info, crf)

    try:
        with tqdm(desc=f"Embedding {output_path.name}", unit="frames") as progress:
            while True:
                raw = read_up_to(
                    reader.stdout,
                    bytes_per_frame * chunk_size,
                )

                if not raw:
                    break

                if len(raw) % bytes_per_frame != 0:
                    raise RuntimeError("FFmpeg returned an incomplete video frame.")

                clip = raw_to_tensor(raw, info, device)

                outputs = model.embed(
                    clip,
                    msgs=message,
                    is_video=True,
                    lowres_attenuation=True,
                )

                writer.stdin.write(tensor_to_raw(outputs["imgs_w"]))
                progress.update(clip.shape[0])

    finally:
        if reader.stdout:
            reader.stdout.close()

        if writer.stdin:
            writer.stdin.close()

    check_process(reader, f"Reading {input_path.name}")
    check_process(writer, f"Writing {output_path.name}")


@torch.inference_mode()
def detect_video(
    model: Any,
    input_path: Path,
    chunk_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    info = video_info(input_path)
    bytes_per_frame = info["width"] * info["height"] * 3

    reader = open_raw_reader(input_path)
    scores: list[torch.Tensor] = []
    frame_count = 0

    try:
        with tqdm(desc=f"Detecting {input_path.name}", unit="frames") as progress:
            while True:
                raw = read_up_to(
                    reader.stdout,
                    bytes_per_frame * chunk_size,
                )

                if not raw:
                    break

                if len(raw) % bytes_per_frame != 0:
                    raise RuntimeError("FFmpeg returned an incomplete video frame.")

                clip = raw_to_tensor(raw, info, device)

                outputs = model.detect(clip, is_video=True)

                scores.append(outputs["preds"][:, 1:].detach().cpu())
                frame_count += clip.shape[0]

                progress.update(clip.shape[0])

    finally:
        if reader.stdout:
            reader.stdout.close()

    check_process(reader, f"Reading {input_path.name}")

    soft_bits = torch.cat(scores, dim=0).mean(dim=0)
    decoded = (soft_bits > 0).to(torch.int64)

    if decoded.numel() != PAYLOAD_BITS:
        raise RuntimeError(
            f"Expected {PAYLOAD_BITS} decoded bits, got {decoded.numel()}."
        )

    return decoded, soft_bits, frame_count


def transcode_h264(
    input_path: Path,
    output_path: Path,
    crf: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    run_command(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(input_path),
            "-map",
            "0:v:0",
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ]
    )


def evaluate(
    decoded: torch.Tensor,
    message_a: torch.Tensor,
    message_b: torch.Tensor,
) -> dict[str, Any]:
    target_a = message_a.squeeze(0).to(torch.int64).cpu()
    target_b = message_b.squeeze(0).to(torch.int64).cpu()

    if not torch.equal(target_b, 1 - target_a):
        raise RuntimeError("Message B must be the bitwise complement of message A.")

    matches_a = int((decoded == target_a).sum().item())
    matches_b = int((decoded == target_b).sum().item())

    if matches_a + matches_b != PAYLOAD_BITS:
        raise RuntimeError("Complementary-message accounting failed.")

    if matches_a > matches_b:
        preference = "A"
    elif matches_b > matches_a:
        preference = "B"
    else:
        preference = "tie"

    return {
        "bits_matching_A": matches_a,
        "bits_matching_B": matches_b,
        "accuracy_A_pct": round(100 * matches_a / PAYLOAD_BITS, 2),
        "accuracy_B_pct": round(100 * matches_b / PAYLOAD_BITS, 2),
        "decoded_preference": preference,
        "decoded_bits": bit_string(decoded),
    }


def save_messages(
    output_dir: Path,
    message_a: torch.Tensor,
    message_b: torch.Tensor,
) -> None:
    (output_dir / "message_A.txt").write_text(
        bit_string(message_a.cpu()),
        encoding="utf-8",
    )

    (output_dir / "message_B.txt").write_text(
        bit_string(message_b.cpu()),
        encoding="utf-8",
    )


def save_results(
    output_dir: Path,
    results: dict[str, dict[str, Any]],
    decoded_bits: dict[str, torch.Tensor],
    message_a: torch.Tensor,
    message_b: torch.Tensor,
) -> None:
    summary_columns = [
        "condition",
        "frames",
        "bits_matching_A",
        "bits_matching_B",
        "accuracy_A_pct",
        "accuracy_B_pct",
        "decoded_preference",
        "decoded_bits",
    ]

    with (output_dir / "results.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=summary_columns)
        writer.writeheader()

        for condition, result in results.items():
            writer.writerow({"condition": condition, **result})

    target_a = message_a.squeeze(0).to(torch.int64).cpu()
    target_b = message_b.squeeze(0).to(torch.int64).cpu()

    with (output_dir / "bit_level_results.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "condition",
                "bit_index",
                "message_A",
                "message_B",
                "decoded",
                "matches_A",
                "matches_B",
            ],
        )

        writer.writeheader()

        for condition, decoded in decoded_bits.items():
            for index, value in enumerate(decoded.tolist()):
                writer.writerow(
                    {
                        "condition": condition,
                        "bit_index": index,
                        "message_A": int(target_a[index]),
                        "message_B": int(target_b[index]),
                        "decoded": int(value),
                        "matches_A": int(value == int(target_a[index])),
                        "matches_B": int(value == int(target_b[index])),
                    }
                )


def main() -> None:
    args = parse_args()

    if not args.input.is_file():
        raise SystemExit(f"Input video not found: {args.input}")

    if not args.official_repo.is_dir():
        raise SystemExit(
            f"Official VideoSeal repository not found: {args.official_repo}"
        )

    if args.chunk_size <= 0:
        raise SystemExit("--chunk_size must be positive.")

    require_binary("ffmpeg")
    require_binary("ffprobe")

    sys.path.insert(0, str(args.official_repo))
    import videoseal

    device = torch.device("cpu")

    model = videoseal.load("videoseal_0.0")
    model.eval().to(device)
    model.compile()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    message_a, message_b = complementary_messages(args.seed, device)
    save_messages(args.output_dir, message_a, message_b)

    paths = {
        "A_only": args.output_dir / "A_only.mp4",
        "B_only": args.output_dir / "B_only.mp4",
        "A_transcode": args.output_dir / "A_transcode.mp4",
        "A_then_A": args.output_dir / "A_then_A.mp4",
        "A_then_B": args.output_dir / "A_then_B.mp4",
        "B_then_A": args.output_dir / "B_then_A.mp4",
    }

    embed_video(
        model,
        args.input,
        paths["A_only"],
        message_a,
        args.chunk_size,
        args.crf,
        device,
    )

    embed_video(
        model,
        args.input,
        paths["B_only"],
        message_b,
        args.chunk_size,
        args.crf,
        device,
    )

    transcode_h264(
        paths["A_only"],
        paths["A_transcode"],
        args.crf,
    )

    embed_video(
        model,
        paths["A_only"],
        paths["A_then_A"],
        message_a,
        args.chunk_size,
        args.crf,
        device,
    )

    embed_video(
        model,
        paths["A_only"],
        paths["A_then_B"],
        message_b,
        args.chunk_size,
        args.crf,
        device,
    )

    embed_video(
        model,
        paths["B_only"],
        paths["B_then_A"],
        message_a,
        args.chunk_size,
        args.crf,
        device,
    )

    results: dict[str, dict[str, Any]] = {}
    decoded_by_condition: dict[str, torch.Tensor] = {}

    for condition, path in paths.items():
        decoded, _scores, frames = detect_video(
            model,
            path,
            args.chunk_size,
            device,
        )

        result = evaluate(decoded, message_a, message_b)
        result["frames"] = frames

        results[condition] = result
        decoded_by_condition[condition] = decoded

        (args.output_dir / f"decoded_{condition}.txt").write_text(
            bit_string(decoded),
            encoding="utf-8",
        )

    save_results(
        args.output_dir,
        results,
        decoded_by_condition,
        message_a,
        message_b,
    )

    print("\nCondition       A match    B match    Decoder preference")

    for condition, result in results.items():
        print(
            f"{condition:<15} "
            f"{result['accuracy_A_pct']:>6.2f}%   "
            f"{result['accuracy_B_pct']:>6.2f}%   "
            f"{result['decoded_preference']}"
        )

    print(f"\nSaved results to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()