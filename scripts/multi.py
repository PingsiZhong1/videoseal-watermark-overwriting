from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

PAYLOAD_BITS = 96
NUM_SLOTS = 3
SLOT_BITS = PAYLOAD_BITS // NUM_SLOTS
TOKEN_BITS = SLOT_BITS - 1

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sequential slot-watermark fine-tuning prototype for VideoSeal v0.0."
    )

    parser.add_argument("--mode", choices=("train", "test"), required=True)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--video_dir", type=Path)

    parser.add_argument(
        "--official_repo",
        type=Path,
        default=Path.home() / "Projects" / "videoseal",
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/multi"),
    )

    parser.add_argument("--checkpoint", type=Path)

    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--size", type=int, default=256)

    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lambda_image", type=float, default=5.0)

    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=250)

    parser.add_argument("--seed", type=int, default=2025)

    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "mps", "cpu"),
        default="auto",
    )

    return parser.parse_args()


def require_binary(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"{name} was not found in PATH.")


def choose_device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")

    if name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available.")
        return torch.device("mps")

    if name == "cpu":
        return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def run(command: list[str]) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        command,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def probe_video(path: Path) -> tuple[float, float]:
    result = run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=avg_frame_rate:format=duration",
            "-of",
            "json",
            str(path),
        ]
    )

    info = json.loads(result.stdout.decode("utf-8"))

    duration = float(info["format"]["duration"])
    rate = info["streams"][0]["avg_frame_rate"]

    numerator, denominator = rate.split("/")
    fps = float(numerator) / float(denominator)

    if duration <= 0 or fps <= 0:
        raise RuntimeError(f"Could not read duration or fps from {path}.")

    return duration, fps


def load_clip(
    path: Path,
    frames: int,
    size: int,
    rng: random.Random,
    device: torch.device,
) -> torch.Tensor:
    duration, fps = probe_video(path)

    clip_duration = frames / fps
    start = rng.uniform(0, max(0.0, duration - clip_duration))

    result = run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-ss",
            f"{start:.3f}",
            "-i",
            str(path),
            "-an",
            "-vf",
            f"scale={size}:{size}:flags=lanczos",
            "-frames:v",
            str(frames),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ]
    )

    bytes_per_frame = size * size * 3
    available = len(result.stdout) // bytes_per_frame

    if available == 0:
        raise RuntimeError(f"FFmpeg returned no frames for {path}.")

    raw = np.frombuffer(
        result.stdout[: available * bytes_per_frame],
        dtype=np.uint8,
    ).reshape(available, size, size, 3)

    if available < frames:
        padding = np.repeat(raw[-1:], frames - available, axis=0)
        raw = np.concatenate((raw, padding), axis=0)

    return (
        torch.from_numpy(raw.copy())
        .permute(0, 3, 1, 2)
        .float()
        .div(255.0)
        .to(device)
    )


def collect_videos(args: argparse.Namespace) -> list[Path]:
    videos: list[Path] = []

    if args.input is not None:
        videos.append(args.input)

    if args.video_dir is not None:
        videos.extend(
            path
            for path in sorted(args.video_dir.rglob("*"))
            if path.suffix.lower() in VIDEO_EXTENSIONS
        )

    videos = sorted(set(videos))

    if not videos:
        raise RuntimeError("Provide --input or --video_dir.")

    for video in videos:
        if not video.is_file():
            raise RuntimeError(f"Video not found: {video}.")

    return videos


def random_token(
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    return torch.randint(
        0,
        2,
        (TOKEN_BITS,),
        generator=generator,
        dtype=torch.float32,
    ).to(device)


def pack_slots(
    tokens: list[torch.Tensor | None],
    device: torch.device,
) -> torch.Tensor:
    if len(tokens) != NUM_SLOTS:
        raise ValueError(f"Expected {NUM_SLOTS} slots.")

    slots: list[torch.Tensor] = []

    for token in tokens:
        if token is None:
            slots.append(torch.zeros(SLOT_BITS, device=device))
        else:
            if token.numel() != TOKEN_BITS:
                raise ValueError(f"Token must contain {TOKEN_BITS} bits.")

            slots.append(
                torch.cat(
                    (
                        torch.ones(1, device=device),
                        token.float(),
                    )
                )
            )

    message = torch.cat(slots).unsqueeze(0)

    if message.shape != (1, PAYLOAD_BITS):
        raise RuntimeError("Packed message has the wrong shape.")

    return message


def make_messages(
    generator: torch.Generator,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    token_a = random_token(generator, device)
    token_b = random_token(generator, device)
    token_c = random_token(generator, device)

    return {
        "A": pack_slots([token_a, None, None], device),
        "AB": pack_slots([token_a, token_b, None], device),
        "ABC": pack_slots([token_a, token_b, token_c], device),
    }


def full_frame_masks(clip: torch.Tensor) -> np.ndarray:
    return np.ones(
        (
            clip.shape[0],
            1,
            clip.shape[-2],
            clip.shape[-1],
        ),
        dtype=np.float32,
    )

def reduce_message_logits(preds: torch.Tensor) -> torch.Tensor:
    logits = preds[:, 1:]

    if logits.ndim > 2:
        logits = logits.flatten(start_dim=2).mean(dim=2)

    return logits


def decode_loss(
    output: dict[str, torch.Tensor],
    target: torch.Tensor,
) -> torch.Tensor:
    logits = reduce_message_logits(output["preds"])
    expected = target.expand(logits.shape[0], -1)

    return F.binary_cross_entropy_with_logits(logits, expected)


def sequential_forward(
    model: Any,
    clip: torch.Tensor,
    messages: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    masks = None

    output_a = model.video_forward(
        clip,
        masks=masks,
        msgs=messages["A"],
    )

    output_ab = model.video_forward(
        output_a["imgs_w"],
        masks=masks,
        msgs=messages["AB"],
    )

    output_abc = model.video_forward(
        output_ab["imgs_w"],
        masks=masks,
        msgs=messages["ABC"],
    )

    message_loss = (
        0.5 * decode_loss(output_a, messages["A"])
        + 1.0 * decode_loss(output_ab, messages["AB"])
        + 1.5 * decode_loss(output_abc, messages["ABC"])
    )

    image_loss = (
        F.mse_loss(output_a["imgs_w"], clip)
        + F.mse_loss(output_ab["imgs_w"], clip)
        + F.mse_loss(output_abc["imgs_w"], clip)
    ) / 3.0

    return message_loss, image_loss


def save_checkpoint(
    path: Path,
    model: Any,
    args: argparse.Namespace,
    step: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "step": step,
            "payload_bits": PAYLOAD_BITS,
            "slot_bits": SLOT_BITS,
            "token_bits": TOKEN_BITS,
            "args": vars(args),
        },
        path,
    )


def load_model(
    official_repo: Path,
    device: torch.device,
    checkpoint: Path | None,
) -> Any:
    if not official_repo.is_dir():
        raise RuntimeError(
            f"Official VideoSeal repository not found: {official_repo}."
        )

    sys.path.insert(0, str(official_repo))

    import videoseal
    from videoseal.augmentation.masks import NoMaskEmbedder

    model = videoseal.load("videoseal_0.0")

    model.augmenter.mask_embedder = NoMaskEmbedder()

    print(
        "Mask embedder:",
        type(model.augmenter.mask_embedder).__name__,
    )

    model.to(device)

    if checkpoint is not None:
        payload = torch.load(checkpoint, map_location=device)
        state_dict = payload.get("state_dict", payload)
        model.load_state_dict(state_dict, strict=True)

    return model


def train(args: argparse.Namespace) -> None:
    videos = collect_videos(args)
    device = choose_device(args.device)

    model = load_model(
        args.official_repo,
        device,
        checkpoint=None,
    )

    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
    )

    rng = random.Random(args.seed)

    message_generator = torch.Generator(
        device="cpu"
    ).manual_seed(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    log_path = args.output_dir / "train_log.csv"

    with log_path.open("w", encoding="utf-8") as file:
        file.write("step,total_loss,message_loss,image_loss,video\n")

        for step in range(1, args.steps + 1):
            video_path = rng.choice(videos)

            clip = load_clip(
                video_path,
                args.frames,
                args.size,
                rng,
                device,
            )

            messages = make_messages(
                message_generator,
                device,
            )

            optimizer.zero_grad(set_to_none=True)

            message_loss, image_loss = sequential_forward(
                model,
                clip,
                messages,
            )

            total_loss = message_loss + args.lambda_image * image_loss

            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            file.write(
                f"{step},"
                f"{total_loss.item():.6f},"
                f"{message_loss.item():.6f},"
                f"{image_loss.item():.6f},"
                f"{video_path.name}\n"
            )

            file.flush()

            if step == 1 or step % args.log_every == 0:
                print(
                    f"step={step} "
                    f"total={total_loss.item():.4f} "
                    f"message={message_loss.item():.4f} "
                    f"image={image_loss.item():.6f}"
                )

            if step % args.save_every == 0:
                save_checkpoint(
                    args.output_dir / f"checkpoint_step_{step}.pt",
                    model,
                    args,
                    step,
                )

    save_checkpoint(
        args.output_dir / "checkpoint_final.pt",
        model,
        args,
        args.steps,
    )

    print(
        f"Saved checkpoint to: "
        f"{args.output_dir / 'checkpoint_final.pt'}"
    )


@torch.inference_mode()
def decode_message(
    model: Any,
    video: torch.Tensor,
) -> torch.Tensor:
    return (
        model.extract_message(
            video,
            aggregation="avg",
        )
        .squeeze(0)
        .to(torch.int64)
    )


def slot_report(
    decoded: torch.Tensor,
    expected: torch.Tensor,
) -> list[dict[str, int]]:
    expected_bits = expected.squeeze(0).to(torch.int64)

    report: list[dict[str, int]] = []

    for slot_index in range(NUM_SLOTS):
        start = slot_index * SLOT_BITS
        end = start + SLOT_BITS

        decoded_slot = decoded[start:end]
        expected_slot = expected_bits[start:end]

        report.append(
            {
                "slot": slot_index + 1,
                "decoded_present": int(decoded_slot[0].item()),
                "expected_present": int(expected_slot[0].item()),
                "token_matches": int(
                    (decoded_slot[1:] == expected_slot[1:]).sum().item()
                ),
                "token_bits": TOKEN_BITS,
                "slot_matches": int(
                    (decoded_slot == expected_slot).sum().item()
                ),
                "slot_bits": SLOT_BITS,
            }
        )

    return report


@torch.inference_mode()
def test(args: argparse.Namespace) -> None:
    if args.input is None or not args.input.is_file():
        raise RuntimeError(
            "Test mode requires --input pointing to one video."
        )

    if args.checkpoint is None or not args.checkpoint.is_file():
        raise RuntimeError("Test mode requires --checkpoint.")

    device = choose_device(args.device)

    model = load_model(
        args.official_repo,
        device,
        checkpoint=args.checkpoint,
    )

    model.eval()

    rng = random.Random(args.seed)

    message_generator = torch.Generator(
        device="cpu"
    ).manual_seed(args.seed)

    clip = load_clip(
        args.input,
        args.frames,
        args.size,
        rng,
        device,
    )

    messages = make_messages(
        message_generator,
        device,
    )

    video_a = model.embed(
        clip,
        msgs=messages["A"],
        is_video=True,
        lowres_attenuation=True,
    )["imgs_w"]

    video_ab = model.embed(
        video_a,
        msgs=messages["AB"],
        is_video=True,
        lowres_attenuation=True,
    )["imgs_w"]

    video_abc = model.embed(
        video_ab,
        msgs=messages["ABC"],
        is_video=True,
        lowres_attenuation=True,
    )["imgs_w"]

    videos = {
        "A": (video_a, messages["A"]),
        "AB": (video_ab, messages["AB"]),
        "ABC": (video_abc, messages["ABC"]),
    }

    results: dict[str, Any] = {}

    for name, (video, expected) in videos.items():
        decoded = decode_message(model, video)
        expected_bits = expected.squeeze(0).to(torch.int64)

        matches = int(
            (decoded == expected_bits).sum().item()
        )

        report = slot_report(decoded, expected)

        results[name] = {
            "full_message_matches": matches,
            "full_message_bits": PAYLOAD_BITS,
            "slots": report,
        }

        print(f"\n{name}: {matches}/{PAYLOAD_BITS}")

        for item in report:
            print(
                f"slot {item['slot']}: "
                f"presence={item['decoded_present']}/"
                f"{item['expected_present']} "
                f"token={item['token_matches']}/"
                f"{item['token_bits']} "
                f"slot={item['slot_matches']}/"
                f"{item['slot_bits']}"
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    (args.output_dir / "test_results.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print(
        f"\nSaved results to: "
        f"{args.output_dir / 'test_results.json'}"
    )


def main() -> None:
    args = parse_args()

    require_binary("ffmpeg")
    require_binary("ffprobe")

    if args.mode == "train":
        train(args)
    else:
        test(args)


if __name__ == "__main__":
    main()