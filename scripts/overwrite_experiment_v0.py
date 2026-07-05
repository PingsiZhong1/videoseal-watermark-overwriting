import argparse
import csv
import sys
from pathlib import Path

import ffmpeg
import numpy as np
import torch
from tqdm import tqdm

OFFICIAL_REPO = Path.home() / "Projects" / "videoseal"
sys.path.insert(0, str(OFFICIAL_REPO))

import videoseal
from videoseal.evals.metrics import bit_accuracy


def video_info(path):
    probe = ffmpeg.probe(str(path))
    stream = next(s for s in probe["streams"] if s["codec_type"] == "video")

    width = int(stream["width"])
    height = int(stream["height"])

    fps_num, fps_den = stream["r_frame_rate"].split("/")
    fps = float(fps_num) / float(fps_den)

    return width, height, fps, int(stream["nb_frames"])


def embed_clip(model, clip, message):
    clip = torch.tensor(clip, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0

    output = model.embed(
        clip,
        msgs=message,
        is_video=True,
        lowres_attenuation=True,
    )["imgs_w"]

    return (output * 255).byte().permute(0, 2, 3, 1).numpy()


def detect_clip(model, clip):
    clip = torch.tensor(clip, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    return model.detect(clip, is_video=True)["preds"][:, 1:]


def embed_video(model, source, target, message, chunk_size, crf):
    width, height, fps, nframes = video_info(source)
    target.parent.mkdir(parents=True, exist_ok=True)

    reader = (
        ffmpeg.input(str(source))
        .output(
            "pipe:",
            format="rawvideo",
            pix_fmt="rgb24",
            s=f"{width}x{height}",
            r=fps,
        )
        .run_async(pipe_stdout=True, pipe_stderr=False)
    )

    writer = (
        ffmpeg.input(
            "pipe:",
            format="rawvideo",
            pix_fmt="rgb24",
            s=f"{width}x{height}",
            r=fps,
        )
        .output(
            str(target),
            vcodec="libx264",
            pix_fmt="yuv420p",
            crf=crf,
            r=fps,
        )
        .overwrite_output()
        .run_async(pipe_stdin=True, pipe_stderr=False)
    )

    frame_size = width * height * 3
    chunk = np.zeros((chunk_size, height, width, 3), dtype=np.uint8)
    count = 0

    for raw in tqdm(
        iter(lambda: reader.stdout.read(frame_size), b""),
        total=nframes,
        desc=f"Embed {target.stem}",
    ):
        chunk[count] = np.frombuffer(raw, np.uint8).reshape(height, width, 3)
        count += 1

        if count == chunk_size:
            writer.stdin.write(embed_clip(model, chunk, message).tobytes())
            count = 0

    if count:
        writer.stdin.write(embed_clip(model, chunk[:count], message).tobytes())

    reader.stdout.close()
    writer.stdin.close()
    reader.wait()
    writer.wait()


def detect_video(model, source, chunk_size):
    width, height, _, nframes = video_info(source)

    reader = (
        ffmpeg.input(str(source))
        .output("pipe:", format="rawvideo", pix_fmt="rgb24")
        .run_async(pipe_stdout=True, pipe_stderr=False)
    )

    frame_size = width * height * 3
    chunk = np.zeros((chunk_size, height, width, 3), dtype=np.uint8)
    count = 0
    predictions = []

    for _ in tqdm(range(nframes), desc=f"Detect {source.stem}"):
        raw = reader.stdout.read(frame_size)

        if not raw:
            break

        chunk[count] = np.frombuffer(raw, np.uint8).reshape(height, width, 3)
        count += 1

        if count == chunk_size:
            predictions.append(detect_clip(model, chunk))
            count = 0

    if count:
        predictions.append(detect_clip(model, chunk[:count]))

    reader.stdout.close()
    reader.wait()

    return torch.cat(predictions, dim=0).mean(dim=0)


def transcode(source, target, crf):
    (
        ffmpeg.output(
            ffmpeg.input(str(source)).video,
            str(target),
            vcodec="libx264",
            pix_fmt="yuv420p",
            crf=crf,
        )
        .overwrite_output()
        .run(quiet=True)
    )


def fixed_message(nbits, seed):
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, 2, (1, nbits), generator=generator).float()


def save_message(message, path):
    bits = "".join(map(str, message[0].int().tolist()))
    path.write_text(bits, encoding="utf-8")


def evaluate(model, name, path, message_a, message_b, chunk_size):
    prediction = detect_video(model, path, chunk_size)

    return {
        "condition": name,
        "bit_accuracy_A": round(bit_accuracy(prediction, message_a).item() * 100, 2),
        "bit_accuracy_B": round(bit_accuracy(prediction, message_b).item() * 100, 2),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/overwrite_v0"),
    )
    parser.add_argument("--chunk_size", type=int, default=16)
    parser.add_argument("--crf", type=int, default=23)
    args = parser.parse_args()

    if not OFFICIAL_REPO.exists():
        raise FileNotFoundError(f"Official repository not found: {OFFICIAL_REPO}")

    if not args.input.is_file():
        raise FileNotFoundError(f"Input video not found: {args.input}")

    model = videoseal.load("videoseal_0.0")
    model.eval()
    model.to(torch.device("cpu"))
    model.compile()

    nbits = model.get_random_msg().shape[-1]
    message_a = fixed_message(nbits, 2025)
    message_b = fixed_message(nbits, 2026)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    save_message(message_a, args.output_dir / "message_A.txt")
    save_message(message_b, args.output_dir / "message_B.txt")

    a_only = args.output_dir / "A_only.mp4"
    b_only = args.output_dir / "B_only.mp4"
    a_transcode = args.output_dir / "A_transcode.mp4"
    a_then_b = args.output_dir / "A_then_B.mp4"

    embed_video(model, args.input, a_only, message_a, args.chunk_size, args.crf)
    embed_video(model, args.input, b_only, message_b, args.chunk_size, args.crf)
    transcode(a_only, a_transcode, args.crf)
    embed_video(model, a_only, a_then_b, message_b, args.chunk_size, args.crf)

    rows = [
        evaluate(model, "A_only", a_only, message_a, message_b, args.chunk_size),
        evaluate(model, "B_only", b_only, message_a, message_b, args.chunk_size),
        evaluate(
            model,
            "A_transcode",
            a_transcode,
            message_a,
            message_b,
            args.chunk_size,
        ),
        evaluate(
            model,
            "A_then_B",
            a_then_b,
            message_a,
            message_b,
            args.chunk_size,
        ),
    ]

    with (args.output_dir / "results.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(
            f"{row['condition']:14} "
            f"A={row['bit_accuracy_A']:5.2f}% "
            f"B={row['bit_accuracy_B']:5.2f}%"
        )


if __name__ == "__main__":
    main()