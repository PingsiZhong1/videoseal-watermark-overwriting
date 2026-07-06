from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import torch

from overwrite_experiment_v0 import (
    bit_string,
    detect_video,
    embed_video,
    require_binary,
    transcode_h264,
)

PAYLOAD_BITS = 96

REGISTRY: dict[str, dict[str, Any]] = {
    "1": {"actor": "A", "parent": 0, "step": 1},
    "2": {"actor": "B", "parent": 1, "step": 2},
    "3": {"actor": "C", "parent": 2, "step": 3},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pointer-based chained watermarking experiment."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/chain3"),
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


def hamming_distance(a: torch.Tensor, b: torch.Tensor) -> int:
    return int(
        (a.to(torch.int64).cpu() != b.to(torch.int64).cpu()).sum().item()
    )


def make_codebook(seed: int) -> dict[int, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)

    code_a = torch.zeros(PAYLOAD_BITS, dtype=torch.int64)
    order = torch.randperm(PAYLOAD_BITS, generator=generator)
    code_a[order[: PAYLOAD_BITS // 2]] = 1

    code_b = 1 - code_a

    one_positions = torch.where(code_a == 1)[0]
    zero_positions = torch.where(code_a == 0)[0]

    flip_ones = one_positions[
        torch.randperm(len(one_positions), generator=generator)[
            : PAYLOAD_BITS // 4
        ]
    ]
    flip_zeros = zero_positions[
        torch.randperm(len(zero_positions), generator=generator)[
            : PAYLOAD_BITS // 4
        ]
    ]

    code_c = code_a.clone()
    code_c[flip_ones] = 0
    code_c[flip_zeros] = 1

    return {
        1: code_a,
        2: code_b,
        3: code_c,
    }


def reconstruct_chain(record_id: int) -> str:
    actors: list[str] = []
    seen: set[int] = set()
    current_id = record_id

    while current_id != 0:
        if current_id in seen:
            return "invalid cycle"

        seen.add(current_id)
        record = REGISTRY.get(str(current_id))

        if record is None:
            return "unknown record"

        actors.append(record["actor"])
        current_id = record["parent"]

    actors.reverse()
    return " -> ".join(actors)


def classify_record(
    decoded_bits: torch.Tensor,
    codebook: dict[int, torch.Tensor],
) -> tuple[int, dict[int, int], int, int, int]:
    distances = {
        record_id: hamming_distance(decoded_bits, codeword)
        for record_id, codeword in codebook.items()
    }

    ranked = sorted(distances.items(), key=lambda item: item[1])

    best_id, best_distance = ranked[0]
    second_distance = ranked[1][1]
    margin = second_distance - best_distance

    return best_id, distances, best_distance, second_distance, margin


def save_codebook(
    output_dir: Path,
    codebook: dict[int, torch.Tensor],
) -> None:
    codes = {
        str(record_id): bit_string(codeword)
        for record_id, codeword in codebook.items()
    }

    (output_dir / "codes.json").write_text(
        json.dumps(codes, indent=2),
        encoding="utf-8",
    )

    (output_dir / "registry.json").write_text(
        json.dumps(REGISTRY, indent=2),
        encoding="utf-8",
    )


def save_results(
    output_dir: Path,
    results: list[dict[str, Any]],
) -> None:
    columns = [
        "condition",
        "frames",
        "expected_record_id",
        "expected_actor",
        "decoded_record_id",
        "decoded_actor",
        "record_correct",
        "bit_accuracy_vs_expected_pct",
        "distance_to_A",
        "distance_to_B",
        "distance_to_C",
        "best_distance",
        "runner_up_distance",
        "decision_margin",
        "registry_chain",
    ]

    with (output_dir / "results.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(results)


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

    codebook = make_codebook(args.seed)
    save_codebook(args.output_dir, codebook)

    messages = {
        record_id: codeword.unsqueeze(0).float().to(device)
        for record_id, codeword in codebook.items()
    }

    paths = {
        "A": args.output_dir / "A.mp4",
        "AB": args.output_dir / "AB.mp4",
        "ABC": args.output_dir / "ABC.mp4",
        "ABC_transcode": args.output_dir / "ABC_transcode.mp4",
    }

    embed_video(
        model,
        args.input,
        paths["A"],
        messages[1],
        args.chunk_size,
        args.crf,
        device,
    )

    embed_video(
        model,
        paths["A"],
        paths["AB"],
        messages[2],
        args.chunk_size,
        args.crf,
        device,
    )

    embed_video(
        model,
        paths["AB"],
        paths["ABC"],
        messages[3],
        args.chunk_size,
        args.crf,
        device,
    )

    transcode_h264(
        paths["ABC"],
        paths["ABC_transcode"],
        args.crf,
    )

    expected = {
        "A": 1,
        "AB": 2,
        "ABC": 3,
        "ABC_transcode": 3,
    }

    results: list[dict[str, Any]] = []

    for condition, path in paths.items():
        decoded_bits, _soft_bits, frames = detect_video(
            model,
            path,
            args.chunk_size,
            device,
        )

        decoded_id, distances, best_distance, second_distance, margin = (
            classify_record(decoded_bits, codebook)
        )

        expected_id = expected[condition]
        expected_code = codebook[expected_id]

        expected_accuracy = round(
            100
            * (
                PAYLOAD_BITS
                - hamming_distance(decoded_bits, expected_code)
            )
            / PAYLOAD_BITS,
            2,
        )

        (args.output_dir / f"decoded_{condition}.txt").write_text(
            bit_string(decoded_bits),
            encoding="utf-8",
        )

        results.append(
            {
                "condition": condition,
                "frames": frames,
                "expected_record_id": expected_id,
                "expected_actor": REGISTRY[str(expected_id)]["actor"],
                "decoded_record_id": decoded_id,
                "decoded_actor": REGISTRY[str(decoded_id)]["actor"],
                "record_correct": decoded_id == expected_id,
                "bit_accuracy_vs_expected_pct": expected_accuracy,
                "distance_to_A": distances[1],
                "distance_to_B": distances[2],
                "distance_to_C": distances[3],
                "best_distance": best_distance,
                "runner_up_distance": second_distance,
                "decision_margin": margin,
                "registry_chain": reconstruct_chain(decoded_id),
            }
        )

    save_results(args.output_dir, results)

    print(
        "\nCondition        Expected   Decoded   Correct   "
        "Best dist   Margin   Registry chain"
    )

    for result in results:
        correct = "yes" if result["record_correct"] else "no"

        print(
            f"{result['condition']:<16} "
            f"{result['expected_actor']:<10} "
            f"{result['decoded_actor']:<9} "
            f"{correct:<9} "
            f"{result['best_distance']:<11} "
            f"{result['decision_margin']:<8} "
            f"{result['registry_chain']}"
        )

    print(f"\nSaved results to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()