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

FIELD_BITS = 4
REPEATS = 8
LOGICAL_BITS = FIELD_BITS * 3
PAYLOAD_BITS = LOGICAL_BITS * REPEATS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pilot chained-token experiment for VideoSeal v0.0."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/chain"),
    )
    parser.add_argument(
        "--official_repo",
        type=Path,
        default=Path.home() / "Projects" / "videoseal",
    )
    parser.add_argument("--chunk_size", type=int, default=16)
    parser.add_argument("--crf", type=int, default=23)
    return parser.parse_args()


def int_to_bits(value: int) -> list[int]:
    if not 0 <= value < 2**FIELD_BITS:
        raise ValueError(f"Value must fit in {FIELD_BITS} bits: {value}")

    return [int(bit) for bit in f"{value:0{FIELD_BITS}b}"]


def bits_to_int(bits: torch.Tensor) -> int:
    return int("".join(str(int(bit)) for bit in bits.tolist()), 2)


def token_record(current_id: int, parent_id: int, step: int) -> dict[str, int]:
    return {
        "current_id": current_id,
        "parent_id": parent_id,
        "step": step,
    }


def make_token(
    current_id: int,
    parent_id: int,
    step: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    logical = torch.tensor(
        int_to_bits(current_id) + int_to_bits(parent_id) + int_to_bits(step),
        dtype=torch.float32,
    )

    physical = logical.repeat_interleave(REPEATS).unsqueeze(0)

    return physical.to(device), logical.to(torch.int64)


def decode_token(
    soft_bits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
    soft_bits = soft_bits.cpu()

    physical = (soft_bits > 0).to(torch.int64)
    logical_scores = soft_bits.reshape(LOGICAL_BITS, REPEATS).mean(dim=1)
    logical = (logical_scores > 0).to(torch.int64)

    record = token_record(
        bits_to_int(logical[0:4]),
        bits_to_int(logical[4:8]),
        bits_to_int(logical[8:12]),
    )

    return physical, logical, record


def reconstruct_chain(
    latest: dict[str, int],
    registry: dict[str, dict[str, Any]],
) -> tuple[str, bool, str]:
    latest_record = registry.get(str(latest["current_id"]))

    if latest_record is None:
        return "", False, "unknown current ID"

    if latest_record["parent"] != latest["parent_id"]:
        return "", False, "parent ID does not match registry"

    if latest_record["step"] != latest["step"]:
        return "", False, "step does not match registry"

    actors: list[str] = []
    seen: set[int] = set()
    current_id = latest["current_id"]

    while current_id != 0:
        if current_id in seen:
            return "", False, "cycle detected in registry"

        seen.add(current_id)
        record = registry.get(str(current_id))

        if record is None:
            return "", False, "missing parent record"

        actors.append(record["actor"])
        current_id = record["parent"]

    actors.reverse()
    return " -> ".join(actors), True, ""


def save_tokens(
    output_dir: Path,
    physical_tokens: dict[str, torch.Tensor],
    logical_tokens: dict[str, torch.Tensor],
) -> None:
    for name, token in physical_tokens.items():
        (output_dir / f"token_{name}.txt").write_text(
            bit_string(token.cpu()),
            encoding="utf-8",
        )

    for name, token in logical_tokens.items():
        (output_dir / f"logical_{name}.txt").write_text(
            bit_string(token.cpu()),
            encoding="utf-8",
        )


def save_results(output_dir: Path, results: list[dict[str, Any]]) -> None:
    fields = [
        "condition",
        "frames",
        "raw_bit_accuracy_pct",
        "logical_bit_accuracy_pct",
        "decoded_current_id",
        "decoded_parent_id",
        "decoded_step",
        "expected_current_id",
        "expected_parent_id",
        "expected_step",
        "latest_token_correct",
        "reconstructed_chain",
        "chain_valid",
        "chain_note",
    ]

    with (output_dir / "results.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fields)
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

    registry: dict[str, dict[str, Any]] = {
        "1": {"parent": 0, "step": 1, "actor": "A"},
        "2": {"parent": 1, "step": 2, "actor": "B"},
        "3": {"parent": 2, "step": 3, "actor": "C"},
    }

    (args.output_dir / "registry.json").write_text(
        json.dumps(registry, indent=2),
        encoding="utf-8",
    )

    token_a, logical_a = make_token(1, 0, 1, device)
    token_b, logical_b = make_token(2, 1, 2, device)
    token_c, logical_c = make_token(3, 2, 3, device)

    physical_tokens = {
        "A": token_a,
        "B": token_b,
        "C": token_c,
    }

    logical_tokens = {
        "A": logical_a,
        "B": logical_b,
        "C": logical_c,
    }

    save_tokens(args.output_dir, physical_tokens, logical_tokens)

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
        token_a,
        args.chunk_size,
        args.crf,
        device,
    )

    embed_video(
        model,
        paths["A"],
        paths["AB"],
        token_b,
        args.chunk_size,
        args.crf,
        device,
    )

    embed_video(
        model,
        paths["AB"],
        paths["ABC"],
        token_c,
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
        "A": (token_a, logical_a, token_record(1, 0, 1)),
        "AB": (token_b, logical_b, token_record(2, 1, 2)),
        "ABC": (token_c, logical_c, token_record(3, 2, 3)),
        "ABC_transcode": (token_c, logical_c, token_record(3, 2, 3)),
    }

    results: list[dict[str, Any]] = []

    for condition, path in paths.items():
        _decoded, soft_bits, frames = detect_video(
            model,
            path,
            args.chunk_size,
            device,
        )

        decoded_physical, decoded_logical, decoded_record = decode_token(soft_bits)

        expected_physical, expected_logical, expected_record = expected[condition]

        raw_matches = int(
            (decoded_physical == expected_physical.squeeze(0).cpu()).sum().item()
        )

        logical_matches = int(
            (decoded_logical == expected_logical.cpu()).sum().item()
        )

        latest_token_correct = decoded_record == expected_record

        chain, chain_valid, chain_note = reconstruct_chain(
            decoded_record,
            registry,
        )

        (args.output_dir / f"decoded_{condition}.txt").write_text(
            bit_string(decoded_physical),
            encoding="utf-8",
        )

        (args.output_dir / f"logical_decoded_{condition}.txt").write_text(
            bit_string(decoded_logical),
            encoding="utf-8",
        )

        results.append(
            {
                "condition": condition,
                "frames": frames,
                "raw_bit_accuracy_pct": round(100 * raw_matches / PAYLOAD_BITS, 2),
                "logical_bit_accuracy_pct": round(
                    100 * logical_matches / LOGICAL_BITS,
                    2,
                ),
                "decoded_current_id": decoded_record["current_id"],
                "decoded_parent_id": decoded_record["parent_id"],
                "decoded_step": decoded_record["step"],
                "expected_current_id": expected_record["current_id"],
                "expected_parent_id": expected_record["parent_id"],
                "expected_step": expected_record["step"],
                "latest_token_correct": latest_token_correct,
                "reconstructed_chain": chain,
                "chain_valid": chain_valid,
                "chain_note": chain_note,
            }
        )

    save_results(args.output_dir, results)

    print("\nCondition        Token correct   Chain                Chain valid")

    for result in results:
        token_ok = "yes" if result["latest_token_correct"] else "no"
        chain_ok = "yes" if result["chain_valid"] else "no"
        chain = result["reconstructed_chain"] or "-"

        print(
            f"{result['condition']:<16} "
            f"{token_ok:<15} "
            f"{chain:<20} "
            f"{chain_ok}"
        )

    print(f"\nSaved results to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
    