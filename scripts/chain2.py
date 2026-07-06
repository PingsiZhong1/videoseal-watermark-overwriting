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

ID_BITS = 3
CURRENT_REPEATS = 8
PARENT_REPEATS = 16
CHECK_REPEATS = 8

CURRENT_BITS = ID_BITS * CURRENT_REPEATS
PARENT_BITS = ID_BITS * PARENT_REPEATS
CHECK_BITS = ID_BITS * CHECK_REPEATS
PAYLOAD_BITS = CURRENT_BITS + PARENT_BITS + CHECK_BITS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Improved chained-token experiment for VideoSeal v0.0."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/chain2"),
    )
    parser.add_argument(
        "--official_repo",
        type=Path,
        default=Path.home() / "Projects" / "videoseal",
    )
    parser.add_argument("--chunk_size", type=int, default=16)
    parser.add_argument("--crf", type=int, default=23)
    return parser.parse_args()


def int_to_bits(value: int) -> torch.Tensor:
    if not 0 <= value < 2**ID_BITS:
        raise ValueError(f"ID must fit in {ID_BITS} bits: {value}")

    return torch.tensor(
        [int(bit) for bit in f"{value:0{ID_BITS}b}"],
        dtype=torch.int64,
    )


def bits_to_int(bits: torch.Tensor) -> int:
    return int("".join(str(int(bit)) for bit in bits.tolist()), 2)


def repeat_bits(bits: torch.Tensor, repeats: int) -> torch.Tensor:
    return bits.repeat_interleave(repeats)


def check_bits(current_bits: torch.Tensor, parent_bits: torch.Tensor) -> torch.Tensor:
    return torch.bitwise_xor(current_bits, parent_bits)


def make_token(
    current_id: int,
    parent_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    current = int_to_bits(current_id)
    parent = int_to_bits(parent_id)
    check = check_bits(current, parent)

    physical = torch.cat(
        [
            repeat_bits(current, CURRENT_REPEATS),
            repeat_bits(parent, PARENT_REPEATS),
            repeat_bits(check, CHECK_REPEATS),
        ]
    ).float()

    fields = {
        "current": current,
        "parent": parent,
        "check": check,
    }

    return physical.unsqueeze(0).to(device), fields


def decode_field(
    soft_bits: torch.Tensor,
    start: int,
    repeats: int,
) -> torch.Tensor:
    field = soft_bits[start : start + ID_BITS * repeats]
    scores = field.reshape(ID_BITS, repeats).mean(dim=1)
    return (scores > 0).to(torch.int64)


def decode_token(
    soft_bits: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, int], bool]:
    soft_bits = soft_bits.cpu()
    physical = (soft_bits > 0).to(torch.int64)

    current = decode_field(soft_bits, 0, CURRENT_REPEATS)
    parent = decode_field(soft_bits, CURRENT_BITS, PARENT_REPEATS)
    check = decode_field(
        soft_bits,
        CURRENT_BITS + PARENT_BITS,
        CHECK_REPEATS,
    )

    expected_check = check_bits(current, parent)
    check_valid = torch.equal(check, expected_check)

    record = {
        "current_id": bits_to_int(current),
        "parent_id": bits_to_int(parent),
    }

    fields = {
        "current": current,
        "parent": parent,
        "check": check,
    }

    return physical, fields, record, check_valid


def reconstruct_chain(
    decoded: dict[str, int],
    registry: dict[str, dict[str, Any]],
    check_valid: bool,
) -> tuple[str, bool, str]:
    if not check_valid:
        return "", False, "token consistency check failed"

    record = registry.get(str(decoded["current_id"]))

    if record is None:
        return "", False, "unknown current ID"

    if record["parent"] != decoded["parent_id"]:
        return "", False, "parent ID does not match registry"

    actors: list[str] = []
    seen: set[int] = set()
    current_id = decoded["current_id"]

    while current_id != 0:
        if current_id in seen:
            return "", False, "cycle detected"

        seen.add(current_id)
        node = registry.get(str(current_id))

        if node is None:
            return "", False, "missing record"

        actors.append(node["actor"])
        current_id = node["parent"]

    actors.reverse()
    return " -> ".join(actors), True, ""


def save_token(
    output_dir: Path,
    name: str,
    token: torch.Tensor,
    fields: dict[str, torch.Tensor],
) -> None:
    (output_dir / f"token_{name}.txt").write_text(
        bit_string(token.cpu()),
        encoding="utf-8",
    )

    text = (
        f"current={bit_string(fields['current'])}\n"
        f"parent={bit_string(fields['parent'])}\n"
        f"check={bit_string(fields['check'])}\n"
    )

    (output_dir / f"fields_{name}.txt").write_text(
        text,
        encoding="utf-8",
    )


def save_results(output_dir: Path, results: list[dict[str, Any]]) -> None:
    columns = [
        "condition",
        "frames",
        "raw_bit_accuracy_pct",
        "decoded_current_id",
        "decoded_parent_id",
        "expected_current_id",
        "expected_parent_id",
        "current_id_correct",
        "parent_id_correct",
        "check_valid",
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

    registry: dict[str, dict[str, Any]] = {
        "1": {"parent": 0, "step": 1, "actor": "A"},
        "2": {"parent": 1, "step": 2, "actor": "B"},
        "3": {"parent": 2, "step": 3, "actor": "C"},
    }

    (args.output_dir / "registry.json").write_text(
        json.dumps(registry, indent=2),
        encoding="utf-8",
    )

    token_a, fields_a = make_token(1, 0, device)
    token_b, fields_b = make_token(2, 1, device)
    token_c, fields_c = make_token(3, 2, device)

    save_token(args.output_dir, "A", token_a, fields_a)
    save_token(args.output_dir, "B", token_b, fields_b)
    save_token(args.output_dir, "C", token_c, fields_c)

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
        "A": (token_a, {"current_id": 1, "parent_id": 0}),
        "AB": (token_b, {"current_id": 2, "parent_id": 1}),
        "ABC": (token_c, {"current_id": 3, "parent_id": 2}),
        "ABC_transcode": (token_c, {"current_id": 3, "parent_id": 2}),
    }

    results: list[dict[str, Any]] = []

    for condition, path in paths.items():
        _decoded, soft_bits, frames = detect_video(
            model,
            path,
            args.chunk_size,
            device,
        )

        decoded_physical, _fields, decoded_record, check_valid = decode_token(
            soft_bits
        )

        expected_physical, expected_record = expected[condition]

        raw_matches = int(
            (
                decoded_physical
                == expected_physical.squeeze(0).cpu().to(torch.int64)
            )
            .sum()
            .item()
        )

        current_ok = (
            decoded_record["current_id"] == expected_record["current_id"]
        )
        parent_ok = (
            decoded_record["parent_id"] == expected_record["parent_id"]
        )

        latest_token_correct = current_ok and parent_ok and check_valid

        chain, chain_valid, chain_note = reconstruct_chain(
            decoded_record,
            registry,
            check_valid,
        )

        (args.output_dir / f"decoded_{condition}.txt").write_text(
            bit_string(decoded_physical),
            encoding="utf-8",
        )

        results.append(
            {
                "condition": condition,
                "frames": frames,
                "raw_bit_accuracy_pct": round(
                    100 * raw_matches / PAYLOAD_BITS,
                    2,
                ),
                "decoded_current_id": decoded_record["current_id"],
                "decoded_parent_id": decoded_record["parent_id"],
                "expected_current_id": expected_record["current_id"],
                "expected_parent_id": expected_record["parent_id"],
                "current_id_correct": current_ok,
                "parent_id_correct": parent_ok,
                "check_valid": check_valid,
                "latest_token_correct": latest_token_correct,
                "reconstructed_chain": chain,
                "chain_valid": chain_valid,
                "chain_note": chain_note,
            }
        )

    save_results(args.output_dir, results)

    print("\nCondition        Token   Current   Parent   Check   Chain          Valid")

    for result in results:
        token = "yes" if result["latest_token_correct"] else "no"
        check = "yes" if result["check_valid"] else "no"
        valid = "yes" if result["chain_valid"] else "no"
        chain = result["reconstructed_chain"] or "-"

        print(
            f"{result['condition']:<16} "
            f"{token:<7} "
            f"{result['decoded_current_id']:<9} "
            f"{result['decoded_parent_id']:<8} "
            f"{check:<7} "
            f"{chain:<14} "
            f"{valid}"
        )

    print(f"\nSaved results to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()