# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import gzip
import json
import statistics
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_dir", type=Path)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as file:
            return [json.loads(line) for line in file]
    return [json.loads(line) for line in path.read_text().splitlines()]


def find_trace(mode_dir: Path, stream: str) -> Path:
    paths = list(mode_dir.glob(f"kv_lifecycle_{stream}_*.jsonl"))
    if not paths:
        paths = list(mode_dir.glob(f"kv_lifecycle_{stream}_*.jsonl.gz"))
    if len(paths) != 1:
        raise ValueError(
            f"Expected one {stream} trace in {mode_dir}, found {len(paths)}"
        )
    return paths[0]


def internal_request_map(core: list[dict[str, Any]]) -> dict[str, str]:
    return {
        event["request_id"].split("-", 1)[0]: event["request_id"]
        for event in core
        if event["event"] == "REQUEST_ADD"
    }


def median_rows(
    rows: list[dict[str, Any]],
    group_keys: tuple[str, ...],
    metrics: tuple[str, ...],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in group_keys)].append(row)
    result = []
    for group, values in sorted(grouped.items()):
        result.append(
            {
                **dict(zip(group_keys, group, strict=True)),
                **{
                    metric: statistics.median(value[metric] for value in values)
                    for metric in metrics
                },
                "samples": len(values),
            }
        )
    return result


def analyze_mode(mode_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    manifest = json.loads((mode_dir / "experiment_manifest.json").read_text())
    core = read_jsonl(find_trace(mode_dir, "core"))
    detail = read_jsonl(find_trace(mode_dir, "detail"))
    cache_enabled = manifest["prefix_cache_enabled"]
    client_to_internal = internal_request_map(core)
    summaries = {
        event["request_id"]: event
        for event in core
        if event["event"] == "REQUEST_SUMMARY"
    }
    lookups = {
        event["request_id"]: event
        for event in detail
        if event["event"] == "PREFIX_LOOKUP"
    }
    schedules: dict[str, list[dict[str, Any]]] = defaultdict(list)
    hashes = {
        event["request_id"]: event
        for event in detail
        if event["event"] == "PROMPT_HASHES"
    }
    for event in detail:
        if event["event"] == "SCHEDULE_STEP":
            schedules[event["request_id"]].append(event)

    probe_rows = []
    throughput_rows = []
    for trial in manifest["trials"]:
        if not trial["measured"]:
            continue
        seed_id = client_to_internal[trial["seed"]["request_id"]]
        seed_hashes = hashes[seed_id]["prompt_block_hashes"]
        for probe in trial["probes"]:
            probe_id = client_to_internal[probe["request_id"]]
            summary = summaries[probe_id]
            lookup = lookups[probe_id]
            first_schedule = schedules[probe_id][0]
            lcp_tokens = probe["longest_common_prefix_tokens"]
            prefix_hit_tokens = lookup["prefix_hit_tokens"]
            assert prefix_hit_tokens <= lcp_tokens
            if cache_enabled:
                assert prefix_hit_tokens % manifest["block_size_tokens"] == 0
            else:
                assert prefix_hit_tokens == 0
            assert summary["prefill_tokens_scheduled"] == (
                probe["prompt_tokens"] - prefix_hit_tokens
            )

            probe_hashes = hashes[probe_id]["prompt_block_hashes"]
            equal_hash_blocks = 0
            for seed_hash, probe_hash in zip(seed_hashes, probe_hashes, strict=True):
                if seed_hash["token_ids_sha256"] != probe_hash["token_ids_sha256"]:
                    break
                equal_hash_blocks += 1
            probe_rows.append(
                {
                    "prefix_cache_enabled": cache_enabled,
                    "suite": trial["suite"],
                    "case": trial["case"],
                    "concurrency": trial["concurrency"],
                    "repetition": trial["repetition"],
                    "probe_index": probe["probe_index"],
                    "request_id": probe_id,
                    "prompt_tokens": probe["prompt_tokens"],
                    "first_changed_token_index": probe["first_changed_token_index"],
                    "longest_common_prefix_tokens": lcp_tokens,
                    "equal_prefix_hash_blocks": equal_hash_blocks,
                    "prefix_hit_tokens": prefix_hit_tokens,
                    "prefix_hit_blocks": lookup["prefix_hit_block_references"],
                    "newly_computed_tokens": probe["prompt_tokens"] - lcp_tokens,
                    "recomputed_common_tokens": max(lcp_tokens - prefix_hit_tokens, 0),
                    "prefill_tokens_scheduled": summary["prefill_tokens_scheduled"],
                    "reused_block_count": first_schedule["reused_block_count"],
                    "allocated_block_count": first_schedule["allocated_block_count"],
                    "reused_block_ids_by_group": first_schedule[
                        "reused_block_ids_by_group"
                    ],
                    "allocated_block_ids_by_group": first_schedule[
                        "allocated_block_ids_by_group"
                    ],
                    "hit_ref_cnt_before_touch_by_group": lookup[
                        "hit_block_ref_cnt_before_touch_by_group"
                    ],
                    "ttft_ms": summary["ttft_ms"],
                    "mean_tpot_ms": summary["mean_tpot_ms"],
                    "total_ms": summary["total_ms"],
                }
            )
        throughput_rows.append(
            {
                "prefix_cache_enabled": cache_enabled,
                "suite": trial["suite"],
                "case": trial["case"],
                "concurrency": trial["concurrency"],
                "repetition": trial["repetition"],
                "requests_per_second": trial["probe_requests_per_second"],
                "output_tokens_per_second": trial["probe_output_tokens_per_second"],
                "logical_tokens_per_second": trial["probe_logical_tokens_per_second"],
                "probe_elapsed_ms": trial["probe_elapsed_s"] * 1000,
            }
        )
    return probe_rows, throughput_rows


def write_relationship_svg(rows: list[dict[str, Any]], path: Path) -> None:
    sequential = [row for row in rows if row["suite"] == "sequential_prompt_change"]
    aggregate = median_rows(
        sequential,
        ("prefix_cache_enabled", "case"),
        ("first_changed_token_index", "prefix_hit_tokens", "ttft_ms"),
    )
    cases = [
        "C_first_token_changed",
        "B_shared_prefix",
        "D_last_four_changed",
        "A_identical",
    ]
    labels = ["C · token 0", "B · token 2048", "D · token 2108", "A · identical"]
    by_key = {(row["prefix_cache_enabled"], row["case"]): row for row in aggregate}
    width, height = 1000, 560
    left, right, top, bottom = 85, 90, 55, 90
    plot_width = width - left - right
    plot_height = height - top - bottom
    max_tokens = 2112
    max_ttft = max(row["ttft_ms"] for row in aggregate) * 1.15

    def token_y(value: float) -> float:
        return top + plot_height * (1 - value / max_tokens)

    def ttft_y(value: float) -> float:
        return top + plot_height * (1 - value / max_ttft)

    x_positions = [
        left + plot_width * (index + 0.5) / len(cases) for index in range(len(cases))
    ]
    root = ET.Element(
        "svg",
        {
            "xmlns": "http://www.w3.org/2000/svg",
            "width": str(width),
            "height": str(height),
            "viewBox": f"0 0 {width} {height}",
        },
    )

    def add(tag: str, attrs: dict[str, Any], text: str | None = None) -> None:
        element = ET.SubElement(
            root, tag, {key: str(value) for key, value in attrs.items()}
        )
        element.text = text

    add("rect", {"width": "100%", "height": "100%", "fill": "white"})
    add(
        "style",
        {},
        "text{font-family:Arial,sans-serif;fill:#202124}"
        ".axis{stroke:#5f6368;stroke-width:1}"
        ".grid{stroke:#dadce0;stroke-width:1}"
        ".on{stroke:#1a73e8;fill:#1a73e8}"
        ".off{stroke:#d93025;fill:#d93025}"
        ".hit{fill:#8ab4f8;opacity:.75}",
    )
    add(
        "text",
        {"x": 500, "y": 28, "text-anchor": "middle", "font-size": 20},
        "Prompt change position → Prefix hit length → TTFT",
    )
    for tick in (0, 512, 1024, 1536, 2048):
        y = token_y(tick)
        add(
            "line",
            {
                "class": "grid",
                "x1": left,
                "y1": f"{y:.1f}",
                "x2": width - right,
                "y2": f"{y:.1f}",
            },
        )
        add(
            "text",
            {
                "x": left - 10,
                "y": f"{y + 4:.1f}",
                "text-anchor": "end",
                "font-size": 12,
            },
            str(tick),
        )
    for index, (case, label, x) in enumerate(
        zip(cases, labels, x_positions, strict=True)
    ):
        on = by_key[(True, case)]
        off = by_key[(False, case)]
        bar_width = 42
        hit_y = token_y(on["prefix_hit_tokens"])
        add(
            "rect",
            {
                "class": "hit",
                "x": f"{x - bar_width / 2:.1f}",
                "y": f"{hit_y:.1f}",
                "width": bar_width,
                "height": f"{top + plot_height - hit_y:.1f}",
            },
        )
        add(
            "text",
            {
                "x": f"{x:.1f}",
                "y": f"{max(hit_y - 8, top + 14):.1f}",
                "text-anchor": "middle",
                "font-size": 12,
            },
            f"hit {on['prefix_hit_tokens']:.0f}",
        )
        add(
            "text",
            {
                "x": f"{x:.1f}",
                "y": height - bottom + 28,
                "text-anchor": "middle",
                "font-size": 12,
            },
            label,
        )
        add(
            "circle",
            {
                "class": "on",
                "cx": f"{x - 10:.1f}",
                "cy": f"{ttft_y(on['ttft_ms']):.1f}",
                "r": 5,
            },
        )
        add(
            "circle",
            {
                "class": "off",
                "cx": f"{x + 10:.1f}",
                "cy": f"{ttft_y(off['ttft_ms']):.1f}",
                "r": 5,
            },
        )
        if index:
            prev_case = cases[index - 1]
            prev_x = x_positions[index - 1]
            for cache_enabled, css_class, offset, current in (
                (True, "on", -10, on),
                (False, "off", 10, off),
            ):
                previous = by_key[(cache_enabled, prev_case)]
                add(
                    "line",
                    {
                        "class": css_class,
                        "fill": "none",
                        "x1": f"{prev_x + offset:.1f}",
                        "y1": f"{ttft_y(previous['ttft_ms']):.1f}",
                        "x2": f"{x + offset:.1f}",
                        "y2": f"{ttft_y(current['ttft_ms']):.1f}",
                    },
                )
    for x1, y1, x2, y2 in (
        (left, top, left, top + plot_height),
        (width - right, top, width - right, top + plot_height),
        (left, top + plot_height, width - right, top + plot_height),
    ):
        add(
            "line",
            {"class": "axis", "x1": x1, "y1": y1, "x2": x2, "y2": y2},
        )
    add(
        "text",
        {
            "x": 20,
            "y": top + plot_height / 2,
            "transform": f"rotate(-90 20 {top + plot_height / 2})",
            "text-anchor": "middle",
            "font-size": 13,
        },
        "Prefix hit tokens (bars)",
    )
    add(
        "text",
        {
            "x": width - 18,
            "y": top + plot_height / 2,
            "transform": f"rotate(90 {width - 18} {top + plot_height / 2})",
            "text-anchor": "middle",
            "font-size": 13,
        },
        "TTFT ms (points)",
    )
    add(
        "text",
        {"x": width - right + 10, "y": top + 4, "font-size": 11},
        f"{max_ttft:.0f} ms",
    )
    add(
        "text",
        {
            "x": width - right + 10,
            "y": top + plot_height + 4,
            "font-size": 11,
        },
        "0 ms",
    )
    legend = (
        (
            "rect",
            {"class": "hit", "x": 320, "y": 520, "width": 16, "height": 10},
            342,
            "Prefix hit tokens",
        ),
        (
            "circle",
            {"class": "on", "cx": 485, "cy": 525, "r": 5},
            497,
            "Cache ON TTFT",
        ),
        (
            "circle",
            {"class": "off", "cx": 625, "cy": 525, "r": 5},
            637,
            "Cache OFF TTFT",
        ),
    )
    for shape, attrs, text_x, label in legend:
        add(shape, attrs)
        add("text", {"x": text_x, "y": 530, "font-size": 12}, label)
    ET.indent(root)
    ET.ElementTree(root).write(
        path,
        encoding="unicode",
        xml_declaration=False,
    )
    with path.open("a", encoding="utf-8") as file:
        file.write("\n")


def main() -> None:
    args = parse_args()
    experiment_dir = args.experiment_dir.resolve()
    all_probe_rows = []
    all_throughput_rows = []
    for mode in ("off", "on"):
        probe_rows, throughput_rows = analyze_mode(experiment_dir / mode)
        all_probe_rows.extend(probe_rows)
        all_throughput_rows.extend(throughput_rows)

    probe_aggregate = median_rows(
        all_probe_rows,
        ("prefix_cache_enabled", "suite", "case", "concurrency"),
        (
            "first_changed_token_index",
            "longest_common_prefix_tokens",
            "equal_prefix_hash_blocks",
            "prefix_hit_tokens",
            "prefix_hit_blocks",
            "newly_computed_tokens",
            "recomputed_common_tokens",
            "prefill_tokens_scheduled",
            "ttft_ms",
            "mean_tpot_ms",
            "total_ms",
        ),
    )
    throughput_aggregate = median_rows(
        all_throughput_rows,
        ("prefix_cache_enabled", "suite", "case", "concurrency"),
        (
            "requests_per_second",
            "output_tokens_per_second",
            "logical_tokens_per_second",
            "probe_elapsed_ms",
        ),
    )
    analysis = {
        "all_invariants_passed": True,
        "probe_rows": all_probe_rows,
        "probe_medians": probe_aggregate,
        "throughput_rows": all_throughput_rows,
        "throughput_medians": throughput_aggregate,
    }
    result_path = experiment_dir / "analysis_summary.json"
    result_path.write_text(json.dumps(analysis, indent=2) + "\n", encoding="utf-8")
    write_relationship_svg(
        all_probe_rows,
        experiment_dir / "prompt_change_prefix_hit_ttft.svg",
    )
    print(result_path)


if __name__ == "__main__":
    main()
