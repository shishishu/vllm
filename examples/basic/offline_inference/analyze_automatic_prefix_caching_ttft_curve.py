# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import csv
import json
import statistics
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from analyze_automatic_prefix_caching import (
    find_trace,
    internal_request_map,
    read_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_dir", type=Path)
    return parser.parse_args()


def linear_regression(points: list[tuple[float, float]]) -> dict[str, float]:
    mean_x = statistics.mean(point[0] for point in points)
    mean_y = statistics.mean(point[1] for point in points)
    sum_xx = sum((x - mean_x) ** 2 for x, _ in points)
    sum_xy = sum((x - mean_x) * (y - mean_y) for x, y in points)
    slope = sum_xy / sum_xx
    intercept = mean_y - slope * mean_x
    residual = sum((y - (intercept + slope * x)) ** 2 for x, y in points)
    total = sum((y - mean_y) ** 2 for _, y in points)
    return {
        "slope_ms_per_prefill_token": slope,
        "intercept_ms": intercept,
        "r_squared": 1 - residual / total,
    }


def causal_work_regression(
    rows: list[dict[str, Any]], prompt_tokens: int
) -> dict[str, float]:
    query_tokens = np.array(
        [row["prefill_tokens_scheduled"] for row in rows], dtype=np.float64
    )
    ttft_ms = np.array([row["median_ttft_ms"] for row in rows], dtype=np.float64)
    causal_attention_pairs = query_tokens * (2 * prompt_tokens - query_tokens + 1) / 2
    design = np.column_stack(
        (np.ones_like(query_tokens), query_tokens, causal_attention_pairs)
    )
    coefficients = np.linalg.lstsq(design, ttft_ms, rcond=None)[0]
    predicted = design @ coefficients
    residual = np.square(ttft_ms - predicted).sum()
    total = np.square(ttft_ms - ttft_ms.mean()).sum()
    return {
        "intercept_ms": float(coefficients[0]),
        "ms_per_query_token": float(coefficients[1]),
        "ms_per_causal_attention_pair": float(coefficients[2]),
        "r_squared": float(1 - residual / total),
    }


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["prefix_hit_tokens"]].append(row)
    result = []
    for hit_tokens, values in sorted(grouped.items()):
        ttfts = [value["ttft_ms"] for value in values]
        result.append(
            {
                "prefix_hit_tokens": hit_tokens,
                "prefix_hit_blocks": values[0]["prefix_hit_blocks"],
                "prefill_tokens_scheduled": values[0]["prefill_tokens_scheduled"],
                "median_ttft_ms": statistics.median(ttfts),
                "min_ttft_ms": min(ttfts),
                "max_ttft_ms": max(ttfts),
                "samples": len(values),
            }
        )
    return result


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def write_svg(rows: list[dict[str, Any]], path: Path) -> None:
    width, height = 1080, 620
    left, right, top, bottom = 90, 40, 60, 90
    plot_width = width - left - right
    plot_height = height - top - bottom
    max_x = max(row["prefix_hit_tokens"] for row in rows)
    max_y = max(row["max_ttft_ms"] for row in rows) * 1.08

    def x_position(value: float) -> float:
        return left + plot_width * value / max_x

    def y_position(value: float) -> float:
        return top + plot_height * (1 - value / max_y)

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
        ".range{stroke:#8ab4f8;stroke-width:2}"
        ".curve{stroke:#1a73e8;stroke-width:3;fill:none}"
        ".point{stroke:#1a73e8;stroke-width:2;fill:white}",
    )
    add(
        "text",
        {"x": width / 2, "y": 30, "text-anchor": "middle", "font-size": 20},
        "Prefix reuse length → TTFT (2112-token prompt)",
    )
    for fraction in (0, 0.25, 0.5, 0.75, 1):
        y_value = max_y * fraction
        y = y_position(y_value)
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
            f"{y_value:.0f}",
        )
    for x_value in (0, 512, 1024, 1536, 2048, 2096):
        x = x_position(x_value)
        add(
            "line",
            {
                "class": "grid",
                "x1": f"{x:.1f}",
                "y1": top,
                "x2": f"{x:.1f}",
                "y2": top + plot_height,
            },
        )
        add(
            "text",
            {
                "x": f"{x:.1f}",
                "y": height - bottom + 25,
                "text-anchor": "middle",
                "font-size": 12,
            },
            str(x_value),
        )
    curve_points = []
    for row in rows:
        x = x_position(row["prefix_hit_tokens"])
        median_y = y_position(row["median_ttft_ms"])
        min_y = y_position(row["min_ttft_ms"])
        max_y_position = y_position(row["max_ttft_ms"])
        add(
            "line",
            {
                "class": "range",
                "x1": f"{x:.1f}",
                "y1": f"{min_y:.1f}",
                "x2": f"{x:.1f}",
                "y2": f"{max_y_position:.1f}",
            },
        )
        curve_points.append(f"{x:.1f},{median_y:.1f}")
    add("polyline", {"class": "curve", "points": " ".join(curve_points)})
    for row in rows:
        add(
            "circle",
            {
                "class": "point",
                "cx": f"{x_position(row['prefix_hit_tokens']):.1f}",
                "cy": f"{y_position(row['median_ttft_ms']):.1f}",
                "r": 4,
            },
        )
    for x1, y1, x2, y2 in (
        (left, top, left, top + plot_height),
        (left, top + plot_height, width - right, top + plot_height),
    ):
        add(
            "line",
            {"class": "axis", "x1": x1, "y1": y1, "x2": x2, "y2": y2},
        )
    add(
        "text",
        {
            "x": 22,
            "y": top + plot_height / 2,
            "transform": f"rotate(-90 22 {top + plot_height / 2})",
            "text-anchor": "middle",
            "font-size": 13,
        },
        "TTFT (ms)",
    )
    add(
        "text",
        {
            "x": left + plot_width / 2,
            "y": height - 20,
            "text-anchor": "middle",
            "font-size": 13,
        },
        "Actual prefix hit tokens (16-token blocks)",
    )
    add(
        "text",
        {"x": width - right, "y": 48, "text-anchor": "end", "font-size": 11},
        "Points: median; vertical line: min–max",
    )
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding="unicode", xml_declaration=False)
    with path.open("a", encoding="utf-8") as file:
        file.write("\n")


def main() -> None:
    args = parse_args()
    experiment_dir = args.experiment_dir.resolve()
    manifest = json.loads((experiment_dir / "experiment_manifest.json").read_text())
    core = read_jsonl(find_trace(experiment_dir, "core"))
    detail = read_jsonl(find_trace(experiment_dir, "detail"))
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
    rows = []
    for trial in manifest["trials"]:
        if not trial["measured"]:
            continue
        probe = trial["probes"][0]
        request_id = client_to_internal[probe["request_id"]]
        lookup = lookups[request_id]
        summary = summaries[request_id]
        target_hit_tokens = trial["target_prefix_hit_tokens"]
        assert probe["longest_common_prefix_tokens"] == target_hit_tokens
        assert lookup["prefix_hit_tokens"] == target_hit_tokens
        assert summary["prefill_tokens_scheduled"] == (
            manifest["prompt_tokens"] - target_hit_tokens
        )
        rows.append(
            {
                "repetition": trial["repetition"],
                "request_id": request_id,
                "prefix_hit_tokens": lookup["prefix_hit_tokens"],
                "prefix_hit_blocks": lookup["prefix_hit_block_references"],
                "prefill_tokens_scheduled": summary["prefill_tokens_scheduled"],
                "ttft_ms": summary["ttft_ms"],
                "mean_tpot_ms": summary["mean_tpot_ms"],
                "total_ms": summary["total_ms"],
            }
        )
    aggregate = aggregate_rows(rows)
    regression = linear_regression(
        [(row["prefill_tokens_scheduled"], row["median_ttft_ms"]) for row in aggregate]
    )
    endpoints = {row["prefix_hit_tokens"]: row for row in aggregate}
    regression["ttft_reduction_0_to_2096_ms"] = (
        endpoints[0]["median_ttft_ms"] - endpoints[2096]["median_ttft_ms"]
    )
    regression["ttft_reduction_0_to_2096_percent"] = (
        regression["ttft_reduction_0_to_2096_ms"] / endpoints[0]["median_ttft_ms"] * 100
    )
    monotonicity_violations = []
    for previous, current in zip(aggregate, aggregate[1:]):
        if current["median_ttft_ms"] > previous["median_ttft_ms"]:
            monotonicity_violations.append(
                {
                    "from_prefix_hit_tokens": previous["prefix_hit_tokens"],
                    "to_prefix_hit_tokens": current["prefix_hit_tokens"],
                    "ttft_increase_ms": (
                        current["median_ttft_ms"] - previous["median_ttft_ms"]
                    ),
                }
            )
    analysis = {
        "all_invariants_passed": True,
        "samples": rows,
        "medians": aggregate,
        "linear_prefill_token_model": regression,
        "causal_work_model": causal_work_regression(
            aggregate, manifest["prompt_tokens"]
        ),
        "monotonicity_violations": monotonicity_violations,
    }
    (experiment_dir / "ttft_curve_summary.json").write_text(
        json.dumps(analysis, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(aggregate, experiment_dir / "prefix_hit_tokens_ttft.csv")
    write_svg(aggregate, experiment_dir / "prefix_hit_tokens_ttft.svg")
    print(experiment_dir / "ttft_curve_summary.json")


if __name__ == "__main__":
    main()
