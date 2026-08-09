# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_dir", type=Path)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def median(values: list[float]) -> float:
    assert values
    return statistics.median(values)


def analyze_run(run_dir: Path) -> dict[str, Any]:
    manifest = json.loads((run_dir / "experiment_manifest.json").read_text())
    core = read_jsonl(next(run_dir.glob("kv_lifecycle_core_*.jsonl")))
    detail = read_jsonl(next(run_dir.glob("kv_lifecycle_detail_*.jsonl")))
    summaries = {
        event["request_id"]: event
        for event in core
        if event["event"] == "REQUEST_SUMMARY"
    }
    steps_by_request: dict[str, list[dict[str, Any]]] = defaultdict(list)
    steps_by_engine_step: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for event in core:
        if event["event"] == "STEP_COMPLETE":
            steps_by_request[event["request_id"]].append(event)
            steps_by_engine_step[event["engine_step"]].append(event)
    schedules_by_request: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in detail:
        if event["event"] == "SCHEDULE_STEP":
            schedules_by_request[event["request_id"]].append(event)

    trial_rows = []
    for trial in manifest["trials"]:
        if not trial["measured"]:
            continue
        ongoing_id = trial["requests"]["ongoing"]
        newcomer_id = trial["requests"]["newcomer"]
        long_id = trial["requests"]["long"]
        ongoing_token_steps = [
            event
            for event in steps_by_request[ongoing_id]
            if event["num_new_output_tokens"] > 0
        ]
        assert len(ongoing_token_steps) == 16
        join_step = steps_by_request[newcomer_id][0]["engine_step"]
        join_step_events = steps_by_engine_step[join_step]
        join_step_ms = max(event["engine_step_ms"] for event in join_step_events)
        ongoing_itl_ms = [
            (current["monotonic_time_ns"] - previous["monotonic_time_ns"]) / 1e6
            for previous, current in zip(
                ongoing_token_steps,
                ongoing_token_steps[1:],
            )
        ]
        join_step_index = next(
            index
            for index, event in enumerate(ongoing_token_steps)
            if event["engine_step"] == join_step
        )
        assert join_step_index > 0
        ongoing_join_step_gap_ms = ongoing_itl_ms[join_step_index - 1]

        long_prefill_chunks = []
        if long_id is not None:
            long_prefill_chunks = [
                event["num_scheduled_tokens"]
                for event in schedules_by_request[long_id]
                if event["phase"] == "PREFILL"
            ]

        trial_rows.append(
            {
                "trial": trial["trial"],
                "condition": trial["condition"],
                "join_engine_step": join_step,
                "join_step_request_count": len(join_step_events),
                "join_step_ms": join_step_ms,
                "ongoing_join_step_gap_ms": ongoing_join_step_gap_ms,
                "ongoing_max_itl_ms": max(ongoing_itl_ms),
                "ongoing_ttft_ms": summaries[ongoing_id]["ttft_ms"],
                "ongoing_mean_tpot_ms": summaries[ongoing_id]["mean_tpot_ms"],
                "ongoing_total_ms": summaries[ongoing_id]["total_ms"],
                "newcomer_ttft_ms": summaries[newcomer_id]["ttft_ms"],
                "newcomer_total_ms": summaries[newcomer_id]["total_ms"],
                "long_ttft_ms": (
                    summaries[long_id]["ttft_ms"] if long_id is not None else None
                ),
                "long_prefill_chunks": long_prefill_chunks,
                "elapsed_after_arrival_ms": trial["elapsed_after_arrival_ms"],
            }
        )

    metrics = [
        "join_step_ms",
        "ongoing_join_step_gap_ms",
        "ongoing_max_itl_ms",
        "ongoing_mean_tpot_ms",
        "ongoing_total_ms",
        "newcomer_ttft_ms",
        "newcomer_total_ms",
        "elapsed_after_arrival_ms",
    ]
    aggregate: dict[str, dict[str, float]] = {}
    for condition in ("baseline", "interference"):
        rows = [row for row in trial_rows if row["condition"] == condition]
        aggregate[condition] = {
            metric: median([row[metric] for row in rows]) for metric in metrics
        }
    interference = aggregate["interference"]
    baseline = aggregate["baseline"]
    deltas = {
        metric: {
            "absolute_ms": interference[metric] - baseline[metric],
            "ratio": interference[metric] / baseline[metric],
        }
        for metric in metrics
    }

    interference_rows = [
        row for row in trial_rows if row["condition"] == "interference"
    ]
    aggregate["interference"]["long_ttft_ms"] = median(
        [row["long_ttft_ms"] for row in interference_rows]
    )
    example_trial = next(
        trial
        for trial in manifest["trials"]
        if trial["measured"] and trial["condition"] == "interference"
    )
    role_by_request_id = {
        request_id: role
        for role, request_id in example_trial["requests"].items()
        if request_id is not None
    }
    example_long_id = example_trial["requests"]["long"]
    example_timeline = []
    for long_schedule in schedules_by_request[example_long_id]:
        if long_schedule["phase"] != "PREFILL":
            continue
        engine_step = long_schedule["engine_step"]
        scheduled_requests = []
        for request_id, role in role_by_request_id.items():
            event = next(
                (
                    event
                    for event in schedules_by_request[request_id]
                    if event["engine_step"] == engine_step
                ),
                None,
            )
            if event is not None:
                scheduled_requests.append(
                    {
                        "role": role,
                        "phase": event["phase"],
                        "scheduled_tokens": event["num_scheduled_tokens"],
                    }
                )
        example_timeline.append(
            {
                "engine_step": engine_step,
                "engine_step_ms": max(
                    event["engine_step_ms"]
                    for event in steps_by_engine_step[engine_step]
                ),
                "requests": scheduled_requests,
            }
        )
    return {
        "chunked_prefill": manifest["chunked_prefill"],
        "max_num_batched_tokens": manifest["max_num_batched_tokens"],
        "repetitions": manifest["repetitions"],
        "workload": manifest["workload"],
        "trial_rows": trial_rows,
        "median": aggregate,
        "interference_minus_baseline": deltas,
        "long_prefill_chunks": [
            row["long_prefill_chunks"] for row in interference_rows
        ],
        "example_timeline": example_timeline,
    }


def main() -> None:
    args = parse_args()
    experiment_dir = args.experiment_dir.resolve()
    runs = [
        analyze_run(experiment_dir / "unchunked"),
        analyze_run(experiment_dir / "chunked"),
    ]
    unchunked = runs[0]["median"]["interference"]
    chunked = runs[1]["median"]["interference"]
    cross_metrics = [
        "join_step_ms",
        "ongoing_join_step_gap_ms",
        "ongoing_max_itl_ms",
        "ongoing_mean_tpot_ms",
        "ongoing_total_ms",
        "newcomer_ttft_ms",
        "newcomer_total_ms",
        "long_ttft_ms",
        "elapsed_after_arrival_ms",
    ]
    output = {
        "runs": runs,
        "chunked_vs_unchunked_interference": {
            metric: {
                "unchunked_ms": unchunked[metric],
                "chunked_ms": chunked[metric],
                "absolute_reduction_ms": unchunked[metric] - chunked[metric],
                "relative_reduction": 1 - chunked[metric] / unchunked[metric],
            }
            for metric in cross_metrics
        },
    }
    result_path = experiment_dir / "analysis_summary.json"
    result_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(result_path)


if __name__ == "__main__":
    main()
