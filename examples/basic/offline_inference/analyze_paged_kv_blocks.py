# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def percentile(values: list[float], fraction: float) -> float:
    assert values
    index = min(math.ceil(len(values) * fraction) - 1, len(values) - 1)
    return sorted(values)[index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trace_dir = args.trace_dir.resolve()
    experiments = json.loads((trace_dir / "experiment_results.json").read_text())
    core = read_jsonl(next(trace_dir.glob("kv_lifecycle_core_*.jsonl")))
    detail = read_jsonl(next(trace_dir.glob("kv_lifecycle_detail_*.jsonl")))

    engine = next(event for event in core if event["event"] == "ENGINE_INIT")
    block_size = engine["block_size_tokens"]
    summaries = {
        event["request_id"]: event
        for event in core
        if event["event"] == "REQUEST_SUMMARY"
    }
    finishes = {
        event["request_id"]: event
        for event in detail
        if event["event"] == "FINISH_BEFORE_RELEASE"
    }
    client_request_specs = {
        request["request_id"]: request
        for experiment in experiments
        for request in experiment["requests"]
    }
    internal_request_ids = {
        request_id.split("-", 1)[0]: request_id for request_id in summaries
    }
    assert client_request_specs.keys() == internal_request_ids.keys()
    request_specs = {
        internal_request_ids[client_request_id]: spec
        for client_request_id, spec in client_request_specs.items()
    }
    assert request_specs.keys() == summaries.keys() == finishes.keys()

    noncontiguous_requests = []
    boundary_rows = []
    for request_id, spec in request_specs.items():
        summary = summaries[request_id]
        finish = finishes[request_id]
        computed_tokens = spec["expected_computed_tokens"]
        mappings = finish["block_mappings_by_group"][0]
        expected_blocks = math.ceil(computed_tokens / block_size)

        assert summary["num_computed_tokens"] == computed_tokens
        assert len(mappings) == expected_blocks
        assert [mapping["logical_block_index"] for mapping in mappings] == list(
            range(expected_blocks)
        )
        assert sum(mapping["valid_tokens"] for mapping in mappings) == computed_tokens
        assert sum(mapping["unused_token_slots"] for mapping in mappings) == (
            expected_blocks * block_size - computed_tokens
        )
        assert summary["num_preemptions"] == 0
        assert summary["recompute_occurred"] is False

        physical_ids = [mapping["physical_block_id"] for mapping in mappings]
        assert len(set(physical_ids)) == len(physical_ids)
        if any(
            current != previous + 1
            for previous, current in zip(physical_ids, physical_ids[1:])
        ):
            noncontiguous_requests.append(request_id)

        if spec["output_tokens"] == 1:
            boundary_rows.append(
                {
                    "prompt_tokens": spec["prompt_tokens"],
                    "logical_blocks": expected_blocks,
                    "last_block_valid_tokens": mappings[-1]["valid_tokens"],
                    "unused_token_slots": mappings[-1]["unused_token_slots"],
                }
            )

    schedule_steps = [event for event in detail if event["event"] == "SCHEDULE_STEP"]
    steps_by_engine_step: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for event in schedule_steps:
        steps_by_engine_step[event["engine_step"]].append(event)
    for events in steps_by_engine_step.values():
        seen_physical_ids: set[int] = set()
        for event in events:
            physical_ids = set(event["block_ids_by_group"][0])
            assert seen_physical_ids.isdisjoint(physical_ids)
            seen_physical_ids.update(physical_ids)

    for experiment in experiments:
        request_ids = [
            internal_request_ids[request["request_id"]]
            for request in experiment["requests"]
        ]
        last_summary = max(
            (summaries[request_id] for request_id in request_ids),
            key=lambda summary: summary["monotonic_time_ns"],
        )
        assert last_summary["final_free_blocks"] == engine["num_free_blocks"]

    allocation_times = [event["allocation_duration_ms"] for event in schedule_steps]
    free_times = [summary["free_duration_ms"] for summary in summaries.values()]
    continuous_batching = [
        {
            "suite": experiment["suite"],
            "concurrency": experiment["concurrency"],
            "elapsed_s": experiment["elapsed_s"],
            "output_tokens_per_second": experiment["output_tokens_per_second"],
            "total_tokens_per_second": experiment["total_tokens_per_second"],
            "max_total_latency_ms": max(
                summaries[internal_request_ids[request["request_id"]]]["total_ms"]
                for request in experiment["requests"]
            ),
        }
        for experiment in experiments
        if experiment["suite"].startswith("continuous_batching")
    ]
    analysis = {
        "block_size": block_size,
        "total_blocks": engine["num_total_blocks"],
        "initial_free_blocks": engine["num_free_blocks"],
        "request_count": len(request_specs),
        "all_invariants_passed": True,
        "preemption_or_recompute_observed": False,
        "noncontiguous_request_count": len(noncontiguous_requests),
        "noncontiguous_request_ids": noncontiguous_requests,
        "allocation_duration_ms": {
            "mean": statistics.mean(allocation_times),
            "p95": percentile(allocation_times, 0.95),
            "max": max(allocation_times),
        },
        "free_duration_ms": {
            "mean": statistics.mean(free_times),
            "p95": percentile(free_times, 0.95),
            "max": max(free_times),
        },
        "prompt_boundary_results": sorted(
            boundary_rows,
            key=lambda row: row["prompt_tokens"],
        ),
        "continuous_batching_results": continuous_batching,
    }
    analysis_path = trace_dir / "analysis_summary.json"
    analysis_path.write_text(json.dumps(analysis, indent=2) + "\n")
    print(analysis_path)


if __name__ == "__main__":
    main()
