# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

from automatic_prefix_caching import run_trial

from vllm import LLM

PROMPT_TOKENS = 2112
BLOCK_SIZE_TOKENS = 16
DEFAULT_HIT_TOKENS = (*range(0, 2049, 128), 2096)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--trace-dir", required=True, type=Path)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--warmup-repetitions", type=int, default=1)
    parser.add_argument("--output-tokens", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.84)
    parser.add_argument(
        "--hit-tokens",
        type=int,
        nargs="+",
        default=DEFAULT_HIT_TOKENS,
    )
    return parser.parse_args()


def distinct_token_ids(tokenizer: Any) -> tuple[int, int]:
    token_ids = []
    for text in (" KV", " cache", " prefix", " alpha"):
        encoded = tokenizer.encode(text, add_special_tokens=False)
        assert encoded
        if encoded[0] not in token_ids:
            token_ids.append(encoded[0])
    assert len(token_ids) >= 2
    return token_ids[0], token_ids[1]


def validate_hit_tokens(hit_tokens: list[int]) -> list[int]:
    points = sorted(set(hit_tokens))
    assert points
    assert points[0] >= 0
    assert points[-1] <= PROMPT_TOKENS - BLOCK_SIZE_TOKENS
    assert all(point % BLOCK_SIZE_TOKENS == 0 for point in points)
    return points


def main() -> None:
    args = parse_args()
    hit_tokens = validate_hit_tokens(args.hit_tokens)
    trace_dir = args.trace_dir.resolve()
    trace_dir.mkdir(parents=True, exist_ok=True)
    os.environ["VLLM_KV_LIFECYCLE_TRACE_DIR"] = str(trace_dir)

    llm = LLM(
        model=args.model,
        max_model_len=3072,
        max_num_seqs=1,
        max_num_batched_tokens=3072,
        max_num_scheduled_tokens=3072,
        block_size=BLOCK_SIZE_TOKENS,
        enable_chunked_prefill=False,
        enable_prefix_caching=True,
        async_scheduling=False,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
    )
    seed_token_id, changed_token_id = distinct_token_ids(llm.get_tokenizer())
    seed_prompt = [seed_token_id] * PROMPT_TOKENS
    total_repetitions = args.warmup_repetitions + args.repetitions
    trials = []

    for repetition in range(total_repetitions):
        measured = repetition >= args.warmup_repetitions
        repetition_points = hit_tokens.copy()
        random.Random(repetition).shuffle(repetition_points)
        for target_hit_tokens in repetition_points:
            probe_prompt = seed_prompt[:target_hit_tokens] + [changed_token_id] * (
                PROMPT_TOKENS - target_hit_tokens
            )
            trial = run_trial(
                llm,
                suite="prefix_hit_ttft_curve",
                case=f"hit_{target_hit_tokens}",
                repetition=repetition,
                measured=measured,
                seed_prompt=seed_prompt,
                probe_prompts=[probe_prompt],
                output_tokens=args.output_tokens,
            )
            trial["target_prefix_hit_tokens"] = target_hit_tokens
            trials.append(trial)

    manifest = {
        "model": args.model,
        "prefix_cache_enabled": True,
        "block_size_tokens": BLOCK_SIZE_TOKENS,
        "prompt_tokens": PROMPT_TOKENS,
        "output_tokens": args.output_tokens,
        "repetitions": args.repetitions,
        "warmup_repetitions": args.warmup_repetitions,
        "target_prefix_hit_tokens": hit_tokens,
        "trial_order_random_seed": "repetition index",
        "trials": trials,
    }
    manifest_path = trace_dir / "experiment_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(manifest_path)


if __name__ == "__main__":
    main()
