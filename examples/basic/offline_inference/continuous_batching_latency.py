# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import json
import os
import time
from pathlib import Path

from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--trace-dir", required=True)
    parser.add_argument("--chunked-prefill", action="store_true")
    parser.add_argument("--max-num-batched-tokens", type=int, required=True)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    return parser.parse_args()


def sampling_params(output_tokens: int) -> SamplingParams:
    return SamplingParams(
        n=1,
        temperature=0,
        max_tokens=output_tokens,
        min_tokens=output_tokens,
        ignore_eos=True,
        stop=None,
        stop_token_ids=None,
    )


def add_request(
    llm: LLM,
    request_id: str,
    seed_token_id: int,
    prompt_tokens: int,
    output_tokens: int,
) -> str:
    return llm.llm_engine.add_request(
        request_id,
        TokensPrompt(prompt_token_ids=[seed_token_id] * prompt_tokens),
        sampling_params(output_tokens),
    )


def run_trial(
    llm: LLM,
    seed_token_id: int,
    trial_name: str,
    include_long_prefill: bool,
    measured: bool,
) -> dict[str, object]:
    ongoing_id = add_request(
        llm,
        f"{trial_name}-ongoing",
        seed_token_id,
        prompt_tokens=128,
        output_tokens=16,
    )

    first_token_seen = False
    while not first_token_seen:
        first_step_outputs = llm.llm_engine.step()
        first_token_seen = bool(first_step_outputs)
        assert not any(output.finished for output in first_step_outputs)
    assert llm.llm_engine.has_unfinished_requests()

    newcomer_id = add_request(
        llm,
        f"{trial_name}-newcomer",
        seed_token_id,
        prompt_tokens=16,
        output_tokens=1,
    )
    long_id = None
    if include_long_prefill:
        long_id = add_request(
            llm,
            f"{trial_name}-long",
            seed_token_id,
            prompt_tokens=2049,
            output_tokens=1,
        )

    start = time.monotonic()
    while llm.llm_engine.has_unfinished_requests():
        llm.llm_engine.step()
    elapsed_ms = (time.monotonic() - start) * 1000

    return {
        "trial": trial_name,
        "condition": "interference" if include_long_prefill else "baseline",
        "measured": measured,
        "elapsed_after_arrival_ms": elapsed_ms,
        "requests": {
            "ongoing": ongoing_id,
            "newcomer": newcomer_id,
            "long": long_id,
        },
    }


def main() -> None:
    args = parse_args()
    trace_dir = Path(args.trace_dir).resolve()
    trace_dir.mkdir(parents=True, exist_ok=True)
    os.environ["VLLM_KV_LIFECYCLE_TRACE_DIR"] = str(trace_dir)

    llm = LLM(
        model=args.model,
        max_model_len=4096,
        max_num_seqs=8,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_scheduled_tokens=args.max_num_batched_tokens,
        block_size=16,
        enable_chunked_prefill=args.chunked_prefill,
        enable_prefix_caching=False,
        async_scheduling=False,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
    )
    tokenizer = llm.get_tokenizer()
    seed_token_id = tokenizer.encode(" KV", add_special_tokens=False)[0]

    trials = [
        run_trial(llm, seed_token_id, "warmup-baseline", False, False),
        run_trial(llm, seed_token_id, "warmup-interference", True, False),
    ]
    for repetition in range(args.repetitions):
        trials.append(
            run_trial(
                llm,
                seed_token_id,
                f"r{repetition}-baseline",
                False,
                True,
            )
        )
        trials.append(
            run_trial(
                llm,
                seed_token_id,
                f"r{repetition}-interference",
                True,
                True,
            )
        )

    manifest = {
        "chunked_prefill": args.chunked_prefill,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "repetitions": args.repetitions,
        "workload": {
            "ongoing": {"prompt_tokens": 128, "output_tokens": 16},
            "newcomer": {"prompt_tokens": 16, "output_tokens": 1},
            "long": {"prompt_tokens": 2049, "output_tokens": 1},
        },
        "trials": trials,
    }
    result_path = trace_dir / "experiment_manifest.json"
    result_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(result_path)


if __name__ == "__main__":
    main()
