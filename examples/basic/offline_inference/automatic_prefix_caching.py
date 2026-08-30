# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--trace-dir", required=True, type=Path)
    parser.add_argument("--prefix-cache", required=True, choices=("on", "off"))
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--warmup-repetitions", type=int, default=1)
    parser.add_argument("--output-tokens", type=int, default=32)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.84)
    return parser.parse_args()


def token_ids_sha256(token_ids: list[int]) -> str:
    payload = json.dumps(token_ids, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def longest_common_prefix(left: list[int], right: list[int]) -> int:
    for index, (left_token, right_token) in enumerate(zip(left, right, strict=True)):
        if left_token != right_token:
            return index
    return len(left)


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


def run_generation(
    llm: LLM,
    prompts: list[list[int]],
    output_tokens: int,
) -> tuple[list[dict[str, Any]], float]:
    start = time.monotonic()
    outputs = llm.generate(
        [TokensPrompt(prompt_token_ids=prompt) for prompt in prompts],
        [sampling_params(output_tokens) for _ in prompts],
        use_tqdm=False,
    )
    elapsed_s = time.monotonic() - start
    requests = []
    for prompt, output in zip(prompts, outputs, strict=True):
        actual_output_tokens = len(output.outputs[0].token_ids)
        assert output.prompt_token_ids == prompt
        assert actual_output_tokens == output_tokens
        requests.append(
            {
                "request_id": output.request_id,
                "prompt_tokens": len(prompt),
                "output_tokens": actual_output_tokens,
                "prompt_token_ids_sha256": token_ids_sha256(prompt),
            }
        )
    return requests, elapsed_s


def make_prompts(
    tokenizer: Any,
) -> tuple[dict[str, tuple[list[int], list[int]]], list[int]]:
    token_ids = []
    for text in (
        " KV",
        " cache",
        " prefix",
        " alpha",
        " beta",
        " gamma",
        " delta",
        " epsilon",
        " zeta",
        " eta",
        " theta",
        " iota",
    ):
        encoded = tokenizer.encode(text, add_special_tokens=False)
        assert encoded
        if encoded[0] not in token_ids:
            token_ids.append(encoded[0])
    assert len(token_ids) >= 10

    common = [token_ids[0]] * 2048
    suffix_a = [token_ids[1]] * 64
    suffix_b = [token_ids[2]] * 64
    base = common + suffix_a
    first_changed = base.copy()
    first_changed[0] = token_ids[3]
    tail_changed = base.copy()
    tail_changed[-4:] = [token_ids[4]] * 4
    return (
        {
            "A_identical": (base, base.copy()),
            "B_shared_prefix": (base, common + suffix_b),
            "C_first_token_changed": (base, first_changed),
            "D_last_four_changed": (base, tail_changed),
        },
        token_ids,
    )


def run_trial(
    llm: LLM,
    suite: str,
    case: str,
    repetition: int,
    measured: bool,
    seed_prompt: list[int],
    probe_prompts: list[list[int]],
    output_tokens: int,
) -> dict[str, Any]:
    assert llm.reset_prefix_cache()
    seed_requests, seed_elapsed_s = run_generation(llm, [seed_prompt], output_tokens)
    probe_requests, probe_elapsed_s = run_generation(llm, probe_prompts, output_tokens)
    lcp_tokens = [
        longest_common_prefix(seed_prompt, probe_prompt)
        for probe_prompt in probe_prompts
    ]
    concurrency = len(probe_prompts)
    return {
        "suite": suite,
        "case": case,
        "repetition": repetition,
        "measured": measured,
        "concurrency": concurrency,
        "seed": seed_requests[0],
        "probes": [
            {
                **request,
                "probe_index": probe_index,
                "longest_common_prefix_tokens": lcp_tokens[probe_index],
                "first_changed_token_index": lcp_tokens[probe_index],
            }
            for probe_index, request in enumerate(probe_requests)
        ],
        "seed_elapsed_s": seed_elapsed_s,
        "probe_elapsed_s": probe_elapsed_s,
        "probe_requests_per_second": concurrency / probe_elapsed_s,
        "probe_output_tokens_per_second": (
            concurrency * output_tokens / probe_elapsed_s
        ),
        "probe_logical_tokens_per_second": (
            sum(len(prompt) + output_tokens for prompt in probe_prompts)
            / probe_elapsed_s
        ),
    }


def main() -> None:
    args = parse_args()
    trace_dir = args.trace_dir.resolve()
    trace_dir.mkdir(parents=True, exist_ok=True)
    os.environ["VLLM_KV_LIFECYCLE_TRACE_DIR"] = str(trace_dir)
    prefix_cache_enabled = args.prefix_cache == "on"

    llm = LLM(
        model=args.model,
        max_model_len=3072,
        max_num_seqs=8,
        max_num_batched_tokens=24576,
        max_num_scheduled_tokens=24576,
        block_size=16,
        enable_chunked_prefill=False,
        enable_prefix_caching=prefix_cache_enabled,
        async_scheduling=False,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
    )
    cases, distinct_token_ids = make_prompts(llm.get_tokenizer())
    total_repetitions = args.warmup_repetitions + args.repetitions
    trials = []

    for repetition in range(total_repetitions):
        measured = repetition >= args.warmup_repetitions
        for case, (seed_prompt, probe_prompt) in cases.items():
            trials.append(
                run_trial(
                    llm,
                    suite="sequential_prompt_change",
                    case=case,
                    repetition=repetition,
                    measured=measured,
                    seed_prompt=seed_prompt,
                    probe_prompts=[probe_prompt],
                    output_tokens=args.output_tokens,
                )
            )

    base_seed, _ = cases["A_identical"]
    concurrency_suffix_ids = distinct_token_ids[2:10]
    concurrent_common = base_seed[:1024]
    concurrent_seed = concurrent_common + base_seed[-64:]
    for repetition in range(total_repetitions):
        measured = repetition >= args.warmup_repetitions
        for concurrency in (1, 4, 8):
            trials.append(
                run_trial(
                    llm,
                    suite="concurrent_identical",
                    case="A_identical",
                    repetition=repetition,
                    measured=measured,
                    seed_prompt=concurrent_seed,
                    probe_prompts=[concurrent_seed.copy() for _ in range(concurrency)],
                    output_tokens=args.output_tokens,
                )
            )
            shared_prompts = [
                concurrent_common + [concurrency_suffix_ids[index]] * 64
                for index in range(concurrency)
            ]
            trials.append(
                run_trial(
                    llm,
                    suite="concurrent_shared_prefix",
                    case="B_shared_prefix",
                    repetition=repetition,
                    measured=measured,
                    seed_prompt=concurrent_seed,
                    probe_prompts=shared_prompts,
                    output_tokens=args.output_tokens,
                )
            )

    manifest = {
        "model": args.model,
        "prefix_cache_enabled": prefix_cache_enabled,
        "block_size_tokens": 16,
        "sequential_common_prefix_tokens": 2048,
        "sequential_prompt_tokens": 2112,
        "concurrent_common_prefix_tokens": 1024,
        "concurrent_prompt_tokens": 1088,
        "output_tokens": args.output_tokens,
        "repetitions": args.repetitions,
        "warmup_repetitions": args.warmup_repetitions,
        "concurrency_levels": [1, 4, 8],
        "trials": trials,
    }
    manifest_path = trace_dir / "experiment_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(manifest_path)


if __name__ == "__main__":
    main()
