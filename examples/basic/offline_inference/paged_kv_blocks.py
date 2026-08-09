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


def run_batch(
    llm: LLM,
    seed_token_id: int,
    suite: str,
    prompt_lengths: list[int],
    output_lengths: list[int],
    concurrency: int,
) -> dict[str, object]:
    assert len(prompt_lengths) == len(output_lengths) == concurrency
    prompts = [
        TokensPrompt(prompt_token_ids=[seed_token_id] * prompt_length)
        for prompt_length in prompt_lengths
    ]
    params = [sampling_params(output_length) for output_length in output_lengths]

    start = time.monotonic()
    outputs = llm.generate(prompts, params, use_tqdm=False)
    elapsed_s = time.monotonic() - start

    requests = []
    for expected_prompt, expected_output, output in zip(
        prompt_lengths,
        output_lengths,
        outputs,
        strict=True,
    ):
        actual_prompt = len(output.prompt_token_ids)
        actual_output = len(output.outputs[0].token_ids)
        assert actual_prompt == expected_prompt
        assert actual_output == expected_output
        requests.append(
            {
                "request_id": output.request_id,
                "prompt_tokens": actual_prompt,
                "output_tokens": actual_output,
                "expected_computed_tokens": actual_prompt + actual_output - 1,
            }
        )

    total_output_tokens = sum(output_lengths)
    total_tokens = sum(prompt_lengths) + total_output_tokens
    return {
        "suite": suite,
        "concurrency": concurrency,
        "elapsed_s": elapsed_s,
        "output_tokens_per_second": total_output_tokens / elapsed_s,
        "total_tokens_per_second": total_tokens / elapsed_s,
        "requests": requests,
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
        max_num_batched_tokens=16384,
        max_num_scheduled_tokens=16384,
        block_size=16,
        enable_chunked_prefill=False,
        enable_prefix_caching=False,
        async_scheduling=False,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
    )
    tokenizer = llm.get_tokenizer()
    seed_token_id = tokenizer.encode(" KV", add_special_tokens=False)[0]

    results = []
    boundary_prompts = [
        15,
        16,
        17,
        31,
        32,
        33,
        255,
        256,
        257,
        511,
        512,
        513,
        2047,
        2048,
        2049,
    ]
    for prompt_length in boundary_prompts:
        results.append(
            run_batch(
                llm,
                seed_token_id,
                suite="prompt_boundary",
                prompt_lengths=[prompt_length],
                output_lengths=[1],
                concurrency=1,
            )
        )

    for prompt_length in (16, 32, 256):
        for output_length in (2, 17, 18):
            results.append(
                run_batch(
                    llm,
                    seed_token_id,
                    suite="decode_boundary",
                    prompt_lengths=[prompt_length],
                    output_lengths=[output_length],
                    concurrency=1,
                )
            )

    concurrent_cases = [
        ([2049], [128]),
        ([15, 257, 513, 2049], [32, 128, 32, 128]),
        (
            [15, 17, 255, 257, 511, 513, 2047, 2049],
            [32, 128, 32, 128, 32, 128, 32, 128],
        ),
    ]
    for prompt_lengths, output_lengths in concurrent_cases:
        results.append(
            run_batch(
                llm,
                seed_token_id,
                suite="continuous_batching_mapping",
                prompt_lengths=prompt_lengths,
                output_lengths=output_lengths,
                concurrency=len(prompt_lengths),
            )
        )

    for concurrency in (1, 4, 8):
        results.append(
            run_batch(
                llm,
                seed_token_id,
                suite="continuous_batching_scaling",
                prompt_lengths=[512] * concurrency,
                output_lengths=[128] * concurrency,
                concurrency=concurrency,
            )
        )

    result_path = trace_dir / "experiment_results.json"
    result_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(result_path)


if __name__ == "__main__":
    main()
