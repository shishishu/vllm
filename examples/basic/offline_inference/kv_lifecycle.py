# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import json
import os
from pathlib import Path

from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--trace-dir", required=True)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trace_dir = Path(args.trace_dir).resolve()
    trace_dir.mkdir(parents=True, exist_ok=True)
    os.environ["VLLM_KV_LIFECYCLE_TRACE_DIR"] = str(trace_dir)

    llm = LLM(
        model=args.model,
        max_model_len=4096,
        max_num_seqs=1,
        max_num_batched_tokens=4096,
        max_num_scheduled_tokens=4096,
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
    for prompt_len in (128, 512, 2048):
        for output_len in (1, 32, 128):
            prompt = TokensPrompt(prompt_token_ids=[seed_token_id] * prompt_len)
            sampling_params = SamplingParams(
                n=1,
                temperature=0,
                max_tokens=output_len,
                min_tokens=output_len,
                ignore_eos=True,
                stop=None,
                stop_token_ids=None,
            )
            output = llm.generate(
                prompt,
                sampling_params,
                use_tqdm=False,
            )[0]
            result = {
                "request_id": output.request_id,
                "prompt_tokens": len(output.prompt_token_ids),
                "output_tokens": len(output.outputs[0].token_ids),
            }
            assert result["prompt_tokens"] == prompt_len
            assert result["output_tokens"] == output_len
            results.append(result)

    result_path = trace_dir / "experiment_results.json"
    result_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(result_path)


if __name__ == "__main__":
    main()
