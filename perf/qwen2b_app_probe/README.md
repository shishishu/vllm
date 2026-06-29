# Qwen 3.5 2B App Probe

This probe measures four application-level request shapes against an already
running vLLM OpenAI-compatible server:

- short prompt + short output, `max_tokens=64`
- short prompt + long output, `max_tokens=512`
- long prompt + short output, `max_tokens=64`
- long prompt + long output, `max_tokens=512`

The short prompt targets about 64 prompt tokens. The long prompt targets about
2048 prompt tokens. If `transformers` can load a tokenizer from `--tokenizer`
or `--model`, the script uses
`AutoTokenizer.from_pretrained(...).apply_chat_template(...)` for prompt token
counting. If not, it falls back to a character-count estimate and prints a
warning.

## Start Server

From the vLLM repository root:

```bash
env ALL_PROXY= all_proxy= \
  HF_HOME=.hf_cache \
  HF_HUB_CACHE=.hf_cache/hub \
  HF_XET_CACHE=.hf_cache/xet \
  VLLM_USE_FLASHINFER_SAMPLER=0 \
  .venv/bin/python -m vllm.entrypoints.openai.api_server \
  --model .hf_cache/hub/models--Qwen--Qwen3.5-2B/snapshots/15852e8c16360a2fea060d615a32b45270f8a8fc \
  --served-model-name qwen3.5-2b \
  --dtype float16 \
  --max-model-len 4096 \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.82 \
  --enforce-eager \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  --host 127.0.0.1 \
  --port 8000
```

Wait until the server logs show it is ready before running the probe.

`prompt_tokens + max_tokens` must fit within the server `--max-model-len`.
`max_tokens` is only the output budget; it is not a guarantee that the model
will generate that many tokens.

## Run Probe

```bash
.venv/bin/python perf/qwen2b_app_probe/run_app_probe.py \
  --model qwen3.5-2b
```

Useful options:

```bash
.venv/bin/python perf/qwen2b_app_probe/run_app_probe.py \
  --base-url http://localhost:8000/v1 \
  --model qwen3.5-2b \
  --tokenizer .hf_cache/hub/models--Qwen--Qwen3.5-2B/snapshots/15852e8c16360a2fea060d615a32b45270f8a8fc \
  --repeat 3 \
  --warmup 1 \
  --shuffle true \
  --temperature 0.0 \
  --timeout 120 \
  --output-dir perf/qwen2b_app_probe/results
```

For exact local tokenizer counts, keep `--model` as the server's served model
name and pass the local model snapshot with `--tokenizer`. If `--model` is only
the served name and no matching tokenizer exists locally, the script will still
run with estimated prompt token counts.

## Outputs

The script writes the measured CSV here:

```text
perf/qwen2b_app_probe/results/app_probe_results.csv
```

CSV fields:

```text
case_name, run_id, target_prompt_tokens, prompt_tokens, prompt_chars,
max_tokens, latency_seconds, ttft_seconds, tpot_seconds, completion_tokens,
total_tokens, one_over_tpot_tokens_per_second, response_chars, response_preview,
error
```

The script also writes full raw request and response records here:

```text
perf/qwen2b_app_probe/results/app_probe_raw.jsonl
```

Each JSONL line contains the full prompt messages, complete response text, full
API `usage`, latency, TTFT, TPOT, error, and `is_warmup`. Warmup requests are
included in JSONL but are not included in the final CSV statistics.

## Reading Results

TTFT is time to first non-empty streamed content chunk. It is more sensitive to
prefill cost, prompt length, queueing, and scheduler behavior.

TPOT is computed as:

```text
(latency_seconds - ttft_seconds) / (completion_tokens - 1)
```

It is more sensitive to decode cost and sustained generation throughput. TPOT
is `-1` when it cannot be calculated.

`one_over_tpot_tokens_per_second`, displayed as `1/TPOT (tok/s)`, is computed
as:

```text
1 / tpot_seconds
```

It estimates steady decode throughput after the first generated token. Use it
with TPOT when comparing decode-heavy behavior. It is `-1` when TPOT cannot be
calculated.

Use the CSV to compare trends across request shapes. Use the JSONL file when
you need to replay or audit the complete input, output, usage, and error data.

If the server is not running or cannot be reached, the script prints a clear
error and records the connection failure in the output files.

## Experiment Results

These results were collected on the local RTX 2080 test machine with Qwen 3.5
2B, `--max-model-len 4096`, `--max-num-seqs 1`, `--dtype float16`,
`--enforce-eager`, and `VLLM_USE_FLASHINFER_SAMPLER=0`.

The stable no-cache run used:

```text
--gpu-memory-utilization 0.72
--warmup 3
--repeat 10
```

The prefix-cache run used:

```text
--gpu-memory-utilization 0.72
--enable-prefix-caching
--warmup 0
--repeat 10
--shuffle false
--order grouped
```

`--warmup 0` was used for the prefix-cache run so the first request in each
group could show cold-prefill behavior before later repeated requests reused
cached prefixes.

### No Prefix Cache

Source files:

```text
perf/qwen2b_app_probe/results/app_probe_results.csv
perf/qwen2b_app_probe/results/app_probe_raw.jsonl
```

Measured rows: 40. Errors: 0.

| Case | Avg latency (s) | Avg TTFT (s) | Avg TPOT (s) | 1/TPOT (tok/s) |
| --- | ---: | ---: | ---: | ---: |
| short_prompt_short_output | 1.757 | 0.073 | 0.0268 | 37.39 |
| short_prompt_long_output | 13.856 | 0.072 | 0.0270 | 37.07 |
| long_prompt_short_output | 2.360 | 0.657 | 0.0270 | 37.05 |
| long_prompt_long_output | 14.532 | 0.625 | 0.0272 | 36.77 |

### Prefix Cache Enabled

Source files:

```text
perf/qwen2b_app_probe/results/prefix_cache_enabled/app_probe_results.csv
perf/qwen2b_app_probe/results/prefix_cache_enabled/app_probe_raw.jsonl
```

Measured rows: 40. Errors: 0.

| Case | Avg latency (s) | Avg TTFT (s) | Avg TPOT (s) | 1/TPOT (tok/s) |
| --- | ---: | ---: | ---: | ---: |
| short_prompt_short_output | 1.826 | 0.121 | 0.0271 | 36.95 |
| short_prompt_long_output | 13.799 | 0.072 | 0.0269 | 37.23 |
| long_prompt_short_output | 1.997 | 0.292 | 0.0271 | 36.95 |
| long_prompt_long_output | 13.703 | 0.233 | 0.0264 | 37.95 |

### Findings

Prefix caching mainly improves TTFT for long prompts:

| Case | No-cache TTFT (s) | Prefix-cache TTFT (s) | Change |
| --- | ---: | ---: | ---: |
| long_prompt_short_output | 0.657 | 0.292 | -56% |
| long_prompt_long_output | 0.625 | 0.233 | -63% |

Decode throughput is mostly unchanged. `1/TPOT (tok/s)` stays near
37 tok/s across both runs, so prefix caching is reducing prefill work rather
than improving steady decode speed.

The benefit is much easier to see on short-output requests because prefill is a
larger fraction of total latency. For long-output requests, decode time
dominates total latency, so the TTFT improvement is less visible in end-to-end
latency.

### Notes

The initial no-cache run with `--gpu-memory-utilization 0.82` hit CUDA OOM on a
long-prompt, long-output request. The failure happened because the KV cache
pool left too little free GPU memory for temporary CUDA buffers. Lowering
`--gpu-memory-utilization` to `0.72` left enough headroom while still providing
more than enough KV cache capacity for this single-concurrency experiment.

When prefix caching is enabled for Qwen 3.5, vLLM reports Mamba prefix caching
as experimental. Treat these numbers as local behavior for this machine and
configuration, not a general hardware-independent benchmark.
