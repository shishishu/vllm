#!/usr/bin/env python3
"""Run an application-level streaming probe against a vLLM OpenAI server."""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_DIR = "perf/qwen2b_app_probe/results"
SYSTEM_PROMPT = "你是一个简洁、准确的中文助手。"
SHORT_TARGET_PROMPT_TOKENS = 64
LONG_TARGET_PROMPT_TOKENS = 2048
MAX_PROMPT_REPEATS = 2048
CSV_FIELDS = [
    "case_name",
    "run_id",
    "target_prompt_tokens",
    "prompt_tokens",
    "prompt_chars",
    "max_tokens",
    "latency_seconds",
    "ttft_seconds",
    "tpot_seconds",
    "completion_tokens",
    "total_tokens",
    "one_over_tpot_tokens_per_second",
    "response_chars",
    "response_preview",
    "error",
]


def parse_bool(value: str) -> bool:
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


class PromptTokenizer:
    """Best-effort tokenizer wrapper with a deterministic character fallback."""

    def __init__(self, tokenizer_name_or_path: str) -> None:
        self.tokenizer: Any | None = None
        self.warning = ""
        try:
            from transformers import AutoTokenizer
        except ImportError:
            self.warning = (
                "transformers is not available; prompt_tokens will use a "
                "character-count estimate."
            )
            return

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name_or_path,
                trust_remote_code=True,
                local_files_only=True,
            )
        except Exception as exc:  # noqa: BLE001
            self.warning = (
                "Could not load tokenizer with "
                "AutoTokenizer.from_pretrained"
                f"({tokenizer_name_or_path!r}): {exc}. "
                "prompt_tokens will use a character-count estimate. "
                "If --model is only a served-model-name, pass a local model "
                "path with --tokenizer when you need exact local token counts."
            )

    @property
    def has_tokenizer(self) -> bool:
        return self.tokenizer is not None

    def _count_tokenized_output(self, tokenized: Any) -> int:
        if hasattr(tokenized, "get"):
            input_ids = tokenized.get("input_ids")
            if input_ids is not None:
                return self._count_tokenized_output(input_ids)
        if isinstance(tokenized, (list, tuple)):
            if not tokenized:
                return 0
            first = tokenized[0]
            if isinstance(first, int):
                return len(tokenized)
            if isinstance(first, (list, tuple)):
                return sum(self._count_tokenized_output(item) for item in tokenized)
        return len(tokenized)

    def count_messages(self, messages: list[dict[str, str]]) -> int:
        if self.tokenizer is not None:
            try:
                tokenized = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                )
                return self._count_tokenized_output(tokenized)
            except Exception:  # noqa: BLE001
                text = "\n".join(message["content"] for message in messages)
                return len(self.tokenizer.encode(text, add_special_tokens=True))

        text = "\n".join(message["content"] for message in messages)
        return max(1, round(len(text) / 1.5))


def build_messages(prompt: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]


def prompt_token_count(prompt: str, tokenizer: PromptTokenizer) -> int:
    return tokenizer.count_messages(build_messages(prompt))


def prompt_for_target(target_tokens: int, tokenizer: PromptTokenizer) -> str:
    intro = (
        "请围绕 vLLM 的一次线上推理请求进行技术分析，重点关注 "
        "prefill、decode、KV cache 和 scheduler 的性能影响。"
    )
    unit = (
        "请求进入 OpenAI-compatible server 后，scheduler 会把它放入等待队列；"
        "prefill 阶段需要处理完整 prompt 并写入初始 KV cache，decode 阶段则"
        "持续复用 KV cache 逐步生成 token。请比较短 prompt 与长 prompt 对首"
        "token 延迟的影响，比较短输出与长输出对持续生成吞吐的影响，并结合"
        "max_model_len、max_num_seqs、block 分配、batch 合并和显存利用率说明"
        "如何判断瓶颈来自 prefill、decode、KV cache 容量还是调度策略。"
    )
    suffix = (
        "请给出面向应用开发者的观察方法，避免泛泛而谈，输出结构化结论。"
    )

    def make_prompt(repeats: int) -> str:
        parts = [intro]
        if repeats:
            parts.extend(unit for _ in range(repeats))
        parts.append(suffix)
        return "\n\n".join(parts)

    low = 0
    high = 1
    while prompt_token_count(make_prompt(high), tokenizer) < target_tokens:
        high *= 2
        if high > MAX_PROMPT_REPEATS:
            raise RuntimeError(
                "Could not reach target prompt tokens before "
                f"{MAX_PROMPT_REPEATS} prompt repeats."
            )

    while low < high:
        mid = (low + high) // 2
        if prompt_token_count(make_prompt(mid), tokenizer) < target_tokens:
            low = mid + 1
        else:
            high = mid

    candidates = [make_prompt(max(0, low - 1)), make_prompt(low)]
    return min(
        candidates,
        key=lambda prompt: abs(prompt_token_count(prompt, tokenizer) - target_tokens),
    )


def build_cases(tokenizer: PromptTokenizer) -> list[dict[str, Any]]:
    short_prompt = prompt_for_target(SHORT_TARGET_PROMPT_TOKENS, tokenizer)
    long_prompt = prompt_for_target(LONG_TARGET_PROMPT_TOKENS, tokenizer)
    return [
        {
            "case_name": "short_prompt_short_output",
            "target_prompt_tokens": SHORT_TARGET_PROMPT_TOKENS,
            "prompt": short_prompt,
            "max_tokens": 64,
        },
        {
            "case_name": "short_prompt_long_output",
            "target_prompt_tokens": SHORT_TARGET_PROMPT_TOKENS,
            "prompt": short_prompt,
            "max_tokens": 512,
        },
        {
            "case_name": "long_prompt_short_output",
            "target_prompt_tokens": LONG_TARGET_PROMPT_TOKENS,
            "prompt": long_prompt,
            "max_tokens": 64,
        },
        {
            "case_name": "long_prompt_long_output",
            "target_prompt_tokens": LONG_TARGET_PROMPT_TOKENS,
            "prompt": long_prompt,
            "max_tokens": 512,
        },
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe Qwen 3.5 2B through a running vLLM OpenAI server."
    )
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Optional local tokenizer path. Defaults to --model.",
    )
    parser.add_argument(
        "--shuffle",
        nargs="?",
        const=True,
        default=True,
        type=parse_bool,
        help="Shuffle measured requests. Use --shuffle false to disable.",
    )
    parser.add_argument("--no-shuffle", dest="shuffle", action="store_false")
    parser.add_argument(
        "--order",
        choices=["round_robin", "grouped"],
        default="round_robin",
        help="Request order when shuffle is disabled.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    return parser.parse_args()


def make_payload(
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    return {
        "model": model,
        "messages": build_messages(prompt),
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }


def post_streaming_chat_completion(
    base_url: str,
    payload: dict[str, Any],
    timeout: float,
) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/chat/completions"
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    response_parts: list[str] = []
    usage: dict[str, Any] = {}
    ttft = -1.0

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                if not line.startswith("data: "):
                    continue

                data_line = line.removeprefix("data: ").strip()
                if data_line == "[DONE]":
                    break

                try:
                    chunk = json.loads(data_line)
                except json.JSONDecodeError:
                    continue

                chunk_usage = chunk.get("usage")
                if chunk_usage:
                    usage = chunk_usage

                choices = chunk.get("choices") or []
                for choice in choices:
                    delta = choice.get("delta") or {}
                    content = delta.get("content") or ""
                    if content:
                        if ttft < 0:
                            ttft = time.perf_counter() - start
                        response_parts.append(str(content))
    except urllib.error.HTTPError as exc:
        latency = time.perf_counter() - start
        detail = exc.read().decode("utf-8", errors="replace")
        return {
            "response_text": "",
            "usage": usage,
            "latency_seconds": latency,
            "ttft_seconds": ttft,
            "error": f"HTTP {exc.code}: {detail}",
        }
    except urllib.error.URLError as exc:
        latency = time.perf_counter() - start
        return {
            "response_text": "",
            "usage": usage,
            "latency_seconds": latency,
            "ttft_seconds": ttft,
            "error": f"Connection failed: {exc.reason}",
        }
    except TimeoutError:
        latency = time.perf_counter() - start
        return {
            "response_text": "".join(response_parts),
            "usage": usage,
            "latency_seconds": latency,
            "ttft_seconds": ttft,
            "error": f"Timed out after {timeout:.1f}s",
        }

    latency = time.perf_counter() - start
    return {
        "response_text": "".join(response_parts),
        "usage": usage,
        "latency_seconds": latency,
        "ttft_seconds": ttft,
        "error": "",
    }


def calc_tpot(latency: float, ttft: float, completion_tokens: Any) -> float:
    if isinstance(completion_tokens, int) and completion_tokens > 1 and ttft >= 0:
        return max(0.0, (latency - ttft) / (completion_tokens - 1))
    return -1.0


def calc_one_over_tpot_tokens_per_second(tpot: float) -> float:
    if tpot > 0:
        return 1 / tpot
    return -1.0


def format_float(value: float) -> str:
    return f"{value:.4f}" if value >= 0 else "-1"


def row_from_result(
    case: dict[str, Any],
    run_id: int,
    result: dict[str, Any],
    local_prompt_tokens: int,
) -> dict[str, Any]:
    usage = result["usage"] or {}
    response_text = result["response_text"]
    latency = result["latency_seconds"]
    ttft = result["ttft_seconds"]
    completion_tokens = usage.get("completion_tokens", "")
    total_tokens = usage.get("total_tokens", "")
    prompt_tokens = usage.get("prompt_tokens", local_prompt_tokens)
    tpot = calc_tpot(latency, ttft, completion_tokens)
    one_over_tpot_tokens_per_second = calc_one_over_tpot_tokens_per_second(tpot)

    return {
        "case_name": case["case_name"],
        "run_id": run_id,
        "target_prompt_tokens": case["target_prompt_tokens"],
        "prompt_tokens": prompt_tokens,
        "prompt_chars": len(case["prompt"]),
        "max_tokens": case["max_tokens"],
        "latency_seconds": format_float(latency),
        "ttft_seconds": format_float(ttft),
        "tpot_seconds": format_float(tpot),
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "one_over_tpot_tokens_per_second": format_float(
            one_over_tpot_tokens_per_second,
        ),
        "response_chars": len(response_text),
        "response_preview": response_text.replace("\n", " ")[:120],
        "error": result["error"],
    }


def raw_record_from_result(
    case: dict[str, Any],
    run_id: int,
    payload: dict[str, Any],
    result: dict[str, Any],
    is_warmup: bool,
) -> dict[str, Any]:
    usage = result["usage"] or {}
    completion_tokens = usage.get("completion_tokens")
    tpot = calc_tpot(
        result["latency_seconds"],
        result["ttft_seconds"],
        completion_tokens,
    )
    return {
        "case_name": case["case_name"],
        "run_id": run_id,
        "is_warmup": is_warmup,
        "request": {
            "model": payload["model"],
            "messages": payload["messages"],
            "max_tokens": payload["max_tokens"],
            "temperature": payload["temperature"],
            "stream": payload["stream"],
        },
        "response_text": result["response_text"],
        "usage": usage,
        "latency_seconds": result["latency_seconds"],
        "ttft_seconds": result["ttft_seconds"],
        "tpot_seconds": tpot,
        "error": result["error"],
    }


def write_raw_record(raw_file: Any, record: dict[str, Any]) -> None:
    raw_file.write(json.dumps(record, ensure_ascii=False) + "\n")
    raw_file.flush()


def numeric_values(rows: list[dict[str, Any]], field: str) -> list[float]:
    values = []
    for row in rows:
        value = row[field]
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if number >= 0:
            values.append(number)
    return values


def avg_text(rows: list[dict[str, Any]], field: str) -> str:
    values = numeric_values(rows, field)
    if not values:
        return "-1"
    return f"{statistics.fmean(values):.4f}"


def print_summary(rows: list[dict[str, Any]], cases: list[dict[str, Any]]) -> None:
    print("\nSummary by case:")
    for case in cases:
        case_rows = [
            row for row in rows
            if row["case_name"] == case["case_name"] and not row["error"]
        ]
        print(
            f"{case['case_name']}: "
            f"count={len(case_rows)}, "
            f"avg_latency={avg_text(case_rows, 'latency_seconds')}, "
            f"avg_ttft={avg_text(case_rows, 'ttft_seconds')}, "
            f"avg_tpot={avg_text(case_rows, 'tpot_seconds')}, "
            f"avg_prompt_tokens={avg_text(case_rows, 'prompt_tokens')}, "
            f"avg_completion_tokens={avg_text(case_rows, 'completion_tokens')}, "
            "avg_1_over_tpot_tokens_per_second="
            f"{avg_text(case_rows, 'one_over_tpot_tokens_per_second')}"
        )


def build_tasks(
    cases: list[dict[str, Any]],
    repeat: int,
    warmup: int,
    shuffle: bool,
    order: str,
) -> list[tuple[dict[str, Any], int, bool]]:
    if order == "grouped":
        warmup_tasks = [
            (case, warmup_id, True)
            for case in cases
            for warmup_id in range(1, warmup + 1)
        ]
        measured_tasks = [
            (case, run_id, False)
            for case in cases
            for run_id in range(1, repeat + 1)
        ]
    else:
        warmup_tasks = [
            (case, warmup_id, True)
            for warmup_id in range(1, warmup + 1)
            for case in cases
        ]
        measured_tasks = [
            (case, run_id, False)
            for run_id in range(1, repeat + 1)
            for case in cases
        ]
    if shuffle:
        random.shuffle(warmup_tasks)
        random.shuffle(measured_tasks)
    return warmup_tasks + measured_tasks


def main() -> int:
    args = parse_args()
    if args.repeat < 1:
        print("--repeat must be >= 1", file=sys.stderr)
        return 2
    if args.warmup < 0:
        print("--warmup must be >= 0", file=sys.stderr)
        return 2

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "app_probe_results.csv"
    raw_path = output_dir / "app_probe_raw.jsonl"

    tokenizer = PromptTokenizer(args.tokenizer or args.model)
    if tokenizer.warning:
        print(tokenizer.warning, file=sys.stderr)

    cases = build_cases(tokenizer)
    local_prompt_tokens = {
        case["case_name"]: prompt_token_count(case["prompt"], tokenizer)
        for case in cases
    }
    for case in cases:
        print(
            f"{case['case_name']}: target_prompt_tokens="
            f"{case['target_prompt_tokens']}, local_prompt_tokens="
            f"{local_prompt_tokens[case['case_name']]}, "
            f"prompt_chars={len(case['prompt'])}, max_tokens={case['max_tokens']}"
        )

    rows: list[dict[str, Any]] = []
    tasks = build_tasks(cases, args.repeat, args.warmup, args.shuffle, args.order)

    with raw_path.open("w", encoding="utf-8") as raw_file:
        for case, run_id, is_warmup in tasks:
            payload = make_payload(
                args.model,
                case["prompt"],
                case["max_tokens"],
                args.temperature,
            )
            result = post_streaming_chat_completion(
                args.base_url,
                payload,
                args.timeout,
            )
            if result["error"].startswith("Connection failed"):
                print(
                    "vLLM OpenAI-compatible server is not reachable at "
                    f"{args.base_url}. Start the server first, then rerun.",
                    file=sys.stderr,
                )

            raw_record = raw_record_from_result(
                case,
                run_id,
                payload,
                result,
                is_warmup,
            )
            write_raw_record(raw_file, raw_record)

            row = row_from_result(
                case,
                run_id,
                result,
                local_prompt_tokens[case["case_name"]],
            )
            phase = "warmup" if is_warmup else "run"
            status = "error" if result["error"] else "ok"
            print(
                f"{case['case_name']} {phase} {run_id}: {status}, "
                f"latency={row['latency_seconds']}s, "
                f"ttft={row['ttft_seconds']}s, "
                f"tpot={row['tpot_seconds']}s, "
                f"completion_tokens={row['completion_tokens']}"
            )

            if not is_warmup:
                rows.append(row)

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} measured rows to {csv_path}")
    print(f"Wrote {len(tasks)} raw records to {raw_path}")
    print_summary(rows, cases)
    return 1 if any(row["error"] for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
