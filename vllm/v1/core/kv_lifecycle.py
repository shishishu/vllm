# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

import vllm.envs as envs
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.request import Request


@dataclass
class _RequestTraceState:
    add_time: float
    first_schedule_time: float | None = None
    first_token_time: float | None = None
    last_token_time: float | None = None
    finish_time: float | None = None
    token_times: list[float] = field(default_factory=list)
    peak_blocks: int = 0
    initial_free_blocks: int = 0
    last_block_ids: tuple[tuple[int, ...], ...] = ()
    pending_steps: list[dict[str, Any]] = field(default_factory=list)


class KVLifecycleTracer:
    """Write opt-in engine-side request and KV lifecycle traces."""

    def __init__(
        self,
        trace_dir: str,
        kv_cache_manager: KVCacheManager,
        engine_config: dict[str, Any],
    ) -> None:
        trace_path = Path(trace_dir)
        trace_path.mkdir(parents=True, exist_ok=True)
        file_suffix = f"{os.getpid()}_{time.time_ns()}"
        self.core_path = trace_path / f"kv_lifecycle_core_{file_suffix}.jsonl"
        self.detail_path = trace_path / f"kv_lifecycle_detail_{file_suffix}.jsonl"
        self._core_file = self.core_path.open("w", encoding="utf-8", buffering=1)
        self._detail_file = self.detail_path.open("w", encoding="utf-8", buffering=1)
        self._kv_cache_manager = kv_cache_manager
        self._states: dict[str, _RequestTraceState] = {}
        self._event_seq = 0
        self._stream_seq = {"core": 0, "detail": 0}
        self._block_sizes: tuple[int, ...] = tuple(
            group.kv_cache_spec.block_size
            for group in kv_cache_manager.kv_cache_config.kv_cache_groups
        )

        group_config = []
        for group_id, group in enumerate(
            kv_cache_manager.kv_cache_config.kv_cache_groups
        ):
            spec = group.kv_cache_spec
            group_config.append(
                {
                    "group_id": group_id,
                    "block_size_tokens": spec.block_size,
                    "page_size_bytes_per_layer": spec.page_size_bytes,
                    "num_layers": len(group.layer_names),
                }
            )
        self._write_both(
            "ENGINE_INIT",
            monotonic_time=time.monotonic(),
            num_total_blocks=kv_cache_manager.block_pool.num_gpu_blocks,
            num_free_blocks=kv_cache_manager.block_pool.get_num_free_blocks(),
            cache_groups=group_config,
            **engine_config,
        )

    @classmethod
    def from_env(
        cls,
        kv_cache_manager: KVCacheManager,
        engine_config: dict[str, Any],
    ) -> "KVLifecycleTracer | None":
        trace_dir = envs.VLLM_KV_LIFECYCLE_TRACE_DIR
        if not trace_dir:
            return None
        return cls(trace_dir, kv_cache_manager, engine_config)

    def _base_event(self, event: str, monotonic_time: float) -> dict[str, Any]:
        self._event_seq += 1
        return {
            "schema_version": 1,
            "event_seq": self._event_seq,
            "event": event,
            "wall_time_ns": time.time_ns(),
            "monotonic_time_ns": int(monotonic_time * 1e9),
        }

    def _write(self, file: TextIO, stream: str, event: dict[str, Any]) -> None:
        self._stream_seq[stream] += 1
        record = dict(event)
        record["stream_seq"] = self._stream_seq[stream]
        file.write(json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n")

    def _write_core(self, event: str, monotonic_time: float, **fields: Any) -> None:
        record = self._base_event(event, monotonic_time)
        record.update(fields)
        self._write(self._core_file, "core", record)

    def _write_detail(self, event: str, monotonic_time: float, **fields: Any) -> None:
        record = self._base_event(event, monotonic_time)
        record.update(fields)
        self._write(self._detail_file, "detail", record)

    def _write_both(self, event: str, monotonic_time: float, **fields: Any) -> None:
        record = self._base_event(event, monotonic_time)
        record.update(fields)
        self._write(self._core_file, "core", record)
        self._write(self._detail_file, "detail", record)

    def _snapshot(self, request_id: str, num_kv_tokens: int) -> dict[str, Any]:
        blocks = self._kv_cache_manager.get_blocks(request_id).blocks
        block_ids = tuple(tuple(block.block_id for block in group) for group in blocks)
        ref_cnts = tuple(tuple(block.ref_cnt for block in group) for group in blocks)
        block_counts = tuple(len(group) for group in blocks)
        total_blocks = sum(block_counts)
        all_ref_cnts = [ref_cnt for group in ref_cnts for ref_cnt in group]
        mappings = tuple(
            tuple(
                {
                    "logical_block_index": logical_block_index,
                    "physical_block_id": block.block_id,
                    "valid_tokens": max(
                        min(
                            num_kv_tokens - logical_block_index * block_size,
                            block_size,
                        ),
                        0,
                    ),
                    "unused_token_slots": block_size
                    - max(
                        min(
                            num_kv_tokens - logical_block_index * block_size,
                            block_size,
                        ),
                        0,
                    ),
                }
                for logical_block_index, block in enumerate(group)
            )
            for group, block_size in zip(
                blocks,
                self._block_sizes,
                strict=True,
            )
        )
        return {
            "block_ids_by_group": block_ids,
            "block_mappings_by_group": mappings,
            "ref_cnt_by_group": ref_cnts,
            "block_count_by_group": block_counts,
            "total_block_references": total_blocks,
            "ref_cnt_min": min(all_ref_cnts, default=None),
            "ref_cnt_max": max(all_ref_cnts, default=None),
            "num_free_blocks": self._kv_cache_manager.block_pool.get_num_free_blocks(),
        }

    def on_request_add(self, request: Request) -> None:
        now = time.monotonic()
        free_blocks = self._kv_cache_manager.block_pool.get_num_free_blocks()
        self._states[request.request_id] = _RequestTraceState(
            add_time=now,
            initial_free_blocks=free_blocks,
        )
        sampling_params = request.sampling_params
        self._write_both(
            "REQUEST_ADD",
            now,
            request_id=request.request_id,
            prompt_tokens=request.num_prompt_tokens,
            max_output_tokens=request.max_tokens,
            min_output_tokens=(sampling_params.min_tokens if sampling_params else None),
            ignore_eos=(sampling_params.ignore_eos if sampling_params else None),
            status=str(request.status),
            num_free_blocks=free_blocks,
        )

    def on_schedule(
        self,
        request: Request,
        engine_step: int,
        schedule_start: float,
        num_computed_tokens_before: int,
        num_scheduled_tokens: int,
        allocation_duration_ms: float,
        free_blocks_before_allocation: int,
        free_blocks_after_allocation: int,
    ) -> None:
        state = self._states.get(request.request_id)
        if state is None:
            return
        if state.first_schedule_time is None:
            state.first_schedule_time = schedule_start

        phase = (
            "PREFILL"
            if num_computed_tokens_before < request.num_prompt_tokens
            else "DECODE"
        )
        reserved_kv_tokens = request.num_computed_tokens
        snapshot = self._snapshot(request.request_id, reserved_kv_tokens)
        current_ids = snapshot["block_ids_by_group"]
        previous_ids = state.last_block_ids
        new_block_ids = tuple(
            tuple(block_id for block_id in group if block_id not in previous)
            for group, previous in zip(current_ids, previous_ids, strict=False)
        )
        if len(previous_ids) < len(current_ids):
            new_block_ids += current_ids[len(previous_ids) :]
        state.last_block_ids = current_ids
        state.peak_blocks = max(state.peak_blocks, snapshot["total_block_references"])
        step = {
            "engine_step": engine_step,
            "phase": phase,
            "schedule_start": schedule_start,
            "num_computed_tokens_before": num_computed_tokens_before,
            "num_scheduled_tokens": num_scheduled_tokens,
            "num_computed_tokens_after_schedule": request.num_computed_tokens,
            "reserved_kv_tokens": reserved_kv_tokens,
            "new_block_count": sum(len(group) for group in new_block_ids),
            "allocation_duration_ms": allocation_duration_ms,
            "free_blocks_before_allocation": free_blocks_before_allocation,
            "free_blocks_after_allocation": free_blocks_after_allocation,
        }
        state.pending_steps.append(step)
        self._write_detail(
            "SCHEDULE_STEP",
            schedule_start,
            request_id=request.request_id,
            **step,
            new_block_ids_by_group=new_block_ids,
            **snapshot,
        )

    def on_step_output(self, request: Request, num_new_tokens: int) -> None:
        state = self._states.get(request.request_id)
        if state is None or not state.pending_steps:
            return
        now = time.monotonic()
        step = state.pending_steps.pop(0)
        if num_new_tokens:
            state.token_times.extend([now] * num_new_tokens)
            if state.first_token_time is None:
                state.first_token_time = now
            state.last_token_time = now

        snapshot = self._snapshot(request.request_id, request.num_computed_tokens)
        state.peak_blocks = max(state.peak_blocks, snapshot["total_block_references"])
        wait_ms = (
            (state.first_schedule_time - state.add_time) * 1000
            if state.first_schedule_time is not None
            else None
        )
        ttft_ms = (
            (state.first_token_time - state.add_time) * 1000
            if state.first_token_time is not None
            else None
        )
        self._write_core(
            "STEP_COMPLETE",
            now,
            request_id=request.request_id,
            engine_step=step["engine_step"],
            phase=step["phase"],
            num_computed_tokens_before=step["num_computed_tokens_before"],
            num_scheduled_tokens=step["num_scheduled_tokens"],
            num_computed_tokens=request.num_computed_tokens,
            num_new_output_tokens=num_new_tokens,
            num_output_tokens=request.num_output_tokens,
            new_block_count=step["new_block_count"],
            allocation_duration_ms=step["allocation_duration_ms"],
            free_blocks_before_allocation=step["free_blocks_before_allocation"],
            free_blocks_after_allocation=step["free_blocks_after_allocation"],
            block_count_by_group=snapshot["block_count_by_group"],
            total_block_references=snapshot["total_block_references"],
            ref_cnt_min=snapshot["ref_cnt_min"],
            ref_cnt_max=snapshot["ref_cnt_max"],
            num_free_blocks=snapshot["num_free_blocks"],
            wait_ms=wait_ms,
            ttft_ms=ttft_ms,
            engine_step_ms=(now - step["schedule_start"]) * 1000,
        )
        self._write_detail(
            "MODEL_STEP_END",
            now,
            request_id=request.request_id,
            num_new_output_tokens=num_new_tokens,
            num_output_tokens=request.num_output_tokens,
            **step,
            **snapshot,
        )

    def on_finish_before_release(self, request: Request) -> None:
        state = self._states.get(request.request_id)
        if state is None:
            return
        now = time.monotonic()
        state.finish_time = now
        snapshot = self._snapshot(request.request_id, request.num_computed_tokens)
        common = {
            "request_id": request.request_id,
            "status": str(request.status),
            "prompt_tokens": request.num_prompt_tokens,
            "output_tokens": request.num_output_tokens,
            "num_computed_tokens": request.num_computed_tokens,
        }
        self._write_core(
            "FINISH_BEFORE_RELEASE",
            now,
            **common,
            block_count_by_group=snapshot["block_count_by_group"],
            total_block_references=snapshot["total_block_references"],
            ref_cnt_min=snapshot["ref_cnt_min"],
            ref_cnt_max=snapshot["ref_cnt_max"],
            num_free_blocks=snapshot["num_free_blocks"],
        )
        self._write_detail("FINISH_BEFORE_RELEASE", now, **common, **snapshot)

    def on_finish_after_release(
        self,
        request: Request,
        free_duration_ms: float,
        free_blocks_before_release: int,
    ) -> None:
        state = self._states.pop(request.request_id, None)
        if state is None:
            return
        now = time.monotonic()
        block_pool = self._kv_cache_manager.block_pool
        released_ids = state.last_block_ids
        released_ref_cnts = tuple(
            tuple(block_pool.blocks[block_id].ref_cnt for block_id in group)
            for group in released_ids
        )
        free_blocks = block_pool.get_num_free_blocks()
        release_ms = (
            (now - state.finish_time) * 1000 if state.finish_time is not None else None
        )
        total_ms = (now - state.add_time) * 1000
        wait_ms = (
            (state.first_schedule_time - state.add_time) * 1000
            if state.first_schedule_time is not None
            else None
        )
        prefill_ms = (
            (state.first_token_time - state.first_schedule_time) * 1000
            if state.first_token_time is not None
            and state.first_schedule_time is not None
            else None
        )
        ttft_ms = (
            (state.first_token_time - state.add_time) * 1000
            if state.first_token_time is not None
            else None
        )
        tpot_ms = None
        if len(state.token_times) > 1:
            tpot_ms = (
                (state.token_times[-1] - state.token_times[0])
                / (len(state.token_times) - 1)
                * 1000
            )
        summary = {
            "request_id": request.request_id,
            "prompt_tokens": request.num_prompt_tokens,
            "output_tokens": request.num_output_tokens,
            "num_computed_tokens": request.num_computed_tokens,
            "wait_ms": wait_ms,
            "prefill_ms": prefill_ms,
            "ttft_ms": ttft_ms,
            "mean_tpot_ms": tpot_ms,
            "total_ms": total_ms,
            "release_ms": release_ms,
            "free_duration_ms": free_duration_ms,
            "peak_block_references": state.peak_blocks,
            "initial_free_blocks": state.initial_free_blocks,
            "free_blocks_before_release": free_blocks_before_release,
            "final_free_blocks": free_blocks,
            "num_preemptions": request.num_preemptions,
            "recompute_occurred": request.num_preemptions > 0,
        }
        self._write_core("FINISH_AFTER_RELEASE", now, **summary)
        self._write_core("REQUEST_SUMMARY", now, **summary)
        self._write_detail(
            "FINISH_AFTER_RELEASE",
            now,
            **summary,
            released_block_ids_by_group=released_ids,
            released_ref_cnt_by_group=released_ref_cnts,
        )
