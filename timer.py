"""Latency profiler for the plan-execute workflow.

Measures wall-clock time at each phase:
  1. Discovery  — spawning MCP servers and listing tools
  2. Planning   — LLM call to decompose the question into steps
  3. Execution  — per step: MCP tool call + optional LLM arg resolution
  4. Summary    — LLM call to synthesise the final answer

Usage:
    uv run python -m workflow.profiler "What assets are at site MAIN?"
    uv run python -m workflow.profiler --runs 3 "What assets are at site MAIN?"
    uv run python -m workflow.profiler --model-id watsonx/ibm/granite-3-3-8b-instruct "..."
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ── data classes ──────────────────────────────────────────────────────────────

@dataclass
class StepTiming:
    step_number: int
    server: str
    task: str
    tool: str
    llm_resolve_s: float = 0.0   # time spent resolving {step_N} placeholders
    tool_call_s: float = 0.0     # time spent in the MCP tool call
    total_s: float = 0.0
    success: bool = True


@dataclass
class RunTiming:
    question: str
    discovery_s: float = 0.0
    planning_s: float = 0.0
    steps: list[StepTiming] = field(default_factory=list)
    summarization_s: float = 0.0
    total_s: float = 0.0

    @property
    def execution_s(self) -> float:
        return sum(s.total_s for s in self.steps)


# ── instrumented runner ───────────────────────────────────────────────────────

class ProfiledRunner:
    """Wraps PlanExecuteRunner and injects timing at each phase boundary."""

    def __init__(self, model_id: str, server_paths: dict | None = None) -> None:
        from llm.litellm import LiteLLMBackend
        from workflow.executor import Executor, DEFAULT_SERVER_PATHS, _has_placeholders, _resolve_args_with_llm, _call_tool
        from workflow.planner import Planner

        self._model_id = model_id
        self._llm = LiteLLMBackend(model_id=model_id)
        self._server_paths = server_paths or DEFAULT_SERVER_PATHS
        self._planner = Planner(self._llm)
        self._executor = Executor(self._llm, self._server_paths)

        # stash references to internal helpers for timed calls
        self._has_placeholders = _has_placeholders
        self._resolve_args_with_llm = _resolve_args_with_llm
        self._call_tool = _call_tool

    async def run(self, question: str) -> RunTiming:
        from workflow.models import Plan, StepResult

        timing = RunTiming(question=question)
        run_start = time.perf_counter()

        # ── 1. Discovery ──────────────────────────────────────────────────────
        t0 = time.perf_counter()
        server_descriptions = await self._executor.get_server_descriptions()
        timing.discovery_s = time.perf_counter() - t0

        # ── 2. Planning ───────────────────────────────────────────────────────
        t0 = time.perf_counter()
        plan = self._planner.generate_plan(question, server_descriptions)
        timing.planning_s = time.perf_counter() - t0

        # ── 3. Execution (step by step) ───────────────────────────────────────
        ordered = plan.resolved_order()
        context: dict[int, StepResult] = {}

        for step in ordered:
            step_start = time.perf_counter()
            st = StepTiming(
                step_number=step.step_number,
                server=step.server,
                task=step.task,
                tool=step.tool or "none",
            )

            server_path = self._server_paths.get(step.server)
            if server_path is None or not step.tool or step.tool.lower() in ("none", "null"):
                # no tool call — record zero times
                result = StepResult(
                    step_number=step.step_number,
                    task=step.task,
                    server=step.server,
                    response=step.expected_output,
                    tool=step.tool,
                    tool_args=step.tool_args,
                )
                st.total_s = time.perf_counter() - step_start
                context[step.step_number] = result
                timing.steps.append(st)
                continue

            try:
                resolved_args = step.tool_args

                if self._has_placeholders(step.tool_args):
                    t_llm = time.perf_counter()
                    resolved_args = await self._resolve_args_with_llm(
                        step.task, step.tool, step.tool_args, context, self._llm
                    )
                    st.llm_resolve_s = time.perf_counter() - t_llm

                t_tool = time.perf_counter()
                response = await self._call_tool(server_path, step.tool, resolved_args)
                st.tool_call_s = time.perf_counter() - t_tool

                result = StepResult(
                    step_number=step.step_number,
                    task=step.task,
                    server=step.server,
                    response=response,
                    tool=step.tool,
                    tool_args=resolved_args,
                )
            except Exception as exc:  # noqa: BLE001
                result = StepResult(
                    step_number=step.step_number,
                    task=step.task,
                    server=step.server,
                    response="",
                    error=str(exc),
                    tool=step.tool,
                    tool_args=step.tool_args,
                )
                st.success = False

            st.total_s = time.perf_counter() - step_start
            context[step.step_number] = result
            timing.steps.append(st)

        # ── 4. Summarization ──────────────────────────────────────────────────
        from workflow.runner import _SUMMARIZE_PROMPT

        results_text = "\n\n".join(
            f"Step {r.step_number} — {r.task} (server: {r.server}):\n"
            + (r.response if r.success else f"ERROR: {r.error}")
            for r in context.values()
        )

        t0 = time.perf_counter()
        self._llm.generate(
            _SUMMARIZE_PROMPT.format(question=question, results=results_text)
        )
        timing.summarization_s = time.perf_counter() - t0

        timing.total_s = time.perf_counter() - run_start
        return timing


# ── reporting ─────────────────────────────────────────────────────────────────

def _bar(value: float, total: float, width: int = 20) -> str:
    filled = int(round(value / total * width)) if total > 0 else 0
    return "█" * filled + "░" * (width - filled)


def print_run(timing: RunTiming, run_index: int | None = None) -> None:
    label = f"Run {run_index}" if run_index is not None else "Result"
    print(f"\n{'═' * 62}")
    print(f"  {label}: {timing.question[:55]}")
    print(f"{'═' * 62}")

    rows = [
        ("Discovery",     timing.discovery_s),
        ("Planning (LLM)", timing.planning_s),
    ]
    for st in timing.steps:
        tag = f"Step {st.step_number} [{st.server}] {st.tool}"
        rows.append((tag, st.total_s))
        if st.llm_resolve_s > 0:
            rows.append((f"  └─ LLM resolve", st.llm_resolve_s))
        if st.tool_call_s > 0:
            rows.append((f"  └─ tool call",   st.tool_call_s))
    rows.append(("Summarization (LLM)", timing.summarization_s))

    col = 32
    for label, t in rows:
        bar = _bar(t, timing.total_s)
        print(f"  {label:<{col}} {t:6.3f}s  {bar}")

    print(f"  {'─' * (col + 30)}")
    print(f"  {'TOTAL':<{col}} {timing.total_s:6.3f}s")


def print_summary(timings: list[RunTiming]) -> None:
    if len(timings) < 2:
        return

    print(f"\n{'═' * 62}")
    print(f"  Summary across {len(timings)} runs")
    print(f"{'═' * 62}")

    def stats(values: list[float]) -> str:
        mn, mx, avg = min(values), max(values), sum(values) / len(values)
        return f"avg={avg:.3f}s  min={mn:.3f}s  max={mx:.3f}s"

    col = 28
    print(f"  {'Phase':<{col}}  avg      min      max")
    print(f"  {'─' * 56}")

    phases = [
        ("Discovery",          [t.discovery_s for t in timings]),
        ("Planning (LLM)",     [t.planning_s for t in timings]),
        ("Execution (total)",  [t.execution_s for t in timings]),
        ("Summarization (LLM)",[t.summarization_s for t in timings]),
        ("TOTAL",              [t.total_s for t in timings]),
    ]
    for name, values in phases:
        print(f"  {name:<{col}}  {stats(values)}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="profiler",
        description="Phase-level latency profiler for the plan-execute workflow.",
    )
    parser.add_argument("question", help="The question to profile.")
    parser.add_argument(
        "--model-id",
        default="watsonx/meta-llama/llama-3-3-70b-instruct",
        metavar="MODEL_ID",
        help="LiteLLM model string (default: watsonx/meta-llama/llama-3-3-70b-instruct).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        metavar="N",
        help="Number of times to run the query (default: 1). Use 3+ for stable averages.",
    )
    return parser


async def _main(args: argparse.Namespace) -> None:
    runner = ProfiledRunner(model_id=args.model_id)
    timings: list[RunTiming] = []

    for i in range(1, args.runs + 1):
        if args.runs > 1:
            print(f"\nRun {i}/{args.runs}...", flush=True)
        t = await runner.run(args.question)
        timings.append(t)
        print_run(t, run_index=i if args.runs > 1 else None)

    print_summary(timings)


def main() -> None:
    from dotenv import load_dotenv
    load_dotenv()
    args = _build_parser().parse_args()
    asyncio.run(_main(args))


if __name__ == "__main__":
    main()
