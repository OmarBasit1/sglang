import time
from dataclasses import asdict, dataclass, field
from threading import Lock
from typing import Any

import numpy as np


@dataclass
class BenchmarkMetrics:
    latency: float = -1.0
    ttft: float = 0.0  # time to first token
    itl: list[float] = field(default_factory=list)  # inter-token latencies (per token)
    tpot: float | None = None  # avg next-token latencies
    prompt_len: int = -1
    output_len: int = 0


class BenchmarkRequestTracker:
    def __init__(self) -> None:
        self.start_time = time.perf_counter()
        self.recent_time = self.start_time
        self._metrics = BenchmarkMetrics()
        self._prev_completion_tokens: int | None = None
        self._received_first_token = False

    def update(
        self, *, prompt_tokens: int | None, completion_tokens: int | None
    ) -> None:
        if completion_tokens is None:
            return

        metrics = self._metrics
        now = time.perf_counter()

        if metrics.prompt_len < 0 and prompt_tokens is not None:
            metrics.prompt_len = int(prompt_tokens)

        prev = self._prev_completion_tokens
        if prev is None:
            prev = 0
        delta = int(completion_tokens) - prev
        if delta <= 0:
            self._prev_completion_tokens = int(completion_tokens)
            return

        if not self._received_first_token:
            metrics.ttft = now - self.start_time
            self._received_first_token = True

        interval = now - self.recent_time
        per_token = interval / delta
        metrics.itl.extend([per_token] * delta)
        metrics.output_len += delta

        self.recent_time = now
        self._prev_completion_tokens = int(completion_tokens)

    def finish(self) -> BenchmarkMetrics:
        if self._metrics.prompt_len < 0:
            raise ValueError("Prompt length not set")
        now = time.perf_counter()
        metrics = self._metrics
        metrics.latency = now - self.start_time
        if metrics.output_len > 1:
            metrics.tpot = (metrics.latency - metrics.ttft) / (metrics.output_len - 1)
        return metrics


@dataclass
class BenchmarkResults:
    completed: int
    total_input: int
    total_output: int
    request_throughput: float
    request_goodput: float
    output_throughput: float
    total_token_throughput: float
    mean_ttft_ms: float
    median_ttft_ms: float
    std_ttft_ms: float
    percentiles_ttft_ms: list[tuple[float, float]]
    mean_tpot_ms: float
    median_tpot_ms: float
    std_tpot_ms: float
    percentiles_tpot_ms: list[tuple[float, float]]
    mean_itl_ms: float
    median_itl_ms: float
    std_itl_ms: float
    percentiles_itl_ms: list[tuple[float, float]]
    mean_e2el_ms: float
    median_e2el_ms: float
    std_e2el_ms: float
    percentiles_e2el_ms: list[tuple[float, float]]


class SGLangBenchmarker:
    def __init__(self, selected_percentiles: list[float] | None = None) -> None:
        self._lock = Lock()
        self._metrics: list[BenchmarkMetrics] = []
        self._start_time: float | None = None
        self._selected_percentiles = (
            [99.0] if selected_percentiles is None else selected_percentiles
        )

    def is_active(self) -> bool:
        with self._lock:
            return self._start_time is not None

    def start_benchmark(self) -> None:
        with self._lock:
            self._metrics.clear()
            self._start_time = time.perf_counter()

    def add_metrics(self, metrics: BenchmarkMetrics | None) -> None:
        if metrics is None:
            return
        with self._lock:
            if self._start_time is None:
                return
            self._metrics.append(metrics)

    def stop_benchmark(self) -> dict[str, Any]:
        with self._lock:
            if self._start_time is None:
                raise ValueError("Benchmark not started")
            start_time = self._start_time
            outputs = list(self._metrics)
            self._start_time = None
            self._metrics.clear()

        dur_s = time.perf_counter() - start_time

        total_input = 0
        actual_output_lens: list[int] = []
        itls: list[float] = []
        tpots: list[float] = []
        ttfts: list[float] = []
        e2els: list[float] = []

        completed = len(outputs)
        good_completed = completed
        for m in outputs:
            total_input += m.prompt_len
            actual_output_lens.append(m.output_len)
            if m.tpot is not None:
                tpots.append(m.tpot)
            itls += m.itl
            ttfts.append(m.ttft)
            e2els.append(m.latency)

        total_output = sum(actual_output_lens)

        results = BenchmarkResults(
            completed=completed,
            total_input=total_input,
            total_output=total_output,
            request_throughput=(completed / dur_s) if dur_s > 0 else 0.0,
            request_goodput=(good_completed / dur_s) if dur_s > 0 else 0.0,
            output_throughput=(total_output / dur_s) if dur_s > 0 else 0.0,
            total_token_throughput=(
                (total_input + total_output) / dur_s if dur_s > 0 else 0.0
            ),
            mean_ttft_ms=float(np.mean(ttfts or 0) * 1000),
            std_ttft_ms=float(np.std(ttfts or 0) * 1000),
            median_ttft_ms=float(np.median(ttfts or 0) * 1000),
            percentiles_ttft_ms=[
                (p, float(np.percentile(ttfts or 0, p) * 1000))
                for p in self._selected_percentiles
            ],
            mean_tpot_ms=float(np.mean(tpots or 0) * 1000),
            std_tpot_ms=float(np.std(tpots or 0) * 1000),
            median_tpot_ms=float(np.median(tpots or 0) * 1000),
            percentiles_tpot_ms=[
                (p, float(np.percentile(tpots or 0, p) * 1000))
                for p in self._selected_percentiles
            ],
            mean_itl_ms=float(np.mean(itls or 0) * 1000),
            std_itl_ms=float(np.std(itls or 0) * 1000),
            median_itl_ms=float(np.median(itls or 0) * 1000),
            percentiles_itl_ms=[
                (p, float(np.percentile(itls or 0, p) * 1000))
                for p in self._selected_percentiles
            ],
            mean_e2el_ms=float(np.mean(e2els or 0) * 1000),
            std_e2el_ms=float(np.std(e2els or 0) * 1000),
            median_e2el_ms=float(np.median(e2els or 0) * 1000),
            percentiles_e2el_ms=[
                (p, float(np.percentile(e2els or 0, p) * 1000))
                for p in self._selected_percentiles
            ],
        )

        return asdict(results)


benchmarker = SGLangBenchmarker()
