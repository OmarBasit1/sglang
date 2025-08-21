import json
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

@dataclass
class PerfStats:
    ttft_mean: float
    ttft_p90: float
    ttft_p99: float
    tpot_mean: float
    tpot_p90: float
    tpot_p99: float


def load_trace_into_df(trace_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    with open(trace_path) as f:
        trace = json.load(f)['traceEvents']
    df_freqs = []
    df_stats = []
    df_batches = []
    df_requests = []
    flattened_trace = []
    for t in trace:
        if isinstance(t, dict):
            flattened_trace.append(t)
        elif isinstance(t, list):
            for item in t:
                if isinstance(item, dict):
                    flattened_trace.append(item)
                else:
                    print(f"WARNING: unexpected item in trace: {item}")
        else:
            print(f"WARNING: unexpected type in trace: {t}")
    for t in flattened_trace:
        if not isinstance(t, dict) and isinstance(t, list):
            if len(t) > 1:
                print(len(t))
            t = t[0]
        if t.get('name') == 'stats':
            df_stats.append({
                'ts': t['ts'],
                **t['args'],
            })
        elif t.get('ph') == 'X':
            df_batches.append({
                'ts': t['ts'],
                **t['args'],
            })
        elif t.get('name') == 'request_end':
            df_requests.append({
                'ts': t['ts'],
                **t['args'],
            })
    df_stats = pd.DataFrame(df_stats)
    df_batches = pd.DataFrame(df_batches)
    df_requests = pd.DataFrame(df_requests)

    df_batches = df_batches.drop('ts', axis=1)
    df_stats = df_stats.join(df_batches, how='inner')

    return df_stats, df_requests


def calc_perf_stats(df_stats: pd.DataFrame, df_requests: pd.DataFrame) -> PerfStats:
    ttft_arr = (df_requests['prefill_completed_at'] - df_requests['arrived_at']).to_numpy()
    tpot_arr = ((df_requests['completed_at'] - df_requests['prefill_completed_at'])/df_requests['num_decode_tokens']).to_numpy()

    return PerfStats(
        ttft_mean=float(np.mean(ttft_arr)),
        ttft_p90=float(percentile_or_nan(ttft_arr, q=90)),
        ttft_p99=float(percentile_or_nan(ttft_arr, q=99)),
        tpot_mean=float(np.mean(tpot_arr)),
        tpot_p90=float(percentile_or_nan(tpot_arr, q=90)),
        tpot_p99=float(percentile_or_nan(tpot_arr, q=99)),
    )



def percentile_or_nan(a, q):
    if len(a) > 0:
        return np.percentile(a, q)
    else:
        return np.nan

if __name__ == '__main__':
    if len(sys.argv) > 1:
        expr_root = Path(sys.argv[1])
    else:
        expr_root = Path('/export2/home/kong102/vidur/simulator_output')
    df = []

    for expr_dir in sorted(expr_root.glob('*')):
        if not expr_dir.is_dir():
            continue
        trace_path = expr_dir / 'chrome_trace.json'

        try:
            df_stats, df_requests = load_trace_into_df(trace_path)
            s = calc_perf_stats(df_stats, df_requests)
        except FileNotFoundError:
            print(f'WARNING: log not found, skipping: {trace_path}')
            continue
        except AssertionError:
            print(f'WARNING: error parsing log, skipping: {trace_path}')
            continue
        # df_stats, df_requests = load_trace_into_df(trace_path)
        # s = calc_perf_stats(df_stats, df_requests)

        df.append({
            'expr_dir': expr_dir.name,
            **asdict(s),
        })
    try:
        pd.DataFrame(df).to_csv(expr_root / 'metrics.csv', index=False)
    except PermissionError:
        save_path = Path.home() / 'metrics.csv'
        pd.DataFrame(df).to_csv(save_path, index=False)
        print(f'No permission to save to expr_dir. Saved to: {save_path}')