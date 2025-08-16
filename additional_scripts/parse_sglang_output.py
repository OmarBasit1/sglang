import itertools
import json
import sys
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re


@dataclass
class PerfStats:
    throughput_rps: float   # Requests per second
    # ttft_mean: float
    # ttft_p99: float
    # tbt_mean_of_mean: float
    # tbt_p99_of_mean: float
    # tbt_old_mean: float
    # tbt_old_p99: float
    power_w: float
    energy_per_token: float
    freq_mhz_mean: float
    freq_mhz_p10: float
    freq_mhz_p50: float
    freq_mhz_p90: float
    # # mem_util_mean: float
    # # mem_util_p10: float
    # # mem_util_p50: float
    # # mem_util_p90: float
    # running_queue_len_mean: float
    # waiting_queue_len_mean: float
    expr_duration_s: float
    num_requests: int
    num_tokens_decoded: int
    num_tokens_prefilled: int
    # # num_preempted_reqs: float
    generated_tokens_per_second: float
    input_tokens_per_second: float


def calc_perf_stats(expr_dir: Path) -> PerfStats:
    df_perf_metric_decode, df_perf_metric_prefill, df_power = load_logs(expr_dir)
    df_perf_metric_decode_steady, df_perf_metric_prefill_steady, df_power_steady = extract_steady_region(df_perf_metric_decode,
                                                                                                         df_perf_metric_prefill,
                                                                                                         df_power)

    duration = df_perf_metric_decode_steady['now'].max(
    ) - df_perf_metric_prefill_steady['now'].min()

    # Calculate power/energy/freq within only the steady region
    freq_arr_list = []
    # Sum energy across all GPU_i_power_w columns
    energy_j_steady = 0.0
    for col in df_power_steady.columns:
        if col.startswith('GPU_') and col.endswith('_power_w'):
            energy_j_steady += np.trapezoid(
                df_power_steady[col], df_power_steady['Timestamp'])
        if col.startswith('GPU_') and col.endswith('_freq_mhz'):
            freq_arr_list.append(df_power_steady[col].to_numpy())
    power_w = energy_j_steady / duration

    # unique request IDs = num requests served
    unique_req_ids = set()
    for req_id_row in df_perf_metric_prefill_steady['req_ids_iter']:
        req_ids = list(eval(req_id_row))
        unique_req_ids.update(req_ids)
    for req_id_row in df_perf_metric_decode_steady['req_ids_iter']:
        req_ids = list(eval(req_id_row))
        unique_req_ids.update(req_ids)

    id_decode_prefilled_dict = dict()
    for req_id_row, req_precomputed_tokens_row, req_total_prefilled_tokens_row in df_perf_metric_decode_steady[['req_ids_iter', 'req_precomputed_tokens_iter', 'req_total_prefilled_tokens']].itertuples(index=False, name=None):
        req_ids = list(eval(req_id_row))
        req_precomputed_tokens = list(eval(req_precomputed_tokens_row))
        req_total_prefilled_tokens = list(eval(req_total_prefilled_tokens_row))
        for id, decoded, prefilled in zip(req_ids, req_precomputed_tokens, req_total_prefilled_tokens):
            if id not in id_decode_prefilled_dict:
                id_decode_prefilled_dict[id] = (decoded, prefilled)
            else:
                existing_decoded, existing_prefilled = id_decode_prefilled_dict[id]
                id_decode_prefilled_dict[id] = (max(decoded, existing_decoded), existing_prefilled)

    # -1 because the first token is generated in prefill not decode
    total_decoded = sum(decoded - 1 for decoded, _ in id_decode_prefilled_dict.values())
    total_prefilled = sum(prefilled for _, prefilled in id_decode_prefilled_dict.values())

    # the 1st token that was generated in prefill is also counted here
    total_generated = sum(decoded for decoded, _ in id_decode_prefilled_dict.values())
    total_input = sum(prefilled - 1 for _, prefilled in id_decode_prefilled_dict.values())

    return PerfStats(
        num_requests=len(unique_req_ids),
        throughput_rps=len(unique_req_ids) / duration,
        power_w=power_w,
        freq_mhz_mean=float(np.mean(freq_arr_list)),
        freq_mhz_p10=float(percentile_or_nan(
            freq_arr_list, q=10)),
        freq_mhz_p50=float(percentile_or_nan(
            freq_arr_list, q=50)),
        freq_mhz_p90=float(percentile_or_nan(
            freq_arr_list, q=90)),
        # running_queue_len_mean=global_running_queue_len.mean(),
        # waiting_queue_len_mean=global_waiting_queue_len.mean(),
        expr_duration_s=duration,
        num_tokens_decoded= total_decoded,
        num_tokens_prefilled=total_prefilled,
        energy_per_token=energy_j_steady / (total_decoded + total_prefilled),
        generated_tokens_per_second=total_generated / duration,
        input_tokens_per_second=total_input / duration
    )


def percentile_or_nan(a, q):
    if len(a) > 0:
        return np.percentile(a, q)
    else:
        return np.nan


def load_logs(expr_dir: Path) -> Tuple[
    pd.DataFrame,   # decode
    pd.DataFrame,   # prefill
    pd.DataFrame,   # power
]:
    try:
        inference_csv_path = next(expr_dir.glob('perf_metric_*_decode.csv'))
    except StopIteration:
        raise FileNotFoundError()
    # decode
    df_perf_metric_decode = pd.read_csv(inference_csv_path)
    # Extract the numbers in place of * in 'perf_metric_*_decode.csv'
    match = re.search(r'perf_metric_(\d+)_decode\.csv', inference_csv_path.name)
    if match:
        pid = int(match.group(1))
    else:
        raise ValueError("Could not extract PID from filename")
    # prefill
    inference_prefill_csv_path = next(expr_dir.glob(f'perf_metric_{pid}_prefill.csv'))
    df_perf_metric_prefill = pd.read_csv(inference_prefill_csv_path)
    # Skip the first two rows, first for test by inference engine, second test by sender
    df_perf_metric_prefill = df_perf_metric_prefill.iloc[2:]  
    # power
    df_power = pd.read_csv(expr_dir / f'power_log_{pid}.csv')

    df_perf_metric_decode = df_perf_metric_decode.dropna()
    df_perf_metric_prefill = df_perf_metric_prefill.dropna()

    return df_perf_metric_decode, df_perf_metric_prefill, df_power


def extract_steady_region(
    df_perf_metric_decode: pd.DataFrame,
    df_perf_metric_prefill: pd.DataFrame,
    df_power: pd.DataFrame,
    clip_minutes: float = 60.0
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Drop the first minute of data from df_perf_metric_*
    """
    steady_start_time_df_perf_metric = df_perf_metric_prefill['now'].min() + clip_minutes
    steady_end_time_df_perf_metric = df_perf_metric_decode['now'].max() - clip_minutes

    df_perf_metric_decode_steady = df_perf_metric_decode[df_perf_metric_decode['now'].between(
        steady_start_time_df_perf_metric, steady_end_time_df_perf_metric)].reset_index(drop=True)

    df_perf_metric_prefill_steady = df_perf_metric_prefill[df_perf_metric_prefill['now'].between(
        steady_start_time_df_perf_metric, steady_end_time_df_perf_metric)].reset_index(drop=True)

    df_power_steady = df_power[df_power['Timestamp'].between(
        steady_start_time_df_perf_metric, steady_end_time_df_perf_metric)].reset_index(drop=True)

    return df_perf_metric_decode_steady, df_perf_metric_prefill_steady, df_power_steady


if __name__ == '__main__':
    if len(sys.argv) >= 2:
        expr_root = Path(sys.argv[1])
    else:
        expr_root = Path('/export2/obasit/ClusterLevelServing/sglang_logs') / \
            'test_logs' 

    df_stats = []
    for expr_dir in sorted(expr_root.glob('*')):
        if not expr_dir.is_dir():
            continue
        print('expr_dir: ', expr_dir)
        df_stats.append({
            'expr_dir': expr_dir.name,
            **asdict(calc_perf_stats(expr_dir))
        })
    df_stats = pd.DataFrame(df_stats)
    df_stats.to_csv(expr_root / 'metrics.csv', index=False)
