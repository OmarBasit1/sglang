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
    energy_j: float
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
    raw_logs_dict = load_logs(expr_dir)
    raw_logs_dict_steady = extract_steady_region(raw_logs_dict)

    perfstats_list = []
    for k, v in raw_logs_dict_steady.items():
        decode, prefill, power = v
        perfstats = calc_perf_stats_single_instance(k, decode, prefill, power)
        perfstats_list.append((k, perfstats))

    total_requests = sum(p.num_requests for k, p in perfstats_list if "prefill" in k)
    total_duration_prefill = max(p.expr_duration_s for k, p in perfstats_list if "prefill" in k)
    total_duration = max(p.expr_duration_s for _, p in perfstats_list)
    total_energy = sum(p.energy_j for _, p in perfstats_list)
    total_decode = sum(p.num_tokens_decoded for k, p in perfstats_list if "decode" in k)
    total_prefill = sum(p.num_tokens_prefilled for k, p in perfstats_list if "decode" in k)
    total_generated = sum(p.generated_tokens_per_second for k, p in perfstats_list if "decode" in k)
    total_input = sum(p.input_tokens_per_second for k, p in perfstats_list if "prefill" in k)

    total_perfstats = PerfStats(
        throughput_rps=total_requests / total_duration_prefill,
        power_w=total_energy / total_duration,
        energy_j=total_energy,
        energy_per_token=total_energy / (total_decode + total_prefill),
        freq_mhz_mean=0,
        freq_mhz_p10=0,
        freq_mhz_p50=0,
        freq_mhz_p90=0,
        expr_duration_s=total_duration,
        num_requests=total_requests,
        num_tokens_decoded=total_decode,
        num_tokens_prefilled=total_prefill,
        generated_tokens_per_second=total_generated,
        input_tokens_per_second=total_input
    )
    perfstats_list.append(('total', total_perfstats))
    return perfstats_list


def calc_perf_stats_single_instance(root_name: str,
                                    df_perf_metric_decode_steady: pd.DataFrame,
                                    df_perf_metric_prefill_steady: pd.DataFrame,
                                    df_power_steady: pd.DataFrame) -> PerfStats:
    
    # Calculate duration using min and max from both decode and prefill dfs
    decode_min = df_perf_metric_decode_steady['now'].min() if not df_perf_metric_decode_steady.empty else None
    decode_max = df_perf_metric_decode_steady['now'].max() if not df_perf_metric_decode_steady.empty else None
    prefill_min = df_perf_metric_prefill_steady['now'].min() if not df_perf_metric_prefill_steady.empty else None
    prefill_max = df_perf_metric_prefill_steady['now'].max() if not df_perf_metric_prefill_steady.empty else None

    min_time = min([t for t in [decode_min, prefill_min] if t is not None])
    max_time = max([t for t in [decode_max, prefill_max] if t is not None])
    duration = max_time - min_time

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
    # prefer prefill
    unique_req_ids = set()
    if "prefill" in root_name:
        for req_id_row in df_perf_metric_prefill_steady['req_ids_iter']:
            req_ids = list(eval(req_id_row))
            unique_req_ids.update(req_ids)
    else:
        for req_id_row in df_perf_metric_decode_steady['req_ids_iter']:
            req_ids = list(eval(req_id_row))
            unique_req_ids.update(req_ids)
    unique_req_ids = {req_id for req_id in unique_req_ids if "HEALTH_CHECK" not in str(req_id)}

    # # ttft calculated done as a difference of time req entered queue with time batch finished
    # id_start_time_end_time_dict = dict()
    # last_rows_req_ids = []
    # for req_id_row, req_queue_start_time_row, last_batch_finished_time_row in df_perf_metric_prefill_steady[['req_ids_iter', 'req_queue_start_time', 'last_batch_finished_time']].itertuples(index=False, name=None):
    #     req_ids = list(eval(req_id_row))
    #     req_queue_start_time = list(eval(req_queue_start_time_row))
    #     for id, start_time in zip(req_ids, req_queue_start_time):
    #         id_start_time_end_time_dict[id] = (start_time, 0.0)
    #     for last_req_id in last_rows_req_ids:
    #         id_start_time_end_time_dict[last_req_id] = (id_start_time_end_time_dict[last_req_id][0], last_batch_finished_time_row)
    #     last_rows_req_ids = req_ids
    # id_start_time_end_time_dict = {k: v for k, v in id_start_time_end_time_dict.items() if "HEALTH_CHECK" not in str(k)}
    # ttft_list = [end_time - start_time if end_time > start_time else 0 for start_time, end_time in id_start_time_end_time_dict.values()]

    # decode df has info about both input tokens and generated tokens
    id_decode_prefilled_dict = dict()
    if "decode" in root_name:
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
    else:
        for req_id_row, req_total_prefilled_tokens_row in df_perf_metric_prefill_steady[['req_ids_iter', 'req_total_prefilled_tokens']].itertuples(index=False, name=None):
            req_ids = list(eval(req_id_row))
            req_total_prefilled_tokens = list(eval(req_total_prefilled_tokens_row))
            for id, prefilled in zip(req_ids, req_total_prefilled_tokens):
                id_decode_prefilled_dict[id] = (0, prefilled)
        total_decoded = 0
        total_prefilled = sum(prefilled for _, prefilled in id_decode_prefilled_dict.values())
        total_input = sum(prefilled - 1 for _, prefilled in id_decode_prefilled_dict.values())
        total_generated = sum(prefilled for _, prefilled in id_decode_prefilled_dict.values()) - total_input
    

    return PerfStats(
        num_requests=len(unique_req_ids),
        throughput_rps=len(unique_req_ids) / duration,
        # ttft_mean=float(np.mean(ttft_list)),
        # ttft_p99=float(percentile_or_nan(ttft_list, q=99)),
        power_w=power_w,
        energy_j=energy_j_steady,
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

def load_logs(expr_dir: Path) -> dict:
    logs = {}
    for subfolder in sorted(expr_dir.iterdir()):
        if subfolder.is_dir():
            try:
                logs[subfolder.name] = load_logs_prefill_decode_power_logs(subfolder)
            except Exception as e:
                print(f"Skipping {subfolder} due to error: {e}")
    return logs

def load_logs_prefill_decode_power_logs(expr_dir: Path) -> Tuple[
    pd.DataFrame,   # decode
    pd.DataFrame,   # prefill
    pd.DataFrame,   # power
]:
    
    # Read all decode CSVs and concatenate
    # Read decode CSV if it exists
    decode_csv_paths = list(expr_dir.glob('perf_metric_*_decode.csv'))
    if len(decode_csv_paths) > 1:
        raise FileNotFoundError("More than one perf_metric_*_decode.csv file found in the directory")
    if decode_csv_paths:
        df_perf_metric_decode = pd.read_csv(decode_csv_paths[0])
    else:
        df_perf_metric_decode = pd.DataFrame()

    # Read prefill CSV if it exists, skip first two rows
    prefill_csv_paths = list(expr_dir.glob('perf_metric_*_prefill.csv'))
    if len(prefill_csv_paths) > 1:
        raise FileNotFoundError("More than one perf_metric_*_prefill.csv file found in the directory")
    if prefill_csv_paths:
        df_perf_metric_prefill = pd.read_csv(prefill_csv_paths[0])
        df_perf_metric_prefill = df_perf_metric_prefill[~df_perf_metric_prefill['req_ids_iter'].astype(str).str.contains('HEALTH_CHECK')].iloc[2:]
    else:
        df_perf_metric_prefill = pd.DataFrame()

    # Read the single power log CSV
    power_log_files = list(expr_dir.glob('power_log_*.csv'))
    if len(power_log_files) != 1:
        raise FileNotFoundError("There should be exactly one power_log_*.csv file in the directory")
    df_power = pd.read_csv(power_log_files[0])

    df_perf_metric_decode = df_perf_metric_decode.dropna()
    df_perf_metric_prefill = df_perf_metric_prefill.dropna()

    return df_perf_metric_decode, df_perf_metric_prefill, df_power


def extract_steady_region(
    raw_logs_dict: dict,
    clip_minutes: float = 0.0
) -> dict:
    """
    Drop the first and last clip_minutes of data from df_perf_metric_*
    """
    # Gather all decode and prefill logs
    decode_dfs = []
    prefill_dfs = []
    power_dfs = []

    for logs in raw_logs_dict.values():
        if isinstance(logs, tuple) and len(logs) == 3:
            decode_df, prefill_df, power_df = logs
            if not decode_df.empty:
                decode_dfs.append(decode_df)
            if not prefill_df.empty:
                prefill_dfs.append(prefill_df)
            if not power_df.empty:
                power_dfs.append(power_df)

    # Concatenate all logs
    df_perf_metric_decode_all = pd.concat(decode_dfs, ignore_index=True) if decode_dfs else pd.DataFrame()
    df_perf_metric_prefill_all = pd.concat(prefill_dfs, ignore_index=True) if prefill_dfs else pd.DataFrame()

    # Find min and max times
    decode_min = df_perf_metric_decode_all['now'].min() if not df_perf_metric_decode_all.empty else None
    decode_max = df_perf_metric_decode_all['now'].max() if not df_perf_metric_decode_all.empty else None
    prefill_min = df_perf_metric_prefill_all['now'].min() if not df_perf_metric_prefill_all.empty else None
    prefill_max = df_perf_metric_prefill_all['now'].max() if not df_perf_metric_prefill_all.empty else None

    # Use the earliest start and latest end
    global_min = min([t for t in [decode_min, prefill_min] if t is not None])
    global_max = max([t for t in [decode_max, prefill_max] if t is not None])

    # Optionally clip minutes from start/end
    clip_seconds = clip_minutes * 60
    steady_start = global_min + clip_seconds
    steady_end = global_max - clip_seconds

    # Filter steady region
    raw_logs_dict_steady = {}
    for key, logs in raw_logs_dict.items():
        if isinstance(logs, tuple) and len(logs) == 3:
            decode_df, prefill_df, power_df = logs
            decode_df_steady = decode_df[(decode_df['now'] >= steady_start) & (decode_df['now'] <= steady_end)] if not decode_df.empty else pd.DataFrame()
            prefill_df_steady = prefill_df[(prefill_df['now'] >= steady_start) & (prefill_df['now'] <= steady_end)] if not prefill_df.empty else pd.DataFrame()
            power_df_steady = power_df[(power_df['Timestamp'] >= steady_start) & (power_df['Timestamp'] <= steady_end)] if not power_df.empty else pd.DataFrame()
            raw_logs_dict_steady[key] = (decode_df_steady, prefill_df_steady, power_df_steady)
    return raw_logs_dict_steady


if __name__ == '__main__':
    if len(sys.argv) >= 2:
        expr_root = Path(sys.argv[1])
    else:
        expr_root = Path('/export2/obasit/ClusterLevelServing/sglang_logs') / \
            'test_logs' 

    # structure of log files should be like this:
    # |-> expr_root
    # |  |-> mixed_logs_test
    # |  |  |-> prefill_and_decode
    # |  |  |  |-> perf_metric_*_decode.csv
    # |  |  |  |-> perf_metric_*_prefill.csv
    # |  |  |  |-> power_log_*.csv
    # |  |
    # |  |-> disag_1P1D_test
    # |  |  |-> prefill_1
    # |  |  |  |-> perf_metric_*_prefill.csv
    # |  |  |  |-> power_log_*.csv
    # |  |  |-> decode_1
    # |  |  |  |-> perf_metric_*_decode.csv
    # |  |  |  |-> power_log_*.csv
    # |  |
    # |  |-> disag_2P1D_test
    # |  |  |-> prefill_1
    # |  |  |  |-> perf_metric_*_prefill.csv
    # |  |  |  |-> power_log_*.csv
    # |  |  |-> prefill_2
    # |  |  |  |-> perf_metric_*_prefill.csv
    # |  |  |  |-> power_log_*.csv
    # |  |  |-> decode_1
    # |  |  |  |-> perf_metric_*_decode.csv
    # |  |  |  |-> power_log_*.csv
    # ...

    df_stats = []
    for expr_dir in sorted(expr_root.glob('*')):
        if not expr_dir.is_dir():
            continue
        if not any(child.is_dir() for child in expr_dir.iterdir()):
            continue
        print('expr_dir: ', expr_dir)
        perfstats_list = calc_perf_stats(expr_dir)
        for key, perfstats in perfstats_list:
            df_stats.append({
                'expr_dir': expr_dir.name,
                'instance': key,
                **asdict(perfstats)
            })
    df_stats = pd.DataFrame(df_stats)
    df_stats.to_csv(expr_root / 'metrics.csv', index=False)
