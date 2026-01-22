#!/usr/bin/env python3
"""
log_stats.py

Usage:
    python parse_log.py <logfile> <n>

功能：
 - 从给定文件中查找形如 "[Init 0.2066][Queue 0.0000][Prefill 0.0003]" 的匹配。
     对于每个匹配，提取 Init 和 Prefill 的数值（Prefill 必须紧跟在 Init 和 Queue 之后）。
 - 将匹配按顺序每隔 n 条记录切分为一组（组大小为 n）。例如 n=5 时，记录 0..4 属于组 0，5..9 属于组 1，以此类推。
 - 交互提示输入要统计的组范围（先最小，再最大）。空回车表示不限制。
 - 统计所选组内所有匹配中的数值总和、条目数，并计算两种平均：
         * average_per_entry = total_sum / total_entries
         * average_normalized = total_sum / (selected_group_count * n)
     （第二种对应“除以组数和 n”的实现，其中 n 现在为每组包含的记录数）

说明与假设：
 - 将每个匹配的 Init 和 Prefill 当作两个独立的数值条目进行统计。
 - 分组方式为固定块大小（chunk），不是按总组数平摊余数。
 - 如果你希望采用不同策略（如 round-robin 或每组不满时合并最后一组），可进一步调整。

"""
import argparse
import re
import sys
from math import ceil
from typing import List, Tuple, Any


def parse_args():
    p = argparse.ArgumentParser(description="统计日志中 Init/Prefill 或 TP GPU/Prefill/Decode 值并按组汇总 (n 表示每组记录数)")
    p.add_argument('file', help='要分析的日志文件路径')
    p.add_argument('n', type=int, help='每组包含 n 条记录（组大小）')
    # 支持同时指定多个模式（例如 --ttft --token）
    p.add_argument('--ttft', action='store_true', help='使用 Init/Prefill 的统计（默认）')
    p.add_argument('--gpu', action='store_true', help='使用 TP GPU / Prefill / Decode 的统计')
    p.add_argument('--lora', action='store_true', help='使用 LORA 的统计')
    p.add_argument('--prepare', action='store_true', help='使用 PREPARE 的统计')
    p.add_argument('--token', action='store_true', help='使用 Token 模式统计 Prefill Token: this=... 的 this 值')
    p.add_argument('--kvn', action='store_true', help='使用 KV 带斜杠的统计，例如 [KV 0.219 / 0.194]，分别统计斜杠前后两个值')
    p.add_argument('--kvo', action='store_true', help='使用 KV 单值统计，例如 [KV 0.445417]，统计该 kv 值')
    p.add_argument('--min', type=int, default=None, help='非交互式: 最小组编号（包含），回车或不提供表示不限制')
    p.add_argument('--max', type=int, default=None, help='非交互式: 最大组编号（包含），回车或不提供表示不限制')
    return p.parse_args()


PATTERN_TTFT = re.compile(r"\[Init\s+([0-9]*\.?[0-9]+)\]\[Queue\s+[0-9]*\.?[0-9]+\]\[Prefill\s+([0-9]*\.?[0-9]+)\](?:\[Reqs\s+([0-9]+)\])?")
PATTERN_GPU = re.compile(r"\[TP GPU\s+([0-9]*\.?[0-9]+)\]\(Prefill\s+([0-9]*\.?[0-9]+)\)\(Decode=([0-9]*\.?[0-9]+)\)")
PATTERN_LORA = re.compile(r"\[LORA\s*\(\s*([0-9]*\.?[0-9]+)\s*,\s*([0-9]*\.?[0-9]+)\s*\)\]")
PATTERN_PREPARE = re.compile(r"\[PREPARE\s+([0-9]*\.?[0-9]+)\]\[Prefill\s+([0-9]*\.?[0-9]+)\]\s*\[Other\s+([0-9]*\.?[0-9]+)\]\s*\[PROCESS\s+([0-9]*\.?[0-9]+)\]")
PATTERN_TOKEN_PREFILL = re.compile(r"Prefill Token:\s*this=([0-9]*\.?[0-9]+)")
PATTERN_TOKEN_DECODE = re.compile(r"Decode Token:\s*this=([0-9]*\.?[0-9]+)")
PATTERN_KVN = re.compile(r"\[KV\s+([0-9]*\.?[0-9]+)\s*/\s*([0-9]*\.?[0-9]+)\]")
PATTERN_KVO = re.compile(r"\[KV\s+([0-9]*\.?[0-9]+)\s*\]")


def find_ttft_matches(text: str) -> List[Tuple[Any, Any, Any]]:
    """返回每个匹配的 (init, prefill, reqs) 列表（TTFT 模式）。

    如果日志行包含 `[Reqs N]` 则第三项为整数 N，否则为 None。
    """
    results: List[Tuple[Any, Any, Any]] = []
    for m in PATTERN_TTFT.finditer(text):
        try:
            init_v = float(m.group(1))
            prefill_v = float(m.group(2))
        except Exception:
            continue
        reqs_v = None
        try:
            if m.lastindex and m.lastindex >= 3 and m.group(3) is not None:
                reqs_v = int(m.group(3))
        except Exception:
            reqs_v = None
        results.append((init_v, prefill_v, reqs_v))
    return results


def find_gpu_matches(text: str) -> List[Tuple[float, float, float]]:
    """返回每个匹配的 (tp_gpu, prefill, decode) 列表（GPU 模式）。"""
    results = []
    for m in PATTERN_GPU.finditer(text):
        try:
            tp_v = float(m.group(1))
            prefill_v = float(m.group(2))
            decode_v = float(m.group(3))
            results.append((tp_v, prefill_v, decode_v))
        except Exception:
            continue
    return results


def find_lora_matches(text: str) -> List[Tuple[float, float]]:
    """返回每个匹配的 (lora_a, lora_b) 列表（LORA 模式）。"""
    results = []
    for m in PATTERN_LORA.finditer(text):
        try:
            a = float(m.group(1))
            b = float(m.group(2))
            results.append((a, b))
        except Exception:
            continue
    return results


def find_prepare_matches(text: str) -> List[Tuple[float, float, float, float]]:
    """返回每个匹配的 (prepare, prefill, other, process) 列表（PREPARE 模式）。

    不再匹配或返回 KV 字段。
    """
    results: List[Tuple[float, float, float, float]] = []
    for m in PATTERN_PREPARE.finditer(text):
        try:
            prepare_v = float(m.group(1))
            prefill_v = float(m.group(2))
            other_v = float(m.group(3))
            process_v = float(m.group(4))
            results.append((prepare_v, prefill_v, other_v, process_v))
        except Exception:
            continue
    return results


def find_token_matches(text: str) -> List[Tuple[Any, Any]]:
    """返回每行中 Prefill Token 和 Decode Token 的 this 值对列表。

    每个元素为 (prefill_this, decode_this)，如果某一项缺失则为 None。
    这样可以按原有的分组逻辑把记录按行顺序分块，每条记录包含两项可能的值。
    """
    results: List[Tuple[Any, Any]] = []
    for line in text.splitlines():
        pre = None
        dec = None
        m1 = PATTERN_TOKEN_PREFILL.search(line)
        if m1:
            try:
                pre = float(m1.group(1))
            except Exception:
                pre = None
        m2 = PATTERN_TOKEN_DECODE.search(line)
        if m2:
            try:
                dec = float(m2.group(1))
            except Exception:
                dec = None

        if m1 or m2:
            results.append((pre, dec))

    return results


def find_kvn_matches(text: str) -> List[Tuple[float, float]]:
    """返回每个匹配的 (before, after) 列表，例如匹配 `[KV 0.2197161439 / 0.194054]`。"""
    results: List[Tuple[float, float]] = []
    for m in PATTERN_KVN.finditer(text):
        try:
            a = float(m.group(1))
            b = float(m.group(2))
            results.append((a, b))
        except Exception:
            continue
    return results


def find_kvo_matches(text: str) -> List[float]:
    """返回每个匹配的 kv 单值列表，例如匹配 `[KV 0.445417]`。"""
    results: List[float] = []
    for m in PATTERN_KVO.finditer(text):
        try:
            v = float(m.group(1))
            results.append(v)
        except Exception:
            continue
    return results


def split_into_n_groups(records: List[Any], n: int) -> List[List[Any]]:
    """按顺序把 records 每 n 条切为一组（chunk）。

    返回一个组列表，最后一组可能少于 n 条。
    """
    if n <= 0:
        raise ValueError('n 必须为正整数（组大小）')
    total = len(records)
    groups: List[List[Any]] = []
    for i in range(0, total, n):
        groups.append(records[i:i + n])
    return groups


def prompt_range(n: int) -> Tuple[int, int]:
    """交互提示最小和最大组索引，空回车表示不限制。返回 (min_idx, max_idx) 的闭区间。
    若输入越界会自动 clamp 到 [0, n-1]。"""
    raw_min = input(f"请输入最小组编号 (0..{n-1})，回车表示不限制: ").strip()
    raw_max = input(f"请输入最大组编号 (0..{n-1})，回车表示不限制: ").strip()

    if raw_min == '':
        min_idx = 0
    else:
        try:
            min_idx = int(raw_min)
        except ValueError:
            print('最小组输入无效，使用 0')
            min_idx = 0

    if raw_max == '':
        max_idx = n - 1
    else:
        try:
            max_idx = int(raw_max)
        except ValueError:
            print(f'最大组输入无效，使用 {n-1}')
            max_idx = n - 1

    # clamp
    min_idx = max(0, min(min_idx, n - 1))
    max_idx = max(0, min(max_idx, n - 1))

    if min_idx > max_idx:
        print('注意：最小组号大于最大组号，交换之')
        min_idx, max_idx = max_idx, min_idx

    return min_idx, max_idx


def get_range_from_args(min_arg: int, max_arg: int, n: int) -> Tuple[int, int]:
    """如果用户通过命令行提供了 --min/--max，则使用并 clamp；否则返回 None 来表示需交互。

    返回 (min_idx, max_idx)。
    """
    # If both are None, signal to caller to prompt interactively
    if min_arg is None and max_arg is None:
        return None  # type: ignore

    # Interpret None as unbounded
    if min_arg is None:
        min_idx = 0
    else:
        min_idx = int(min_arg)

    if max_arg is None:
        max_idx = n - 1
    else:
        max_idx = int(max_arg)

    # clamp
    min_idx = max(0, min(min_idx, n - 1))
    max_idx = max(0, min(max_idx, n - 1))
    if min_idx > max_idx:
        min_idx, max_idx = max_idx, min_idx
    return min_idx, max_idx


def main():
    args = parse_args()
    try:
        with open(args.file, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
    except Exception as e:
        print('无法打开文件：', e)
        sys.exit(2)

    # 支持同时指定多个统计模式；若未指定，则默认 ttft
    modes_to_run = []
    if args.ttft:
        modes_to_run.append('ttft')
    if args.gpu:
        modes_to_run.append('gpu')
    if args.lora:
        modes_to_run.append('lora')
    if args.prepare:
        modes_to_run.append('prepare')
    if args.token:
        modes_to_run.append('token')
    if args.kvn:
        modes_to_run.append('kvn')
    if args.kvo:
        modes_to_run.append('kvo')
    if not modes_to_run:
        modes_to_run.append('ttft')

    # 打印每个被请求模式的头（摘要信息将在下方详细列出）
    n = args.n
    if n <= 0:
        print('n 必须为正整数')
        sys.exit(2)

    # 先为所有模式收集记录并按 n 分组 -- 这样我们能在一次交互中让用户选择组范围
    mode_info = {}
    for mode in modes_to_run:
        if mode == 'ttft':
            records = find_ttft_matches(text)
        elif mode == 'token':
            records = find_token_matches(text)
        elif mode == 'gpu':
            records = find_gpu_matches(text)
        elif mode == 'lora':
            records = find_lora_matches(text)
        elif mode == 'prepare':
            records = find_prepare_matches(text)
        elif mode == 'kvn':
            records = find_kvn_matches(text)
        elif mode == 'kvo':
            records = find_kvo_matches(text)
        else:
            records = []

        groups = split_into_n_groups(records, n)
        mode_info[mode] = {
            'records': records,
            'groups': groups,
            'num_groups': len(groups),
            'total_records': len(records),
        }

    # 显示每个模式的匹配数量和组数
    print('\n模式摘要:')
    for mode in modes_to_run:
        info = mode_info[mode]
        if info['total_records'] == 0:
            print(f"  - {mode}: 未找到匹配记录")
        else:
            print(f"  - {mode}: 找到 {info['total_records']} 条匹配记录，每 {n} 条为一组 → {info['num_groups']} 组 (0..{max(0, info['num_groups']-1)})")

    # 统一选择范围（优先使用命令行提供的 --min/--max，否则交互询问一次）
    max_groups_overall = max((info['num_groups'] for info in mode_info.values()), default=0)
    if max_groups_overall == 0:
        print('在所选模式中未找到任何匹配记录，退出。')
        sys.exit(0)

    rng = get_range_from_args(args.min, args.max, max_groups_overall)
    if rng is None:
        min_idx, max_idx = prompt_range(max_groups_overall)
    else:
        min_idx, max_idx = rng

    # 对每个模式分别裁剪范围并执行原有统计逻辑
    for mode in modes_to_run:
        print(f"\n=== 统计模式: {mode} ===")
        info = mode_info[mode]
        num_groups = info['num_groups']
        if num_groups == 0:
            print(f"跳过 {mode}：未找到匹配记录。")
            continue

        # clamp range to this mode's available groups
        min_idx_mode = max(0, min(min_idx, num_groups - 1))
        max_idx_mode = max(0, min(max_idx, num_groups - 1))
        if min_idx_mode > max_idx_mode:
            min_idx_mode, max_idx_mode = max_idx_mode, min_idx_mode

        selected_group_indices = list(range(min_idx_mode, max_idx_mode + 1))
        groups = info['groups']
        selected_groups = [groups[i] for i in selected_group_indices]
        selected_group_count = len([g for g in selected_groups if g])

        # 根据模式执行对应的聚合和输出（保留原有行为，但使用 precomputed groups）

        if mode == 'ttft':
            init_sum = prefill_sum = 0.0
            init_count = prefill_count = 0
            reqs_sum = 0
            reqs_count = 0
            for g in selected_groups:
                for (init_v, prefill_v, reqs_v) in g:
                    init_sum += init_v
                    prefill_sum += prefill_v
                    init_count += 1
                    prefill_count += 1
                    if reqs_v is not None:
                        try:
                            reqs_sum += int(reqs_v)
                            reqs_count += 1
                        except Exception:
                            pass

            total_sum = init_sum + prefill_sum
            total_entries = init_count + prefill_count

            print('\n统计范围: 组', selected_group_indices)
            print('选中组中非空组数:', selected_group_count)
            print('\nInit 总和:', init_sum)
            print('Init 条目数:', init_count)
            if init_count > 0:
                print('Init 平均 (init_sum / init_count):', init_sum / init_count)
            else:
                print('Init 平均: 无可用条目')

            print('\nPrefill 总和:', prefill_sum)
            print('Prefill 条目数:', prefill_count)
            if prefill_count > 0:
                print('Prefill 平均 (prefill_sum / prefill_count):', prefill_sum / prefill_count)
            else:
                print('Prefill 平均: 无可用条目')

            # Reqs 统计
            print('\nReqs 总和:', reqs_sum)
            print('Reqs 条目数:', reqs_count)
            if reqs_count > 0:
                print('Reqs 平均 (reqs_sum / reqs_count):', reqs_sum / reqs_count)
            else:
                print('Reqs 平均: 无可用条目')

            denom = selected_group_count * n if selected_group_count > 0 else None
            if denom:
                avg_normalized = total_sum / denom
                print(f'按 (selected_group_count * n) 归一化平均 (total_sum / ({selected_group_count} * {n})): {avg_normalized}')
                print('按 (selected_group_count * n) 归一化 Init 平均 (init_sum / (selected_group_count * n)):', init_sum / denom)
                print('按 (selected_group_count * n) 归一化 Prefill 平均 (prefill_sum / (selected_group_count * n)):', prefill_sum / denom)
            else:
                print('按 (selected_group_count * n) 归一化平均: 无可用组')

        elif mode == 'token':
            # 每条记录为 (prefill_this, decode_this)
            prefill_sum = decode_sum = 0.0
            prefill_count = decode_count = 0
            for g in selected_groups:
                for (pre, dec) in g:
                    if pre is not None:
                        prefill_sum += pre
                        prefill_count += 1
                    if dec is not None:
                        decode_sum += dec
                        decode_count += 1

            total_sum = prefill_sum + decode_sum
            total_entries = prefill_count + decode_count

            print('\n统计范围: 组', selected_group_indices)
            print('选中组中非空组数:', selected_group_count)
            print('\nPrefill Token this 总和:', prefill_sum)
            print('Prefill Token this 条目数:', prefill_count)
            if prefill_count > 0:
                print('Prefill Token this 平均 (prefill_sum / prefill_count):', prefill_sum / prefill_count)
            else:
                print('Prefill Token this 平均: 无可用条目')

            print('\nDecode Token this 总和:', decode_sum)
            print('Decode Token this 条目数:', decode_count)
            if decode_count > 0:
                print('Decode Token this 平均 (decode_sum / decode_count):', decode_sum / decode_count)
            else:
                print('Decode Token this 平均: 无可用条目')

            print('\n合计 数值总和:', total_sum)
            print('合计 条目数:', total_entries)
            if total_entries > 0:
                print('按条目平均 (total_sum / total_entries):', total_sum / total_entries)
            else:
                print('按条目平均: 无可用条目')

            denom = selected_group_count * n if selected_group_count > 0 else None
            if denom:
                print('按 (selected_group_count * n) 归一化 Prefill Token 平均 (prefill_sum / denom):', prefill_sum / denom)
                print('按 (selected_group_count * n) 归一化 Decode Token 平均 (decode_sum / denom):', decode_sum / denom)
            else:
                print('按 (selected_group_count * n) 归一化平均: 无可用组')

        elif mode == 'gpu':
            tp_sum = prefill_sum = decode_sum = 0.0
            tp_count = prefill_count = decode_count = 0
            for g in selected_groups:
                for (tp_v, prefill_v, decode_v) in g:
                    tp_sum += tp_v
                    prefill_sum += prefill_v
                    decode_sum += decode_v
                    tp_count += 1
                    prefill_count += 1
                    decode_count += 1

            print('\n统计范围: 组', selected_group_indices)
            print('选中组中非空组数:', selected_group_count)
            print('\nTP GPU 总和:', tp_sum)
            print('TP GPU 条目数:', tp_count)
            if tp_count > 0:
                print('TP GPU 平均 (tp_sum / tp_count):', tp_sum / tp_count)
            else:
                print('TP GPU 平均: 无可用条目')

            print('\nPrefill 总和:', prefill_sum)
            print('Prefill 条目数:', prefill_count)
            if prefill_count > 0:
                print('Prefill 平均 (prefill_sum / prefill_count):', prefill_sum / prefill_count)
            else:
                print('Prefill 平均: 无可用条目')

            print('\nDecode 总和:', decode_sum)
            print('Decode 条目数:', decode_count)
            if decode_count > 0:
                print('Decode 平均 (decode_sum / decode_count):', decode_sum / decode_count)
            else:
                print('Decode 平均: 无可用条目')

            denom = selected_group_count * n if selected_group_count > 0 else None
            if denom:
                print('按 (selected_group_count * n) 归一化 TP GPU 平均 (tp_sum / (selected_group_count * n)):', tp_sum / denom)
                print('按 (selected_group_count * n) 归一化 Prefill 平均 (prefill_sum / (selected_group_count * n)):', prefill_sum / denom)
                print('按 (selected_group_count * n) 归一化 Decode 平均 (decode_sum / (selected_group_count * n)):', decode_sum / denom)
            else:
                print('按 (selected_group_count * n) 归一化平均: 无可用组')

        elif mode == 'lora':
            a_sum = b_sum = 0.0
            a_count = b_count = 0
            for g in selected_groups:
                for (a, b) in g:
                    a_sum += a
                    b_sum += b
                    a_count += 1
                    b_count += 1

            total_sum = a_sum + b_sum
            total_entries = a_count + b_count

            print('\n统计范围: 组', selected_group_indices)
            print('选中组中非空组数:', selected_group_count)
            print('\nLORA a 总和:', a_sum)
            print('LORA a 条目数:', a_count)
            if a_count > 0:
                print('LORA a 平均 (a_sum / a_count):', a_sum / a_count)
            else:
                print('LORA a 平均: 无可用条目')

            print('\nLORA b 总和:', b_sum)
            print('LORA b 条目数:', b_count)
            if b_count > 0:
                print('LORA b 平均 (b_sum / b_count):', b_sum / b_count)
            else:
                print('LORA b 平均: 无可用条目')

            print('\n合计 数值总和:', total_sum)
            print('合计 条目数 (a+b):', total_entries)
            if total_entries > 0:
                print('按条目平均 (total_sum / total_entries):', total_sum / total_entries)
            else:
                print('按条目平均: 无可用条目')

            denom = selected_group_count * n if selected_group_count > 0 else None
            if denom:
                print('按 (selected_group_count * n) 归一化 LORA a 平均 (a_sum / denom):', a_sum / denom)
                print('按 (selected_group_count * n) 归一化 LORA b 平均 (b_sum / denom):', b_sum / denom)
            else:
                print('按 (selected_group_count * n) 归一化平均: 无可用组')

        elif mode == 'kvn':
            # 每条记录为 (before, after)
            before_sum = after_sum = 0.0
            before_count = after_count = 0
            for g in selected_groups:
                for (bef, aft) in g:
                    try:
                        before_sum += float(bef)
                        before_count += 1
                    except Exception:
                        pass
                    try:
                        after_sum += float(aft)
                        after_count += 1
                    except Exception:
                        pass

            total_sum = before_sum + after_sum
            total_entries = before_count + after_count

            print('\n统计范围: 组', selected_group_indices)
            print('选中组中非空组数:', selected_group_count)
            print('\nKV (before) 总和:', before_sum)
            print('KV (before) 条目数:', before_count)
            if before_count > 0:
                print('KV (before) 平均 (before_sum / before_count):', before_sum / before_count)
            else:
                print('KV (before) 平均: 无可用条目')

            print('\nKV (after) 总和:', after_sum)
            print('KV (after) 条目数:', after_count)
            if after_count > 0:
                print('KV (after) 平均 (after_sum / after_count):', after_sum / after_count)
            else:
                print('KV (after) 平均: 无可用条目')

            print('\n合计 数值总和:', total_sum)
            print('合计 条目数:', total_entries)
            if total_entries > 0:
                print('按条目平均 (total_sum / total_entries):', total_sum / total_entries)
            else:
                print('按条目平均: 无可用条目')

            denom = selected_group_count * n if selected_group_count > 0 else None
            if denom:
                print('按 (selected_group_count * n) 归一化 KV (before) 平均 (before_sum / denom):', before_sum / denom)
                print('按 (selected_group_count * n) 归一化 KV (after) 平均 (after_sum / denom):', after_sum / denom)
            else:
                print('按 (selected_group_count * n) 归一化平均: 无可用组')

        elif mode == 'kvo':
            # 每条记录为单个 kv 值
            kv_sum = 0.0
            kv_count = 0
            for g in selected_groups:
                for v in g:
                    try:
                        kv_sum += float(v)
                        kv_count += 1
                    except Exception:
                        pass

            print('\n统计范围: 组', selected_group_indices)
            print('选中组中非空组数:', selected_group_count)
            print('\nKV 总和:', kv_sum)
            print('KV 条目数:', kv_count)
            if kv_count > 0:
                print('KV 平均 (kv_sum / kv_count):', kv_sum / kv_count)
            else:
                print('KV 平均: 无可用条目')

            denom = selected_group_count * n if selected_group_count > 0 else None
            if denom:
                print('按 (selected_group_count * n) 归一化 KV 平均 (kv_sum / denom):', kv_sum / denom)
            else:
                print('按 (selected_group_count * n) 归一化平均: 无可用组')

        elif mode == 'prepare':
            prepare_sum = prefill_sum = other_sum = process_sum = 0.0
            prepare_count = prefill_count = other_count = process_count = 0
            for g in selected_groups:
                for (prepare_v, prefill_v, other_v, process_v) in g:
                    prepare_sum += prepare_v
                    prefill_sum += prefill_v
                    other_sum += other_v
                    process_sum += process_v
                    prepare_count += 1
                    prefill_count += 1
                    other_count += 1
                    process_count += 1

            total_sum = prepare_sum + prefill_sum + other_sum + process_sum
            total_entries = prepare_count + prefill_count + other_count + process_count

            print('\n统计范围: 组', selected_group_indices)
            print('选中组中非空组数:', selected_group_count)
            print('\nPREPARE 总和:', prepare_sum)
            print('PREPARE 条目数:', prepare_count)
            if prepare_count > 0:
                print('PREPARE 平均 (prepare_sum / prepare_count):', prepare_sum / prepare_count)
            else:
                print('PREPARE 平均: 无可用条目')

            print('\nPrefill 总和:', prefill_sum)
            print('Prefill 条目数:', prefill_count)
            if prefill_count > 0:
                print('Prefill 平均 (prefill_sum / prefill_count):', prefill_sum / prefill_count)
            else:
                print('Prefill 平均: 无可用条目')

            print('\nOther 总和:', other_sum)
            print('Other 条目数:', other_count)
            if other_count > 0:
                print('Other 平均 (other_sum / other_count):', other_sum / other_count)
            else:
                print('Other 平均: 无可用条目')

            print('\nProcess 总和:', process_sum)
            print('Process 条目数:', process_count)
            if process_count > 0:
                print('Process 平均 (process_sum / process_count):', process_sum / process_count)
            else:
                print('Process 平均: 无可用条目')

            print('\n合计 数值总和:', total_sum)
            print('合计 条目数:', total_entries)
            if total_entries > 0:
                print('按条目平均 (total_sum / total_entries):', total_sum / total_entries)
            else:
                print('按条目平均: 无可用条目')

            denom = selected_group_count * n if selected_group_count > 0 else None
            if denom:
                print('按 (selected_group_count * n) 归一化 PREPARE 平均 (prepare_sum / denom):', prepare_sum / denom)
                print('按 (selected_group_count * n) 归一化 Prefill 平均 (prefill_sum / denom):', prefill_sum / denom)
                print('按 (selected_group_count * n) 归一化 Other 平均 (other_sum / denom):', other_sum / denom)
                print('按 (selected_group_count * n) 归一化 Process 平均 (process_sum / denom):', process_sum / denom)
            else:
                print('按 (selected_group_count * n) 归一化平均: 无可用组')

    # 处理完成（根据模式已输出相应统计）


if __name__ == '__main__':
    main()
