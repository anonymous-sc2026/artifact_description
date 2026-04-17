#!/usr/bin/env python3

import pandas as pd
import numpy as np
import sys, os
from glob import glob

def compute_kv_cache(df, task_cols, finished_mask=None):
    """
    Compute per-task cumulative token usage (KV cache) across rounds.
    
    Args:
        df : pd.DataFrame
            Dataframe containing one column per task with tokens generated per round.
        task_cols : list of str
            List of column names corresponding to tasks.
        finished_mask : pd.DataFrame, optional
            Boolean DataFrame same shape as df[task_cols], True where task is finished.
            If None, cumulative sum resets only if a task is never active again.
    
    Returns:
        pd.DataFrame : cumulative tokens per task (KV cache)
    """
    # Copy to avoid modifying original data
    token_cumsum = df[task_cols].copy()

    for col in task_cols:
        # Mask where the task is currently generating tokens
        active = df[col] != 0

        # If no finished_mask is provided, treat last non-zero as end of task
        if finished_mask is None:
            # Assume task is finished if it never becomes non-zero again
            finished = (~active[::-1].cummax())[::-1]
        else:
            finished = finished_mask[col]

        # Grouping: increment group counter at each finished point
        group = finished.cumsum()

        # Only accumulate tokens when task is active
        token_cumsum[col] = df[col].where(active, 0).groupby(group).cumsum()

    return token_cumsum

def augment_csv(file_path: str):
    df = pd.read_csv(file_path)

    task_cols = [c for c in df.columns if c.startswith("task_")]
    tokens = df[task_cols].to_numpy()

    decode_mask = tokens == 1
    prefill_mask = tokens > 1

    token_cumsum = compute_kv_cache(df, task_cols)
 
    df["kv_cache_total"] = (token_cumsum ** 2).sum(axis=1)


    df["num_decode_requests"] = (tokens * decode_mask).sum(axis=1)
    df["prefill_tokens"] = (tokens * prefill_mask).sum(axis=1)
    df["num_prefill_requests"] = prefill_mask.sum(axis=1)
    df = df.loc[:, ~df.columns.str.startswith("task_")]
    # Compute time difference between consecutive rows
    df['round_time'] = df['timestamp'].diff()
    # Drop the first two rows
    df = df.iloc[2:].reset_index(drop=True)
    df["scheduling_time"] = df['prev_round_duration'] - df["prev_fp_duration"]
    df["diff"] = abs(df["round_time"] - df["prev_round_duration"])

       # Vérifie les entiers manquants dans la colonne "round"
    rounds = df["round"].dropna().astype(int)
    missing = sorted(set(range(rounds.min(), rounds.max() + 1)) - set(rounds))

    if missing:
        first_missing = missing[0]
        print(f"{file_path} Missing rounds: {missing}")
         # rounds to inspect: r-1 and r+1
        neighbors = [first_missing - 1, first_missing + 1]
        cols_to_show = ["round", "num_decode_requests", "prefill_tokens",
                        "num_prefill_requests", "prev_fp_duration",
                        "prev_round_duration", "timestamp"]

        subset = df[df["round"].isin(neighbors)][cols_to_show]
        print(subset)
    else:
        print("✅ Column 'round' is complete (all integers from min to max are present).")

    df = df[~((df["num_decode_requests"] == 0) &
          (df["prefill_tokens"] == 0) &
          (df["num_prefill_requests"] == 0))]
    
    print(df["diff"].mean())

    

    df = df[["round", "num_decode_requests", "prefill_tokens", "num_prefill_requests", "kv_cache_total", "prev_fp_duration", "prev_round_duration", "timestamp"]]
    df = df.rename(columns={
        "prev_fp_duration": "fp_duration",
        "prev_round_duration": "round_duration"
    })
    
    # Force integer types
    df[["round", "num_decode_requests", "prefill_tokens", "num_prefill_requests", "kv_cache_total"]] = (
        df[["round", "num_decode_requests", "prefill_tokens", "num_prefill_requests", "kv_cache_total"]].astype(int)
    )
    
    
    n_requests = df["num_decode_requests"] + df["num_prefill_requests"]
    df["kv_cache_per_req"] = df["kv_cache_total"] / n_requests

    df["filename"] = file_path
    # print(df.sort_values("round_duration"))
    return df

def extend_output(output_csv: str, new_csv: str):
    """
    Append the contents of new_csv to output_csv after augmenting it.
    If output_csv does not exist, it will be created.
    Handles empty DataFrames gracefully.
    """
    df_new = augment_csv(new_csv)

    if not os.path.exists(output_csv) or df_new.empty:
        # write CSV (creates if missing, handles empty DataFrame)
        df_new.to_csv(output_csv, index=False, mode='w')
    else:
        # append without header
        df_new.to_csv(output_csv, index=False, mode='a', header=False)

    print(f"Extended {output_csv} with {len(df_new)} rows")

def build_csv(output_csv, glob_pattern):
    if os.path.exists(output_csv):
        os.remove(output_csv)
    header_written = False

        
    file_list = glob(glob_pattern)
    n = len(file_list)
    # Iterate over all *_monitoring.csv files in the current folder
    for i, file_path in enumerate(file_list):
    # for file_path in glob("out-of-order-discard-most-urgent_BurstGPT_1e-07_5_1000_0_1_rnd-deadlines_monitoring.csv"):
        print(f"Augmenting {file_path} {i+1}/{n}")
        df = augment_csv(file_path)  # your function that processes each CSV

        # Append to the final CSV
        if not header_written:
            df.to_csv(output_csv, index=False, mode="w")  # write header the first time
            header_written = True
        else:
            df.to_csv(output_csv, index=False, mode="a", header=False)  # append without header

    print(f"full CSV written to: {output_csv}")


def build_csv_2(output_csv, glob_pattern):
    if os.path.exists(output_csv):
        os.remove(output_csv)
    file_list = glob(glob_pattern)
    n = len(file_list)
    # Iterate over all *_monitoring.csv files in the current folder
    for i, file_path in enumerate(file_list):
    # for file_path in glob("out-of-order-discard-most-urgent_BurstGPT_1e-07_5_1000_0_1_rnd-deadlines_monitoring.csv"):
        print(f"Augmenting {file_path} {i+1}/{n}")
        extend_output(output_csv, file_path)
    print(f"full CSV written to: {output_csv}")

if __name__ == "__main__":
    build_csv_2("monitoring-monotonic_nvidia-a100-80gb-pcie_1_mistral.csv", "monitoring_data/*_monitoring.csv")
