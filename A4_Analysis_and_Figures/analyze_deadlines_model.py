#!/usr/bin/env python3
import pandas as pd
import glob
import re
import sys, os
from statistics import geometric_mean as gmean
import traceback
from tqdm import tqdm
from utils import TimeInterpolator

def enrich_df(df,  urgent_threshold=86400):
    """
    Compute additional metrics and update the DataFrame in place.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe to enrich
    interpolator : object
        Must have a method batch_durations(real_input_tokens, real_output_tokens, batch_size)
        that returns (prefill_seq, decode_seq)
    urgent_threshold : int or float, optional
        Deadline threshold (in seconds) to consider a request urgent.

    Returns
    -------
    pd.DataFrame
        Updated DataFrame with added columns.
    """

    # Urgent requests
    df["is_urgent"] = df["deadline"] < urgent_threshold

    # Deadline metrics
    df["missed_deadline"] = df["finish_time"] > df["deadline"]
    df["lateness"] = (df["finish_time"] - df["deadline"]).clip(lower=0)
    df["lateness_ratio"] = df["finish_time"] / df["deadline"]

    # Useful output tokens for urgent requests that met deadline
    urgent_and_on_time = df["is_urgent"] & (~df["missed_deadline"])
    df['useful_output_tokens_urgent'] = df.loc[urgent_and_on_time, "nb_output_tokens"].sum()

    return df

def merge_intervals(intervals):
    if not intervals:
        return []
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged

def analyse_csv(filename):
    try : 
        df = pd.read_csv(filename)
    except pd.errors.EmptyDataError:
        return None
   
    # Basic sanity check
    if df.empty:
        return None
        
    # Only enrich if 'sequential_work' does not exist
    if 'useful_output_tokens_urgent' not in df.columns :
        enrich_df(df)
        df.to_csv(filename, index=False)  # overwrite the original CSV

    useful_output_tokens_urgent = df['useful_output_tokens_urgent'].iloc[0]
    
    urgent_df = df[df["is_urgent"]]
    nb_urgent = len(urgent_df)
    missed_urgent = urgent_df["missed_deadline"].sum()
    makespan = df["finish_time"].max()


    success_ratio_urgent = (
        100.0 * (nb_urgent - missed_urgent) / nb_urgent
        if nb_urgent > 0 else None
    )
    total_tasks = len(df)
    missed = df["missed_deadline"].sum()

    success_ratio = 100.0 * (total_tasks - missed) / total_tasks

    avg_lateness = df.loc[df["missed_deadline"], "lateness"].mean()
    max_lateness = df["lateness"].max()

    geom_late = (
        gmean(df.loc[df["missed_deadline"], "lateness_ratio"])
        if missed > 0 else None
    )

    geom_on_time = (
        gmean(df.loc[~df["missed_deadline"], "lateness_ratio"])
        if missed < total_tasks else None
    )

    # -------------------------
    # UDS metric for urgent requests
    # -------------------------
    urgents = urgent_df.copy()
    if len(urgents) > 0:
        urgents['slack'] = urgents['deadline'] - urgents['submit_time']  # S_i

    # --------------------
    # GPU time accounting
    # --------------------
    intervals = df[["submit_time", "finish_time"]].values.tolist()
    merged_intervals = merge_intervals(intervals)
    merged_duration = sum(end - start for start, end in merged_intervals)

    # --------------------
    # Throughput metrics
    # --------------------
    urgent_tokens = df.loc[df["is_urgent"], "nb_output_tokens"].sum()

    total_output_tokens = df["nb_output_tokens"].sum()
    useful_output_tokens = df.loc[~df["missed_deadline"], "nb_output_tokens"].sum()

    system_throughput = (
        total_output_tokens / merged_duration if merged_duration > 0 else 0.0
    )

    useful_throughput = (
        useful_output_tokens / merged_duration if merged_duration > 0 else 0.0
    )

    goodput = (
        urgent_tokens / merged_duration if merged_duration > 0 else 0.0
    )

    useful_throughput_urgent = (
        useful_output_tokens_urgent / merged_duration
        if merged_duration > 0 else 0.0
    )

    nb_tasks = len(df)
    mean_interarrival_time = df["submit_time"].max() / nb_tasks

    return {
        "success_ratio": success_ratio,
        "success_ratio_urgent": success_ratio_urgent,
        "avg_lateness": avg_lateness,
        "max_lateness": max_lateness,
        "geom_late": geom_late,
        "geom_on_time": geom_on_time,
        "system_throughput": system_throughput,
        "useful_throughput": useful_throughput,
        "merged_duration": merged_duration,
        "total_output_tokens": total_output_tokens,
        "useful_throughput_urgent": useful_throughput_urgent,
        "makespan": makespan,
        "mean_interarrival_time": mean_interarrival_time,
        "goodput": goodput,
    }


def is_csv_empty(path):
    from pathlib import Path
    path = Path(path)

    # 1) fichier absent
    if not path.exists():
        return True

    # 2) fichier présent mais taille nulle
    if path.stat().st_size < 10:
        return True

    # 3) fichier avec headers uniquement
    df = pd.read_csv(path)
    return df.empty

def analyze_model(model):
    print (f"Analyzing model: {model}")
    nb_req_filter= 100
    print(f"Skipping nb_req != {nb_req_filter}")

    files = glob.glob(f"*deadlines_{model}.csv")

    if not files:
        print("No files found.")
        return 
    

    results = []

    files_sorted = sorted(files, key=lambda f: tuple(f.split('_')[2:]))

    for file  in tqdm(files_sorted, desc="Processing files"):
        if is_csv_empty(file):
            continue

        xp = file.replace(f"_{model}.csv", "")

        parts = xp.split("_", 8)
        # print(xp)
        # print(parts)

        if len(parts) != 9:
            print(f"Malformed filename: {file}")
            continue
        method, trace, rt_scaling, sla_factor, nb_req, skip_lines, nb_lora_adpt, percent_urgent, deadline_type = parts
        if int(nb_req) != nb_req_filter:
            continue

        try:
            metrics = analyse_csv(file)
        except Exception as e:
            print(f"Error analyzing {file}: {e}")
            traceback.print_exc()
            continue

        # print(f"{file=}", end=" ")
        # print(metrics["success_ratio_urgent"])

        # print(metrics["goodput"])
        results.append({
            "xp": xp,
            "method": method,
            "model": model,
            "trace": trace,
            "rt_scaling": float(rt_scaling),
            "sla_factor": float(sla_factor),
            "nb_req": int(nb_req),
            "skip_lines": int(skip_lines),
            "nb_lora_adpt": int(nb_lora_adpt),
            "deadline_type": deadline_type,
            "percent_urgent": float(percent_urgent),
            
           
            # arrival management
            "makespan": round(metrics["makespan"], 2),
            "mean_interarrival_time": round(metrics["mean_interarrival_time"], 4),

           
            # Deadline compliance
            "success_ratio": metrics["success_ratio"],
            "success_ratio_urgent": metrics["success_ratio_urgent"],
            
            # Lateness
            "avg_lateness": round(metrics["avg_lateness"], 2) if metrics["avg_lateness"] is not None else -1,
            "max_lateness": round(metrics["max_lateness"], 2) if metrics["max_lateness"] is not None else -1,

            # Throughput
            "system_throughput": round(metrics["system_throughput"], 2),
            "useful_throughput": round(metrics["useful_throughput"], 2),
            "useful_throughput_urgent": round(metrics["useful_throughput_urgent"], 2),
            "goodput": round(metrics["goodput"], 2),

            # Relative lateness (geom)
            "geom_late": round(metrics["geom_late"], 2) if metrics["geom_late"] is not None else -1,
            "geom_on_time": round(metrics["geom_on_time"], 2) if metrics["geom_on_time"] is not None else -1,
        })

    if not results:
        print("No valid results.")
        return 
    
    df = pd.DataFrame(results)


    filename = f"results_deadline_{model}.csv"
    df.to_csv(filename, index=False)
    print(f"CSV file saved to {filename}")

def get_model_list_set():
    model_list_set = set()

    for file in glob.glob("*deadlines_*.csv"):
        if "monitoring" in file or "log" in file:
            continue
        model = file.split("deadlines_")[1].replace(".csv", "")
        model_list_set.add(model)
        
    print (f"Found models: {model_list_set}")
    return model_list_set

if __name__ == "__main__":
    model_list_set = get_model_list_set()
    # model = 'mistral_nvidia-a100-80gb-pcie_4'
    # analyze_model(model)
    
    for model in model_list_set:
        analyze_model(model)
        