import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import pandas as pd
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator, RBFInterpolator
import numpy as np
import logging
import bisect
import sys
from choose_adapter import WeightedAdapterSelector
import random
import hashlib

VLLM_BATCH_TPS = "./results/vllm_batch_64_tps.csv" # old model not used any more

class TimeInterpolator:
    """
    Interpolator for estimating 'time' based on input_len, output_len, and batch characteristics.
    Uses LinearNDInterpolator with fallback to NearestNDInterpolator for extrapolation.
    """

    def __init__(self, file_path: str, batch_mode):
        self.batch_mode = batch_mode
        # Read the CSV file
        df = pd.read_csv(file_path)


        # Add the (0, 0, 0) -> 0 anchor point if not already present
        anchor_point = {"full_input_len": 0, "full_output_len": 0, "batch_size": 0, "prefill_tps": 0.0, "decode_tps": 0.0}
        if not ((df["full_input_len"] == 0) & (df["full_output_len"] == 0) & (df["batch_size"] == 0)).any():
            df = pd.concat([pd.DataFrame([anchor_point]), df], ignore_index=True)

        self.max_batch_size = df["batch_size"].max()
        self.max_input_len = df["full_input_len"].max()
        self.max_output_len = df["full_output_len"].max()
            
        # Build interpolators
        points = df[["full_input_len", "full_output_len", "batch_size"]].values
        values = df["prefill_tps"].values

        self.prefill_linear = LinearNDInterpolator(points, values)
        self.prefill_rbf = RBFInterpolator(points, values)

        values = df["decode_tps"].values

        self.decode_linear = LinearNDInterpolator(points, values)
        self.decode_rbf = RBFInterpolator(points, values)
        

    def find_closest_prefill_match(self, batch_input_len, batch_output_len, batch_size, max_search=10000):
        """Find the nearest input_len to batch_input_len with a valid positive TPS prediction."""
    
        def is_valid(value):
            return value is not None and not np.isnan(value) and value > 0

        for i in range(max_search):
            # Search forward
            test_len = batch_input_len + i
            point = np.array([test_len, batch_output_len, batch_size])
            est = self.prefill_linear(point)
            if is_valid(est):
                return float(est)
        
            # Search backward (skip if same as forward)
            if i < batch_input_len:
                test_len = batch_input_len - i
                point = np.array([test_len, batch_output_len, batch_size])
                est = self.prefill_linear(point)
                if is_valid(est):
                    return float(est)

        return -1



    def find_closest_decode_match(self, batch_input_len, batch_output_len, batch_size, max_search=10000):
        """Find the nearest input_len to batch_input_len with a valid positive TPS prediction."""
    
        def is_valid(value):
            return value is not None and not np.isnan(value) and value > 0

        for i in range(max_search):
            # print(i)
            # Search forward
            test_len = batch_output_len + i
            point = np.array([batch_input_len, test_len, batch_size])
            est = self.decode_linear(point)
            if is_valid(est):
                return float(est)
        
            # Search backward (skip if same as forward)
            if i < batch_output_len:
                test_len = batch_output_len - i
                point = np.array([batch_input_len, test_len, batch_size])
                est = self.decode_linear(point)
                if is_valid(est):
                    return float(est)


        for i in range(max_search):
            # Search forward
            test_len = batch_input_len + i * 10
            point = np.array([test_len, batch_output_len, batch_size])
            est = self.prefill_linear(point)
            if is_valid(est):
                return float(est)
        
            # Search backward (skip if same as forward)
            if i < batch_input_len:
                test_len = batch_input_len - i * 10
                point = np.array([test_len, batch_output_len, batch_size])
                est = self.prefill_linear(point)
                if is_valid(est):
                    return float(est)
        return -1
    
        
    def batch_durations(self, batch_input_len: int, batch_output_len: int, batch_size: int):
        if batch_size == 0:
            return 0, 0;
        if (batch_input_len == 0) or  (batch_output_len == 0):
            return 0, 0;


        batch_size = min( self.max_batch_size, batch_size)
        batch_output_len = min( self.max_output_len, batch_output_len)
        batch_input_len = min( self.max_input_len, batch_input_len)
            

        # print(f"{batch_size} {batch_input_len} {batch_output_len}")
        point = np.array([batch_input_len, batch_output_len, batch_size])

        # Try linear interpolation
        estimated_prefill_tps = self.prefill_linear(point)
        estimated_decode_tps = self.decode_linear(point)

        # estimated_prefill_tps = self.prefill_rbf(point.reshape(1, -1)).item()
        # estimated_decode_tps = self.decode_rbf(point.reshape(1, -1)).item()

        # Fallback to RBFInterpolator if outside convex hull
        if estimated_prefill_tps is None or np.isnan(estimated_prefill_tps) or estimated_prefill_tps <= 0:
            estimated_prefill_tps = self.prefill_rbf(point.reshape(1, -1)).item()
        else:
            estimated_prefill_tps = float(estimated_prefill_tps)

        if estimated_decode_tps is None or np.isnan(estimated_decode_tps) or estimated_decode_tps <= 0:
            # print(f"ext0={estimated_decode_tps}")
            estimated_decode_tps = self.decode_rbf(point.reshape(1, -1)).item()
        else:
            estimated_decode_tps = float(estimated_decode_tps)

        if estimated_prefill_tps <= 0 :
            estimated_prefill_tps = self.find_closest_prefill_match(batch_input_len, batch_output_len, batch_size)

        if estimated_decode_tps <= 0 :
            # print(f"ext1={estimated_decode_tps}")
            estimated_decode_tps = self.find_closest_decode_match(batch_input_len, batch_output_len, batch_size)
            # print(f"ext2={estimated_decode_tps}")

            
        if estimated_prefill_tps <= 0 or estimated_decode_tps <= 0:
            raise ValueError(f"Estimated TPS must be positive and non-zero. {batch_input_len} ; {batch_output_len} ; {batch_size} but are : {estimated_prefill_tps} and {estimated_decode_tps}")

        batch_prefill_duration = batch_input_len / estimated_prefill_tps
        batch_decode_duration = batch_output_len / estimated_decode_tps

        if self.batch_mode: # in batch mode executing a batch is roughly 20% faster
            batch_prefill_duration /= 1.2 
            batch_decode_duration /= 1.2
        
        return batch_prefill_duration, batch_decode_duration
    
    def task_durations(self, task_input_len, task_output_len, batch_input_len: int, batch_output_len: int, batch_size: int):
        while True:
            try:
                batch_prefill_duration, batch_decode_duration = self.batch_durations(batch_input_len + task_input_len, batch_output_len + task_output_len, batch_size+1)
                break;
            except e:
                batch_input_len += 1
                batch_output_len +=1

        task_prefill_duration = (task_input_len / (batch_input_len + task_input_len)) * batch_prefill_duration * (batch_size + 1)
        task_decode_duration = (task_output_len / (batch_output_len + task_output_len)) *batch_decode_duration * (batch_size + 1)
        return task_prefill_duration, task_decode_duration

    def estimate_time(self, task_input_len: int, task_output_len: int, batch) -> float:
        # Apportion batch time to task
        task_prefill_duration = (task_input_len / batch.nb_input_tokens) * batch.batch_prefill_duration if batch.nb_input_tokens > 0 else 0
        task_decode_duration = (task_output_len / batch.nb_output_tokens) * batch.batch_decode_duration if batch.nb_output_tokens > 0 else 0

        return float(task_prefill_duration + task_decode_duration)
    

@dataclass
class Task:
    id: str
    type: str
    input_tokens: int
    output_tokens: int
    remaining_input_tokens= -1
    remaining_output_tokens= -1
    prefill_duration= -1
    decode_duration= -1 
    finish_time = -1
    start_time= -1 
    adapter: int
    release_time: int
    deadline: int
    soft_deadline: int
    seq_duration: int
    duration_val: Optional["int"] = -1
    
    def update_remaining_tokens(self, current_time):
        """
        Update remaining input (prefill) tokens and output (decode) tokens based on current_time,
        assuming prefill must complete before decode starts.
        """

        if current_time <= self.start_time:
            # Task has not started yet
            self.remaining_input_tokens = self.input_tokens
            self.remaining_output_tokens = self.output_tokens

        elif current_time < self.start_time + self.prefill_duration:
            # Currently in prefill phase
            elapsed_prefill = current_time - self.start_time
            remain_fraction = 1 - (elapsed_prefill / self.prefill_duration)
            self.remaining_input_tokens = int(self.input_tokens * remain_fraction)
            # Decode not started yet, all output tokens remain
            self.remaining_output_tokens = self.output_tokens

        elif current_time < self.finish_time:
            # Currently in decode phase
            self.remaining_input_tokens = 0  # Prefill is done
            elapsed_decode = current_time - (self.start_time + self.prefill_duration)
            remain_fraction = 1 - (elapsed_decode / self.decode_duration)
            self.remaining_output_tokens = int(self.output_tokens * remain_fraction)

        else:
            # Task finished
            self.remaining_input_tokens = 0
            self.remaining_output_tokens = 0
    
    def duration(self, batch) -> float:
        self.duration_val = batch.interpolator.estimate_time(self.input_tokens, self.output_tokens, batch)
        return self.duration_val
    def __str__(self):
        return (f"\tTask(id={self.id}: input_tokens={self.input_tokens}, "
                f"output_tokens={self.output_tokens}, adapter={self.adapter}, "
                f"release_time={self.release_time}, start_time={self.start_time}, finish_time={self.finish_time}, deadline={self.deadline})")


def compute_seed_from_column(df, column='release_time'):
    """
    Compute a single 32-bit seed from all values of a DataFrame column.
    
    Parameters:
        df (pd.DataFrame): The DataFrame containing the column.
        column (str): Name of the column to use (default 'release_time').
    
    Returns:
        int: A 32-bit seed suitable for random.Random(seed).
    """
    # Concatenate all values as strings
    concat_str = ''.join(df[column].astype(str))
    # Compute MD5 and reduce to 32-bit integer
    md5_int = int(hashlib.md5(concat_str.encode()).hexdigest(), 16)
    seed = md5_int % (2**32)
    return seed


def read_tasks_from_csv(filepath: str, model, release_time_scaling, sla_factor, max_rows = None, skip_lines = 0, nb_lora_adapters = 2, random_deadlines=False, percentage_of_urgent_task = 100.0) -> List[Task]:
    selector = WeightedAdapterSelector([i for i in range(nb_lora_adapters)], window_size=20, seed=42)
    tasks = []


    
    df = pd.read_csv(filepath, nrows=None if max_rows < 0 else max_rows, skiprows=range(1, skip_lines+1))

    seed = compute_seed_from_column(df)
    urgent_random = random.Random(seed+1)
    if random_deadlines:
        my_random = random.Random(seed)
    


    # we both shift deadlines and release time to orgin time of 0. 
    min_release_time = df['release_time'].min()
    df['release_time'] = df['release_time'] - min_release_time
    df['release_time']  = df['release_time'] * release_time_scaling
    if 'deadline' in df.columns:
        df['deadline'] = df['deadline'] - min_release_time
        
    print(f"avg input tokens: {df['input_tokens'].mean()}")
    print(f"avg output tokens: {df['output_tokens'].mean()}")
    print(df)
    
    
    ONE_DAY = 24 * 60 * 60
    for idx,row in df.iterrows():
        if int(row["input_tokens"]) <= 0 and int(row["output_tokens"]) <= 0:
               continue
        input_tokens = max(1, int(row["input_tokens"]))
        output_tokens = max(1, int(row["output_tokens"]))

        prefill_tps, decode_tps = model.compute_tps()
        # print(f"{prefill_tps=} {decode_tps=}", end=" ")
        prefill_duration, decode_duration = input_tokens / prefill_tps, output_tokens / decode_tps
        # print(f"{input_tokens=} {output_tokens=}", end=" ")
        
        seq_duration = prefill_duration + decode_duration
        # print(f"{prefill_duration=} {decode_duration=} {seq_duration=}")
        release_time = row['release_time']
        if random_deadlines:
            deadline = release_time + my_random.uniform(0,sla_factor * (prefill_duration + decode_duration))
        else:
            if 'deadline' in df.columns:
                deadline = float(row['deadline'])
            else:
                deadline = release_time + sla_factor * seq_duration
        
        is_urgent = urgent_random.random() < percentage_of_urgent_task / 100
        if not is_urgent:
            deadline = release_time + ONE_DAY 
        
        # print(f"{idx=}\t{release_time=}\t{deadline=} {seq_duration=}")
        task = Task(
            type = "compute",
            id      = row["id"] if "id" in row and row["id"] != "" else str(idx),
            adapter = int(row["adapter"]) if "adapter" in row and row["adapter"] != "" else selector.next_third(),  # Round robin adapter assignment
            input_tokens  = int(row["input_tokens"]),
            output_tokens = int(row["output_tokens"]),
            release_time  = release_time,
            deadline = deadline,
            seq_duration = seq_duration,
            soft_deadline = deadline
            
            # deadline = release_time + int(row["input_tokens"])/100
        )
        tasks.append(task)
    return tasks, nb_lora_adapters



def read_tasks_from_csv_h100(filepath: str, interpolator: TimeInterpolator, release_time_scaling, sla_factor, max_rows = None, skip_lines = 0, nb_lora_adapters = 2, random_deadlines=False, percentage_of_urgent_task = 100.0) -> List[Task]:
    selector = WeightedAdapterSelector([i for i in range(nb_lora_adapters)], window_size=20, seed=42)
    tasks = []


    
    df = pd.read_csv(filepath, nrows=None if max_rows < 0 else max_rows, skiprows=range(1, skip_lines+1))

    seed = compute_seed_from_column(df)
    urgent_random = random.Random(seed+1)
    if random_deadlines:
        my_random = random.Random(seed)
    


    # we both shift deadlines and release time to orgin time of 0. 
    min_release_time = df['release_time'].min()
    df['release_time'] = df['release_time'] - min_release_time
    df['release_time']  = df['release_time'] * release_time_scaling
    if 'deadline' in df.columns:
        df['deadline'] = df['deadline'] - min_release_time
        
    print(f"avg input tokens: {df['input_tokens'].mean()}")
    print(f"avg output tokens: {df['output_tokens'].mean()}")
    print(df)
    
    
    ONE_DAY = 24 * 60 * 60
    for idx,row in df.iterrows():
        if int(row["input_tokens"]) <= 0 and int(row["output_tokens"]) <= 0:
               continue
        input_tokens = max(1, int(row["input_tokens"]))
        output_tokens = max(1, int(row["output_tokens"]))

        
        while True:
            try:
                # print(f"{input_tokens=} {output_tokens=}")
                prefill_duration, decode_duration = interpolator.batch_durations(input_tokens, output_tokens, 1)
                break
            except ValueError:
                # Décrémente, mais jamais en dessous de 1
                input_tokens = int(max(1, input_tokens/2))
                output_tokens = int(max(1, output_tokens/2))

        seq_duration = prefill_duration + decode_duration

        release_time = row['release_time']
        if random_deadlines:
            deadline = release_time + my_random.uniform(0,sla_factor * (prefill_duration + decode_duration))
        else:
            if 'deadline' in df.columns:
                deadline = float(row['deadline'])
            else:
                deadline = release_time + sla_factor * seq_duration
        
        is_urgent = urgent_random.random() < percentage_of_urgent_task / 100
        if not is_urgent:
            deadline = release_time + ONE_DAY 
        
        # print(f"{idx=}\t{release_time=}\t{deadline=} {seq_duration=}")
        task = Task(
            type = "compute",
            id      = row["id"] if "id" in row and row["id"] != "" else str(idx),
            adapter = int(row["adapter"]) if "adapter" in row and row["adapter"] != "" else selector.next_third(),  # Round robin adapter assignment
            input_tokens  = int(row["input_tokens"]),
            output_tokens = int(row["output_tokens"]),
            release_time  = release_time,
            deadline = deadline,
            seq_duration = seq_duration,
            soft_deadline = deadline
            
            # deadline = release_time + int(row["input_tokens"])/100
        )
        tasks.append(task)
    return tasks, nb_lora_adapters


