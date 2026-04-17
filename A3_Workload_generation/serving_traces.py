#!/usr/bin/env python3

import sys, os
import numpy as np
import threading
import queue
import csv
import time
import argparse
import asyncio
import torch
import torch.distributed as dist
from pathlib import Path
from datetime import datetime
from vllm import SamplingParams
import atexit
import random 
import pandas as pd
import math 

from server_adapater import mistral_start,  model_server_start, NUM_GPUS

from shared_context import SharedContext, get_global_shared_context, SharedData, create_global_shm_shared_data, Algorithm, DEFAULT_SHM_NAME

import logging
from concurrent.futures import as_completed
from csv_utils import read_tasks_from_csv, read_tasks_from_csv_h100, TimeInterpolator, VLLM_BATCH_TPS
from performance_model.analyze_monitoring import extend_output
from performance_model.interpolator import perf_model_has_converged
from performance_model.round_duration import RoundDurationModel

def load_requests(filename):
    requests = []
    with open(filename, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for i, row in enumerate(reader):
            adapter_id = int(row["adapter_id"].strip())
            prompt = row["prompt"].strip()
            requests.append((f"req-{i}", adapter_id, prompt))
    return requests

async def collect_full_output(generator):
    final_output = None
    async for output in generator:
        final_output = output
    return final_output


async def serve_generations(gens):
    outputs = await asyncio.gather(
        *[collect_full_output(gen) for gen in gens]
    )
    return outputs

async def serve_generation(gen,idx):
    output = await asyncio.wait_for(
        collect_full_output(gen),
        timeout=1200
    )
    return {"idx": idx, "output": output}


def nb_tokens_in_prompts(prompts, tokenizer):
    """
    Returns the total number of non-padding tokens across all prompts.

    Args:
        prompts (list[str]): list of input prompt strings
        tokenizer (transformers.PreTrainedTokenizer): tokenizer instance

    Returns:
        int: total number of tokens (excluding padding)
    """
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    # Count only tokens not equal to pad_token_id
    num_tokens = (inputs["input_ids"] != tokenizer.pad_token_id).sum().item()
    return num_tokens

def generate_random_prompt(num_tokens, tokenizer):
    """
    Generates a random prompt with exactly `num_tokens` using the tokenizer.

    Args:
        tokenizer: Hugging Face tokenizer.
        num_tokens (int): Number of tokens in the generated prompt.

    Returns:
        str: Decoded text from randomly generated tokens.
    """
    vocab_size = tokenizer.vocab_size  # Get vocab size
    random_token_ids = torch.randint(low=0, high=vocab_size, size=(num_tokens,)).tolist()  # Generate random token IDs
    random_text = tokenizer.decode(random_token_ids, skip_special_tokens=True)  # Decode to text
    return random_text

def realtime_wait_for_task(ref_time, rel_release_time):
    abs_release_time = ref_time + rel_release_time
    cur_time = time.monotonic()
    wait_time = max(0, abs_release_time-cur_time)
    time.sleep(wait_time) 


def wait_for_no_runnning_requests(shared_context):
    ready = all(x == 0 for x in shared_context.nb_running_requests)
    while not ready:
        time.sleep(0.1)
        ready = all(x == 0 for x in shared_context.nb_running_requests)


def seq_submitter(shared_context, tasks, nb_adapters, adapters, tokenizer):
    q=shared_context.sub_exec_queue

    shared_context.init_nb_running_requests(nb_adapters)
    shared_context.init_current_in_tokens(nb_adapters)
    shared_context.init_current_out_tokens(nb_adapters)
   
    tasks.sort(key=lambda t: t.release_time)
    prompts = {}
    for task in tasks:
        prompts[task.id] = generate_random_prompt(task.input_tokens, tokenizer)
    
    
    start_time = time.monotonic()
    shared_context.start_time=start_time

    for task in tasks:
        prompt = prompts[task.id]                     
        payload_list=[{
            "idx": task.id,
            "lora_adapter": adapters[task.adapter] if nb_adapters > 0 else None,
            "lora_id": task.adapter,
            "prompt": prompt,
            "nb_output_tokens": task.output_tokens,
            "nb_input_tokens": task.input_tokens,
            "deadline": task.deadline
        }]
        wait_for_no_runnning_requests(shared_context)
        now = time.monotonic() - start_time 
        print(f"Submitting task {task.id} at time {now:.3f}/{task.release_time:.3f}")
   
        shared_context.increment_nb_running_requests(1, task.adapter)
        shared_context.increment_current_in_tokens(task.input_tokens, task.adapter)
        shared_context.increment_current_out_tokens(task.output_tokens, task.adapter)
   
        q.put(payload_list)

    q.put(None)  # Sentinel to signal the executor to exit
    q.join()    
    shared_context.start_time = -1
    print("Sequential submitter has finished")


# We model the prediction error on the number of generated tokens as log-normal.
#
# Rationale:
# - The number of tokens is strictly positive.
# - Empirically, prediction errors tend to be proportional to the scale
#   (i.e. relative errors are more stable than absolute errors).
# - This suggests a multiplicative noise model:
#
#       tokens_actual = tokens_predicted * exp(epsilon)
#
#   where epsilon ~ N(0, sigma).
#
# - Taking logs gives an additive model:
#
#       log(tokens_actual) = log(tokens_predicted) + epsilon
#
# - If the multiplicative effects influencing generation length
#   (prompt structure, sampling randomness, stopping behavior, etc.)
#   combine approximately independently, the central limit theorem
#   implies epsilon is approximately Gaussian.
#
# Therefore the token count follows a log-normal distribution around
# the prediction, which ensures positivity and models relative errors
# more realistically than an additive Gaussian model.

def noisy_estimate(true_value, rng, sigma=0.0):
    if sigma == 0.0:
        return true_value
    eps = rng.gauss(0, sigma)
    return max(1, int(true_value * math.exp(eps)))

# case for multiplicative
def noisy_estimate_mult(true_value, rng, sigma=0.0):
    if sigma == 0.0:
        return true_value
    eps = rng.gauss(0, sigma)
    return max(1, int(true_value * (1 + eps)))


def realtime_submitter(shared_context, tasks, nb_adapters, adapters, tokenizer, sigma):
    q=shared_context.sub_exec_queue

    rng = random.Random(42)

    shared_context.init_nb_running_requests(nb_adapters)
    shared_context.init_current_in_tokens(nb_adapters)
    shared_context.init_current_out_tokens(nb_adapters)
   
    tasks.sort(key=lambda t: t.release_time)
    prompts = {}
    real_nb_input_tokens = {}
    nb_tokens_estimated = {}
    for task in tasks:
        prompts[task.id] = generate_random_prompt(task.input_tokens, tokenizer)
        real_nb_input_tokens[task.id] = nb_tokens_in_prompts(prompts[task.id], tokenizer)
        nb_tokens_estimated[task.id] = noisy_estimate(task.output_tokens, rng, sigma)

    start_time = time.monotonic()
    shared_context.start_time=start_time
    global_shared_data.set_start_time(time.time())

    prev_release_time = None
    payload_list = []
    for task in tasks:
        prompt = prompts[task.id]        
        global_shared_data.set_num_output_tokens(int(task.id), nb_tokens_estimated[task.id]) 
        global_shared_data.set_num_input_tokens(int(task.id), real_nb_input_tokens[task.id])
        global_shared_data.set_deadline(int(task.id), task.deadline)
        shared_context.increment_nb_running_requests(1, task.adapter)
        shared_context.increment_current_in_tokens(task.input_tokens, task.adapter)
        shared_context.increment_current_out_tokens(task.output_tokens, task.adapter)
        if prev_release_time is None or task.release_time == prev_release_time:
            payload_list.append({
                "idx": task.id,
                "lora_adapter": adapters[task.adapter] if nb_adapters > 0 else None,
                "lora_id": task.adapter,
                "prompt": prompt,
                "nb_output_tokens": task.output_tokens,
                "nb_input_tokens": task.input_tokens,
                "nb_output_tokens_estimated":  nb_tokens_estimated[task.id],
                "deadline": task.deadline
            })
        else:
            realtime_wait_for_task(start_time, prev_release_time)
            now = time.monotonic() - start_time 
            print(f"Submitting task {task.id} at time {now:.3f}/{prev_release_time:.3f}")
            q.put(payload_list)
            payload_list = [{
                "idx": task.id,
                "lora_adapter": adapters[task.adapter] if nb_adapters > 0 else None,
                "lora_id": task.adapter,
                "prompt": prompt,
                "nb_output_tokens": task.output_tokens,
                "nb_input_tokens": task.input_tokens,
                "nb_output_tokens_estimated":  nb_tokens_estimated[task.id],
                "deadline": task.deadline
            }]
        prev_release_time = task.release_time

    # flush last batch
    if payload_list:
        realtime_wait_for_task(start_time, prev_release_time)
        task_id = payload_list[0]["idx"]
        now = time.monotonic() - start_time 
        print(f"Submitting task {task_id} at time {now:.3f}/{prev_release_time:.3f}")
        q.put(payload_list)

    q.put(None)  # Sentinel to signal the executor to exit
    q.join()    
    shared_context.start_time = -1
    print("Real-time submitter has finished")


def baseline_submitter(shared_context, tasks, nb_adapters, adapters, tokenizer):


    q=shared_context.sub_exec_queue

    shared_context.init_nb_running_requests(nb_adapters)
    shared_context.init_current_in_tokens(nb_adapters)
    shared_context.init_current_out_tokens(nb_adapters)
   
    tasks.sort(key=lambda t: t.release_time)
    prompts = {}
    for task in tasks:
        prompts[task.id] = generate_random_prompt(task.input_tokens, tokenizer)
    
    
    start_time = time.monotonic()
    shared_context.start_time=start_time

    for task in tasks:
        prompt = prompts[task.id]                     
        payload_list=[{
            "idx": task.id,
            "lora_adapter": adapters[task.adapter] if nb_adapters > 0 else None,
            "lora_id": task.adapter,
            "prompt": prompt,
            "nb_output_tokens": task.output_tokens,
            "nb_input_tokens": task.input_tokens,
            "deadline": task.deadline
        }]
        realtime_wait_for_task(start_time, task.release_time)
        now = time.monotonic() - start_time 
        print(f"Submitting task {task.id} at time {now:.3f}/{task.release_time:.3f}")
   
        shared_context.increment_nb_running_requests(1, task.adapter)
        shared_context.increment_current_in_tokens(task.input_tokens, task.adapter)
        shared_context.increment_current_out_tokens(task.output_tokens, task.adapter)
   
        q.put(payload_list)

    q.put(None)  # Sentinel to signal the executor to exit
    q.join()    
    shared_context.start_time = -1
    print("Baseline submitter has finsihed")


def executor(shared_context : SharedContext, engine, tokenizer, loop):
    recv_q = shared_context.sub_exec_queue
    send_q = shared_context.exec_coll_queue

    one_year_in_ms= 86400*365*1000

    while shared_context.start_time < 0 :
        pass

    
    start_time = shared_context.start_time
    while True:
        payload_list = recv_q.get()
        if payload_list is None:
            # Sentinel value to signal end of stream
            recv_q.task_done()
            break
        
        idx_list = []
        gens = []
        real_nb_tokens = []
        for payload in payload_list:
            idx = str(payload['idx'])
            idx_list.append(idx)
            adapter = payload['lora_adapter']
            nb_output_tokens = payload['nb_output_tokens']
            prompt = payload['prompt']
            real_nb_tokens.append(nb_tokens_in_prompts(prompt, tokenizer))
            priority = payload.get('priority', 0)
            if not (0 <= priority <= one_year_in_ms):
                raise ValueError(f"Priority {priority} is outside supported range 0-{one_year_in_ms}")
                
            sampling_params = SamplingParams(
                temperature=0.6,
                max_tokens=max(1,nb_output_tokens),
                min_tokens=max(1,nb_output_tokens),
                stop=["something_that_will_000_never_occur_097554"]
            )        
            
            gens.append(engine.generate(request_id = str(idx), 
                                  prompt = prompt, 
                                  sampling_params = sampling_params, 
                                  lora_request = adapter, 
                                  priority = priority,
                                )
            )

        for i, gen in enumerate(gens):
            payload = payload_list[i]
            idx = str(payload['idx'])
            adapter = payload['lora_adapter']
            prompt = payload['prompt']
            nb_output_tokens = payload['nb_output_tokens']

            gen_submit_time = time.monotonic() - start_time
            try: 
                future = asyncio.run_coroutine_threadsafe(serve_generation(gen, idx), loop)
                
            except Exception as e:
                print(f"Error submitting coroutine: {e}")
                import traceback
                traceback.print_exc()

            def record_completion_time(future):
                future.completion_time = time.monotonic()

            future.add_done_callback(record_completion_time)
            
            request={
                "idx": idx,
                "submit_time": gen_submit_time,
                "real_nb_input_tokens": real_nb_tokens[i],
                "nb_input_tokens": payload['nb_input_tokens'],
                "nb_output_tokens": nb_output_tokens,
                "nb_output_tokens_estimated": payload.get('nb_output_tokens_estimated', None),
                "deadline": payload['deadline'],
                "lora_adapter": adapter,
                "lora_id": payload['lora_id'],
                "future": future,
                "submit_time": gen_submit_time 
            }
            shared_context.add_request(request)
            send_q.put(request)
        # print(f"executor starts execution of task(s): {', '.join(idx_list)}")
        # print_stat(engine)
        recv_q.task_done()

    recv_q.join() 
    send_q.put(None)   
    send_q.join()    
    print("Executor has finsihed")

def get_token_from_output(obj):
    while isinstance(obj, list):
        obj = obj[0]

    # OpenAI / vLLM server API format
    if hasattr(obj, "usage") and obj.usage is not None:
        prompt_tokens = obj.usage.prompt_tokens
        completion_tokens = obj.usage.completion_tokens
        return prompt_tokens, completion_tokens

    # vLLM Python engine format
    if hasattr(obj, "prompt_token_ids"):
        prompt_tokens = len(obj.prompt_token_ids)
        completion_tokens = len(obj.outputs[0].token_ids)
        return prompt_tokens, completion_tokens

    raise ValueError("Unknown object format for token usage")
                     
def collector(shared_context, data, col_names, log_file_path):
    if log_file_path is not None:
        f = open(log_file_path,"a")
    else:
        f = sys.stdout

    def collect_future():
        result = future.result()
        idx = result['idx']
        output = result['output']

        request = requests[idx]
        # print(output)
        completed_at = getattr(future, 'completion_time', None)
        if completed_at:
            finish_time = completed_at - start_time
        else:
            finish_time = None
      
        prompt_tokens, completion_tokens = get_token_from_output(output)
        request['real_nb_output_tokens'] =  completion_tokens
        request['real_nb_input_tokens'] = prompt_tokens
        request['finish_time'] = finish_time
    
        duration = request['finish_time']-request['submit_time']
        adapter_id = request['lora_id']

        # duration_2 =  request["statistic_dict"]["first_schedule_time"] -start_time
        print(
        f"idx: {request['idx']}, adapter: {adapter_id}\tsubmit: {request['submit_time']:.3f}"
        f"\tfinish: {request['finish_time']:.3f}\tduration: {duration:.3f}"
        f"\tdeadline: {request['deadline']:.3f}\t"
        f"{request['real_nb_input_tokens']}/{request['nb_input_tokens']}:" 
        f"{request['real_nb_output_tokens']}/{request['nb_output_tokens']} ;"
        f"Running_request: {shared_context.nb_running_requests}"
        ,file=f)
        f.flush()

        
        shared_context.increment_nb_running_requests(-1, adapter_id ) 
        shared_context.increment_current_in_tokens(-request['nb_input_tokens'], adapter_id)
        shared_context.increment_current_out_tokens(-request['nb_output_tokens'], adapter_id)
       

    q=shared_context.exec_coll_queue
    requests = {}
    futures = []
    handled_futures = set() 

    while shared_context.start_time < 0 :
        pass
   
    start_time = shared_context.start_time
    while True:
        try:
            request = q.get(timeout=0.1)  # wait max 100ms for new request
        except queue.Empty:
            pass
        else: 
            if request is None:
                q.task_done()
                break;
            requests[request['idx']] = request
            now = time.monotonic() - start_time
            # print(f"[{now:.3f}] Collector received request: {request['idx']}")
            futures.append(request['future'])
            q.task_done()

        ready_futures = [fut for fut in futures if fut.done() and fut not in handled_futures]
        handled_futures.update(ready_futures)  # add all at once efficiently

        for future in ready_futures:
            collect_future()

    #  The submitter has finished sending future
    q.join() 

    print("Collector handle remaining futures...")   
    remaining_futures = [fut for fut in futures if fut not in handled_futures]
    for future in as_completed(remaining_futures):
        collect_future()
        
    end_time = time.monotonic()
    whole_duration = end_time - start_time
    sorted_requests = [requests[idx] for idx in sorted(requests, key=lambda x: int(x))]
    col_names[:]= ["idx", "adapter", "submit_time", "finish_time", "duration", "deadline",  "real_nb_input_tokens", "nb_input_tokens", "est_nb_output_tokens", "real_nb_output_tokens", "nb_output_tokens" ]
    for request in sorted_requests:
        duration = request['finish_time']-request['submit_time']
        val = [request['idx'], request['lora_id'], request['submit_time'], request['finish_time'], duration, request['deadline'],  request['real_nb_input_tokens'], request['nb_input_tokens'], request['nb_output_tokens_estimated'], request['real_nb_output_tokens'], request['nb_output_tokens']]
        line = val
        data.append(line)

    if log_file_path is not None:
        f.close()
    global_shared_data.set_finished()
    print("Collector has finsihed")
    print(f"Whole duration: {whole_duration}")
    
def monitor(nb_tasks, file_path, store_monitoring):
    if not store_monitoring:
        return
    round = -1

    nb_tokens = np.zeros(nb_tasks)
    with open(file_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["round", "prev_fp_duration", "prev_round_duration", "timestamp"] + [f"task_{i}" for i in range(nb_tasks)])
        while not global_shared_data.is_finished(): 
            new_round, prev_forward_pass_duration, prev_round_duration, now, computed_tokens = global_shared_data.fetch_updated_progress(nb_tasks)
            new_round = int(new_round)
            if new_round > round: 
                round = new_round
                # Tokens generated during this round
                delta_tokens = computed_tokens - nb_tokens
                writer.writerow([new_round, prev_forward_pass_duration, prev_round_duration, now] + delta_tokens.tolist())
                nb_tokens[:] = computed_tokens.astype(int)  # update in place
            else :
                time.sleep(0.005)
    print (f"Monitoring written to: {file_path}")


# ---- Async engine workflow ----
def submitter_executor_mt(tasks, method, batch_mode, nb_lora_adapters, adapters, engine, tokenizer, sigma, loop, monitoring_file_path, nb_requests, store_monitoring, log_file_path=None):

    # Setup
    shared_context = get_global_shared_context(msg="client: ")
    shared_context.sub_exec_queue = queue.Queue()
    shared_context.exec_coll_queue = queue.Queue()
    data = []
    col_names = []

    if "baseline" in method:
        submitter_thread = threading.Thread(target=baseline_submitter, args=(shared_context, tasks, nb_lora_adapters, adapters, tokenizer))
    elif "out-of-order" in method:
        submitter_thread = threading.Thread(target=realtime_submitter, args=(shared_context, tasks,  nb_lora_adapters, adapters, tokenizer, sigma))
    elif "sequentialize" in method: 
        submitter_thread = threading.Thread(target=seq_submitter, args=(shared_context, tasks,  nb_lora_adapters, adapters, tokenizer))
    else: 
        print(f"Unknow method {method}. Skipping...")
        return pd.DataFrame(data, columns=col_names)


    executor_thread = threading.Thread(target=executor, args=(shared_context, engine, tokenizer, loop))
    collector_thread = threading.Thread(target=collector, args=(shared_context, data, col_names, log_file_path))
    monitor_thread = threading.Thread(target=monitor, args=(nb_requests, monitoring_file_path, store_monitoring))
    monitor_thread.start()
    
    executor_thread.start()
    submitter_thread.start()
    collector_thread.start()
    
    submitter_thread.join()
    executor_thread.join()
    collector_thread.join()
    monitor_thread.join()

    df = pd.DataFrame(data, columns=col_names)
    print(df)
    return df

def set_logging_level(verbose_level):
    level_map = {
    0: logging.NOTSET,    # No output (you may want to use a custom handler to suppress all)
    1: logging.CRITICAL,
    2: logging.ERROR,
    3: logging.WARNING,
    4: logging.INFO,
    5: logging.DEBUG,
    }

    # Default to WARNING if out of range
    log_level = level_map.get(min(5,verbose_level), logging.WARNING)

    # Configure logging
    logging.basicConfig(
        level=log_level,
        format="LOG: %(message)s"
    )

    if verbose_level == 0:
       logging.disable()

def get_tracename(trace):
    trace = os.path.basename(trace)    # remove path
    trace = os.path.splitext(trace)[0] # remove extension

    if trace.startswith("BurstGPT"):
        after = trace[len("BurstGPT"):]
        # Case: BurstGPTXXX_*
        if after and "_" in after:
            suffix = after.split("_", 1)[0]  # part immediately after BurstGPT, before first _
            if len(suffix):
                return f"BurstGPT-{suffix}"
            else:
                return f"BurstGPT"
        
        # Case: BurstGPT_*
        elif after.startswith("_"):
            return "BurstGPT"
        # Case: BurstGPTXXX (no underscore)
        elif after:
            return f"BurstGPT-{after}"
        else:
            return "BurstGPT"
    elif "azure-trace" in trace:
        return "azure-trace"
    else:
        sys.exit(f"Do not know how to name {trace}")

def generate_csv_file_name(random_deadlines, method, trace, release_time_scaling, sla_factor, nb_requests, skip_lines, nb_adapters, percent_urgent, model_name, sigma, prefix=""):

    tracename = get_tracename(trace)
    
    if random_deadlines:
        ddl= "rnd-deadlines"
    else:
        ddl = "det-deadlines"

    if sigma == 0:
        sigma_str_ = ""
    else: 
        sigma_str_ = f"sigma={sigma}_"

    if prefix != "":
        filename = f"{method}_{tracename}_{release_time_scaling}_{sla_factor}_" \
            f"{nb_requests}_{skip_lines}_{nb_adapters}_{percent_urgent}_{ddl}_{model_name}_{sigma_str_}{prefix}.csv"
    else:
        filename = f"{method}_{tracename}_{release_time_scaling}_{sla_factor}_" \
            f"{nb_requests}_{skip_lines}_{nb_adapters}_{percent_urgent}_{ddl}_{model_name}.csv"

    
    output_dir = "./emulation_results_openai"
    # output_dir = "./emulation_results_with_percent"

    # output_dir = "./emulation_results"

    # Create directory if not exists
    os.makedirs(output_dir, exist_ok=True)

    # Full path to save file
    file_path = os.path.join(output_dir, filename)
 
    return file_path

def run(engine, adapters, tokenizer, loop, args, model_path):
    trace = args.trace
    nb_requests = int(args.nb_requests)
    nb_adapters = args.nb_adapters
    
    release_time_scaling = float(args.release_time_scaling)
    timeout = args.timeout
    sla_factor = args.sla_factor
    max_batch_size = args.max_batch_size
    skip_lines = args.skip_lines
    verbose = args.verbose
    display_schedule = args.display_schedule
    force_csv = args.force_csv
    method = args.method
    random_deadlines = args.random_deadlines
    store_monitoring = args.store_monitoring
    percent_urgent = args.percent_urgent
    model_name = args.model_name
    sigma = args.sigma
    hw = args.hw

    file_path = generate_csv_file_name(random_deadlines, method, trace, release_time_scaling, sla_factor, nb_requests, skip_lines, nb_adapters, percent_urgent, model_name, sigma=sigma, prefix=hw)
    monitoring_file_path = generate_csv_file_name(random_deadlines, method, trace, release_time_scaling, sla_factor, nb_requests, skip_lines, nb_adapters, percent_urgent, model_name, sigma=sigma, prefix=f"{hw}_monitoring")
    if "sequentialize" in method or store_monitoring:
        log_file_path = generate_csv_file_name(random_deadlines, method, trace, release_time_scaling, sla_factor, nb_requests, skip_lines, nb_adapters, percent_urgent, model_name, sigma=sigma, prefix=f"{hw}_log")
    else:
        log_file_path = None
    
    print(f"{method=} {trace=} {release_time_scaling=} {sla_factor=} " \
           f"{nb_requests=} {skip_lines=} {nb_adapters=} {percent_urgent=} {model_name=}")

    # Detect batch mode based on string suffix
    if method.endswith("_batch"):
        batch_mode = True
        method.replace("_batch", "")
    else:
        batch_mode = False
    

    if using_h100_flag and model_name == "mistral":
        print("Using iterpolateor for H100/Mistral")
        interpolator = TimeInterpolator(VLLM_BATCH_TPS, batch_mode)
        tasks, nb_lora_adapters  = read_tasks_from_csv_h100(trace, interpolator, release_time_scaling, sla_factor, max_rows = nb_requests, skip_lines=skip_lines, nb_lora_adapters=nb_adapters, percentage_of_urgent_task=percent_urgent, random_deadlines=random_deadlines)
    else: 
        print("Using XGBoosted model")
        model = RoundDurationModel.load_model(model_path)
        tasks, nb_lora_adapters  = read_tasks_from_csv(trace, model, release_time_scaling, sla_factor, max_rows = nb_requests, skip_lines=skip_lines, nb_lora_adapters=nb_adapters, percentage_of_urgent_task=percent_urgent, random_deadlines=random_deadlines)

    max_release_time = max(t.release_time for t in tasks)
    if args.store_monitoring:
        max_release_time_threshold = 2 * (len(tasks) - 1) # on average one task every 10 seconds
    else:
        max_release_time_threshold = 2 * (len(tasks) - 1) # on average one task every 2 seconds
    if max_release_time > max_release_time_threshold and max_release_time > 300: # above that, there is no pressure on the scheduler so skip it this case 
        df = pd.DataFrame()
        monitoring_file_path = None
    else:
        df = submitter_executor_mt(tasks, method, batch_mode, nb_lora_adapters, adapters, engine, tokenizer, sigma, loop, monitoring_file_path, nb_requests, store_monitoring, log_file_path=log_file_path)
        
    # Save DataFrame as CSV
    df.to_csv(file_path, index=False)
    print(f"DataFrame saved to {file_path}")
    
    return monitoring_file_path


def create_log_file_name(model_name, hw):
    script_dir = Path(__file__).resolve().parent
    
    log_dir = script_dir / "log"
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = log_dir / f"{model_name}_{hw}_{timestamp}.log"
    
    return log_file
    
def send_1_token_request_and_wait(engine, lora_request, loop):

    async def _run():
        prompt = "a"   # ~1 token

        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=1,
            min_tokens=1
        )
        
        gen = engine.generate(request_id = str(0), prompt=prompt, sampling_params=sampling_params, lora_request=lora_request)

        async for _ in gen:
            pass

    future = asyncio.run_coroutine_threadsafe(_run(), loop)
    future.result()


def sync_with_server(engine, adapters, nb_adapters, loop):
        print("Asking for restart")
        global_shared_data.set_start_time(0) #ensure it is not negative                
        global_shared_data.restart() # Ask vLLM server to restart its internal state from scratch
        # Send one request to be sure the server do at least on forward pass and has processed the restart command and is ready for new requests
        if nb_adapters > 0:
            lora_request = adapters[0]
        else: 
            lora_request = None

        send_1_token_request_and_wait(engine, lora_request, loop) 
        # The server should have done at least ine forward pass since the restart command, 
        # so it should have resterated during one of these forward pass and thus reset the restart flag.
        # If not, it means the server has not yet processed the restart command, which is an error..
        assert(not global_shared_data.ask_for_restart()) 
        # Reset start time to be set again by the submitter of the next experiment. The server is always blocked on negative start time
        # until the submitter set it to a positive value, so this ensure the server will not process any request until the submitter of the next experiment start it by setting a positive start time.
        global_shared_data.set_start_time(-1) 

def loop_xp(args, engine, tokenizer, adapters, loop, sched_policy, full_monitoring_csv):
    release_time_scaling_list = parse_arg_to_list(args.release_time_scaling, as_float=True)
    sla_factor_list           = parse_arg_to_list(args.sla_factor, as_float=True)
    skip_lines_list           = parse_arg_to_list(args.skip_lines)
    percent_urgent_list       = parse_arg_to_list(args.percent_urgent, handle_minus_1=True, as_float=True)
    sigma_list                = parse_arg_to_list(args.sigma, handle_minus_1=False, as_float=True)

    model_path, model_id = global_shared_data.get_perf_model_path_and_id(args.hw, args.model_name)
    global_shared_data.set_perf_model_id(model_id)

    for args.release_time_scaling in release_time_scaling_list:
        for args.sla_factor in sla_factor_list:
            for args.skip_lines in skip_lines_list:
                for args.percent_urgent in percent_urgent_list:
                    for args.sigma in sigma_list:
                        if args.store_monitoring: 
                            file_path = generate_csv_file_name(args.random_deadlines, args.method, args.trace, args.release_time_scaling, 
                                                            args.sla_factor, args.nb_requests, args.skip_lines, args.nb_adapters, args.percent_urgent, 
                                                            args.model_name, args.sigma, prefix=f"{hw}_monitoring")
                        else:
                            file_path = generate_csv_file_name(args.random_deadlines, args.method, args.trace, args.release_time_scaling, 
                                                            args.sla_factor, args.nb_requests, args.skip_lines, args.nb_adapters, args.percent_urgent, 
                                                            args.model_name, args.sigma, prefix=hw)
                                          
                        if not args.force_csv and os.path.exists(file_path):
                            print(f"{file_path}' already exists. Skipping.")
                            continue
                        
                        global_shared_data.reset() # reset the shared memory but number of procs, model_id and algorithm variant stay the same as they were already set.
                        
                        if sched_policy == "deadline":
                            sync_with_server(engine, adapters, args.nb_adapters, loop)
                
                        print(f"{args.release_time_scaling=} {args.sla_factor=} {args.skip_lines=} {args.percent_urgent=}")
                        monitoring_output=run(engine, adapters, tokenizer, loop, args, model_path)
                        
                        if args.store_monitoring and monitoring_output is not None: 
                            extend_output(full_monitoring_csv, monitoring_output)
                            if perf_model_has_converged(full_monitoring_csv):
                                return True
    return False


def create_event_loop():
    # Create new asyncio event loop (not yet running)
    loop = asyncio.new_event_loop()

    # Event to signal that loop is started
    loop_started_event = threading.Event()

    def run_loop():
        asyncio.set_event_loop(loop)      # Set loop for this thread
        loop_started_event.set()          # Signal loop is ready
        loop.run_forever()                # Run loop indefinitely

    # Start the loop in a background thread
    loop_thread = threading.Thread(target=run_loop, daemon=True)
    loop_thread.start()

    # Wait for loop to be ready before submitting requests
    loop_started_event.wait()
    return loop

def clean_shm():
    global global_shm
    # Clean up when finished
    if global_shm is not None: 
        global_shm.close()
        # unregister to avoind destryed
        from multiprocessing import resource_tracker
        resource_tracker.unregister(global_shm._name, "shared_memory")
        global_shm=None


def create_global_shared_memory():
    # set_logging_level(args.verbose)
    global global_shm, global_shared_data
    global_shm = None
    retry_count = 0
    while global_shm is None and retry_count < 5:
        global_shm, global_shared_data, new_shared_data = create_global_shm_shared_data(int(args.nb_requests), shm_name=DEFAULT_SHM_NAME )
        retry_count += 1

    if global_shm is None:
        sys.exit("Failed to create shared memory after multiple attempts. Exiting.")
        
    return new_shared_data
        
def parse_arg_to_list(arg, handle_minus_1=False, as_float=False):
    """
    Convert an argument to a list of numbers.

    - Comma-separated: "1,2,3" → [1,2,3]
    - Range with colon: "0:10" → [0,1,...,10], "0:100:10" → [0,10,...,100]
    - Single number: "5" → [5]
    - Special: "-1" with handle_minus_1=True → [0,10,20,...,100]
    """
    arg = str(arg).strip()

    if handle_minus_1 and arg == "-1":
        vals = range(0, 101, 10)
        return [float(v) if as_float else int(v) for v in vals]

    # Range with colon
    if ":" in arg:
        parts = arg.split(":")
        if len(parts) == 2:
            start, end = map(float if as_float else int, parts)
            step = 1
        elif len(parts) == 3:
            start, end, step = map(float if as_float else int, parts)
        else:
            raise ValueError(f"Invalid range format: {arg}")

        values = []
        current = start
        if step == 0:
            raise ValueError("Step cannot be zero")
        if step > 0:
            while current <= end:
                values.append(current)
                current += step
        else:
            while current >= end:
                values.append(current)
                current += step
        return values

    # Comma-separated
    if "," in arg:
        return [float(v.strip()) if as_float else int(v.strip()) for v in arg.split(",")]

    # Single value
    return [float(arg) if as_float else int(arg)]


# ---- Entrypoint ---
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Schedule LoRA inference tasks.")
    parser.add_argument("trace", help="CSV file with task list or string nb_tasks:inter_arrival_lambda:seed:nb_input_tokens:nb_output_tokens:nb_lora_adapaters")
    parser.add_argument("-n", "--nb-requests", type=str, default = "10", help="Number of request to consider (default: -1 for all)")
    parser.add_argument("-a", "--nb-adapters", type=int, default = 0, help="Number of adaptors (default: 1)")
    parser.add_argument("-r", "--release-time-scaling", type=str, default=1,    help="Scaling factor applied to release times (default: 1.0)")
    parser.add_argument("-f", "--sla-factor", type=str, default="5",     help="Multiplier applied to the sequential runtime to compute the task deadline (default: 5.0)")
    parser.add_argument("-t", "--timeout", type=float, default=10,     help="slice len or tiumeout value (used for slice/timeout strategy). Default: 10")
    parser.add_argument("-b", "--max-batch-size", type=float, default=1024,     help="Maximum batch size - any larger batches will be commited forcibly (default: 1024)")
    parser.add_argument("-l", "--skip-lines", type=str, default="0",     help="Number of lines to skip in the traces, to somulate on a different portions (default: 0)")
    parser.add_argument("-p", "--percent-urgent", type=str, default="100.0", help="Percentage of urgent tasks. Non urgent task have their deadline set to +1 day (86400s). Must be in [0, 100] or -1 to do 0-100 by step of 10. (default: 100.0)")
    parser.add_argument("-v", "--verbose", type=int, default=0, help="Verbosity level from 0 to 5")
    parser.add_argument("-d", "--display_schedule",  action='store_true', help="Flag to display the schedule")
    parser.add_argument("-g", "--nb-gpus", type=int, default = NUM_GPUS, help=f"Number of GPUs to use when bootstrapping the model (default: {NUM_GPUS})")
    parser.add_argument("-M", "--model-name", type=str, default = "mistral", help=f"Model name use to bootstrapping the model (default: 'mistral'" , choices=["mistral", "llama", "qwen"])
    parser.add_argument("-s", "--sigma", type=str, default="0", help="Standard deviation of the Gaussian noise used to perturb the output token estimate (default: 0, i.e., perfect oracle). We simulate prediction errors by perturbing the true output length with multiplicative Gaussian noise: pred = perfect*(1+epsilon), with epsilon drawn from N(0,sigma^2).")
    parser.add_argument("--force-csv", action='store_true', help="force recompute the csv file")
    parser.add_argument("--store-monitoring",action='store_true', help="Enable monitoring data storage. Works only with the out-of-order scheduler and when the monitoring code is enabled inside scheduler.py.")
    parser.add_argument("--random-deadlines", action='store_true', help="random or deterministic deadlines")
    # parser.add_argument("-m","--method", choices=["dyn_prog", "dyn_prog_org", "greedy", "greedy_batch", "baseline", "slice", "slice_batch", "idle_timeout"], required=True, help="Which scheduling method to run")
    parser.add_argument("-m","--method", choices=["baseline", "out-of-order-discard-most-urgent", "out-of-order-edf", "sequentialize"], required=True, help="Which scheduling method to run")
    args = parser.parse_args()

    new_shared_data = create_global_shared_memory()
    
    global using_h100_flag
    atexit.register(clean_shm) # close do not unlink at exit of the process so we can connect to it again  


    hw = f"{torch.cuda.get_device_name(0).lower().replace(' ', '-')}_{args.nb_gpus}"

    using_h100_flag = hw == "nvidia-h100-nvl_1"
    # using_h100_flag = False


    if using_h100_flag:
        if not (args.model_name == "qwen"
                or (args.model_name == "mistral" and args.nb_adapters >= 1)):
            sys.exit(
                "Invalid configuration for H100: "
                "when using the H100 GPU, either use model 'qwen', "
                "or use model 'mistral' with at least one LoRA adapter (nb_adapters >= 1). "
                "Otherwise use A100 GPU(s)."
            )

    args.hw = hw
    full_monitoring_csv = f"./performance_model/monitoring-monotonic_{hw}_{args.model_name}.csv"
    if args.store_monitoring: 
        sched_policy = "deadline"
        global_shared_data.set_algorithm_variant(Algorithm.MONITORING)
    
    elif "out-of-order" in args.method:
        sched_policy = "deadline"
        if "discard-most-urgent" in args.method:
            global_shared_data.set_algorithm_variant(Algorithm.OUT_OF_ORDER_DISCARD_MOST_URGENT)
        elif "edf" in args.method:
            global_shared_data.set_algorithm_variant(Algorithm.EDF)
        else:
            sys.exit(f"Do not know what to do with {args.method=}")
    else:
        sched_policy = "fcfs"

    loop = create_event_loop()
    log_file = create_log_file_name(args.model_name, hw)
    print(f"{new_shared_data=}")
    if using_h100_flag and args.model_name == "mistral":
        engine, tokenizer, adapters = mistral_start(loop, args.nb_adapters, sched_policy=sched_policy, low_mem=("low-mem" in args.method)) # "out-of-order",
    else :
        engine, tokenizer, adapters = model_server_start(args.model_name, sched_policy=sched_policy, nb_gpus=args.nb_gpus,  
                                                     nb_adapaters=args.nb_adapters, log_file=log_file, new_shared_data=new_shared_data, using_h100=using_h100_flag)
    

    converged = loop_xp(args, engine, tokenizer, adapters, loop, sched_policy, full_monitoring_csv)
      

    clean_shm()

    # Shutting down engine:
    # loop.call_soon_threadsafe(engine.shutdown)
    # future = asyncio.run_coroutine_threadsafe(engine.shutdown(), loop)
    # future.result()  # Optional: Wait until shutdown completes

    if dist.is_initialized():
        dist.destroy_process_group()

    # Finally, stop the loop safely
    loop.call_soon_threadsafe(loop.stop)
    
    if args.store_monitoring and converged:
        print ("MODEL HAS CONVERGED")
        sys.exit(2)
    
    sys.exit(0)

  