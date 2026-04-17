import vllm
print(f"vLLM version: {vllm.__version__}")
print(f"from: {vllm.__file__}")

from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.lora.request import LoRARequest
import httpx
import asyncio
import requests
import subprocess
import time
import os
import sys
import signal

import torch.distributed as dist
from transformers import AutoTokenizer
import logging
from openai import AsyncOpenAI

logging.getLogger("vllm").setLevel(logging.WARNING)
# ---- CONFIG ----
MISTRAL_MODEL_NAME= "./models/Mistral-7B-Instruct-v0.2"
QWEN_MODEL_NAME= "Qwen/Qwen3-30B-A3B-Instruct-2507"
LLAMA_MODEL_NAME= "./models/nvidia_Llama-3.1-70B-Instruct-FP8"

NUM_GPUS = 1

MISTRAL_ADAPTER_PATHS = {
    0: "./adapters/mistral/mistral-lora-finetuned_Rank_64_v1",
    1: "./adapters/mistral/mistral-lora-finetuned_Rank_64_v2",
    2: "./adapters/mistral/mistral-lora-finetuned_Rank_64_v3",
    3: "./adapters/mistral/mistral-lora-finetuned_Rank_64_v4",
    4: "./adapters/mistral/mistral-lora-finetuned_Rank_64_v5",
    5: "./adapters/mistral/mistral-lora-finetuned_Rank_64_v6",
    6: "./adapters/mistral/mistral-lora-finetuned_Rank_64_v7",
    7: "./adapters/mistral/mistral-lora-finetuned_Rank_64_v8",
    8: "./adapters/mistral/mistral-lora-finetuned_Rank_64_v9",
    9: "./adapters/mistral/mistral-lora-finetuned_Rank_64_v10",
    10: "./adapters/mistral/mistral-lora-finetuned_Rank_64_v11",
    11: "./adapters/mistral/mistral-lora-finetuned_Rank_64_v12",
    12: "./adapters/mistral/mistral-lora-finetuned_Rank_64_v13",
    13: "./adapters/mistral/mistral-lora-finetuned_Rank_64_v14",
    14: "./adapters/mistral/mistral-lora-finetuned_Rank_64_v15",
    15: "./adapters/mistral/mistral-lora-finetuned_Rank_64_v16",
    16: "./adapters/mistral/mistral-lora-finetuned_Rank_64_v17",
    17: "./adapters/mistral/mistral-lora-finetuned_Rank_64_v18",
    18: "./adapters/mistral/mistral-lora-finetuned_Rank_64_v19",
    19: "./adapters/mistral/mistral-lora-finetuned_Rank_64_v20"
}


LLAMA_ADAPTER_PATHS = {
    0: "./adapters/Llama/MetaMathQA_Meta-Llama-3.1-70B-BNB-NF4_LORA_ADAPTER_96rank_v0",
    1: "./adapters/Llama/MetaMathQA_Meta-Llama-3.1-70B-BNB-NF4_LORA_ADAPTER_96rank_v1",
    2: "./adapters/Llama/MetaMathQA_Meta-Llama-3.1-70B-BNB-NF4_LORA_ADAPTER_96rank_v2",
    3: "./adapters/Llama/MetaMathQA_Meta-Llama-3.1-70B-BNB-NF4_LORA_ADAPTER_96rank_v3",
}


def create_lora_adapters_loop(engine, adapter_paths, loop):
    adapters = {}
    for adapter_id, path in adapter_paths.items():
        # print(f"Adding LoRA adapter {adapter_id}")
        adapter_name = path.split("/")[-1]
        lora_req = LoRARequest(adapter_name, adapter_id + 1, path)
        future = asyncio.run_coroutine_threadsafe(engine.add_lora(lora_req), loop)
        future.result()  # wait for completion or handle asynchronously
        adapters[adapter_id] = lora_req
    return adapters


def create_lora_adapters(engine, adapter_paths):
    adapters = {}
    for adapter_id, path in adapter_paths.items():
        print(f"Adding LoRA adpater {adapter_id}")
        adapter_name = path.split("/")[-1]
        lora_req = LoRARequest(adapter_name, adapter_id+1, path)
        engine.add_lora(lora_req)  # load into engine
        adapters[adapter_id] = lora_req
    return adapters



def create_lora_requests_for_llm():
    requests = {}
    for aid, path in ADAPTER_PATHS.items():
        name = path.split("/")[-1]
        requests[aid] = LoRARequest(name, aid, path)
    return requests

class RemoteEngine:
    def __init__(self, model, lora_id_to_name=[], base_url="http://localhost:8000/v1", nb_connections = 4000):
        limits = httpx.Limits(
            max_connections=nb_connections,
            max_keepalive_connections=nb_connections
        )

        transport = httpx.AsyncHTTPTransport(retries=3)
        
        timeout = httpx.Timeout(
            connect=10.0,
            read=3600.0,
            write=600.0,
            pool=3600.0
        )

        http_client = httpx.AsyncClient(
            limits=limits,
            transport=transport,
            timeout=timeout
        )

        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key="EMPTY",
            http_client=http_client
        )
        

        self.model = model
        self.lora_id_to_name = lora_id_to_name
        self.use_lora = True if len(self.lora_id_to_name) > 0 else False
    
    def generate(self,
             request_id,
             prompt,
             sampling_params,
             lora_request, # None or an integer bewteen 0 and len(self.lora_id_to_name_) -1
             priority=0):

        async def fake_generator():

            extra_body = {
                "priority": priority,
                "min_tokens": sampling_params.min_tokens,
                "ignore_eos": True
            }

             # Add LoRA request only if Not None is valid
            if lora_request is not None and self.use_lora:
                extra_body["lora_request"] = lora_request_name
               
            resp = await self.client.completions.create(
                model=self.model,
                prompt=prompt,
                max_tokens=sampling_params.max_tokens,
                temperature=sampling_params.temperature,
                stop=sampling_params.stop,
                extra_body=extra_body,
                extra_headers={"x-request-id": str(request_id)}
            )

            yield [resp]

        return fake_generator()

def count_vllm_workers():
    """
    Run nvidia-smi and count the number of VLLM workers.
    Each worker (VLLM::Worker_TP*) corresponds to one GPU used by vLLM.
    """

    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            check=True
        )

        count = 0
        for line in result.stdout.splitlines():
            if "VLLM::Worker_TP" in line:
                count += 1

        return count

    except Exception:
        return 0
    
def count_vllm_engine():
    """
    Run nvidia-smi and count the number of VLLM workers.
    Each worker (VLLM::EngineCore) corresponds to one GPU used by vLLM.
    """

    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            check=True
        )

        count = 0
        for line in result.stdout.splitlines():
            if "VLLM::EngineCore" in line:
                count += 1

        return count

    except Exception:
        return 0

def select_free_gpus(nb_gpus: int, max_used_mem_gb: float = 1.0):
    """
    Selects the `nb_gpus` least loaded GPUs based on free memory,
    keeping only GPUs with at mots `max_used_mem_gb` GB free.

    Returns a list of GPU indices (integers). Returns an empty list if none qualify.

    Args:
        nb_gpus (int): Number of GPUs you want to select.
        min_free_mem_gb (float): Minimum free memory required per GPU in GB.
    """
    try:
        # Query nvidia-smi for free memory and GPU index
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,index", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True
        )

        gpu_infos = []
        # Parse each line: "<used_memory>, <gpu_index>"
        for line in result.stdout.strip().split("\n"):
            mem_used_str, idx_str = line.split(",")
            mem_used = int(mem_used_str.strip()) / 1024  # convert MiB -> GiB
            idx = int(idx_str.strip())
            print(f"{idx=} {mem_used=} {max_used_mem_gb=}")
            # Keep GPU only if it does not used too much  memory
            if mem_used <= max_used_mem_gb:
                gpu_infos.append((mem_used, idx))

        # Sort GPUs by less used memory furst
        gpu_infos.sort(reverse=False, key=lambda x: x[0])

        if len(gpu_infos) < nb_gpus:
            print("Error selecting GPUs")
            return None

        # Take the top nb_gpus GPU indices
        selected = [idx for _, idx in gpu_infos[:nb_gpus]]
        return selected

    except Exception as e:
        print("Error selecting GPUs:", e)
        return None

def stop_server_on_port(port=8000, wait_time=2):
    try:
        result = subprocess.run(
            ["lsof", "-ti", f"tcp:{port}"],
            capture_output=True,
            text=True,
            check=False
        )

        pid = result.stdout.strip()

        if not pid:
            print(f"No process listening on port {port}")
            return

        pid = int(pid)

        print(f"Stopping server (PID={pid})...")
        os.kill(pid, signal.SIGTERM)

        time.sleep(wait_time)

        try:
            os.kill(pid, 0)  # check if process still exists
            print("Process still alive, forcing kill...")
            os.kill(pid, signal.SIGKILL)
        except OSError:
            print("Server stopped cleanly.")

    except Exception as e:
        print(f"Error while stopping server: {e}")

def model_server_start(model_name, sched_policy="fcfs", nb_gpus = 1, nb_adapaters=0, log_file="vllm.log", new_shared_data=True, using_h100=False):
    global MODEL_NAME
    if model_name == "mistral":
        MODEL_NAME = MISTRAL_MODEL_NAME
    elif model_name == "llama":
        MODEL_NAME = LLAMA_MODEL_NAME
        if nb_adapaters > 0 :
            sys.exit("Error: adapters not yet supported for this 'llama' model")
    elif model_name == "qwen":
        MODEL_NAME = QWEN_MODEL_NAME
        if nb_adapaters > 0 :
            sys.exit("Error: adapters not yet supported for this 'qwen' model")
    else:
        sys.exit(f"Unknown model name {model_name}")
    return vLLM_server_start(sched_policy, nb_gpus, nb_adapaters, log_file, new_shared_data, using_h100=using_h100)

    
def vLLM_server_start(sched_policy="fcfs", nb_gpus = 1, nb_adapaters=0, log_file="vllm.log", new_shared_data=True, using_h100=False):

    print(f"Starting vLLM server ({MODEL_NAME}")
    lora_id_to_name = [f"adapter{i}" for i in range(nb_adapaters)]

    # tester if server has already started
    error = False
    r= None
    try:
        r = requests.get("http://localhost:8000/health", timeout=1)
        if r.status_code == 200:
            print("vLLM server already running")
        if nb_gpus == 1:
            vllm_workers = count_vllm_engine()
        else:
            vllm_workers = count_vllm_workers()
        if vllm_workers == nb_gpus:
            print(f"Number of GPU is correct")
            if new_shared_data:
                print("Error: we created a new shared data so it not shared with the server. Restart the server")
                error = True
        else: 
            print(f"Error: mismatch in GPU count: expected {nb_gpus}, found {vllm_workers} vLLM workers")
            print("Restarting server...")
            stop_server_on_port(8000)
            raise Exception("Server needs to be restarted")
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to vLLM server: {e}")
        print("Launching new vLLM server... - log file:", str(log_file))
        free_gpus = select_free_gpus(nb_gpus)

        if free_gpus is None: 
            sys.exit("Error: not enough free GPUS available")

        print("Selected GPUs:", free_gpus)

        # Set environment variable so vLLM sees only the chosen GPUs
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, free_gpus))
        print("CUDA_VISIBLE_DEVICES set to:", os.environ["CUDA_VISIBLE_DEVICES"])

        
        log = open(log_file, "w")

        cmd = [
            "python",
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model", MODEL_NAME,
            "--tensor-parallel-size", str(nb_gpus),
            "--scheduling-policy", sched_policy
        ]

        if using_h100 and MODEL_NAME == QWEN_MODEL_NAME:
            cmd.extend(["--max-model-len", "200000"])

        if nb_adapaters > 0:
            lora_modules = [
                f"{lora_id_to_name[i]}={MISTRAL_ADAPTER_PATHS[i]}"
                for i in range(nb_adapaters)
            ]
            
            cmd += [
                "--enable-lora",
                "--max-lora-rank", "64",
                "--lora-modules",
                *lora_modules
            ]

        subprocess.Popen(
            cmd,
            stdout=log,
            stderr=log,
        )

        # Wait for the server to start
        started = False
        if MODEL_NAME == QWEN_MODEL_NAME:
            timeout = 90
        elif MODEL_NAME == MISTRAL_MODEL_NAME:
            timeout = 30
        else:
            timeout = 60

        for _ in range(timeout):
            try:
                r = requests.get("http://localhost:8000/health", timeout=1)
                if r.status_code == 200:
                    print("vLLM server ready")
                    started = True
                    break
            except:
                pass
            time.sleep(1)

        if not started:
            sys.exit(f"Error: vLLM server for {MODEL_NAME} did not start within {timeout} seconds. Check the log file for details.")
        
        print("vLLM server has started!")
    
    if error:
        sys.exit("Exiting...")

    engine = RemoteEngine(MODEL_NAME, lora_id_to_name=lora_id_to_name)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # adapters = create_lora_adapters_loop(engine, MISTRAL_ADAPTER_PATHS, loop)

    return engine, tokenizer, lora_id_to_name


def mistral_start(loop, nb_adapters, sched_policy='fcfs', low_mem = False):
    print(f"Starting mistral engine ({MISTRAL_MODEL_NAME})...")
    engine_args = AsyncEngineArgs(
        model=MISTRAL_MODEL_NAME,
        enable_lora=True,
        # enable_chunked_prefill=True,
        max_lora_rank=64,
        tensor_parallel_size=NUM_GPUS,
        max_loras = nb_adapters, 
    )

    if low_mem:
        # engine_args.gpu_memory_utilization =  0.22625 # one adapter
        major_minor = float('.'.join(vllm.__version__.split('.')[:2]))
        if major_minor < 0.10:
            engine_args.gpu_memory_utilization =  0.22665 #20 adapaters on H100
        else:
            engine_args.gpu_memory_utilization =  0.2 #20 adapaters on H100
        # engine_args.max_model_len = 4096
    else:
        engine_args.gpu_memory_utilization =  0.9 # default
    
    print(f"{engine_args.gpu_memory_utilization=}")

    engine_args.scheduling_policy = sched_policy
    
    print(f"{engine_args.scheduling_policy=}")

    tokenizer = AutoTokenizer.from_pretrained(MISTRAL_MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    engine = AsyncLLMEngine.from_engine_args(engine_args)

    if sched_policy == "prority":
        engine.do_log_stats = lambda *a, **kw: None


    future = asyncio.run_coroutine_threadsafe(engine.get_model_config(), loop)
    model_config = future.result()  # wait for completion or handle asynchronously
    print(f"{model_config.max_model_len=}")


    # import pprint
    # pprint.pprint(model_config.__dict__)
    adapters = create_lora_adapters_loop(engine, MISTRAL_ADAPTER_PATHS, loop)
    return engine, tokenizer, adapters

def llama_start(nb_adapters):
    engine_args = AsyncEngineArgs(
        model=LLAMA_MODEL_NAME,
        enable_lora=True,
        # enable_chunked_prefill=True,
        max_lora_rank=128,
        tensor_parallel_size=NUM_GPUS,
        max_loras = nb_adapters, 
        max_model_len=8192,           # Lower context if possible
        # quantization="fp8",
        # dtype="fp8",  # Specify dtype here inside engine args
    )

    tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    engine = AsyncLLMEngine.from_engine_args(engine_args)
    adapters = create_lora_adapters(engine, LLAMA_ADAPTER_PATHS)
    return engine, tokenizer, adapters


async def cleanup(engine):
    await engine.shutdown()
    if dist.is_initialized():
        dist.destroy_process_group()
