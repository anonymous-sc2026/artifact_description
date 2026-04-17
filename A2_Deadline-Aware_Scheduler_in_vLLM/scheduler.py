# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import itertools
import time
from collections import defaultdict
from collections.abc import Iterable
from typing import Any, Optional, Union


from vllm.config import VllmConfig
from vllm.distributed.kv_events import EventPublisherFactory, KVEventBatch
from vllm.distributed.kv_transfer.kv_connector.factory import (
    KVConnectorFactory)
from vllm.distributed.kv_transfer.kv_connector.v1 import (KVConnectorBase_V1,
                                                          KVConnectorRole)
from vllm.logger import init_logger
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.v1.core.encoder_cache_manager import (EncoderCacheManager,
                                                compute_encoder_budget)
from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
from vllm.v1.core.sched.interface import SchedulerInterface
from vllm.v1.core.sched.output import (CachedRequestData, NewRequestData,
                                       SchedulerOutput)
from vllm.v1.core.sched.request_queue import (SchedulingPolicy, FCFSRequestQueue,
                                              create_request_queue, )
from vllm.v1.core.sched.utils import check_stop, remove_all
from vllm.v1.engine import (EngineCoreEventType, EngineCoreOutput,
                            EngineCoreOutputs)
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.metrics.stats import SchedulerStats
from vllm.v1.outputs import DraftTokenIds, KVConnectorOutput, ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.spec_decode.metrics import SpecDecodingStats
from vllm.v1.structured_output import StructuredOutputManager

logger = init_logger(__name__)

### BEGIN MOD ####
import sys
import numpy as np
import pandas as pd
from itertools import chain
import math
# Absolute path to your shared_context.py location
shared_context_path = "<your abs. path here>"

if shared_context_path not in sys.path:
    sys.path.insert(0, shared_context_path)

from shared_context import create_shared_data_from_shm, SharedData, Algorithm
from performance_model.round_duration import RoundDurationModel
from multiprocessing import shared_memory
from dataclasses import dataclass

@dataclass
class Metrics:
    remaining_time: np.ndarray
    remaining_decode_tokens: np.ndarray
    remaining_prefill_tokens: np.ndarray
    decode_tps_requirement: np.ndarray
    max_concurency_array: np.ndarray
    is_critical: np.ndarray    
    is_deprioritized: np.ndarray
    has_priority: np.ndarray
    must_be_scheduled: np.ndarray

    critical_id: int 
    critical_max_concurency: int | np.integer
    critical_remaining_decode_tokens: int | np.integer 

    sorted_request_ids: np.ndarray
    sorted_max_concurency_array: np.ndarray
    deschedulable_ids_ordered: np.ndarray
    max_requests: int
    sort_ids: np.ndarray
    nb_priority_requests: int
    in_prefill: np.ndarray
    prefill_ids_set: set

    forced: np.ndarray
### END MOD ####

class Scheduler(SchedulerInterface):

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        structured_output_manager: StructuredOutputManager,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        include_finished_set: bool = False,
        log_stats: bool = False,
    ) -> None:

        ### BEGIN MOD ####
        self.shm = None
        self.shared_data=None
        self.model_id = 0 #No model by default, will be set at the first request
        self.start_time = -1
        self.round = None
        self.last_algo_variant = None
        self.init_time = time.perf_counter()
        ### END MOD ####
        
        self.vllm_config = vllm_config
        self.scheduler_config = vllm_config.scheduler_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.kv_cache_config = kv_cache_config
        self.kv_events_config = vllm_config.kv_events_config
        self.parallel_config = vllm_config.parallel_config
        self.log_stats = log_stats
        self.structured_output_manager = structured_output_manager
        self.is_encoder_decoder = vllm_config.model_config.is_encoder_decoder
        # include_finished_set controls whether a separate set of finished
        # request ids should be included in the EngineCoreOutputs returned
        # by update_from_outputs(). This is currently used in the multi-engine
        # case to track request lifetimes efficiently.
        self.finished_req_ids_dict: Optional[dict[int, set[str]]] = (
            defaultdict(set) if include_finished_set else None)

        # Scheduling constraints.
        self.max_num_running_reqs = self.scheduler_config.max_num_seqs
        self.max_num_scheduled_tokens = \
            self.scheduler_config.max_num_batched_tokens
        self.max_model_len = self.scheduler_config.max_model_len
        self.enable_kv_cache_events = (
            self.kv_events_config is not None
            and self.kv_events_config.enable_kv_cache_events)

        # Create KVConnector for the Scheduler. Note that each Worker
        # will have a corresponding KVConnector with Role=WORKER.
        # KV Connector pushes/pull of remote KVs for P/D and offloading.
        self.connector = None
        if self.vllm_config.kv_transfer_config is not None:
            assert len(self.kv_cache_config.kv_cache_groups) == 1, (
                "Multiple KV cache groups are not currently supported "
                "with KV connectors")
            assert not self.is_encoder_decoder, (
                "Encoder-decoder models are not currently supported "
                "with KV connectors")
            self.connector = KVConnectorFactory.create_connector(
                config=self.vllm_config, role=KVConnectorRole.SCHEDULER)

        self.kv_event_publisher = EventPublisherFactory.create(
            self.kv_events_config,
            self.parallel_config.data_parallel_rank,
        )

        num_gpu_blocks = self.cache_config.num_gpu_blocks
        assert num_gpu_blocks is not None and num_gpu_blocks > 0

        self.block_size = self.cache_config.block_size

        self.dcp_world_size = \
            vllm_config.parallel_config.decode_context_parallel_size
        # Note(hc): The scheduler’s block_size must be multiplied
        # by dcp_world_size, since block hashes are computed on the
        # original full token sequence at a granularity of
        # original_block_size × dcp_world_size.
        if self.dcp_world_size > 1:
            self.block_size *= self.dcp_world_size

        # req_id -> Request
        self.requests: dict[str, Request] = {}
        # Scheduling policy
        if self.scheduler_config.policy == "priority":
            self.policy = SchedulingPolicy.PRIORITY
        elif self.scheduler_config.policy == "fcfs":
            self.policy = SchedulingPolicy.FCFS
        elif self.scheduler_config.policy == "deadline":
            self.policy = SchedulingPolicy.DEADLINE
        else:
            raise ValueError(
                f"Unknown scheduling policy: {self.scheduler_config.policy}")
        # Priority queues for requests.
        self.waiting = create_request_queue(self.policy)
        self.running: list[Request] = []

        # The request IDs that are finished in between the previous and the
        # current steps. This is used to notify the workers about the finished
        # requests so that they can free the cached states for those requests.
        # This is flushed at the end of each scheduling step.
        self.finished_req_ids: set[str] = set()

        # KV Connector: requests in process of async KV loading or recving
        self.finished_recving_kv_req_ids: set[str] = set()

        # Encoder-related.
        # Calculate encoder cache size if applicable
        # NOTE: For now we use the same budget for both compute and space.
        # This can be changed when we make encoder cache for embedding caching
        # across requests.
        encoder_compute_budget, encoder_cache_size = compute_encoder_budget(
            model_config=vllm_config.model_config,
            scheduler_config=vllm_config.scheduler_config,
            mm_registry=mm_registry,
        )

        # NOTE(woosuk): Here, "encoder" includes the vision encoder (and
        # projector if needed) for MM models as well as encoder-decoder
        # transformers.
        self.max_num_encoder_input_tokens = encoder_compute_budget
        # NOTE: For the models without encoder (e.g., text-only models),
        # the encoder cache will not be initialized because cache size is 0
        # for these models.
        self.encoder_cache_manager = EncoderCacheManager(
            cache_size=encoder_cache_size)

        speculative_config = vllm_config.speculative_config
        self.use_eagle = False
        self.num_spec_tokens = self.num_lookahead_tokens = 0
        if speculative_config:
            self.num_spec_tokens = speculative_config.num_speculative_tokens
            if speculative_config.use_eagle():
                self.use_eagle = True
                self.num_lookahead_tokens = self.num_spec_tokens

        # Create the KV cache manager.
        self.kv_cache_manager = KVCacheManager(
            kv_cache_config=kv_cache_config,
            max_model_len=self.max_model_len,
            enable_caching=self.cache_config.enable_prefix_caching,
            use_eagle=self.use_eagle,
            log_stats=self.log_stats,
            enable_kv_cache_events=self.enable_kv_cache_events,
            dcp_world_size=self.dcp_world_size,
        )
        self.use_pp = self.parallel_config.pipeline_parallel_size > 1
    ### BEGIN MOD #### 
    def __del__(self):
        try:
            if self.shm:
                self.shm.unlink() # destroy this segment.  
        except Exception:
            # Destructor should not raise exceptions
            pass
    ### END MOD ####
    # === Inputs ===
    # These arrays should already exist in your scheduler context:
    # self.num_output_tokens[request_ids]
    # self.num_input_tokens[request_ids]
    # computed_tokens
    # self.scheduler_config.long_prefill_token_threshold

    ### BEGIN MOD ####
    def compute_rounds_left(self, request_ids, computed_tokens):
        # Extract the per-request info
        num_output_tokens = self.num_output_tokens[request_ids]
        num_input_tokens  = self.num_input_tokens[request_ids]
        threshold         = self.scheduler_config.long_prefill_token_threshold

        # ---- Remaining tokens ----
        # Prefill: tokens not yet processed from the input
        remaining_prefill = np.maximum(num_input_tokens - computed_tokens, 0)
        # Decode: tokens still to be generated
        remaining_decode  = np.maximum(num_output_tokens - np.maximum(computed_tokens - num_input_tokens, 0), 0)

        # ---- Prefill rounds ----
        # If threshold > 0 → prefill done in chunks
        # If threshold == 0 → all prefill in one round if any left
        if threshold > 0:
            prefill_rounds = np.ceil(remaining_prefill / threshold)
        else:
            prefill_rounds = (remaining_prefill > 0).astype(float)
      
        # ---- Total rounds left ----
        # Prefill requests: prefill_rounds + decode tokens
        # Decode requests: only decode tokens (since prefill_rounds = 0) -  1 token per rounf
        rounds_left = prefill_rounds + remaining_decode
        if len(prefill_rounds):
            nb_prefill_rounds= np.max(prefill_rounds.astype(int))
        else:
            nb_prefill_rounds=0
        return rounds_left, remaining_decode, remaining_prefill, nb_prefill_rounds


    def estimate_kv_blocks(self, request, predicted_tokens, kv_manager, reserved_blocks):

        # ---- Reproduce the scheduler logic used in vLLM ----
        # When the request has not started yet (no tokens computed),
        # the scheduler checks whether some prefix tokens are already
        # cached in the KV cache.

        if request.num_computed_tokens == 0:

            # Query the KV cache manager to find cached prefix blocks
            # and the number of tokens that can be reused.
            new_blocks, num_new_local_tokens = \
                kv_manager.get_computed_blocks(request)

            # In this simulator we assume there is no external KV connector
            # (which could provide additional cached tokens from another node).
            num_external_tokens = 0

            # Total number of tokens that are already computed
            # thanks to prefix caching.
            num_computed_tokens = (
                num_new_local_tokens + num_external_tokens
            )

        else:
            # If the request is already running, the scheduler
            # directly uses the number of tokens already computed.
            num_computed_tokens = request.num_computed_tokens

        # ---- Simulate the future token growth of the request ----
        # We estimate how many tokens will exist after generating
        # the predicted number of new tokens.

        num_tokens_need_slot = min(
            num_computed_tokens + predicted_tokens,
            kv_manager.max_model_len
        )

        # Convert the number of tokens into KV cache blocks.
        # Blocks are allocated in chunks of block_size tokens.
        blocks_needed = math.ceil(
            num_tokens_need_slot / kv_manager.block_size
        )

        # Number of blocks currently allocated to this request.
        current_blocks = len(
            kv_manager.get_blocks(request.request_id).blocks[0]
        )

        # Additional blocks that would need to be allocated.
        extra_blocks = max(0, blocks_needed - current_blocks)

        # Number of blocks currently free in the KV cache.
        free_blocks = kv_manager.block_pool.get_num_free_blocks()

        # Effective free blocks after accounting for blocks
        # that were already reserved during the simulation.
        effective_free = free_blocks - reserved_blocks

        # If we do not have enough free blocks, the request
        # cannot be admitted.
        if extra_blocks > effective_free:
            return None

        # Otherwise return the number of blocks that must be reserved.
        return extra_blocks
    ### END MOD ####

    def schedule(self) -> SchedulerOutput:
        # NOTE(woosuk) on the scheduling algorithm:
        # There's no "decoding phase" nor "prefill phase" in the scheduler.
        # Each request just has the num_computed_tokens and
        # num_tokens_with_spec. num_tokens_with_spec =
        # len(prompt_token_ids) + len(output_token_ids) + len(spec_token_ids).
        # At each step, the scheduler tries to assign tokens to the requests
        # so that each request's num_computed_tokens can catch up its
        # num_tokens_with_spec. This is general enough to cover
        # chunked prefills, prefix caching, speculative decoding,
        # and the "jump decoding" optimization in the future.
       
        ### BEGIN MOD ####
        #########################################################################################################
        ################### CUSTOM DEADLINE STRATEGY ############################################################
        #########################################################################################################
        def deadline_strategy(variant):
            t0 = time.monotonic()
            self.round += 1
            scheduler_start_time = time.time()
            now = scheduler_start_time - self.start_time
            # verbose = (self.round % 100 == 1)
            verbose = False
            verbose_kv = True
            if verbose: 
                logger.info(f"########## round {self.round} — now={now:.3f} - Overhead={self.overhead:.6f} - Cum. overhead={self.cum_ovreahead:.3f} ##########")     
            elif self.round % 100 == 0:
                logger.info(f"########## round {self.round} — now={now:.3f} - Overhead={self.overhead:.6f} - Cum. overhead={self.cum_ovreahead:.3f} ##########")    
         
            n_waiting = len(self.waiting)
            n_running = len(self.running)
            nb_requests = n_waiting + n_running

            waiting_request_ids = np.empty(n_waiting, dtype=int)
            running_request_ids = np.empty(n_running, dtype=int)
            computed_tokens = np.empty(nb_requests, dtype=int)
            num_tokens = np.empty(nb_requests, dtype=int)
            id_to_request = {}

            # KV cache management
            reserved_blocks = 0

            new_requests_ids_set = set()
            finished_requests_ids_set = set(self.last_requests_set)  # start assuming all might finish 
        
            # Fill running part
            for i, r in enumerate(self.running):
                if not r.is_finished():
                    req_id = id_to_int(r.request_id)
                    running_request_ids[i] = req_id
                    id_to_request[req_id] = r
                    computed_tokens[i] = r.num_computed_tokens
                    num_tokens[i] = r.num_tokens
                    finished_requests_ids_set.discard(req_id)  # remove from finished if still running
                    if req_id not in self.last_requests_set:
                        new_requests_ids_set.add(req_id)
            
            # Fill waiting part
            for i, r in enumerate(self.waiting):
                if not r.is_finished():
                    req_id = id_to_int(r.request_id)
                    waiting_request_ids[i] = req_id
                    id_to_request[req_id] = r
                    computed_tokens[n_running + i] = r.num_computed_tokens
                    num_tokens[n_running + i] = r.num_tokens
                    finished_requests_ids_set.discard(req_id)  # remove from finished if still waiting
                    if req_id not in self.last_requests_set:
                        new_requests_ids_set.add(req_id)
            

            
            request_ids = np.concatenate([running_request_ids, waiting_request_ids])
            nb_new_requests = len(new_requests_ids_set)   
            request_ids_set = set(request_ids)         
            self.last_requests_set = request_ids_set
            t1 = time.monotonic()


            if verbose :
                print(f"{self.skip_round=}")

            if nb_new_requests == 0 and len(finished_requests_ids_set)== 0 and self.skip_round:
                if len(self.last_scheduled_ids_set) > 0:
                    critical_remaining_time = self.deadlines[self.last_critical_id] - now
                    critical_idx = np.flatnonzero(request_ids == self.last_critical_id)[0]
                    critical_computed_tokens = computed_tokens[critical_idx]

                    remaining_decode  = max(self.num_output_tokens[self.last_critical_id] - 
                                            max(critical_computed_tokens - self.num_input_tokens[self.last_critical_id], 0), 0)
                    required_tps =  remaining_decode /critical_remaining_time
                    idx = np.round(required_tps * 10).astype(int)
                    idx = np.clip(idx, 0, len(self.max_concurrency_lookup_table) - 1)
                    critical_max_concurency = self.max_concurrency_lookup_table[idx]

                    if critical_max_concurency == self.last_critical_max_concurency:
                        self.skip_round = True
                    else:
                        self.skip_round = False



                    running_scheduled_ids_set = self.last_scheduled_ids_set.intersection(set(running_request_ids))
                    waiting_set_scheduled_ids_set = self.last_scheduled_ids_set.intersection(set(waiting_request_ids))
                    running_requests = len(running_scheduled_ids_set)

                    if verbose:
                        print("Skipping")
                        print(f"{sorted(self.last_scheduled_ids_set)=}")
                        print(f"{running_scheduled_ids_set=}")
                        print(f"{waiting_set_scheduled_ids_set=}")
                        print(f"{critical_max_concurency=} == {self.last_critical_max_concurency=} ?")

                    # Build a rank map for quick ordering
                    rank = {rid: pos for pos, rid in enumerate(self.last_sorted_request_ids)}
                    
                    # Sort running tasks by their rank
                    self.running.sort(key=lambda r: rank.get(id_to_int(r.request_id)))

                    return running_requests, waiting_set_scheduled_ids_set, self.scheduler_config.long_prefill_token_threshold      
        
            if verbose: 
                print(f"{nb_requests=}")
                print(f"{nb_new_requests=}")
                print(f"{new_requests_ids_set=}")
                print(f"{len(finished_requests_ids_set)=}")
                print(f"{finished_requests_ids_set=}")
                print(f"{request_ids=}")
                print(f"{self.max_decode_tps=}")

            if nb_requests==0 :
                #if there is no on time_request schedule all of them
                self.last_scheduled_ids_set= np.array([])
                waiting_set_scheduled_ids_set = set()
                running_request_ids = []
                running_requests = 0
                self.skip_round = False
              
                return running_requests, waiting_set_scheduled_ids_set, self.scheduler_config.long_prefill_token_threshold      
            
    
            scheduled_ids_set = self.last_scheduled_ids_set - finished_requests_ids_set
                
                
            _, remaining_decode_tokens, remaining_prefill_tokens, _ = self.compute_rounds_left(request_ids, computed_tokens)

            in_prefill = remaining_prefill_tokens > 0
            prefill_ids_set = set(request_ids[in_prefill])
            scheduled_last_round = np.isin(request_ids, list(scheduled_ids_set))
            
            this_round_tokens = np.where(in_prefill, remaining_prefill_tokens, 1)
        
            id_to_tokens = dict(zip(request_ids, this_round_tokens))
            # request that are almost finished are not cindidered for criticality
            forced = (remaining_decode_tokens < 5) & ~in_prefill 

            force_schedule_set = set(request_ids[forced])
            if verbose: 
                print(f"{force_schedule_set=}")

            def update_metrics(delta=0):
                remaining_time = self.deadlines[request_ids] - now - delta
                decode_tps_requirement = remaining_decode_tokens / remaining_time
                                 
                # max_concurency_array = np.floor(self.decode_req_interp(np.zeros(nb_requests), np.zeros(nb_requests), decode_tps_requirement)).astype(int)
                
                idx = np.round(decode_tps_requirement * 10).astype(int)
                # Clip indices to stay in bounds
                idx = np.clip(idx, 0, len(self.max_concurrency_lookup_table) - 1)
                # Lookup concurrency
                max_concurency_array = self.max_concurrency_lookup_table[idx]

                is_deprioritized = (self.deadlines[request_ids] == np.inf)
                nb_deprioritized = np.count_nonzero(is_deprioritized)
                nb_priority_requests = nb_requests - nb_deprioritized
                could_deprioritized = (decode_tps_requirement > self.max_decode_tps) | (remaining_time < -delta)
                has_priority = ~is_deprioritized & ~could_deprioritized

            
                # Sort IDs by descending priority
                # Sort indices by descending decode_tps_requirement, breaking ties with ascending remaining_time
                # A request is critical if its concurecy is bellow the the number of priority requests,
                # and either (1) it is not in prefill, or (2) it is in prefill but can accomodate
                # many requests (max_concurency_array > 10).
                is_critical = has_priority & (max_concurency_array < nb_priority_requests) & (
                    (~in_prefill) | (max_concurency_array > 10)
                ) 
                must_be_scheduled = is_critical & scheduled_last_round
            
                # Find the first critical ID
                # Determine the "critical" task to prioritize
                # 1. If there are tasks that must be scheduled this round, consider only them
                # 2. Otherwise, if there are critical tasks (max_concurency_array < nb_requests), consider these
                # 3. If neither exist, there is no critical task; set defaults
                # if np.any(must_be_scheduled & has_priority & ~forced):
                #     mask = must_be_scheduled & has_priority & ~forced # Only tasks that must be scheduled
                # elif np.any(is_critical &  has_priority & ~forced):
                #     mask = is_critical & has_priority & ~forced      # Only critical tasks
                # elif np.any(~in_prefill & has_priority & ~forced):
                #     mask = ~in_prefill & has_priority & ~forced     # No prefill tasks
                # else:
                #     mask = np.ones(len(max_concurency_array), dtype=bool) # all tasks


                if np.any(must_be_scheduled & has_priority):
                    mask = must_be_scheduled & has_priority # Only tasks that must be scheduled
                elif np.any(is_critical &  has_priority):
                    mask = is_critical & has_priority      # Only critical tasks
                elif np.any(~in_prefill & has_priority):
                    mask = ~in_prefill & has_priority     # No prefill tasks
                else:
                    mask = np.ones(len(max_concurency_array), dtype=bool) # all tasks

                # Find the index of the task with the smallest nb_requests within the selected mask
                idx_within_mask = np.argmin(max_concurency_array[mask])
                # Map this index back to the original request_ids array
                critical_idx = np.flatnonzero(mask)[idx_within_mask]
                # Select the corresponding request_id and number of requests
                critical_id = request_ids[critical_idx]
                critical_max_concurency = max_concurency_array[critical_idx] 
                critical_remaining_decode_tokens = remaining_decode_tokens[critical_idx]
                
                max_requests = min(critical_max_concurency, nb_requests)

                # is_deprioritized = np.logical_or(self.deadlines[request_ids] == np.inf, 
                #                                  max_concurency_array < max_requests)
          
                # print(f"{remaining_time.shape=}\n{max_concurency_array=}\n{must_be_scheduled=}\n{is_deprioritized=}")

                # Sort requests using multiple keys with np.lexsort (last key is primary):
                # 1. is_deprioritized       -> False (not deprioritized) first, True later
                # 2. -must_be_scheduled      -> True (must be scheduled) first, False later (inverted for descending order)
                # 3. max_concurency_array       -> lower number of requests first
                # 4. remaining_time          -> lower remaining time first
                sort_ids = np.lexsort((remaining_time, max_concurency_array, ~must_be_scheduled, could_deprioritized, is_deprioritized, ~forced))
                sorted_request_ids = request_ids[sort_ids]
                sorted_max_concurency_array = max_concurency_array[sort_ids]
        
                
                if(verbose):
                    print(f"{critical_id=} - {critical_max_concurency=} {critical_remaining_decode_tokens=}")

                # non critcal task( task that can be desceduled) are tasks that are cirrentley scehduled 
                # and that have accomodate  more than teh current number of requests in parallele
                # nc_mask = ( sorted_max_concurency_array > nb_requests) # & np.isin(sorted_request_ids, scheduled_ids_set)
                # deschedulable_ids_ordered = sorted_request_ids[nc_mask][::-1]  # least prioirty tasks first

                return Metrics(
                    remaining_time=remaining_time,
                    remaining_decode_tokens=remaining_decode_tokens,
                    remaining_prefill_tokens=remaining_prefill_tokens,
                    decode_tps_requirement=decode_tps_requirement,
                    max_concurency_array=max_concurency_array,
                    is_critical=is_critical,
                    must_be_scheduled=must_be_scheduled,
                    is_deprioritized=is_deprioritized,
                    critical_id=critical_id,
                    critical_max_concurency=critical_max_concurency,
                    critical_remaining_decode_tokens=critical_remaining_decode_tokens,
                    sorted_request_ids=sorted_request_ids,
                    sorted_max_concurency_array=sorted_max_concurency_array,
                    deschedulable_ids_ordered= None, #deschedulable_ids_ordered,
                    max_requests=max_requests,
                    nb_priority_requests=nb_priority_requests,
                    has_priority = has_priority,
                    sort_ids=sort_ids ,
                    in_prefill=in_prefill,
                    prefill_ids_set=prefill_ids_set,
                    forced = forced
                )

            delta = 0
            
            metrics = update_metrics(delta=delta)

            if len(finished_requests_ids_set) == 0:

                if nb_new_requests>0:
                    mask_new_req = np.isin(request_ids, list(new_requests_ids_set))
                    new_requests_max_concurence = metrics.max_concurency_array[mask_new_req]
                    skip_new =  np.all(new_requests_max_concurence >= nb_requests)
                else: 
                    new_requests_max_concurence = None
                    skip_new = self.skip_round

                nb_in_decode = nb_requests - len(prefill_ids_set)

                skip_prefill = (
                    len(prefill_ids_set) == 0
                    or (nb_in_decode >= metrics.max_requests and skip_new)
                    # or np.any((~in_prefill) & (~metrics.is_deprioritized) & (remaining_decode_tokens <= 5))
                    # or skip_new

                )

                if skip_prefill:
                    sorted_decode_ids = metrics.sorted_request_ids[~in_prefill[metrics.sort_ids]]
                    nb_selected = min(metrics.max_requests, len(sorted_decode_ids))
                    if nb_selected > 0 or len(force_schedule_set) > 0 :
                        scheduled_ids_set = set(sorted_decode_ids[:nb_selected])
                        scheduled_ids_set.update(force_schedule_set)

                        running_scheduled_ids_set = scheduled_ids_set.intersection(set(running_request_ids))
                        waiting_set_scheduled_ids_set = scheduled_ids_set.intersection(set(waiting_request_ids))
                        running_requests = len(running_scheduled_ids_set)
                        self.last_scheduled_ids_set = scheduled_ids_set
                        self.last_sorted_request_ids = metrics.sorted_request_ids

                        if verbose:
                            print("Prefill Skipping")
                            print(f"{metrics.critical_max_concurency=} < 10 and {nb_in_decode=} >= {metrics.max_requests=}")
                            # print(f"or\t {~in_prefill})\n\t&({~metrics.is_deprioritized})\n\t&({remaining_decode_tokens <= 5})")
                            print(f"or\t{self.skip_round=} & all {new_requests_max_concurence=} < {nb_requests=} & {nb_new_requests=}")
                            print(f"{sorted(self.last_scheduled_ids_set)=}")
                            print(f"{running_scheduled_ids_set=}")
                            print(f"{waiting_set_scheduled_ids_set=}")

                        # Build a rank map for quick ordering
                        rank = {rid: pos for pos, rid in enumerate(metrics.sorted_request_ids)}
                        
                        # Sort running tasks by their rank
                        self.running.sort(key=lambda r: rank.get(id_to_int(r.request_id)))

                        self.last_critical_max_concurency = metrics.critical_max_concurency
                        self.last_critical_id = metrics.critical_id
                        self.skip_round = True
                    
                        return running_requests, waiting_set_scheduled_ids_set, self.scheduler_config.long_prefill_token_threshold  

            

            # if we have few requests, no need to test deprioritization    
            if metrics.critical_max_concurency < metrics.nb_priority_requests:
                # Boolean masks for depririritize tasks
                mask1 = metrics.remaining_time <= -delta # already passed the deadline-delta
                mask2 = metrics.decode_tps_requirement > self.max_decode_tps # TPS requierement unacheivable
                mask3 = metrics.max_concurency_array < metrics.max_requests # requests that cannot accomodate enough concurent requests 

                # Combine masks
                combined_mask = mask1 | mask2 | mask3

                # Select request IDs and convert to set
                deprioritize_request_ids = request_ids[combined_mask]
            else:
                deprioritize_request_ids = None
            
            t2 = time.monotonic()
            if verbose: 
                    df = pd.DataFrame({
                        "request_id": request_ids[metrics.sort_ids],
                        "prefill_tokens": metrics.remaining_prefill_tokens[metrics.sort_ids],
                        "decode_tokens": metrics.remaining_decode_tokens[metrics.sort_ids],
                        "remaining_time": metrics.remaining_time[metrics.sort_ids],
                        "decode_tps": metrics.decode_tps_requirement[metrics.sort_ids],
                        "max_conc": metrics.max_concurency_array[metrics.sort_ids],
                        "crit": np.where(metrics.is_critical[metrics.sort_ids], "*", ""), 
                        "prio": np.where(metrics.has_priority[metrics.sort_ids], "*", ""),
                        "must": np.where(metrics.must_be_scheduled[metrics.sort_ids], "*", "")
                    })
                    df["prev_sched"] = df["request_id"].apply(lambda rid: "*" if rid in scheduled_ids_set else "")
                  
                    pd.set_option('display.max_columns', None)
                    pd.set_option('display.width', 1000)
                    print(df.to_string(index=True))

                    # print(f"{len(metrics.deschedulable_ids_ordered)=}\n{metrics.deschedulable_ids_ordered=}\n")
                    print(f"{metrics.critical_max_concurency=}\n{metrics.max_requests=}\n{deprioritize_request_ids=}")

            t3 = time.monotonic()
            if deprioritize_request_ids is not None and len(deprioritize_request_ids)>0:
                self.deadlines[deprioritize_request_ids] = np.inf

                metrics = update_metrics()

                if verbose: 
                    print(f"After deprioritization:\n\t{metrics.remaining_time=}\n\t{metrics.decode_tps_requirement=}\n\t{metrics.max_concurency_array=}")
                
    
                
                # Sort IDs by descending priority

            t4 = time.monotonic()
            t4_0, t4_1 = 0, 0
            scheduled_ids_set = set(request_ids[metrics.must_be_scheduled]) 
            scheduled_ids_set.update(force_schedule_set)
            
            self.skip_round = True
          
            if verbose: 
                print(f"{len(scheduled_ids_set)=} <=>  {metrics.max_requests=}")
   
            # Remove excess if needed
            if len(scheduled_ids_set) > metrics.max_requests:
                nb_to_remove = len(scheduled_ids_set) - metrics.max_requests
                # Find lowest priority requests that are in scheduled_ids
                scheduled_ids_arr = np.fromiter(scheduled_ids_set, dtype=metrics.sorted_request_ids.dtype)
                mask = np.isin(metrics.sorted_request_ids[::-1], scheduled_ids_arr)
                to_remove = metrics.sorted_request_ids[::-1][mask][:nb_to_remove]
                if verbose: 
                    print(f"{nb_to_remove=}\n{to_remove=}")
                scheduled_ids_set = scheduled_ids_set - set(to_remove)

            elif len(scheduled_ids_set) < metrics.max_requests:
                # Add requests if needed
                id_to_unsorted_index = None
                nb_to_add = metrics.max_requests - len(scheduled_ids_set)
                if verbose: 
                    print(f"{len(scheduled_ids_set)=} < {metrics.max_requests=} ?")
                    print(f"True: adding requests {nb_to_add=}")
                
                # to_add = sorted_requests[mask][:nb_to_add]
                # print(f"{nb_to_add=}\n{to_add=}")
                nb_added = 0
                to_add = set()
                nb_prefill_req= 0
                tot_prefill_tokens = 0
                nb_prefill_scheduled = np.sum(metrics.must_be_scheduled & metrics.in_prefill)    
                nb_critical_in_decode = np.sum(~metrics.in_prefill & metrics.is_critical)
                
                tried_prioritized_prefill_req = False

               
                token_budget = self.max_num_scheduled_tokens   
                for id in scheduled_ids_set:
                    token_budget -= id_to_tokens[id] 
                    
                max_prefill_tokens = np.inf
                candidates = np.setdiff1d(
                    metrics.sorted_request_ids,
                    np.fromiter(scheduled_ids_set, dtype=metrics.sorted_request_ids.dtype),
                    assume_unique=True
                )
                for idx, id_to_try in enumerate(candidates):
                    if verbose: 
                        print(f"{idx=} {id_to_try=} {token_budget=} {nb_added=}/{nb_to_add=}")
                    if token_budget < 0:
                        break
                    if nb_added >= nb_to_add:
                        break
                    if id_to_try in scheduled_ids_set:
                        continue
                    if id_to_try not in metrics.prefill_ids_set:
                        # check KV cache space
                        extra_blocks = self.estimate_kv_blocks(
                            request=id_to_request[id_to_try],
                            predicted_tokens=1,
                            kv_manager=self.kv_cache_manager,
                            reserved_blocks=reserved_blocks
                        )
                        if extra_blocks is None:
                            if verbose_kv: 
                                logger.info(f"💾🚫 - Decode: No more blocks for {id_to_try=}")
                            continue   # cannot admit
                        else: 
                            reserved_blocks += extra_blocks

                        if verbose:
                            print(f"\tAdding decode request: {id_to_try=}")
                        to_add.add(id_to_try)
                        nb_added += 1
                        token_budget -= id_to_tokens[id_to_try] 
                    else:
                        # For a prefill request:
                        # We only schedule it if adding this request does not increase the prefill penalty 
                        # to the point that the "critical request" (the highest-priority request) 
                        # would require less concurrent resources than targeted (max_requests) to finish on time.
                        # In other words, we check whether scheduling this prefill request would violate 
                        # the timing or resource constraints of the critical request. 
                        # If it does, we skip this prefill request to ensure 
                        # the critical request can still meet its deadline. 
                        if id_to_unsorted_index is None:
                            id_to_unsorted_index = dict(zip(request_ids, range(len(request_ids)))) 
                        # Here we do not want to try to compute prefill for deprioritized tasks if there stil have priroity tasks in prefill 
                        idx_to_try = id_to_unsorted_index[id_to_try]

                        if not metrics.is_deprioritized[idx_to_try]: #this is priority requests
                            tried_prioritized_prefill_req = True
                        else:
                            if tried_prioritized_prefill_req:
                                continue
        
                        remaining_prefill_tokens = metrics.remaining_prefill_tokens[idx_to_try]

                        if remaining_prefill_tokens > max_prefill_tokens:
                            if verbose:
                                print(f"Skip: {remaining_prefill_tokens=} > {max_prefill_tokens=}")
                            continue

                        
                        # Manage KV cache
                        extra_blocks = self.estimate_kv_blocks(
                            request=id_to_request[id_to_try],
                            predicted_tokens=remaining_prefill_tokens,
                            kv_manager=self.kv_cache_manager,
                            reserved_blocks=reserved_blocks
                        )

                        if extra_blocks is None:
                            if verbose_kv: 
                                logger.info(f"💾❌ - Prefill: No more blocks for {id_to_try=}")
                            continue  # KV memory would overflow
                        else: 
                            reserved_blocks += extra_blocks

                        nb_prefill_req += 1
                        tot_prefill_tokens += remaining_prefill_tokens


                        t4_0 = time.monotonic()
                        prefill_penalty = self.prefill_penalty(metrics.max_requests - nb_prefill_req, nb_prefill_req, tot_prefill_tokens) * 1.2
                        t4_1 = time.monotonic()
                        # compute what's ahapping after the prefill phase of this task and the decode phase of the critical task
                        new_decode_tps_requirement = None
                        if verbose:
                            print(f"\t### {id_to_try=}")
                            print(f"\t\t{prefill_penalty=}")
                            print(f"\t\t{metrics.critical_id=}")
                            print(f"\t\t{metrics.max_requests=}, {nb_prefill_req=}, {tot_prefill_tokens=}")
                        
                        new_remaining_time = self.deadlines[metrics.critical_id] - now - delta - prefill_penalty
                        
                        if new_remaining_time <= 0:
                            # Deadline will be missed by doing prefill so set max decode tps requirement
                            new_decode_tps_requirement = self.max_decode_tps
                        else:
                            new_decode_tps_requirement = (
                                metrics.critical_remaining_decode_tokens - 1
                            ) / new_remaining_time
                        
                        this_remaining_time = self.deadlines[id_to_try] - now - delta - prefill_penalty            
                        if not metrics.has_priority[idx_to_try]:
                            tps_all = np.array([new_decode_tps_requirement])
                        else:
                            this_decode_tps_requirement = (self.num_output_tokens[id_to_try] - 1) / this_remaining_time
                            tps_all = np.array([new_decode_tps_requirement, this_decode_tps_requirement])
       
                        zeros = np.zeros_like(tps_all)
                        max_concurency = self.max_concurrency_lookup_table[0]
                        # logger.info(f"Max concurrency={max_concurency}, self.max_tps={self.max_decode_tps}")

                        admission_threshold = min(max(nb_critical_in_decode + nb_prefill_scheduled + 1, len(scheduled_ids_set)+nb_added)
                                                  , metrics.max_requests)
                        admission_threshold = max(admission_threshold, 10)
                        if max_concurency < admission_threshold : # skip this one
                            if verbose: 
                                print(f"\t\t🔴 NOT OK : {max_concurency=} < {admission_threshold=}")
                            nb_prefill_req -= 1
                            tot_prefill_tokens -= remaining_prefill_tokens
                            max_prefill_tokens = remaining_prefill_tokens
                        else: 
                            if verbose: 
                                print(f"\t\t🟢 OK : {new_decode_tps_requirement=}\n\t\t{max_concurency=} >= {admission_threshold=}")
                            to_add.add(id_to_try)
                            nb_added += 1
                            nb_prefill_scheduled += 1
                            token_budget -= id_to_tokens[id_to_try] 
                            self.skip_round = False # we need to check on next round
                            

                        # if nb_added==1 and metrics.critical_id is None:
                        #     metrics.critical_id = id_to_try
                        #     metrics.critical_remaining_decode_tokens = metrics.remaining_decode_tokens[metrics.sort_ids][idx]
                        #     if verbose:
                        #         print(f"{metrics.critical_id=} {metrics.critical_remaining_decode_tokens=}")
                scheduled_ids_set = scheduled_ids_set.union(to_add) 

            t5 = time.monotonic()
                # # Previously we added tasks without considering deschedulable tasks.
                # # If we now have more scheduled tasks than allowed, remove the excess
                # if len(scheduled_ids) > max_requests: 
                #     excess = len(scheduled_ids) - max_requests  # number of tasks to remove
                #     ids_to_deschedule = [rid for rid in deschedulable_ids_ordered if rid in scheduled_ids][:excess]
                #     # Remove them from scheduled_ids
                #     scheduled_ids = np.setdiff1d(scheduled_ids, ids_to_deschedule, assume_unique=True)
            
        
            if verbose:
                print(f"{scheduled_ids_set=}")

            
            # mask = ~np.isin(scheduled_ids, request_ids)  # True for elements not in request_ids
            # if np.any(mask):
            #     print("1. These IDs are not in request_ids:", scheduled_ids[mask])
            #     assert False, "Some scheduled_ids are not in request_ids"

            

            running_scheduled_ids_set = scheduled_ids_set.intersection(set(running_request_ids))
            waiting_scheduled_ids_set = scheduled_ids_set.intersection(set(waiting_request_ids))
            running_requests = len(running_scheduled_ids_set)
            self.last_scheduled_ids_set = scheduled_ids_set
            self.last_sorted_request_ids = metrics.sorted_request_ids
            self.last_critical_max_concurency = metrics.critical_max_concurency
            self.last_critical_id = metrics.critical_id
            if verbose:
                print(f"{running_request_ids=}")
                print(f"{waiting_request_ids=}")
                print(f"{running_scheduled_ids_set=}")
                print(f"{waiting_scheduled_ids_set=}")


            if running_requests>0 : 
                running_speed = metrics.decode_tps_requirement[:len(running_request_ids)]
                # Sort the running requests in decreasing order of their decode TPS:
                # - `zip(running_speed, self.running)` pairs each request with its current decode TPS.
                # - `sorted(..., key=lambda x: x[0], reverse=True)` sorts the pairs by TPS in descending order.
                # - The list comprehension extracts the requests in the new sorted order.
                # After this, `self.running` is reordered so that the highest decode TPS requests come first.
                self.running = [r for _, r in sorted(zip(running_speed, self.running), key=lambda x: x[0], reverse=True)]

            if len(waiting_scheduled_ids_set) > 0 :
                # We must reorder only the waiting queue, but `sort_ids` provides a
                # global ordering over *all* requests (running first, then waiting).
                # Running requests occupy global positions [0, n_running),
                # and waiting requests occupy [n_running, n_running + n_waiting).
                #
                # Philosophy:
                # 1. Identify which sorted positions correspond to waiting requests.
                # 2. Convert those global positions to local indices within `self.waiting`.
                # 3. Rebuild the waiting queue in that new order.
                
                # Step 1 — select the waiting segment in the global sorted list
                mask = (metrics.sort_ids >= n_running)

                # Step 2 — convert global positions to local indices of the waiting queue
                waiting_local = metrics.sort_ids[mask] - n_running

                # Step 3 — rebuild the waiting queue in the new sorted order
                # reorder waiting queue
                self.waiting.reorder_by_indices(waiting_local)
                if verbose:
                    # Build the reordered list of waiting request IDs
                    new_waiting_ids = [self.waiting[i].request_id for i in waiting_local]

                    # Print them
                    print("New waiting order:", new_waiting_ids)

            t6 = time.monotonic()
            if verbose: 
                delta_init_time = t1-t0
                delta_build_critical_time = t2-t1
                delta_df_time = t3-t2
                delta_deprio_time = t4-t3
                delta_sched_id_time = t5-t4
                delta_finalize_time = t6-t5
                delta_prefill_time = t4_1 - t4_0

                self.init_time += delta_init_time
                self.build_critical_time += delta_build_critical_time
                self.df_time += delta_df_time
                self.deprio_time += delta_deprio_time
                self.sched_id_time += delta_sched_id_time
                self.finalize_time += delta_finalize_time
                self.prefill_time += delta_prefill_time

                print(f"{self.init_time=} {delta_init_time=}")
                print(f"{self.build_critical_time=} {delta_build_critical_time=}")
                print(f"{self.df_time=} {delta_df_time=}")
                print(f"{self.deprio_time=} {delta_deprio_time=}")
                print(f"{self.sched_id_time=} - {self.prefill_time=} | {delta_sched_id_time=} - {delta_prefill_time=}")
                print(f"{self.finalize_time=} {delta_finalize_time=}")
            

            return running_requests, waiting_scheduled_ids_set, self.scheduler_config.long_prefill_token_threshold      
        ### END MOD ####
        ### BEGIN MOD ####
        #########################################################################################################
        ################### EDF STRATEGY ######################################################################
        #########################################################################################################
        def plain_edf():
            # plain EDF with token budget for prefill tasks 
            self.round += 1
            verbose = False
            n_waiting = len(self.waiting)
            n_running = len(self.running)
            nb_requests = n_waiting + n_running

            waiting_request_ids = np.empty(n_waiting, dtype=int)
            running_request_ids = np.empty(n_running, dtype=int)
            computed_tokens = np.empty(nb_requests, dtype=int)
            

            # Fill running part
            for i, r in enumerate(self.running):
                if not r.is_finished():
                    req_id = id_to_int(r.request_id)
                    running_request_ids[i] = req_id
                    computed_tokens[i] = r.num_computed_tokens
                                
            # Fill waiting part
            for i, r in enumerate(self.waiting):
                if not r.is_finished():
                    req_id = id_to_int(r.request_id)
                    waiting_request_ids[i] = req_id
                    computed_tokens[n_running + i] = r.num_computed_tokens
            

            
            request_ids = np.concatenate([running_request_ids, waiting_request_ids])
            deadlines = self.deadlines[request_ids]
            order = np.argsort(deadlines)
            sorted_request_ids = request_ids[order]
                        
            # Tokens needed for this round
            _, _, remaining_prefill_tokens, _ =  self.compute_rounds_left(request_ids, computed_tokens)
            in_prefill = remaining_prefill_tokens > 0
            this_round_tokens = np.where(in_prefill, remaining_prefill_tokens, 1)
            id_to_tokens = dict(zip(request_ids, this_round_tokens))

            # ---- EDF selection with token budget ----
            scheduled_ids_set = set()
            token_budget = self.max_num_scheduled_tokens

            for idx in sorted_request_ids:
                if token_budget <= 0:
                    break
                scheduled_ids_set.add(idx)
                token_budget -= id_to_tokens[idx]
    
            running_scheduled_ids_set = scheduled_ids_set.intersection(set(running_request_ids))
            waiting_scheduled_ids_set = scheduled_ids_set.intersection(set(waiting_request_ids))
            running_requests = len(running_scheduled_ids_set)
            
            if running_requests>0 : 
                running_deadlines = self.deadlines[running_request_ids]
                # Sort running request by EDF
                self.running = [r for _, r in sorted(zip(running_deadlines, self.running), key=lambda x: x[0])]
                if self.round % 100 == 0 and verbose:
                    print(f"running_ids={[id_to_int(r.request_id) for r in self.running]}")
            

            if len(waiting_scheduled_ids_set) > 0 :    
                # positions globales correspondant aux waiting
                mask = (order >= n_running)

                # indices locaux dans self.waiting
                waiting_local = order[mask] - n_running

                # reorder waiting queue
                self.waiting.reorder_by_indices(waiting_local)
                            
            return running_requests, waiting_scheduled_ids_set, self.scheduler_config.long_prefill_token_threshold    
        ### END MOD ####

        ### BEGIN MOD ####
        # OLD stuff not used any more
        def edf_strategy():
            
            token_budget = self.max_num_scheduled_tokens
            self.round += 1
            scheduler_start_time = time.time()
            now = scheduler_start_time - self.start_time
         
            n_waiting = len(self.waiting)
            n_running = len(self.running)
            nb_requests = n_waiting + n_running

            waiting_request_ids = np.empty(n_waiting, dtype=int)
            running_request_ids = np.empty(n_running, dtype=int)
            computed_tokens = np.empty(nb_requests, dtype=int)
            num_tokens = np.empty(nb_requests, dtype=int)
            id_to_request = {}

            # Fill running part
            for i, r in enumerate(self.running):
                if not r.is_finished():
                    running_request_ids[i] = r.request_id
                    id_to_request[id_to_int(r.request_id)] = r
                    computed_tokens[i] = r.num_computed_tokens
                    num_tokens[i] = r.num_tokens
                    
            # Fill waiting part
            for i, r in enumerate(self.waiting):
                if not r.is_finished():
                    waiting_request_ids[i] = r.request_id
                    id_to_request[id_to_int(r.request_id)] = r
                    computed_tokens[n_running + i] = r.num_computed_tokens
                    num_tokens[n_running + i] = r.num_tokens
            
        
            request_ids = np.concatenate([running_request_ids, waiting_request_ids])

            if len(request_ids)==0:
                #if there is no on time_request schedule all of them 
                running_requests = len(running_request_ids)
                waiting_set_scheduled_ids = set(waiting_request_ids)
                return running_requests, waiting_set_scheduled_ids, self.scheduler_config.long_prefill_token_threshold      


            return running_requests, waiting_set_scheduled_ids, self.scheduler_config.long_prefill_token_threshold      
       ### BEGIN MOD ####

       ### BEGIN MOD ####
        def id_to_int(string_id: str) -> int:
            """
            Convert a string ID into an integer quickly.
            Handles pure numeric strings or IDs with '-' like 'cpml-1234-0'.
            """
            
            # If '-' in the ID, take the first numeric-looking part
            if '-' in string_id:
                string_id  = string_id.split('-')[1]
            
            return int(string_id)

        ### END MOD ####

        ### BEGIN MOD ####
        # Montoring code 
        def monitoring(round_start_time):
            self.round += 1

            if self.prev_finish_time is not None: 
                prev_forward_pass_duration =  round_start_time - self.prev_finish_time
                prev_round_duration =  round_start_time - self.prev_round_start_time 
            else : 
                prev_forward_pass_duration = -1
                prev_round_duration = -1
            
            self.prev_round_start_time  = round_start_time

            scheduler_start_time = time.time()
            now = scheduler_start_time - self.start_time
            
            n_waiting = len(self.waiting)
            n_running = len(self.running)

            waiting_request_ids = np.empty(n_waiting, dtype=int)
            running_request_ids = np.empty(n_running, dtype=int)
            computed_tokens = np.empty(n_waiting + n_running, dtype=int)
            
            # Fill running part
            for i, r in enumerate(self.running):
                if not r.is_finished():
                    running_request_ids[i] = id_to_int(r.request_id)
                    computed_tokens[i] = r.num_computed_tokens
                    
            # Fill waiting part
            for i, r in enumerate(self.waiting):
                if not r.is_finished():
                    waiting_request_ids[i] = id_to_int(r.request_id)
                    computed_tokens[n_running + i] = r.num_computed_tokens
            
            request_ids = np.concatenate([running_request_ids, waiting_request_ids])
            
            # compute difference between new and old
            previous_computed_tokens = self.shared_data.get_computed_tokens(request_ids)
            tokens_progress = computed_tokens - previous_computed_tokens

            self.shared_data.record_token_progress(self.round, prev_forward_pass_duration, prev_round_duration, now, request_ids, tokens_progress)

            running_requests = len(running_request_ids)
            waiting_set_scheduled_ids = set(waiting_request_ids)
  
            return running_requests, waiting_set_scheduled_ids, self.scheduler_config.long_prefill_token_threshold      


        round_start_time = time.monotonic()

        scheduled_new_reqs: list[Request] = []
        scheduled_resumed_reqs: list[Request] = []
        scheduled_running_reqs: list[Request] = []
        preempted_reqs: list[Request] = []

        req_to_new_blocks: dict[str, KVCacheBlocks] = {}
        num_scheduled_tokens: dict[str, int] = {}
        token_budget = self.max_num_scheduled_tokens
        # Encoder-related.
        scheduled_encoder_inputs: dict[str, list[int]] = {}
        encoder_compute_budget = self.max_num_encoder_input_tokens
        # Spec decode-related.
        scheduled_spec_decode_tokens: dict[str, list[int]] = {}

        # For logging.
        scheduled_timestamp = time.monotonic()
        ### END MOD ####
        
        ### BEGIN MOD ####
        # code dto reste the scheduler internal state to enable workflow of XP without rebooting the server
        def restart_deadline_scheduler():
            self.round = -1
            logger.info("Restarting deadline scheduler. Round reset to -1.")
            model_id = self.shared_data.get_perf_model_id()
            if model_id != self.model_id:
                self.model_id = model_id
                model_path = self.shared_data.get_perf_model_path_from_id(model_id)
                logger.info(f"Loading model for id {model_id} from path {model_path}")
                self.model = RoundDurationModel.load_model(model_path)

                self.prefill_penalty = self.model.predict
           
                step = 0.1
                self.max_concurrency_lookup_table, tps_array = self.model.max_concurrency_lookup_table(step=step)

                # Enforce monotonicity: propagate the maximum value forward along the
                # descending TPS array, then reverse it so the final table is aligned
                # with ascending TPS order.
                self.max_concurrency_lookup_table = np.maximum.accumulate(
                    self.max_concurrency_lookup_table
                )[::-1]

                self.max_decode_tps= np.floor(tps_array[0]).astype(int)  
                # logger.info(f"Max concurrency lookup table: {self.max_concurrency_lookup_table}")
                # Now, Given a TPS value/array 'tps', to use the lookup index safely:
                # 1. Round to nearest 0.1 step → index = round(tps / 0.1)
                # 2. Clip the index to the valid range of the lookup table
                # 3. Read the corresponding concurrency
                # idx = int(round(tps / 0.1))
                # idx = np.clip(idx, 0, len(self.max_concurrency_lookup_table) - 1)
                # concurrency = self.max_concurrency_lookup_table[idx]

                self.prefill_penalty(10,10,1024)
                self.prefill_penalty(1,1,1024)
            else:
                logger.info(f"Deadline scheduler restart requested but model id {model_id} is the same as current. No model reload, but resetting internal state.")

            
            self.active_requests = np.zeros(self.nb_requests, dtype=bool) 
            self.deadlines = self.shared_data.get_deadline_np_array()
            self.num_output_tokens = self.shared_data.get_output_tokens_np_array()
            self.num_input_tokens = self.shared_data.get_input_tokens_np_array()
            self.prev_finish_time = None     
            self.prev_round_start_time = None
      
            self.last_scheduled_requests_set = set()
            self.last_scheduled_ids_set = set()
            self.last_requests_set = set()
            self.overhead = 0
            self.cum_ovreahead = 0
            self.skip_round = False
            self.init_time = 0
            self.build_critical_time = 0
            self.build_critical_time_1 = 0
            self.build_critical_time_2 = 0
            self.build_critical_time_3 = 0
            self.build_critical_time_4 = 0
            self.df_time = 0
            self.deprio_time = 0
            self.sched_id_time = 0
            self.finalize_time = 0
            self.prefill_time = 0
    
            self.get_start_time_next_round = False
        ### BEGIN MOD ####
         
        ### BEGIN MOD ####
        # Here we are on the critcal path of the scheduler
        # We add a test if we are doing deadlien scheduling
        # logger.info(f"Scheduling policy: {self.policy}")
        if self.policy == SchedulingPolicy.DEADLINE:
            if self.shared_data == None or self.nb_requests != self.shared_data.get_num_requests():
                if self.shm:
                    self.shm.close()
                self.shm, self.shared_data = create_shared_data_from_shm(shm_name = "my_shared_block")
                self.nb_requests = self.shared_data.get_num_requests()
                logger.info(f"Initialized shared data for deadline scheduler with {self.nb_requests} requests.")
            
            # if self.round is None it means that no ontitilzation has been perfomed
            # so, we loop until te client ask us to perform the first initialization 
            # (which will be done at the first scheduling round) by setting the start time in shared data. T
            # his is to ensure that the scheduler is properly initialized before we start scheduling and
            while True:
                if self.shared_data.ask_for_restart():
                    restart_deadline_scheduler()
                    self.shared_data.restart_done()        
                if  self.round != None:
                    break
                logger.info("Waiting for first initialization signal from client...")
                time.sleep(0.1)
            
            i = 0
            while self.shared_data.get_start_time() < 0:
                if i % 100 == 0:
                    logger.info("Waiting for scheduler start time to be set...")
                time.sleep(0.1)
                i+=1

            
            if self.start_time != self.shared_data.get_start_time():
                self.start_time = self.shared_data.get_start_time()
                logger.info(f"Scheduler start time: {self.start_time}")

            start_sched=time.monotonic()
            algo_variant = self.shared_data.get_algorithm_variant()
            if algo_variant != self.last_algo_variant:
                logger.info(f"Scheduling algorithm variant changed from {self.last_algo_variant} to {algo_variant}")
                self.last_algo_variant = algo_variant
            if algo_variant ==  Algorithm.OUT_OF_ORDER_DISCARD_MOST_URGENT or algo_variant == Algorithm.OUT_OF_ORDER_DISCARD_LEAST_URGENT: 
                running_requests, waiting_set_scheduled_ids, saved_long_prefill_token_threshold = deadline_strategy(algo_variant)
            elif algo_variant == Algorithm.EDF :
                running_requests, waiting_set_scheduled_ids, saved_long_prefill_token_threshold = plain_edf()
            elif algo_variant == Algorithm.MONITORING: 
                running_requests, waiting_set_scheduled_ids, saved_long_prefill_token_threshold = monitoring(round_start_time)
            else : 
                sys.exit(f"[Warning] Unknow variant '{algo_variant}' for SchedulingPolicy == DEADLINE.\nFalling back to baseline.")
            
            self.overhead = time.monotonic() - start_sched
            self.cum_ovreahead += self.overhead
        ### END MOD ####

        # print(f"{self.round=} {running_requests=} {waiting_request_ids=}")
        # First, schedule the RUNNING requests.sort 
        # self.round += 1
        # if self.round%1000 == 0 :
        #         requests_ids = ["Running: "]
        #         for i, r in enumerate(self.running):
        #             if not r.is_finished():
        #                 req_id = r.request_id
        #                 requests_ids.append(req_id)
        #         requests_ids.append("\nWaiting: ")
        #         for i, r in enumerate(self.waiting):
        #             if not r.is_finished():
        #                 req_id = r.request_id
        #                 requests_ids.append(req_id)

        #         print(f"{requests_ids=}")

        
        req_index = 0
        while req_index < len(self.running) and token_budget > 0 :
            request = self.running[req_index]
            
            ### BEGIN MOD ####
            # Here we are in teh ain loop where requestes are processed in order
            if self.policy == SchedulingPolicy.DEADLINE:
                # req_id = id_to_int(request.request_id)
                # is_active = self.active_requests[req_id]
                # # print(f"{self.round=} {in_decode=} {running_requests=}")
                if running_requests <= 0:
                    break;
            ### END MOD ####

            num_new_tokens = (request.num_tokens_with_spec +
                              request.num_output_placeholders -
                              request.num_computed_tokens)
            if (0 < self.scheduler_config.long_prefill_token_threshold <
                    num_new_tokens):
                num_new_tokens = (
                    self.scheduler_config.long_prefill_token_threshold)
            num_new_tokens = min(num_new_tokens, token_budget)

            # Make sure the input position does not exceed the max model len.
            # This is necessary when using spec decoding.
            num_new_tokens = min(
                num_new_tokens,
                self.max_model_len - 1 - request.num_computed_tokens)

            # Schedule encoder inputs.
            encoder_inputs_to_schedule = None
            new_encoder_compute_budget = encoder_compute_budget
            if request.has_encoder_inputs:
                (encoder_inputs_to_schedule, num_new_tokens,
                 new_encoder_compute_budget
                 ) = self._try_schedule_encoder_inputs(
                     request, request.num_computed_tokens, num_new_tokens,
                     encoder_compute_budget)

            if num_new_tokens == 0:
                # The request cannot be scheduled because one of the following
                # reasons:
                # 1. No new tokens to schedule. This may happen when
                #    (1) PP>1 and we have already scheduled all prompt tokens
                #    but they are not finished yet.
                #    (2) Async scheduling and the request has reached to either
                #    its max_total_tokens or max_model_len.
                # 2. The encoder budget is exhausted.
                # 3. The encoder cache is exhausted.
                # NOTE(woosuk): Here, by doing `continue` instead of `break`,
                # we do not strictly follow the FCFS scheduling policy and
                # allow the lower-priority requests to be scheduled.
                req_index += 1
                continue

            while True:
                new_blocks = self.kv_cache_manager.allocate_slots(
                    request,
                    num_new_tokens,
                    num_lookahead_tokens=self.num_lookahead_tokens)
                if new_blocks is None:
                    # The request cannot be scheduled.
                    # Preempt the lowest-priority request.
                    if self.policy == SchedulingPolicy.PRIORITY:
                        preempted_req = max(
                            self.running,
                            key=lambda r: (r.priority, r.arrival_time),
                        )
                        self.running.remove(preempted_req)
                        if preempted_req in scheduled_running_reqs:
                            scheduled_running_reqs.remove(preempted_req)
                    else:
                        preempted_req = self.running.pop()

                    self.kv_cache_manager.free(preempted_req)
                    self.encoder_cache_manager.free(preempted_req)
                    preempted_req.status = RequestStatus.PREEMPTED
                    preempted_req.num_computed_tokens = 0
                    if self.log_stats:
                        preempted_req.record_event(
                            EngineCoreEventType.PREEMPTED, scheduled_timestamp)

                    self.waiting.prepend_request(preempted_req)
                    preempted_reqs.append(preempted_req)
                    if preempted_req == request:
                        # No more request to preempt.
                        can_schedule = False
                        break
                else:
                    # The request can be scheduled.
                    can_schedule = True
                    break
            if not can_schedule:
                break
            assert new_blocks is not None

            # Schedule the request.
            scheduled_running_reqs.append(request)
            req_to_new_blocks[request.request_id] = new_blocks
            num_scheduled_tokens[request.request_id] = num_new_tokens
            token_budget -= num_new_tokens
            req_index += 1

            ### BEGIN MOD ####
            if self.policy == SchedulingPolicy.DEADLINE:
                    running_requests -= 1
            ### END MOD ####

            # Speculative decode related.
            if request.spec_token_ids:
                num_scheduled_spec_tokens = (num_new_tokens +
                                             request.num_computed_tokens -
                                             request.num_tokens)
                if num_scheduled_spec_tokens > 0:
                    # Trim spec_token_ids list to num_scheduled_spec_tokens.
                    del request.spec_token_ids[num_scheduled_spec_tokens:]
                    scheduled_spec_decode_tokens[request.request_id] = (
                        request.spec_token_ids)

            # Encoder-related.
            if encoder_inputs_to_schedule:
                scheduled_encoder_inputs[request.request_id] = (
                    encoder_inputs_to_schedule)
                # Allocate the encoder cache.
                for i in encoder_inputs_to_schedule:
                    self.encoder_cache_manager.allocate(request, i)
                encoder_compute_budget = new_encoder_compute_budget

        # Record the LoRAs in scheduled_running_reqs
        scheduled_loras: set[int] = set()
        if self.lora_config:
            scheduled_loras = set(
                req.lora_request.lora_int_id for req in scheduled_running_reqs
                if req.lora_request and req.lora_request.lora_int_id > 0)
            assert len(scheduled_loras) <= self.lora_config.max_loras

        # Use a temporary RequestQueue to collect requests that need to be
        # skipped and put back at the head of the waiting queue later
        skipped_waiting_requests = create_request_queue(self.policy)

        # Next, schedule the WAITING requests.
        if not preempted_reqs:
            while self.waiting and token_budget > 0:
                if len(self.running) == self.max_num_running_reqs:
                    break

                request = self.waiting.peek_request()
                
                ### BEGIN MOD ####
                if self.policy == SchedulingPolicy.DEADLINE:
                    req_id = id_to_int(request.request_id)
                    # is_active= self.active_requests[req_id]
                    # # print(f"{self.round=} {in_decode=} {req_id=} {waiting_set_scheduled_ids=}")
                    if req_id not in waiting_set_scheduled_ids:
                        self.waiting.pop_request()
                        skipped_waiting_requests.prepend_request(request)
                        continue
                ### END MOD ####

                # KVTransfer: skip request if still waiting for remote kvs.
                if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                    is_ready = self._update_waiting_for_remote_kv(request)
                    if is_ready:
                        request.status = RequestStatus.WAITING
                    else:
                        logger.debug(
                            "%s is still in WAITING_FOR_REMOTE_KVS state.",
                            request.request_id)
                        self.waiting.pop_request()
                        skipped_waiting_requests.prepend_request(request)
                        continue

                # Skip request if the structured output request is still waiting
                # for FSM compilation.
                if request.status == RequestStatus.WAITING_FOR_FSM:
                    structured_output_req = request.structured_output_request
                    if structured_output_req and structured_output_req.grammar:
                        request.status = RequestStatus.WAITING
                    else:
                        self.waiting.pop_request()
                        skipped_waiting_requests.prepend_request(request)
                        continue

                # Check that adding the request still respects the max_loras
                # constraint.
                if (self.lora_config and request.lora_request and
                    (len(scheduled_loras) == self.lora_config.max_loras and
                     request.lora_request.lora_int_id not in scheduled_loras)):
                    # Scheduling would exceed max_loras, skip.
                    self.waiting.pop_request()
                    skipped_waiting_requests.prepend_request(request)
                    continue

                num_external_computed_tokens = 0
                load_kv_async = False

                # Get already-cached tokens.
                if request.num_computed_tokens == 0:
                    # Get locally-cached tokens.
                    new_computed_blocks, num_new_local_computed_tokens = \
                        self.kv_cache_manager.get_computed_blocks(
                            request)

                    # Get externally-cached tokens if using a KVConnector.
                    if self.connector is not None:
                        num_external_computed_tokens, load_kv_async = (
                            self.connector.get_num_new_matched_tokens(
                                request, num_new_local_computed_tokens))

                        if num_external_computed_tokens is None:
                            # The request cannot be scheduled because
                            # the KVConnector couldn't determine
                            # the number of matched tokens.
                            self.waiting.pop_request()
                            skipped_waiting_requests.prepend_request(request)
                            continue

                    # Total computed tokens (local + external).
                    num_computed_tokens = (num_new_local_computed_tokens +
                                           num_external_computed_tokens)
                # KVTransfer: WAITING reqs have num_computed_tokens > 0
                # after async KV recvs are completed.
                else:
                    new_computed_blocks = (
                        self.kv_cache_manager.create_empty_block_list())
                    num_new_local_computed_tokens = 0
                    num_computed_tokens = request.num_computed_tokens

                encoder_inputs_to_schedule = None
                new_encoder_compute_budget = encoder_compute_budget

                # KVTransfer: loading remote KV, do not allocate for new work.
                if load_kv_async:
                    assert num_external_computed_tokens > 0
                    num_new_tokens = 0
                # Number of tokens to be scheduled.
                else:
                    # We use `request.num_tokens` instead of
                    # `request.num_prompt_tokens` to consider the resumed
                    # requests, which have output tokens.
                    num_new_tokens = request.num_tokens - num_computed_tokens
                    if (0 < self.scheduler_config.long_prefill_token_threshold
                            < num_new_tokens):
                        num_new_tokens = (
                            self.scheduler_config.long_prefill_token_threshold)

                    # chunked prefill has to be enabled explicitly to allow
                    # pooling requests to be chunked
                    if not self.scheduler_config.chunked_prefill_enabled and \
                        num_new_tokens > token_budget:
                        self.waiting.pop_request()
                        skipped_waiting_requests.prepend_request(request)
                        continue

                    num_new_tokens = min(num_new_tokens, token_budget)
                    assert num_new_tokens > 0

                    # Schedule encoder inputs.
                    if request.has_encoder_inputs:
                        (encoder_inputs_to_schedule, num_new_tokens,
                         new_encoder_compute_budget
                         ) = self._try_schedule_encoder_inputs(
                             request, num_computed_tokens, num_new_tokens,
                             encoder_compute_budget)
                        if num_new_tokens == 0:
                            # The request cannot be scheduled.
                            break

                # Handles an edge case when P/D Disaggregation
                # is used with Spec Decoding where an
                # extra block gets allocated which
                # creates a mismatch between the number
                # of local and remote blocks.
                effective_lookahead_tokens = (0 if request.num_computed_tokens
                                              == 0 else
                                              self.num_lookahead_tokens)

                # Determine if we need to allocate cross-attention blocks.
                if self.is_encoder_decoder and request.has_encoder_inputs:
                    # TODO(russellb): For Whisper, we know that the input is
                    # always padded to the maximum length. If we support other
                    # encoder-decoder models, this will need to be updated if we
                    # want to only allocate what is needed.
                    assert ("whisper"
                            in self.vllm_config.model_config.model.lower()), (
                                "Whisper is the only supported "
                                "encoder-decoder model.")
                    num_encoder_tokens = MULTIMODAL_REGISTRY.\
                        get_encdec_max_encoder_len(
                        self.vllm_config.model_config)
                else:
                    num_encoder_tokens = 0

                new_blocks = self.kv_cache_manager.allocate_slots(
                    request,
                    num_new_tokens + num_external_computed_tokens,
                    num_new_local_computed_tokens,
                    new_computed_blocks,
                    num_lookahead_tokens=effective_lookahead_tokens,
                    delay_cache_blocks=load_kv_async,
                    num_encoder_tokens=num_encoder_tokens,
                )

                if new_blocks is None:
                    # The request cannot be scheduled.
                    break

                # KVTransfer: the connector uses this info to determine
                # if a load is needed. Note that
                # This information is used to determine if a load is
                # needed for this request.
                if self.connector is not None:
                    self.connector.update_state_after_alloc(
                        request,
                        new_computed_blocks + new_blocks,
                        num_external_computed_tokens,
                    )

                # Request was already popped from self.waiting
                # unless it was re-added above due to new_blocks being None.
                request = self.waiting.pop_request()
                if load_kv_async:
                    # If loading async, allocate memory and put request
                    # into the WAITING_FOR_REMOTE_KV state.
                    skipped_waiting_requests.prepend_request(request)
                    request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
                    continue

                req_index += 1
                self.running.append(request)
                if self.log_stats:
                    request.record_event(EngineCoreEventType.SCHEDULED,
                                         scheduled_timestamp)
                if request.status == RequestStatus.WAITING:
                    scheduled_new_reqs.append(request)
                elif request.status == RequestStatus.PREEMPTED:
                    scheduled_resumed_reqs.append(request)
                else:
                    raise RuntimeError(
                        f"Invalid request status: {request.status}")

                if self.lora_config and request.lora_request:
                    scheduled_loras.add(request.lora_request.lora_int_id)
                req_to_new_blocks[request.request_id] = (
                    self.kv_cache_manager.get_blocks(request.request_id))
                num_scheduled_tokens[request.request_id] = num_new_tokens
                token_budget -= num_new_tokens
                request.status = RequestStatus.RUNNING
                request.num_computed_tokens = num_computed_tokens
                # Count the number of prefix cached tokens.
                if request.num_cached_tokens < 0:
                    request.num_cached_tokens = num_computed_tokens
                # Encoder-related.
                if encoder_inputs_to_schedule:
                    scheduled_encoder_inputs[request.request_id] = (
                        encoder_inputs_to_schedule)
                    # Allocate the encoder cache.
                    for i in encoder_inputs_to_schedule:
                        self.encoder_cache_manager.allocate(request, i)
                    encoder_compute_budget = new_encoder_compute_budget

        # Put back any skipped requests at the head of the waiting queue
        if skipped_waiting_requests:
            self.waiting.prepend_requests(skipped_waiting_requests)

        # Check if the scheduling constraints are satisfied.
        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
        assert total_num_scheduled_tokens <= self.max_num_scheduled_tokens
        assert token_budget >= 0
        assert len(self.running) <= self.max_num_running_reqs
        # Since some requests in the RUNNING queue may not be scheduled in
        # this step, the total number of scheduled requests can be smaller than
        # len(self.running).
        assert (len(scheduled_new_reqs) + len(scheduled_resumed_reqs) +
                len(scheduled_running_reqs) <= len(self.running))

        # Get the longest common prefix among all requests in the running queue.
        # This can be potentially used for cascade attention.
        num_common_prefix_blocks = [0] * len(
            self.kv_cache_config.kv_cache_groups)
        if self.running:
            any_request = self.running[0]
            num_common_prefix_blocks = (
                self.kv_cache_manager.get_num_common_prefix_blocks(
                    any_request, len(self.running)))

        # Construct the scheduler output.
        new_reqs_data = [
            NewRequestData.from_request(
                req, req_to_new_blocks[req.request_id].get_block_ids())
            for req in scheduled_new_reqs
        ]
        cached_reqs_data = self._make_cached_request_data(
            scheduled_running_reqs,
            scheduled_resumed_reqs,
            num_scheduled_tokens,
            scheduled_spec_decode_tokens,
            req_to_new_blocks,
        )
        structured_output_request_ids, grammar_bitmask = (
            self.get_grammar_bitmask(self.running,
                                     scheduled_spec_decode_tokens))
        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=new_reqs_data,
            scheduled_cached_reqs=cached_reqs_data,
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=total_num_scheduled_tokens,
            scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
            scheduled_encoder_inputs=scheduled_encoder_inputs,
            num_common_prefix_blocks=num_common_prefix_blocks,
            # finished_req_ids is an existing state in the scheduler,
            # instead of being newly scheduled in this step.
            # It contains the request IDs that are finished in between
            # the previous and the current steps.
            finished_req_ids=self.finished_req_ids,
            free_encoder_mm_hashes=self.encoder_cache_manager.
            get_freed_mm_hashes(),
            structured_output_request_ids=structured_output_request_ids,
            grammar_bitmask=grammar_bitmask,
        )

        # NOTE(Kuntai): this function is designed for multiple purposes:
        # 1. Plan the KV cache store
        # 2. Wrap up all the KV cache load / save ops into an opaque object
        # 3. Clear the internal states of the connector
        if self.connector is not None:
            meta = self.connector.build_connector_meta(scheduler_output)
            scheduler_output.kv_connector_metadata = meta

        # collect KV cache events from KV cache manager
        events = self.kv_cache_manager.take_events()

        # collect KV cache events from connector
        if self.connector is not None:
            connector_events = self.connector.take_events()
            if connector_events:
                if events is None:
                    events = list(connector_events)
                else:
                    events.extend(connector_events)

        # publish collected KV cache events
        if events:
            batch = KVEventBatch(ts=time.time(), events=events)
            self.kv_event_publisher.publish(batch)

        self._update_after_schedule(scheduler_output)

        
        ### BEGIN MOD ####
        if self.policy == SchedulingPolicy.DEADLINE:
            self.scheduler_config.long_prefill_token_threshold = int(saved_long_prefill_token_threshold)
            self.prev_finish_time = time.monotonic()
        ### END MOD ####

        return scheduler_output

    def _update_after_schedule(
        self,
        scheduler_output: SchedulerOutput,
    ) -> None:
        # Advance the number of computed tokens for the request AFTER
        # the request is scheduled.
        # 1. The scheduler_output of the current step has to include the
        #    original number of scheduled tokens to determine input IDs.
        # 2. Advance the number of computed tokens here allowing us to
        #    schedule the prefill request again immediately in the next
        #    scheduling step.
        # 3. If some tokens (e.g. spec tokens) are rejected later, the number of
        #    computed tokens will be adjusted in update_from_output.
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        for req_id, num_scheduled_token in num_scheduled_tokens.items():
            request = self.requests[req_id]
            request.num_computed_tokens += num_scheduled_token

            # NOTE: _free_encoder_inputs relies on num_computed_tokens, which
            # may be updated again in _update_from_output for speculative
            # decoding. However, it is safe to call the method here because
            # encoder inputs are always part of the prompt, not the output,
            # and thus are unaffected by speculative decoding.
            if request.has_encoder_inputs:
                self._free_encoder_inputs(request)

        # Clear the finished request IDs.
        # NOTE: We shouldn't do self.finished_req_ids.clear() here because
        # it will also affect the scheduler output.
        self.finished_req_ids = set()

    def _make_cached_request_data(
        self,
        running_reqs: list[Request],
        resumed_reqs: list[Request],
        num_scheduled_tokens: dict[str, int],
        spec_decode_tokens: dict[str, list[int]],
        req_to_new_blocks: dict[str, KVCacheBlocks],
    ) -> CachedRequestData:
        req_ids: list[str] = []
        new_token_ids: list[list[int]] = []
        new_block_ids: list[Optional[tuple[list[int], ...]]] = []
        num_computed_tokens: list[int] = []

        use_connector = self.connector is not None
        for req in itertools.chain(running_reqs, resumed_reqs):
            req_id = req.request_id
            req_ids.append(req_id)
            num_tokens = (num_scheduled_tokens[req_id] -
                          len(spec_decode_tokens.get(req_id, ())))
            if self.use_pp:
                # When using PP, the scheduler sends the sampled tokens back,
                # because there's no direct communication between the first-
                # stage worker and the last-stage worker. Otherwise, we don't
                # need to send the sampled tokens back because the model runner
                # will cache them.
                token_ids = req.all_token_ids[req.num_computed_tokens:req.
                                              num_computed_tokens + num_tokens]
                new_token_ids.append(token_ids)
            elif use_connector:
                # When using a KVConnector, we add a placeholder to avoid index
                # out of bounds errors. TODO: Remove this once the KVConnector
                # is updated to handle token IDs properly.
                new_token_ids.append([])
            new_block_ids.append(
                req_to_new_blocks[req_id].get_block_ids(allow_none=True))
            num_computed_tokens.append(req.num_computed_tokens)
        # Because resumed_reqs is usually empty, it is more efficient to do
        # in-place appending so that we don't need to allocate a new list.
        resumed_from_preemption = [False] * len(running_reqs)
        resumed_from_preemption += [True] * len(resumed_reqs)

        return CachedRequestData(
            req_ids=req_ids,
            resumed_from_preemption=resumed_from_preemption,
            new_token_ids=new_token_ids,
            new_block_ids=new_block_ids,
            num_computed_tokens=num_computed_tokens,
        )

    def _try_schedule_encoder_inputs(
        self,
        request: Request,
        num_computed_tokens: int,
        num_new_tokens: int,
        encoder_compute_budget: int,
    ) -> tuple[list[int], int, int]:
        """
        Determine which encoder inputs need to be scheduled in the current step,
        and update `num_new_tokens` and encoder token budget accordingly.

        An encoder input will be scheduled if:
        - Its output tokens overlap with the range of tokens being computed
        in this step, i.e.,
        [num_computed_tokens, num_computed_tokens + num_new_tokens).
        - It is not already computed and stored in the encoder cache.
        - There is sufficient encoder token budget to process it.
        - The encoder cache has space to store it.

        If an encoder input cannot be scheduled due to cache or budget
        limitations, the method adjusts `num_new_tokens` to schedule only the
        decoder tokens up to just before the unschedulable encoder input.

        Note that num_computed_tokens includes both locally cached
        blocks and externally cached blocks (via KVConnector).
        """
        if num_new_tokens == 0 or not request.has_encoder_inputs:
            return [], num_new_tokens, encoder_compute_budget
        encoder_inputs_to_schedule: list[int] = []
        mm_positions = request.mm_positions
        assert mm_positions is not None
        assert len(mm_positions) > 0

        # NOTE: since scheduler operates on the request level (possibly with
        # multiple encoder inputs per request), we need to create temporary
        # trackers for accounting at the encoder input level.
        mm_hashes_to_schedule = set()
        num_tokens_to_schedule = 0
        for i, pos_info in enumerate(mm_positions):
            start_pos = pos_info.offset
            num_encoder_tokens = pos_info.length

            # The encoder output is needed if the two ranges overlap:
            # [num_computed_tokens, num_computed_tokens + num_new_tokens) and
            # [start_pos, start_pos + num_encoder_tokens)
            if start_pos >= num_computed_tokens + num_new_tokens:
                # The encoder input is not needed in this step.
                break

            if self.is_encoder_decoder and num_computed_tokens > 0:
                assert start_pos == 0, (
                    "Encoder input should be processed at the beginning of "
                    "the sequence when encoder-decoder models are used.")
                # Encoder input has already been computed
                # The calculation here is a bit different. We don't turn encoder
                # output into tokens that get processed by the decoder and
                # reflected in num_computed_tokens. Instead, start_pos reflects
                # the position where we need to ensure we calculate encoder
                # inputs. This should always be 0 to ensure we calculate encoder
                # inputs before running the decoder.  Once we've calculated some
                # decoder tokens (num_computed_tokens > 0), then we know we
                # already calculated encoder inputs and can skip here.
                continue
            elif start_pos + num_encoder_tokens <= num_computed_tokens:
                # The encoder input is already computed and stored
                # in the decoder's KV cache.
                continue

            if not self.is_encoder_decoder:
                # We are not using the encoder cache for encoder-decoder models,
                # yet.
                if request.mm_hashes[i] in mm_hashes_to_schedule:
                    # The same encoder input has already been scheduled in the
                    # current step.
                    continue

                if self.encoder_cache_manager.check_and_update_cache(
                        request, i):
                    # The encoder input is already computed and cached from a
                    # previous step.
                    continue

            # If no encoder input chunking is allowed, we do not want to
            # partially schedule a multimodal item. If the scheduled range would
            # only cover part of the mm input, roll back to before the mm item.
            if (self.scheduler_config.disable_chunked_mm_input
                    and num_computed_tokens < start_pos
                    and (num_computed_tokens + num_new_tokens)
                    < (start_pos + num_encoder_tokens)):
                num_new_tokens = start_pos - num_computed_tokens
                break

            if not self.encoder_cache_manager.can_allocate(
                    request, i, encoder_compute_budget,
                    num_tokens_to_schedule):
                # The encoder cache is full or the encoder budget is exhausted.
                # NOTE(woosuk): We assume that the encoder input tokens should
                # be processed altogether, as the encoder usually uses
                # bidirectional attention.
                if num_computed_tokens < start_pos:
                    # We only schedule the decoder tokens just before the
                    # encoder input.
                    num_new_tokens = start_pos - num_computed_tokens
                else:
                    # Because of prefix caching, num_computed_tokens is greater
                    # than start_pos even though its encoder input is not
                    # available. In this case, we can't schedule any token for
                    # the request in this step.
                    num_new_tokens = 0
                break

            num_tokens_to_schedule += num_encoder_tokens
            encoder_compute_budget -= num_encoder_tokens
            mm_hashes_to_schedule.add(request.mm_hashes[i])
            encoder_inputs_to_schedule.append(i)

        return (
            encoder_inputs_to_schedule,
            num_new_tokens,
            encoder_compute_budget,
        )

    def get_grammar_bitmask(
        self,
        requests: list[Request],
        scheduled_spec_decode_tokens: dict[str, list[int]],
    ):
        # NOTE: structured_output_request_ids maps
        # a request's (request that uses structured output)
        # request_id to its index in the batch.
        # This will help us determine to slice the grammar bitmask
        # and only applies valid mask for requests that
        # uses structured decoding.
        structured_output_request_ids: dict[str, int] = {}
        for i, req in enumerate(requests):
            if req.use_structured_output:
                # PERF: in case of chunked prefill,
                # request might not include any new tokens.
                # Therefore, we might introduce some additional
                # cycle to fill in the bitmask, which could be a big no-op.
                structured_output_request_ids[req.request_id] = i

        if not structured_output_request_ids:
            bitmask = None
        else:
            bitmask = self.structured_output_manager.grammar_bitmask(
                self.requests,
                structured_output_request_ids,
                scheduled_spec_decode_tokens,
            )
        return structured_output_request_ids, bitmask

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> dict[int, EngineCoreOutputs]:
        sampled_token_ids = model_runner_output.sampled_token_ids
        logprobs = model_runner_output.logprobs
        prompt_logprobs_dict = model_runner_output.prompt_logprobs_dict
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        pooler_outputs = model_runner_output.pooler_output
        num_nans_in_logits = model_runner_output.num_nans_in_logits

        outputs: dict[int, list[EngineCoreOutput]] = defaultdict(list)
        spec_decoding_stats: Optional[SpecDecodingStats] = None

        # NOTE(woosuk): As len(num_scheduled_tokens) can be up to 1K or more,
        # the below loop can be a performance bottleneck. We should do our best
        # to avoid expensive operations inside the loop.
        stopped_running_reqs: set[Request] = set()
        stopped_preempted_reqs: set[Request] = set()
        for req_id, num_tokens_scheduled in num_scheduled_tokens.items():
            assert num_tokens_scheduled > 0
            request = self.requests.get(req_id)
            if request is None:
                # The request is already finished. This can happen if the
                # request is aborted while the model is executing it (e.g.,
                # in pipeline parallelism).
                continue

            req_index = model_runner_output.req_id_to_index[req_id]
            generated_token_ids = sampled_token_ids[
                req_index] if sampled_token_ids else []

            scheduled_spec_token_ids = (
                scheduler_output.scheduled_spec_decode_tokens.get(req_id))
            if scheduled_spec_token_ids:
                num_draft_tokens = len(scheduled_spec_token_ids)
                num_accepted = len(generated_token_ids) - 1
                num_rejected = num_draft_tokens - num_accepted
                # num_computed_tokens represents the number of tokens
                # processed in the current step, considering scheduled
                # tokens and rejections. If some tokens are rejected,
                # num_computed_tokens is decreased by the number of rejected
                # tokens.
                request.num_computed_tokens -= num_rejected
                spec_decoding_stats = self.make_spec_decoding_stats(
                    spec_decoding_stats,
                    num_draft_tokens=num_draft_tokens,
                    num_accepted_tokens=num_accepted)

            stopped = False
            new_logprobs = None
            new_token_ids = generated_token_ids
            kv_transfer_params = None
            status_before_stop = request.status

            # Check for stop and update request status.
            if new_token_ids:
                new_token_ids, stopped = self._update_request_with_output(
                    request, new_token_ids)

            # Stop checking for pooler models.
            pooler_output = None
            if pooler_outputs:
                pooler_output = pooler_outputs[req_index]
                stopped = check_stop(request, self.max_model_len,
                                     pooler_output)

            if stopped:
                kv_transfer_params = self._free_request(request)
                if status_before_stop == RequestStatus.RUNNING:
                    stopped_running_reqs.add(request)
                else:
                    stopped_preempted_reqs.add(request)

            # Extract sample logprobs if needed.
            if request.sampling_params is not None \
                and request.sampling_params.logprobs is not None and logprobs:
                # NOTE: once we support N tokens per step (spec decode),
                # the outer lists can be of length > 1.
                new_logprobs = logprobs.slice(req_index, req_index + 1)

            if new_token_ids and self.structured_output_manager.should_advance(
                    request):
                # NOTE: structured_output_request
                # should not be None if use_structured_output, we have
                # checked above, so safe to ignore type warning
                request.structured_output_request.grammar.accept_tokens(  # type: ignore[union-attr]
                    req_id, new_token_ids)

            if num_nans_in_logits is not None and req_id in num_nans_in_logits:
                request.num_nans_in_logits = num_nans_in_logits[req_id]

            # Get prompt logprobs for this request.
            prompt_logprobs_tensors = prompt_logprobs_dict.get(req_id)
            if new_token_ids or pooler_output is not None \
                or kv_transfer_params:

                # Add EngineCoreOutput for this Request.
                outputs[request.client_index].append(
                    EngineCoreOutput(
                        request_id=req_id,
                        new_token_ids=new_token_ids,
                        finish_reason=request.get_finished_reason(),
                        new_logprobs=new_logprobs,
                        new_prompt_logprobs_tensors=prompt_logprobs_tensors,
                        pooling_output=pooler_output,
                        stop_reason=request.stop_reason,
                        events=request.take_events(),
                        kv_transfer_params=kv_transfer_params,
                        trace_headers=request.trace_headers,
                        num_cached_tokens=request.num_cached_tokens,
                    ))
            else:
                # Invariant: EngineCore returns no partial prefill outputs.
                assert not prompt_logprobs_tensors

        # Remove the stopped requests from the running and waiting queues.
        if stopped_running_reqs:
            self.running = remove_all(self.running, stopped_running_reqs)
        if stopped_preempted_reqs:
            # This is a rare case and unlikely to impact performance.
            self.waiting.remove_requests(stopped_preempted_reqs)

        # KV Connector: update state for finished KV Transfers.
        if model_runner_output.kv_connector_output:
            self._update_from_kv_xfer_finished(
                model_runner_output.kv_connector_output)

        # Create EngineCoreOutputs for all clients that have requests with
        # outputs in this step.
        engine_core_outputs = {
            client_index: EngineCoreOutputs(outputs=outs)
            for client_index, outs in outputs.items()
        }

        finished_req_ids = self.finished_req_ids_dict
        if finished_req_ids:
            # Include ids of requests that finished since last outputs
            # were sent.
            for client_index, finished_set in finished_req_ids.items():
                # Set finished request set in EngineCoreOutputs for this client.
                if (eco := engine_core_outputs.get(client_index)) is not None:
                    eco.finished_requests = finished_set
                else:
                    engine_core_outputs[client_index] = EngineCoreOutputs(
                        finished_requests=finished_set)
            finished_req_ids.clear()

        if (stats := self.make_stats(spec_decoding_stats)) is not None:
            # Return stats to only one of the front-ends.
            if (eco := next(iter(engine_core_outputs.values()), None)) is None:
                # We must return the stats even if there are no request
                # outputs this step.
                engine_core_outputs[0] = eco = EngineCoreOutputs()
            eco.scheduler_stats = stats

        return engine_core_outputs

    def _update_request_with_output(
        self,
        request: Request,
        new_token_ids: list[int],
    ) -> tuple[list[int], bool]:
        # Append generated tokens and check for stop. Note that if
        # a request is still being prefilled, we expect the model runner
        # to return empty token ids for the request.
        stopped = False
        for num_new, output_token_id in enumerate(new_token_ids, 1):
            request.append_output_token_ids(output_token_id)

            # Check for stop and update request state.
            # This must be called before we make the EngineCoreOutput.
            stopped = check_stop(request, self.max_model_len)
            if stopped:
                del new_token_ids[num_new:]  # Trim new tokens if needed.
                break
        return new_token_ids, stopped

    def _free_encoder_inputs(self, request: Request) -> None:
        cached_encoder_input_ids = (
            self.encoder_cache_manager.get_cached_input_ids(request))
        # OPTIMIZATION: Avoid list(set) if the set is empty.
        if not cached_encoder_input_ids:
            return

        # Here, we use list(set) to avoid modifying the set while iterating
        # over it.
        for input_id in list(cached_encoder_input_ids):
            mm_positions = request.mm_positions[input_id]
            start_pos = mm_positions.offset
            num_tokens = mm_positions.length
            if self.is_encoder_decoder and request.num_computed_tokens > 0:
                # With Whisper, as soon as we've generated a single token,
                # we know we're done with the encoder input. Cross Attention
                # KVs have been calculated and cached already.
                self.encoder_cache_manager.free_encoder_input(
                    request, input_id)
            elif start_pos + num_tokens <= request.num_computed_tokens:
                # The encoder output is already processed and stored
                # in the decoder's KV cache.
                self.encoder_cache_manager.free_encoder_input(
                    request, input_id)

    def update_draft_token_ids(
        self,
        draft_token_ids: DraftTokenIds,
    ) -> None:
        for req_id, spec_token_ids in zip(
                draft_token_ids.req_ids,
                draft_token_ids.draft_token_ids,
        ):
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                # The request may have been finished. Skip.
                continue

            # Add newly generated spec token ids to the request.
            if not spec_token_ids:
                # NOTE(woosuk): request.spec_token_ids should be updated.
                request.spec_token_ids.clear()
            elif self.structured_output_manager.should_advance(request):
                metadata = request.structured_output_request
                request.spec_token_ids = metadata.grammar.validate_tokens(  # type: ignore[union-attr]
                    spec_token_ids)
            else:
                request.spec_token_ids = spec_token_ids

    def get_request_counts(self) -> tuple[int, int]:
        """Returns (num_running_reqs, num_waiting_reqs)."""
        return len(self.running), len(self.waiting)

    def add_request(self, request: Request) -> None:
        self.waiting.add_request(request)
        self.requests[request.request_id] = request
        if self.log_stats:
            request.record_event(EngineCoreEventType.QUEUED)

    def finish_requests(
        self,
        request_ids: Union[str, Iterable[str]],
        finished_status: RequestStatus,
    ) -> None:
        """Handles the finish signal from outside the scheduler.

        For example, the API server can abort a request when the client
        disconnects.
        """
        assert RequestStatus.is_finished(finished_status)
        if isinstance(request_ids, str):
            request_ids = (request_ids, )
        else:
            request_ids = set(request_ids)

        running_requests_to_remove = set()
        waiting_requests_to_remove = []
        valid_requests = []

        # First pass: collect requests to remove from queues
        for req_id in request_ids:
            request = self.requests.get(req_id)
            if request is None:
                # Invalid request ID.
                continue

            valid_requests.append(request)
            if request.status == RequestStatus.RUNNING:
                running_requests_to_remove.add(request)
            else:
                waiting_requests_to_remove.append(request)

        # Remove all requests from queues at once for better efficiency
        if running_requests_to_remove:
            self.running = remove_all(self.running, running_requests_to_remove)
        if waiting_requests_to_remove:
            self.waiting.remove_requests(waiting_requests_to_remove)

        # Second pass: set status and free requests
        for request in valid_requests:
            request.status = finished_status
            self._free_request(request)

    def _free_request(self, request: Request) -> Optional[dict[str, Any]]:
        assert request.is_finished()

        delay_free_blocks, kv_xfer_params = self._connector_finished(request)
        self.encoder_cache_manager.free(request)
        request_id = request.request_id
        self.finished_req_ids.add(request_id)
        if self.finished_req_ids_dict is not None:
            self.finished_req_ids_dict[request.client_index].add(request_id)

        if not delay_free_blocks:
            self._free_blocks(request)

        return kv_xfer_params

    def _free_blocks(self, request: Request):
        assert request.is_finished()
        self.kv_cache_manager.free(request)
        del self.requests[request.request_id]

    def get_num_unfinished_requests(self) -> int:
        return len(self.waiting) + len(self.running)

    def has_finished_requests(self) -> bool:
        return len(self.finished_req_ids) > 0

    def reset_prefix_cache(self) -> bool:
        return self.kv_cache_manager.reset_prefix_cache()

    def make_stats(
        self,
        spec_decoding_stats: Optional[SpecDecodingStats] = None,
    ) -> Optional[SchedulerStats]:
        if not self.log_stats:
            return None
        prefix_cache_stats = self.kv_cache_manager.make_prefix_cache_stats()
        assert prefix_cache_stats is not None
        return SchedulerStats(
            num_running_reqs=len(self.running),
            num_waiting_reqs=len(self.waiting),
            kv_cache_usage=self.kv_cache_manager.usage,
            prefix_cache_stats=prefix_cache_stats,
            spec_decoding_stats=spec_decoding_stats,
            num_corrupted_reqs=sum(req.is_output_corrupted
                                   for req in self.running),
        )

    def make_spec_decoding_stats(
        self,
        spec_decoding_stats: Optional[SpecDecodingStats],
        num_draft_tokens: int,
        num_accepted_tokens: int,
    ) -> Optional[SpecDecodingStats]:
        if not self.log_stats:
            return None
        if spec_decoding_stats is None:
            spec_decoding_stats = SpecDecodingStats.new(self.num_spec_tokens)
        spec_decoding_stats.observe_draft(
            num_draft_tokens=num_draft_tokens,
            num_accepted_tokens=num_accepted_tokens)
        return spec_decoding_stats

    def shutdown(self) -> None:
        if self.kv_event_publisher:
            self.kv_event_publisher.shutdown()
        if self.connector is not None:
            self.connector.shutdown()

    ########################################################################
    # KV Connector Related Methods
    ########################################################################

    def get_kv_connector(self) -> Optional[KVConnectorBase_V1]:
        return self.connector

    def _connector_finished(
            self, request: Request) -> tuple[bool, Optional[dict[str, Any]]]:
        """
        Invoke the KV connector request_finished() method if applicable.

        Returns optional kv transfer parameters to be included with the
        request outputs.
        """
        if self.connector is None:
            return False, None

        (block_ids, ) = self.kv_cache_manager.get_block_ids(request.request_id)
        return self.connector.request_finished(request, block_ids)

    def _update_waiting_for_remote_kv(self, request: Request) -> bool:
        """
        KV Connector: check if the request_id is finished_recving.

        The finished_recving_kv_req_ids list is populated
        on the previous steps()'s update_from_output based
        on the worker side connector.

        When the kv transfer is ready, we cache the blocks
        and the request state will be moved back to WAITING from
        WAITING_FOR_REMOTE_KV.
        """
        assert self.connector is not None
        if request.request_id not in self.finished_recving_kv_req_ids:
            return False

        # Now that the blocks are ready, actually cache them.
        (block_ids, ) = self.kv_cache_manager.get_block_ids(request.request_id)
        num_computed_tokens = len(block_ids) * self.block_size
        # Handle the case where num request tokens less than one block.
        num_computed_tokens = min(num_computed_tokens, request.num_tokens)
        if num_computed_tokens == request.num_tokens:
            num_computed_tokens -= 1
        # This will cache the blocks iff caching is enabled.
        self.kv_cache_manager.cache_blocks(request, num_computed_tokens)

        # Update the request state for scheduling.
        request.num_computed_tokens = num_computed_tokens

        # Return that we are ready.
        self.finished_recving_kv_req_ids.remove(request.request_id)
        return True

    def _update_from_kv_xfer_finished(self,
                                      kv_connector_output: KVConnectorOutput):
        """
        KV Connector: update the scheduler state based on the output.

        The Worker side connectors add finished_recving and
        finished_sending reqs to the output.
        * if finished_sending: free the blocks
        # if finished_recving: add to state so we can
            schedule the request during the next step.
        """

        if self.connector is not None:
            self.connector.update_connector_output(kv_connector_output)

        # KV Connector:: update recv and send status from last step.
        for req_id in (kv_connector_output.finished_recving or ()):
            logger.debug("Finished recving KV transfer for request %s", req_id)
            self.finished_recving_kv_req_ids.add(req_id)
        for req_id in (kv_connector_output.finished_sending or ()):
            logger.debug("Finished sending KV transfer for request %s", req_id)
            self._free_blocks(self.requests[req_id])
