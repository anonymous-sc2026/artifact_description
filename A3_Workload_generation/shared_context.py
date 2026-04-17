# Customize this to the absolute path of the directory containing
# the performance_model folder
prefix_dir = "<absolute_path>"

import threading
import math
import numpy as np
import sys
from multiprocessing import shared_memory
from typing import Tuple

class Algorithm:
    BASELINE = 0
    OUT_OF_ORDER_DISCARD_MOST_URGENT = 1
    OUT_OF_ORDER_DISCARD_LEAST_URGENT = 2
    EDF = 3
    MONITORING = 4

_shared_context = None

class SharedContext:
    def __init__(self):
        self.sub_exec_queue = None
        self.exec_coll_queue = None
        self.start_time= -1

        # last_decode_tps and its lock and updated flag
        self.last_decode_tps = math.inf
        self.last_decode_tps_lock = threading.Lock()
        self.last_decode_tps_updated = True
        
        # nb_running_requests and its lock and updated flag
        self.nb_running_requests = []
        self.nb_running_requests_lock =  []
        self.nb_running_requests_updated = []
        
        # current_in_tokens and its lock and updated flag
        self.current_in_tokens = []
        self.current_in_tokens_lock = []
        self.current_in_tokens_updated = []
        
        # current_out_tokens and its lock and updated flag
        self.current_out_tokens = []
        self.current_out_tokens_lock = []
        self.current_out_tokens_updated = []
        

        self.request_dict = {}

    def add_request(self, request):
        self.request_dict[request["idx"]]=request

    def get_request(self, idx):
           # Use dict.get which returns None by default if idx not found
        return self.request_dict.get(idx)
    
    # -- last_decode_tps methods --
    def set_last_decode_tps(self, value):
        # print(f"last decode tps: {value}")
        with self.last_decode_tps_lock:
            self.last_decode_tps = value
            self.last_decode_tps_updated = True

    def update_last_decode_tps(self, old_value):
        if old_value is None or self.last_decode_tps_updated:
            with self.last_decode_tps_lock:
                self.last_decode_tps_updated = False
                return self.last_decode_tps
        return old_value

    def increment_last_decode_tps(self, amount=1):
        with self.last_decode_tps_lock:
            self.last_decode_tps += amount
            self.last_decode_tps_updated = True
            return self.last_decode_tps

    # -- nb_running_requests methods --
    def init_nb_running_requests(self, nb_adapters):
        nb_adapters = max(1,nb_adapters) # if no adapaters we still need an array of size one 
        self.nb_running_requests = [0] * nb_adapters
        self.nb_running_requests_updated = [True] * nb_adapters
        self.nb_running_requests_lock = [threading.Lock() for _ in range(nb_adapters)]

    def set_nb_running_requests(self, value, adapter):
        with self.nb_running_requests_lock[adapter] :
            self.nb_running_requests[adapter] = value
            self.nb_running_requests_updated[adapter] = True

    
    def update_nb_running_requests(self, old_value, adapter):
        if old_value is None or self.nb_running_requests_updated[adapter]:
            with self.nb_running_requests_lock[adapter] :
                self.nb_running_requests_updated[adapter] = False
                return self.nb_running_requests[adapter]
        return old_value

    def increment_nb_running_requests(self, amount, adapter):
        with self.nb_running_requests_lock[adapter] :
            self.nb_running_requests[adapter] += amount
            self.nb_running_requests_updated[adapter] = True
            return self.nb_running_requests[adapter]

    # -- current_in_tokens methods --
    def init_current_in_tokens(self, nb_adapters):
        nb_adapters = max(1,nb_adapters) # if no adapaters we still need an array of size one 
        self.current_in_tokens = [0] * nb_adapters
        self.current_in_tokens_updated = [True] * nb_adapters
        self.current_in_tokens_lock = [threading.Lock() for _ in range(nb_adapters)]

    def set_current_in_tokens(self, value, adapter):
        with self.current_in_tokens_lock[adapter] :
            self.current_in_tokens[adapter] = value
            self.current_in_tokens_updated[adapter] = True

    def update_current_in_tokens(self, old_value, adapter):
        if old_value is None or self.current_in_tokens_updated[adapter]:
            with self.current_in_tokens_lock[adapter]:
                self.current_in_tokens_updated[adapter] = False
                return self.current_in_tokens[adapter]
        return old_value

    def increment_current_in_tokens(self, amount, adapter):
        with self.current_in_tokens_lock[adapter]:
            self.current_in_tokens[adapter] += amount
            self.current_in_tokens_updated[adapter] = True
            return self.current_in_tokens[adapter]

    # -- current_out_tokens methods --
    def init_current_out_tokens(self, nb_adapters):
        nb_adapters = max(1,nb_adapters) # if no adapaters we still need an array of size one 
        self.current_out_tokens = [0] * nb_adapters
        self.current_out_tokens_updated = [True] * nb_adapters
        self.current_out_tokens_lock = [threading.Lock() for _ in range(nb_adapters)]

    def set_current_out_tokens(self, value, adapter):
        with self.current_out_tokens_lock[adapter] :
            self.current_out_tokens[adapter] = value
            self.current_out_tokens_updated[adapter] = True

    def update_current_out_tokens(self, old_value, adapter):
        if old_value is None or self.current_out_tokens_updated[adapter]:
            with self.current_out_tokens_lock[adapter]:
                self.current_out_tokens_updated[adapter] = False
                return self.current_out_tokens[adapter]
        return old_value

    def increment_current_out_tokens(self, amount, adapter):
        with self.current_out_tokens_lock[adapter]:
            self.current_out_tokens[adapter] += amount
            self.current_out_tokens_updated[adapter] = True
            return self.current_out_tokens[adapter]



import traceback

def get_global_shared_context(msg = "global: "):
    global _shared_context
    # print(f"{msg}{id(_shared_context)=}")
    # if not _shared_context:
    #     print("========> None")
    # print("Stack trace:")
    # traceback.print_stack()
    if _shared_context is None:
        _shared_context = SharedContext()    
    return _shared_context


DEFAULT_SHM_NAME = "my_shared_block"



class SharedData:
    _model_path_to_id  = {
        #start at 1 to avoid confusion with 0 which is the default value in the shared array and means "no model"
        f"{prefix_dir}/performance_model/pickled_model/monitoring-monotonic_nvidia-h100-nvl_1_mistral.pkl": 1, 
        f"{prefix_dir}/performance_model/pickled_model/monitoring-monotonic_nvidia-a100-80gb-pcie_1_mistral.pkl": 2,
        f"{prefix_dir}/performance_model/pickled_model/monitoring-monotonic_nvidia-a100-80gb-pcie_2_mistral.pkl": 3,
        f"{prefix_dir}/performance_model/pickled_model/monitoring-monotonic_nvidia-a100-80gb-pcie_4_qwen.pkl": 4,
        f"{prefix_dir}/performance_model/pickled_model/monitoring-monotonic_nvidia-h100-nvl_1_qwen.pkl": 5,
        f"{prefix_dir}/performance_model/pickled_model/monitoring-monotonic_nvidia-a100-80gb-pcie_4_mistral.pkl": 6,

    }

    _id_to_model_path = {v: k for k, v in _model_path_to_id .items()}

    class _StoredValue:
        num_requests = 0
        start_time=1
        algo_variant=2
        round=3
        prev_forward_pass_duration=4
        prev_round_duration=5
        now=6
        finished=7
        restart = 8
        perf_model = 9
        run_level = 10

    _HEADER_SIZE = 11 # number of elements in the shared array reserved for metadata (like num_requests, start_time, algo_variant, etc) should be 1 more than the highest index in self._StoredValue

    def __init__(self, num_requests, flat_array=None):
        """
        flat_array: 1D numpy array containing concatenated data for all tasks
        """
        self.attached = False
        self.num_requests = num_requests
        self.num_elems = self._HEADER_SIZE + 4 * num_requests # 4 arrays of size num_requests for input tokens, output tokens, deadlines and computed tokens
        if flat_array is not None:
            assert len(flat_array) == self.num_elems
            assert isinstance(flat_array, np.ndarray)
            flat_array[self._StoredValue.num_requests]= self.num_requests
            self.computed_tokens = flat_array[self._HEADER_SIZE+self.num_requests*3:]
            self.input_tokens = flat_array[self._HEADER_SIZE : self._HEADER_SIZE + self.num_requests]
            self.output_tokens = flat_array[self._HEADER_SIZE + self.num_requests  : self._HEADER_SIZE + self.num_requests * 2]
            self.deadlines = flat_array[self._HEADER_SIZE + self.num_requests * 2 : self._HEADER_SIZE + self.num_requests * 3]
            self.attached = True
            self.check()
        self.flat_array = flat_array
    
    def check(self):
        if self.flat_array is None or len(self.flat_array) == 0:
            raise RuntimeError("flat_array not set")
        if len(self.flat_array) != self.num_elems:
            raise RuntimeError(f"flat_array length {len(self.flat_array)} does not match expected num_elems {self.num_elems}")
        if not self.attached:
            raise RuntimeError("flat_array not attached")
    
    def check_attached(self):
        if not self.attached:
            raise RuntimeError("flat_array not attached")

    def restart(self):
        self.check_attached()
        self.flat_array[self._StoredValue.restart] = 1

    def ask_for_restart(self):
        self.check_attached()
        return self.flat_array[self._StoredValue.restart] == 1

    def restart_done(self):
        self.check_attached()
        self.flat_array[self._StoredValue.restart] = 0

    
    def reset(self):
        self.check_attached()
        algo_variant = self.flat_array[self._StoredValue.algo_variant]
        model_id = self.flat_array[self._StoredValue.perf_model]
        self.flat_array.fill(0)
        self.flat_array[self._StoredValue.algo_variant] = int(algo_variant)
        self.flat_array[self._StoredValue.num_requests] = self.num_requests
        self.flat_array[self._StoredValue.perf_model] = int(model_id)
        self.check()
    
    def get_num_elems(self):
        return self.num_elems

    def get_num_requests(self):
        return self.num_requests
    
    def attach_flat_array(self, flat_array):
        assert isinstance(flat_array, np.ndarray)
        assert len(flat_array) == self.num_elems
        self.flat_array = flat_array
        self.flat_array[self._StoredValue.num_requests] = self.num_requests
        self.computed_tokens = flat_array[self._HEADER_SIZE+self.num_requests*3:]
        self.input_tokens = flat_array[self._HEADER_SIZE : self._HEADER_SIZE + self.num_requests]
        self.output_tokens = flat_array[self._HEADER_SIZE + self.num_requests  : self._HEADER_SIZE + self.num_requests * 2]
        self.deadlines = flat_array[self._HEADER_SIZE + self.num_requests * 2 : self._HEADER_SIZE + self.num_requests * 3]
        self.attached= True
        self.check()

    def get_start_time(self):
        self.check_attached()
        return self.flat_array[self._StoredValue.start_time]
    
    def set_start_time(self, start_time):
        self.check_attached()
        self.flat_array[self._StoredValue.start_time] = start_time
    
    def get_num_input_tokens(self, task_id):
        self.check_attached()
        return int(self.input_tokens[task_id])
    
    def set_num_input_tokens(self, task_id, num_tokens):
        self.check_attached()
        self.input_tokens[task_id] = num_tokens

    def get_input_tokens_np_array(self):
        self.check_attached()
        return self.input_tokens
    
    def get_num_output_tokens(self, task_id):
        self.check_attached()
        return int(self.output_tokens[task_id])
    
    def get_output_tokens_np_array(self):
        self.check_attached()
        return self.output_tokens

    def set_num_output_tokens(self, task_id, num_tokens):
        self.check_attached()
        self.output_tokens[task_id] = num_tokens

    def get_deadline(self, task_id):
        self.check_attached()
        return self.deadlines[task_id]

    def get_deadline_np_array(self):
        self.check_attached()
        return self.deadlines
    
    def set_deadline(self, task_id, deadline):
        self.check_attached()
        self.deadlines[task_id] = deadline

    def set_algorithm_variant(self, value):
        self.check_attached()
        self.flat_array[self._StoredValue.algo_variant] = int(value) 

    def get_algorithm_variant(self):
        
        return int(self.flat_array[self._StoredValue.algo_variant])

    def get_computed_tokens(self, ids):
        return self.computed_tokens[ids]

    def record_token_progress(self, round, prev_forward_pass_duration, prev_round_duration, now, request_ids, tokens_progress):
        self.check_attached()
        self.flat_array[self._StoredValue.round] = round
        self.flat_array[self._StoredValue.prev_forward_pass_duration] = prev_forward_pass_duration
        self.flat_array[self._StoredValue.prev_round_duration] = prev_round_duration
        self.flat_array[self._StoredValue.now] = now

        self.computed_tokens[request_ids] += tokens_progress
    
    def fetch_updated_progress(self, nb_tasks):
        self.check_attached()
        assert(nb_tasks <= self.num_requests)
        return self.flat_array[self._StoredValue.round], self.flat_array[self._StoredValue.prev_forward_pass_duration], self.flat_array[self._StoredValue.prev_round_duration], self.flat_array[self._StoredValue.now], self.computed_tokens[:nb_tasks]

    def set_not_finished(self):
        self.check_attached()
        self.flat_array[self._StoredValue.finished] = 0

    def set_finished(self):
        self.check_attached()
        self.flat_array[self._StoredValue.finished] = 1
    
    def is_finished(self):
        self.check_attached()
        return self.flat_array[self._StoredValue.finished] == 1

    def set_perf_model_id(self, model_id):
        self.check_attached()
        self.flat_array[self._StoredValue.perf_model] = model_id

    def get_perf_model_id(self):
        self.check_attached()
        return int(self.flat_array[self._StoredValue.perf_model])

    def get_perf_model_path_and_id(self, hw, model_name):
        model_path = f"{prefix_dir}/performance_model/pickled_model/monitoring-monotonic_{hw}_{model_name}.pkl"
        if model_path not in self._model_path_to_id :
            raise RuntimeError(f"Performance model not found for hw={hw}, model_name={model_name}. Expected path: {model_path}")
        return model_path, self._model_path_to_id[model_path]

    def get_perf_model_path_from_id(self, id):
        return self._id_to_model_path.get(id, None) # return None if not found
        
    def get_perf_model_id_from_path(self, path):
        return self._model_path_to_id .get(path, None) # return None if not found


def create_shared_data_from_shm(shm_name=DEFAULT_SHM_NAME
                                ) -> Tuple[shared_memory.SharedMemory, SharedData]:
    # Attach to the existing shared memory block
    print(f"Trying to connect to '{shm_name}'")
    shm = shared_memory.SharedMemory(name=shm_name)
    print("Success.")
    # Read the number of requests stored in the shared memory
    elem = np.ndarray((1,), dtype=np.float64, buffer=shm.buf)
    nb_requests = int(elem[0])
    print(f"{nb_requests=}")
    
    # Create the SharedData object
    shared_data = SharedData(nb_requests)
    
    # Create the shared array for the number of elements
    shared_array = np.ndarray((shared_data.get_num_elems(),), dtype=np.float64, buffer=shm.buf)
    
    # Attach the shared array to the SharedData object
    shared_data.attach_flat_array(shared_array)
    
    return shm, shared_data


def create_global_shm_shared_data(num_requests, shm_name=DEFAULT_SHM_NAME):

    if num_requests <= 0:
        raise RuntimeError(f"Do not handle (num_requests=)")
    
    try: 
        shm, shared_data = create_shared_data_from_shm(shm_name)
        print("Found another shared memory managed by a vLLM server.")
        if shared_data.num_requests < num_requests:
            print(f"Not enough space in this shared memory. Expected to handled {num_requests} requests but found {shared_data.num_requests}.\nStoping vLLM server...")


        print("Enough space for this shared memory")
        return shm, shared_data, False
    except FileNotFoundError:
        print("Creating a new shm")
        # Create a new shared memery of float64)
        shared_data = SharedData(num_requests)
        num_elems = shared_data.get_num_elems()
        # If it exists from a previous run,  destroy it 
        try:
            shm = shared_memory.SharedMemory(name=shm_name, create=False)
            shm.close()
            shm.unlink()  
        except FileNotFoundError: 
            pass # did not exists 
        shm = shared_memory.SharedMemory(name=shm_name, create=True, size=8 * num_elems)
        array = np.ndarray((num_elems,), dtype=np.float64, buffer=shm.buf)
        array.fill(0)
        shared_data.attach_flat_array(array)

        return shm, shared_data, True
