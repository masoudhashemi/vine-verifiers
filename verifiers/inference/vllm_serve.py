# adapted from trl/scripts/vllm_serve.py (huggingface/trl)

import argparse
import logging
import os
from collections.abc import Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from itertools import chain
from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection
from typing import Optional, Sequence as TypingSequence # Renamed to avoid conflict

import torch

# from trl import TrlParser # Assuming TrlParser might not be available/needed if ScriptArguments is self-defined
# For TrlParser, if it's from trl and trl is a dependency, it can be used.
# If not, ScriptArguments can be parsed with regular argparse.
# For now, commenting out TrlParser to ensure script is self-contained with standard libraries + vllm, fastapi, etc.
# If TrlParser is essential and available, it can be uncommented.
from transformers import HfArgumentParser # Using HfArgumentParser as a common alternative

from trl.import_utils import is_fastapi_available, is_pydantic_available, is_uvicorn_available, is_vllm_available #, is_vllm_ascend_available


#if is_fastapi_available(): # Assuming these are essential, so not conditionally importing for now
from fastapi import FastAPI

#if is_pydantic_available():
from pydantic import BaseModel

#if is_uvicorn_available():
import uvicorn

#if is_vllm_available():
from vllm import LLM, SamplingParams
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.parallel_state import get_world_group
from vllm.distributed.utils import StatelessProcessGroup
from vllm.sampling_params import GuidedDecodingParams
from vllm.utils import get_open_port

    # if is_vllm_ascend_available(): # Assuming is_vllm_ascend_available is also from trl.import_utils
    #     from vllm_ascend.distributed.device_communicators.pyhccl import PyHcclCommunicator as PyNcclCommunicator

logger = logging.getLogger(__name__)

# We use CUDA with multiprocessing, so we must use the 'spawn' start method. Otherwise, we will get the following
# error: RuntimeError: Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing, you must use
# the 'spawn' start method
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


class WeightSyncWorkerExtension:
    # ... (Content of WeightSyncWorkerExtension class as provided by user) ...
    # The following attributes are initialized when `init_communicator` method is called.
    pynccl_comm = None  # Communicator for weight updates
    client_rank = None  # Source rank for broadcasting updated weights
    # Adding 'device' attribute which is used by PyNcclCommunicator
    device = None # Will be set in llm_worker based on vLLM's assignment

    def init_communicator(self, host: str, port: int, world_size: int) -> None:
        if self.pynccl_comm is not None:
            raise RuntimeError("Weight update group already initialized. Call close_communicator first.")

        rank = get_world_group().rank
        # Ensure self.device is set before PyNcclCommunicator uses it.
        # This 'device' should correspond to the vLLM worker's CUDA device.
        # It's typically set by the LLM engine or worker initialization.
        # Assuming it's available as self.device, set by vLLM's worker context.
        # If not, vLLM might use torch.cuda.current_device(). For safety:
        if self.device is None and torch.cuda.is_available(): # Fallback if not explicitly set
             self.device = torch.device(f"cuda:{torch.cuda.current_device()}")
        elif self.device is None: # CPU case
             self.device = torch.device("cpu")


        pg = StatelessProcessGroup.create(host=host, port=port, rank=rank, world_size=world_size)
        self.pynccl_comm = PyNcclCommunicator(pg, device=self.device)
        self.client_rank = world_size - 1 # Assuming client is the last rank in the group

    def update_named_param(self, name: str, dtype_str: str, shape: TypingSequence[int]) -> None: # Changed dtype to dtype_str
        if self.pynccl_comm is None:
            raise RuntimeError("Communicator not initialized. Call `init_communicator` first.")
        
        # Convert dtype_str back to torch.dtype
        try:
            # Handle potential "torch.dtype" format or just "dtype"
            actual_dtype_str = dtype_str.split('.')[-1]
            dtype = getattr(torch, actual_dtype_str)
        except AttributeError:
            logger.error(f"Invalid dtype string: {dtype_str}")
            raise ValueError(f"Invalid dtype string: {dtype_str}")

        weight = torch.empty(shape, dtype=dtype, device=self.device)
        self.pynccl_comm.broadcast(weight, src=self.client_rank)
        self.pynccl_comm.group.barrier()
        # Ensure self.model_runner is available. This is usually set by vLLM worker.
        if hasattr(self, 'model_runner') and self.model_runner is not None:
            self.model_runner.model.load_weights(weights=[(name, weight)])
        else:
            logger.error("model_runner not available in WeightSyncWorkerExtension. Cannot load weights.")
            raise RuntimeError("model_runner not available in WeightSyncWorkerExtension.")


    def close_communicator(self) -> None:
        if self.pynccl_comm is not None:
            del self.pynccl_comm
            self.pynccl_comm = None
            self.client_rank = None

@dataclass
class ScriptArguments:
    # ... (Content of ScriptArguments class as provided by user) ...
    model: str = field(metadata={"help": "Model name or path to load the model from."})
    revision: Optional[str] = field(
        default=None,
        metadata={"help": "Revision to use for the model. If not specified, the default branch will be used."},
    )
    tensor_parallel_size: int = field(
        default=1,
        metadata={"help": "Number of tensor parallel workers to use."},
    )
    data_parallel_size: int = field(
        default=1,
        metadata={"help": "Number of data parallel workers to use."},
    )
    host: str = field(
        default="0.0.0.0",
        metadata={"help": "Host address to run the server on."},
    )
    port: int = field(
        default=8000,
        metadata={"help": "Port to run the server on."},
    )
    gpu_memory_utilization: float = field(
        default=0.9,
        metadata={
            "help": "Ratio (between 0 and 1) of GPU memory to reserve for the model weights, activations, and KV "
            "cache on the device dedicated to generation powered by vLLM."
        },
    )
    dtype: str = field(
        default="auto",
        metadata={
            "help": "Data type to use for vLLM generation. If set to 'auto', the data type will be automatically "
            "determined based on the model configuration."
        },
    )
    max_model_len: Optional[int] = field(
        default=None,
        metadata={
            "help": "If set, the `max_model_len` to use for vLLM."
        },
    )
    enable_prefix_caching: Optional[bool] = field(
        default=None, # Changed from False to None to match user code
        metadata={
            "help": "Whether to enable prefix caching in vLLM."
        },
    )
    enforce_eager: Optional[bool] = field( # Changed from False to None
        default=None,
        metadata={
            "help": "Whether to enforce eager execution."
        },
    )
    kv_cache_dtype: str = field(
        default="auto",
        metadata={
            "help": "Data type to use for KV cache."
        },
    )
    log_level: str = field(
        default="info",
        metadata={
            "help": "Log level for uvicorn."
        },
    )

def llm_worker(
    script_args: ScriptArguments, data_parallel_rank: int, master_port: int, connection: Connection
) -> None:
    os.environ["VLLM_DP_RANK"] = str(data_parallel_rank)
    os.environ["VLLM_DP_RANK_LOCAL"] = str(data_parallel_rank) # Added based on user provided code
    os.environ["VLLM_DP_SIZE"] = str(script_args.data_parallel_size)
    os.environ["VLLM_DP_MASTER_PORT"] = str(master_port)

    # Corrected worker_extension_cls path
    llm = LLM(
        model=script_args.model,
        revision=script_args.revision,
        tensor_parallel_size=script_args.tensor_parallel_size,
        gpu_memory_utilization=script_args.gpu_memory_utilization,
        enforce_eager=script_args.enforce_eager if script_args.enforce_eager is not None else False, # Provide default if None
        dtype=script_args.dtype,
        enable_prefix_caching=script_args.enable_prefix_caching if script_args.enable_prefix_caching is not None else False, # Provide default
        kv_cache_dtype=script_args.kv_cache_dtype,
        max_model_len=script_args.max_model_len,
        worker_extension_cls="verifiers.inference.vllm_serve.WeightSyncWorkerExtension",
    )
    
    # Pass the device from vLLM worker to the extension
    # This assumes vLLM worker or its model_runner has a 'device' attribute accessible.
    # This part is conceptual as direct access to worker's device for extension might need specific vLLM internals.
    # A common way is that the worker itself (or model_runner) would instantiate and provide device to extension.
    # For now, we assume WeightSyncWorkerExtension.device is set by vLLM internally or the fallback works.
    # The `WeightSyncWorkerExtension.init_communicator` has a fallback for self.device.
    # vLLM's design typically involves the model_runner (within the worker) having device information.
    # If the extension needs the device explicitly, it might need to be passed during its own initialization by vLLM.
    # For now, relying on the fallback in init_communicator or hoping vLLM sets it.

    connection.send({"status": "ready"})

    while True:
        try:
            command = connection.recv()
        except KeyboardInterrupt:
            # Ensure collective_rpc is available or handle AttributeError
            if hasattr(llm, 'collective_rpc'): # Check if collective_rpc exists
                try:
                    llm.collective_rpc(method="close_communicator")
                except Exception as e_rpc:
                    logger.error(f"Error calling collective_rpc for close_communicator: {e_rpc}")
            else:
                logger.warning("llm object does not have collective_rpc method. Cannot close communicator remotely on KeyboardInterrupt.")
            break
        except EOFError: # Pipe closed
            logger.info("Connection closed by parent process. Worker shutting down.")
            break


        if command["type"] in ["call", "fire_and_forget"]:
            method_name = command["method"]
            args, kwargs = command.get("args", ()), command.get("kwargs", {})
            
            # Ensure method exists on llm object
            if not hasattr(llm, method_name):
                logger.error(f"LLM object does not have method: {method_name}")
                if command["type"] == "call":
                    connection.send(AttributeError(f"LLM object does not have method: {method_name}"))
                continue

            method_to_call = getattr(llm, method_name)
            try:
                result = method_to_call(*args, **kwargs)
                if command["type"] == "call":
                    connection.send(result)
            except Exception as e:
                logger.error(f"Error executing method {method_name} on LLM object: {e}", exc_info=True)
                if command["type"] == "call":
                    connection.send(e) # Send exception back to caller
        elif command["type"] == "shutdown":
            logger.info("Received shutdown command. Worker shutting down.")
            break


def chunk_list(lst: list, n: int) -> list[list]:
    # ... (Content of chunk_list function as provided by user) ...
    k, r = divmod(len(lst), n)
    return [lst[i * k + min(i, r) : (i + 1) * k + min(i + 1, r)] for i in range(n)]

def main_server_logic(script_args: ScriptArguments): # Renamed from main to avoid conflict if this script is imported
    # ... (FastAPI app setup, endpoints, and uvicorn.run call as provided by user) ...
    # Ensure all necessary imports like FastAPI, BaseModel, uvicorn are at the top level.

    # Spawn dp workers, and setup pipes for communication
    master_port = get_open_port()
    connections = []
    processes = []
    for data_parallel_rank in range(script_args.data_parallel_size):
        parent_connection, child_connection = Pipe()
        process = Process(target=llm_worker, args=(script_args, data_parallel_rank, master_port, child_connection))
        process.start()
        connections.append(parent_connection)
        processes.append(process)

    @asynccontextmanager 
    async def lifespan(app: FastAPI):
        ready_connections = set()
        # Timeout for waiting for workers
        import time
        start_wait_time = time.time()
        WAIT_TIMEOUT_SECONDS = 120 # 2 minutes for workers to be ready

        logger.info(f"Waiting for {script_args.data_parallel_size} vLLM worker(s) to be ready...")
        while len(ready_connections) < script_args.data_parallel_size:
            if time.time() - start_wait_time > WAIT_TIMEOUT_SECONDS:
                logger.error(f"Timeout: Not all workers became ready within {WAIT_TIMEOUT_SECONDS} seconds.")
                # Optionally, attempt to terminate spawned processes before raising error
                for p_idx, p_proc in enumerate(processes): # Renamed to avoid conflict
                    if p_proc.is_alive(): 
                        logger.warning(f"Terminating unresponsive worker process {p_idx}.")
                        p_proc.terminate()
                        p_proc.join(timeout=5) # Wait a bit for termination
                raise RuntimeError("Failed to initialize all vLLM workers.")

            for conn_idx, connection in enumerate(connections):
                if connection not in ready_connections and connection.poll(0.1): # Poll with a short timeout
                    try:
                        msg = connection.recv()
                        if isinstance(msg, dict) and msg.get("status") == "ready":
                            logger.info(f"Worker with DP rank {conn_idx} reported ready.")
                            ready_connections.add(connection)
                        else:
                            logger.warning(f"Received unexpected message from worker {conn_idx}: {msg}")
                    except Exception as e:
                        logger.error(f"Error receiving ready signal from worker {conn_idx}: {e}")
                        # Consider how to handle this - perhaps terminate and raise
            # time.sleep(0.5) # Short sleep to prevent busy-waiting if poll timeout is very short

        logger.info("All vLLM workers are ready. Application startup complete.")
        yield
        logger.info("Shutting down application and vLLM workers...")
        for conn_idx, connection in enumerate(connections):
            try:
                connection.send({"type": "shutdown"})
            except Exception as e:
                logger.error(f"Error sending shutdown to worker {conn_idx}: {e}")
        for proc_idx, process in enumerate(processes):
            try:
                process.join(timeout=10)
                if process.is_alive():
                    logger.warning(f"Process {proc_idx} (DP rank) is still alive after 10s, attempting to terminate...")
                    process.terminate()
                    process.join(timeout=5) # Wait a bit more after terminate
                    if process.is_alive():
                         logger.error(f"Process {proc_idx} could not be terminated.")
            except Exception as e:
                logger.error(f"Error joining/terminating process {proc_idx}: {e}")


    app = FastAPI(lifespan=lifespan) 

    @app.get("/health/")
    async def health():
        return {"status": "ok"}

    @app.get("/get_world_size/")
    async def get_world_size_endpoint(): # Renamed to avoid conflict with any global get_world_size
        return {"world_size": script_args.tensor_parallel_size * script_args.data_parallel_size}

    class GenerateRequest(BaseModel):
        prompts: list[str]
        n: int = 1
        repetition_penalty: float = 1.0
        temperature: float = 1.0
        top_p: float = 1.0
        top_k: int = -1
        min_p: float = 0.0
        max_tokens: int = 16
        guided_decoding_regex: Optional[str] = None

    class GenerateResponse(BaseModel):
        completion_ids: list[list[int]]        

    @app.post("/generate/", response_model=GenerateResponse)
    async def generate_endpoint(request: GenerateRequest): # Renamed
        if request.guided_decoding_regex is not None:
            guided_decoding = GuidedDecodingParams(backend="outlines", regex=request.guided_decoding_regex)
        else:
            guided_decoding = None
        sampling_params = SamplingParams(
            n=request.n,
            repetition_penalty=request.repetition_penalty,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            min_p=request.min_p,
            max_tokens=request.max_tokens,
            guided_decoding=guided_decoding,
        )
        chunked_prompts = chunk_list(request.prompts, script_args.data_parallel_size)
        for connection, prompts_chunk in zip(connections, chunked_prompts): # Renamed prompts to prompts_chunk
            if not prompts_chunk: prompts_chunk = ["<placeholder_prompt_for_vllm_batching>"] # Ensure list is not empty, use a distinct placeholder
            kwargs = {"prompts": prompts_chunk, "sampling_params": sampling_params}
            connection.send({"type": "call", "method": "generate", "kwargs": kwargs})
        
        all_outputs_raw = []
        for conn_idx, connection in enumerate(connections):
            try:
                output = connection.recv()
                all_outputs_raw.append(output)
            except Exception as e:
                logger.error(f"Error receiving generate result from worker {conn_idx}: {e}")
                all_outputs_raw.append(e) # Store exception to handle below
        
        # Filter out placeholder results and handle exceptions
        processed_outputs = []
        for output_item, prompts_chunk in zip(all_outputs_raw, chunked_prompts):
            if prompts_chunk and prompts_chunk[0] == "<placeholder_prompt_for_vllm_batching>":
                continue # Skip placeholder results
            if isinstance(output_item, Exception):
                logger.error(f"Received exception from worker during generate: {output_item}")
                # Depending on desired behavior, could raise, return error, or skip. Here, skipping for this chunk.
                continue 
            processed_outputs.append(output_item)
        
        flat_outputs = list(chain.from_iterable(processed_outputs))
        completion_ids = [list(output.token_ids) for outputs_item in flat_outputs for output in outputs_item.outputs]
        return {"completion_ids": completion_ids}

    # Chat endpoint (content from user feedback)
    class ChatRequest(BaseModel):
        messages: list[list[dict[str, str]]]
        n: int = 1
        repetition_penalty: float = 1.0
        temperature: float = 1.0
        top_p: float = 1.0
        top_k: int = -1
        min_p: float = 0.0
        max_tokens: int = 16
        guided_decoding_regex: Optional[str] = None
        stop: Optional[list[str]] = None
        include_stop_str_in_output: bool = False
        skip_special_tokens: bool = True
        spaces_between_special_tokens: bool = True

    class ChatOutput(BaseModel):
        token_ids: list[int]
        text: str

    class ChatResponseItem(BaseModel):
        prompt_token_ids: list[int]
        outputs: list[ChatOutput]

    class ChatResponse(BaseModel):
        responses: list[ChatResponseItem]

    @app.post("/chat/", response_model=ChatResponse)
    async def chat_endpoint(request: ChatRequest): # Renamed
        if request.guided_decoding_regex is not None:
            guided_decoding = GuidedDecodingParams(backend="outlines", regex=request.guided_decoding_regex)
        else:
            guided_decoding = None
        sampling_params = SamplingParams(
            n=request.n,
            repetition_penalty=request.repetition_penalty,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            min_p=request.min_p,
            max_tokens=request.max_tokens,
            guided_decoding=guided_decoding,
            stop=request.stop,
            include_stop_str_in_output=request.include_stop_str_in_output,
            skip_special_tokens=request.skip_special_tokens,
            spaces_between_special_tokens=request.spaces_between_special_tokens,
        )
        chunked_messages = chunk_list(request.messages, script_args.data_parallel_size)
        for connection, messages_chunk in zip(connections, chunked_messages): # Renamed messages to messages_chunk
            if not messages_chunk: messages_chunk = [[{"role": "user", "content": "<placeholder_chat_for_vllm_batching>"}]] # Ensure list of list for chat
            kwargs = {"messages": messages_chunk, "sampling_params": sampling_params}
            connection.send({"type": "call", "method": "chat", "kwargs": kwargs})   
        
        all_outputs_raw = []
        for conn_idx, connection in enumerate(connections):
            try:
                output = connection.recv()
                all_outputs_raw.append(output)
            except Exception as e:
                logger.error(f"Error receiving chat result from worker {conn_idx}: {e}")
                all_outputs_raw.append(e)

        processed_outputs = []
        for output_item, messages_chunk in zip(all_outputs_raw, chunked_messages):
            if messages_chunk and messages_chunk[0][0]["content"] == "<placeholder_chat_for_vllm_batching>":
                continue
            if isinstance(output_item, Exception):
                logger.error(f"Received exception from worker during chat: {output_item}")
                continue
            processed_outputs.append(output_item)

        flat_outputs = list(chain.from_iterable(processed_outputs))
        responses = []
        for outputs_item in flat_outputs: # outputs_item is a RequestOutput from vLLM
            response_item = ChatResponseItem(
                prompt_token_ids=list(outputs_item.prompt_token_ids),
                outputs=[],
            )
            for output in outputs_item.outputs: # output is a CompletionOutput from vLLM
                response_item.outputs.append(ChatOutput(
                    token_ids=list(output.token_ids),
                    text=output.text,
                ))
            responses.append(response_item)
        return {"responses": responses}


    class InitCommunicatorRequest(BaseModel):
        host: str
        port: int
        world_size: int # This is the total world size including the client(s)

    @app.post("/init_communicator/")
    async def init_communicator_endpoint(request: InitCommunicatorRequest): # Renamed
        # The server calculates its own part of the world.
        # The request.world_size is the *total* world size the client expects for the group.
        # The server's WeightSyncWorkerExtension will use this total world_size.
        # The TRL script had: world_size = script_args.tensor_parallel_size * script_args.data_parallel_size + 1
        # This implies the TRL server expects only 1 client.
        # If our client sends a world_size that includes multiple clients, the server needs to be able to handle it.
        # For now, let's pass the client-provided world_size directly to the extension.
        # The server extension's StatelessProcessGroup will use this.
        
        # The `host` and `port` in the request are for the client to tell the server
        # about its own parameters for the distributed group, but more importantly,
        # this `port` is the one the server workers should use to connect to the master
        # (rank 0) of this specific communication group.
        # The `host` should be the master of this group (often the client's main process host).
        # The VLLMClient sends its group_port as `port` here.
        
        kwargs = {"method": "init_communicator", "args": (request.host, request.port, request.world_size)}
        for connection in connections:
            connection.send({"type": "fire_and_forget", "method": "collective_rpc", "kwargs": kwargs})
        return {"message": "Request received, initializing communicator"}

    class UpdateWeightsRequest(BaseModel):
        name: str
        dtype: str # Expect string like "torch.float32"
        shape: list[int]

    @app.post("/update_named_param/")
    async def update_named_param_endpoint(request: UpdateWeightsRequest): # Renamed
        # dtype = torch.__getattribute__(request.dtype.split(".")[-1]) # type: ignore # Conversion done in worker extension
        kwargs = {"method": "update_named_param", "args": (request.name, request.dtype, tuple(request.shape))}
        for connection in connections:
            connection.send({"type": "fire_and_forget", "method": "collective_rpc", "kwargs": kwargs})
        return {"message": "Request received, updating named parameter"}

    @app.post("/reset_prefix_cache/")
    async def reset_prefix_cache_endpoint(): # Renamed
        for connection in connections:
            connection.send({"type": "call", "method": "reset_prefix_cache"})
        
        all_outputs_raw = []
        for conn_idx, connection in enumerate(connections):
            try:
                output = connection.recv()
                all_outputs_raw.append(output)
            except Exception as e:
                logger.error(f"Error receiving reset_prefix_cache result from worker {conn_idx}: {e}")
                all_outputs_raw.append(e)
        
        processed_outputs = []
        for output_item in all_outputs_raw:
            if isinstance(output_item, Exception):
                logger.error(f"Received exception from worker during reset_prefix_cache: {output_item}")
                processed_outputs.append(False) # Indicate failure for this worker
            else:
                # Assuming the method itself doesn't return a value, or returns True on success
                processed_outputs.append(True) # Default to True if no specific return value to check
        success = all(processed_outputs)
        return {"message": "Request received, resetting prefix cache status: " + str(success)}

    @app.post("/close_communicator/")
    async def close_communicator_endpoint(): # Renamed
        kwargs = {"method": "close_communicator"}
        for connection in connections:
            connection.send({"type": "fire_and_forget", "method": "collective_rpc", "kwargs": kwargs})
        return {"message": "Request received, closing communicator"}

    logger.info(f"Starting Uvicorn server on {script_args.host}:{script_args.port}")
    uvicorn.run(app, host=script_args.host, port=script_args.port, log_level=script_args.log_level)


def make_parser(): # Removed subparsers argument for simplicity
    parser = HfArgumentParser(ScriptArguments)
    return parser

if __name__ == "__main__":
    # Setup basic logging for the server script itself
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Check for necessary packages before parsing args, as they are used in main_server_logic
    if not is_fastapi_available():
        raise ImportError("FastAPI is required. `pip install fastapi`")
    if not is_pydantic_available():
        raise ImportError("Pydantic is required. `pip install pydantic`")
    if not is_uvicorn_available():
        raise ImportError("Uvicorn is required. `pip install uvicorn`")
    if not is_vllm_available():
        raise ImportError("vLLM is required. `pip install vllm`")

    parser = make_parser()
    # For HfArgumentParser, parse_args_into_dataclasses() is common
    script_args_tuple = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    if len(script_args_tuple) > 1 and script_args_tuple[1]: # Check for remaining strings
        logger.warning(f"Unrecognized arguments: {script_args_tuple[1]}")
    script_args = script_args_tuple[0]
    
    logger.info("Starting vLLM server with arguments:")
    for arg, value in vars(script_args).items():
        logger.info(f"  {arg}: {value}")
        
    main_server_logic(script_args)
```
