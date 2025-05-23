import importlib
import atexit
import logging
import time
from typing import Optional
import torch
from torch import nn
from accelerate import Accelerator
from accelerate.state import AcceleratorState
import requests # Added for requests.Session

# Helper functions
def is_requests_available():
    return importlib.util.find_spec("requests") is not None

def is_vllm_available():
    return importlib.util.find_spec("vllm") is not None

def is_vllm_ascend_available():
    if not is_vllm_available():
        return False
    # This is a simplified check. vLLM's actual check might be more intricate.
    # We're looking for a submodule that indicates Ascend support.
    try:
        import vllm_ascend.distributed.device_communicators.pyhccl
        return True
    except ImportError:
        return False

# VLLMClient class
# Placeholder for the VLLMClient class code from the issue description
# I will add the actual VLLMClient code in the next step.
# For now, I'm just creating the file with the helper functions and imports.

logger = logging.getLogger(__name__)

if is_vllm_available():
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.utils import StatelessProcessGroup

    if is_vllm_ascend_available():
        from vllm_ascend.distributed.device_communicators.pyhccl import PyHcclCommunicator as PyNcclCommunicator

class VLLMClient(nn.Module):
    def __init__(
        self,
        model: str, # Model argument kept for potential future use or logging, but not for server starting
        host: Optional[str] = "localhost",
        server_port: Optional[int] = 8000, 
        group_port: Optional[int] = 22222,
        connection_timeout: Optional[int] = 60, # Added connection_timeout
        # Removed vllm_args, hf_home, tensor_parallel_size, max_model_len, gpu_memory_utilization
        # as client does not start the server.
        **kwargs,
    ):
        super().__init__()

        if not is_vllm_available(): # This check can remain as client still uses vLLM components for comms
            raise ImportError("VLLM is not available. Please install VLLM to use VLLMClient's distributed features.")
        if not is_requests_available():
            raise ImportError("Requests is not available. Please install it to use VLLMClient.")

        self.model_name = model # Store model name for reference if needed
        self.host = host
        self.server_port = server_port
        self.group_port = group_port
        self.connection_timeout = connection_timeout # Store connection_timeout
        self.pynccl_comm = None
        self.session = requests.Session()
        self.accelerator = None # Initialize accelerator attribute

        try:
            if AcceleratorState._shared_state != {}:
                self.accelerator = Accelerator()
                logger.info("Using existing Accelerator instance for VLLMClient.")
            else:
                logger.info("VLLMClient: Accelerator not pre-initialized. Client will operate in standalone mode regarding distributed communication setup.")
        except ImportError:
            logger.info("VLLMClient: Accelerate library not found. Client will operate in standalone mode.")
        except Exception as e: # Broad exception to catch any issue with AcceleratorState
            logger.warning(f"VLLMClient: Failed to access Accelerator state, will operate in standalone mode: {e}")
        
        # Client does not start the server. User must ensure server is running.
        # Call check_server to verify, but do not start it.
        if not self.check_server():
             logger.warning(
                f"vLLM server not available at http://{self.host}:{self.server_port}/health. "
                "Please ensure the vLLM server is running."
            )
             # Depending on strictness, could raise an error here or allow user to proceed
             # and call init_communicator later, which might also fail if server is still down.

    def check_server(self) -> bool:
        """Checks if the vLLM server is running."""
        try:
            response = self.session.get(f"http://{self.host}:{self.server_port}/health", timeout=self.connection_timeout)
            return response.status_code == 200
        except requests.exceptions.ConnectionError:
            return False
        except requests.exceptions.Timeout:
            logger.warning(f"Timeout when checking server health at http://{self.host}:{self.server_port}/health.")
            return False


    def init_communicator(self):
        logger.info("Initializing VLLMClient communicator...")
        
        if not self.check_server():
            error_msg = (
                f"vLLM server not available at http://{self.host}:{self.server_port}/health "
                "before attempting to initialize communicator. Please ensure the server is running."
            )
            logger.error(error_msg)
            raise ConnectionError(error_msg)

        try:
            world_size_url = f"http://{self.host}:{self.server_port}/get_world_size/"
            response = self.session.get(world_size_url, timeout=self.connection_timeout)
            response.raise_for_status()
            vllm_world_size = response.json().get("world_size", 1)
            logger.info(f"VLLM server reports world_size: {vllm_world_size}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get world size from VLLM server: {e}. Assuming standalone vLLM (world_size=1) or server misconfiguration.")
            # Critical error, cannot proceed with communicator init without knowing vLLM's world size.
            raise ConnectionError(f"Could not get world size from vLLM server: {e}")

        client_rank_offset = vllm_world_size # Clients will have ranks starting after vLLM server ranks

        if self.accelerator is not None and self.accelerator.state.distributed_type != 'NO':
            num_client_processes = self.accelerator.num_processes
            client_process_rank_in_own_group = self.accelerator.process_index
            self.rank = client_rank_offset + client_process_rank_in_own_group
            effective_world_size = vllm_world_size + num_client_processes
            device_to_use = self.accelerator.device  # Get device from Accelerator
            logger.info(f"VLLMClient (Accelerate): Rank {self.rank}/{effective_world_size}, Device {device_to_use}")
        else:
            self.rank = client_rank_offset # This client is the first (and only) client process
            effective_world_size = vllm_world_size + 1 # vLLM world + this one client
            # Ensure torch is imported for torch.device
            device_to_use = torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")
            logger.info(f"VLLMClient (Standalone): Rank {self.rank}/{effective_world_size}, Device {device_to_use}")

        # The POST request to /init_communicator/ should use the determined parameters.
        # This call should be made by all client processes that will participate.
        # The VLLM server needs an endpoint like /init_communicator/ that allows new processes to join its group.
        # This is a conceptual representation; actual VLLM API for this might differ.
        init_comm_url = f"http://{self.host}:{self.server_port}/init_communicator/"
        try:
            # This payload tells the vLLM server about this client process wanting to join.
            # The vLLM server would use this to establish its side of the connection for this rank.
            # The `host` and `port` here are for the *client's side* of the P2P communication for the process group.
            # This assumes vLLM server can dynamically add ranks to its communicator.
            # A simpler model might be that vLLM server starts with a fixed world_size, and clients just match.
            # For now, we assume a dynamic join is possible.
            # The 'port' here is the shared group_port for this communication group.
            # The server uses this, along with the client's rank and total world_size,
            # to establish its end of the communication channel within the distributed group.
            payload = {
                "host": "0.0.0.0",  # Client host, often for reference by server
                "port": self.group_port, # Shared group port for this communication group
                "rank": self.rank,
                "world_size": effective_world_size
            }
            response = self.session.post(init_comm_url, json=payload, timeout=self.connection_timeout)
            response.raise_for_status()
            logger.info(f"Successfully registered rank {self.rank} with VLLM server communicator using shared group port {self.group_port}.")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to initialize communicator with VLLM server for rank {self.rank}: {e}")
            self.pynccl_comm = None # Ensure comm is None if registration fails
            raise ConnectionError(f"Could not register with VLLM server communicator: {e}")

        # Initialize StatelessProcessGroup and PyNcclCommunicator (or PyHcclCommunicator for Ascend)
        # The host for StatelessProcessGroup.create should be the host of rank 0 of the *entire* group.
        # This is typically the main vLLM server node.
        # The port is self.group_port, which is the rendezvous port for the distributed group.
        pg = None
        try:
            pg = StatelessProcessGroup.create(
                host=self.host, # Host of the master process (vLLM rank 0)
                port=self.group_port, # Shared group port
                rank=self.rank,
                world_size=effective_world_size
            )
            logger.info(f"StatelessProcessGroup created for rank {self.rank} with master at {self.host}:{self.group_port}.")
        except Exception as e: # Catch broad exceptions as StatelessProcessGroup.create can raise various things
            logger.error(f"Failed to create StatelessProcessGroup for rank {self.rank}: {e}")
            self.pynccl_comm = None
            # Attempt to clean up by informing server this rank is closing, if registration was successful
            if hasattr(self, 'rank'): # Check if rank was assigned before failure
                 self._notify_server_close_communicator()
            raise RuntimeError(f"Could not create process group: {e}")


        if pg is not None: # Proceed only if process group creation was successful
            CommunicatorClass = None
            if is_vllm_ascend_available():
                from vllm_ascend.distributed.device_communicators.pyhccl import PyHcclCommunicator
                CommunicatorClass = PyHcclCommunicator
                logger.info("Using PyHcclCommunicator for Ascend NPU.")
            elif is_vllm_available(): # Default to PyNcclCommunicator if vLLM core is available
                from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
                CommunicatorClass = PyNcclCommunicator
                logger.info("Using PyNcclCommunicator for CUDA GPU.")
            else:
                # This case should ideally be caught by the __init__ check, but as a safeguard:
                logger.error("vLLM (core or Ascend) not available for communicator initialization.")
                self.pynccl_comm = None
                if hasattr(self, 'rank'):
                    self._notify_server_close_communicator()
                raise ImportError("Cannot initialize communicator: vLLM (core or Ascend) not available.")

            try:
                self.pynccl_comm = CommunicatorClass(pg, device=device_to_use)
                logger.info(f"{CommunicatorClass.__name__} initialized successfully for rank {self.rank} on device {device_to_use}.")
                # Register cleanup for this communicator
                atexit.register(self.close_communicator)
                logger.info("Registered close_communicator with atexit.")
            except Exception as e:
                logger.error(f"Failed to initialize {CommunicatorClass.__name__} for rank {self.rank} on device {device_to_use}: {e}")
                self.pynccl_comm = None
                if hasattr(self, 'rank'):
                    self._notify_server_close_communicator() # Notify server even if local communicator setup fails after PG
                raise RuntimeError(f"Could not initialize {CommunicatorClass.__name__}: {e}")
        else: # pg is None
            self.pynccl_comm = None
            # No need to call _notify_server_close_communicator here if pg creation failed,
            # as it means the client never fully joined the group from its perspective.
            # The earlier failure in StatelessProcessGroup.create would have already raised an error.


    def _notify_server_close_communicator(self):
        """Internal helper to notify server about closing communicator for this rank."""
        if hasattr(self, 'rank') and self.check_server(): # Only if rank is defined and server is reachable
            close_comm_url = f"http://{self.host}:{self.server_port}/close_communicator/"
            try:
                logger.info(f"Notifying server: closing communicator for rank {self.rank}.")
                self.session.post(close_comm_url, json={"rank": self.rank}, timeout=self.connection_timeout)
            except requests.exceptions.RequestException as e:
                logger.warning(f"Failed to notify server about closing communicator for rank {self.rank}: {e}")
        else:
            logger.debug("Skipping notification to server for communicator closure (no rank or server down).")


    def close_communicator(self):
        logger.info(f"Closing VLLMClient communicator for rank {getattr(self, 'rank', 'N/A')}.")
        self._notify_server_close_communicator() # Notify the server first

        if hasattr(self, 'pynccl_comm') and self.pynccl_comm is not None:
            # PyNcclCommunicator and its underlying StatelessProcessGroup might have cleanup logic.
            # For example, StatelessProcessGroup might call pg.destroy_process_group().
            # Explicitly deleting or calling a close method if available is good practice.
            # Assuming __del__ of PyNcclCommunicator or its process group handles NCCL/HCCLand torch.dist cleanup.
            try:
                # If PyNcclCommunicator or its pg has an explicit close/destroy method, call it here.
                # e.g., if self.pynccl_comm.pg is the torch.distributed.ProcessGroup:
                # if hasattr(self.pynccl_comm, 'pg') and self.pynccl_comm.pg:
                #     torch.distributed.destroy_process_group(self.pynccl_comm.pg)
                # For now, relying on __del__ or internal cleanup of vLLM's comm objects.
                del self.pynccl_comm
                self.pynccl_comm = None
                logger.info(f"Successfully closed and released communicator resources for rank {getattr(self, 'rank', 'N/A')}.")
            except Exception as e:
                logger.error(f"Error during communicator cleanup for rank {getattr(self, 'rank', 'N/A')}: {e}")
        
        # Unregister from atexit to prevent multiple calls if object is later deleted explicitly
        # Note: atexit does not provide a direct way to unregister a specific function object if lambda or partial used.
        # If registered directly like `atexit.register(self.close_communicator)`,
        # it's hard to unregister this specific instance's method.
        # However, atexit handlers are usually called only once at exit.
        # For simplicity, we don't unregister, assuming atexit handles it.


    def update_named_param(self, name: str, weights: torch.Tensor):
        if not self.pynccl_comm: # Check if communicator is initialized
            error_msg = "Communicator not initialized. Call init_communicator() first."
            logger.error(error_msg + " Cannot update model parameters.")
            raise RuntimeError(error_msg)

        if self.accelerator is not None and self.accelerator.state.distributed_type != 'NO':
            if self.accelerator.is_main_process: # Only main process sends HTTP request and initiates broadcast
                logger.info(f"Main client process (global rank {self.rank}) updating named parameter: {name}")
                dtype_str, shape_list = str(weights.dtype), list(weights.shape)
                url = f"http://{self.host}:{self.server_port}/update_named_param/"
                try:
                    # Server expects dtype as string (e.g., "torch.float32") and shape as list
                    response = self.session.post(url, json={"name": name, "dtype": dtype_str, "shape": shape_list}, timeout=self.connection_timeout)
                    response.raise_for_status()
                except requests.exceptions.RequestException as e:
                    logger.error(f"HTTP request for updating {name} failed: {e}")
                    # Propagate a more specific error or handle retry if appropriate
                    raise RuntimeError(f"Failed to inform server about parameter update for {name}: {e}")

                # Broadcast weights to all processes in the combined group (vLLM server + clients)
                # The source rank for broadcast is the global rank of this main client process.
                logger.info(f"Broadcasting weights for {name} from global rank {self.rank} (main client process).")
                self.pynccl_comm.broadcast(weights, src=self.rank)
                logger.info(f"Main client (rank {self.rank}) waiting for combined group barrier after broadcasting {name}.")
                self.pynccl_comm.group.barrier() # Barrier for the entire group
                logger.info(f"Combined group barrier passed for {name} after main client broadcast.")

            # All client processes (including the main one) in the accelerate group must synchronize.
            # This ensures that non-main client processes wait for the main client to complete
            # the HTTP request and the broadcast to the combined (vLLM+clients) group.
            logger.info(f"Client process (local rank {self.accelerator.process_index}, global rank {self.rank}) waiting for accelerate client group barrier for {name}.")
            # Use torch.distributed.barrier if accelerate uses it and it's initialized.
            if self.accelerator.distributed_type != "NO" and torch.distributed.is_initialized():
                 torch.distributed.barrier()
            # Fallback or alternative for other distributed setups if accelerate provides one.
            elif hasattr(self.accelerator, "wait_for_everyone"): # Generic accelerate barrier
                 self.accelerator.wait_for_everyone()
            logger.info(f"Accelerate client group barrier passed for {name} for local rank {self.accelerator.process_index}.")

        else: # Standalone client (not using accelerate for multi-client orchestration)
            logger.info(f"Standalone client (global rank {self.rank}) updating named parameter: {name}")
            dtype_str, shape_list = str(weights.dtype), list(weights.shape)
            url = f"http://{self.host}:{self.server_port}/update_named_param/"
            try:
                response = self.session.post(url, json={"name": name, "dtype": dtype_str, "shape": shape_list}, timeout=self.connection_timeout)
                response.raise_for_status()
            except requests.exceptions.RequestException as e:
                logger.error(f"HTTP request for updating {name} failed: {e}")
                raise RuntimeError(f"Failed to inform server about parameter update for {name}: {e}")

            logger.info(f"Broadcasting weights for {name} from global rank {self.rank} (standalone client).")
            self.pynccl_comm.broadcast(weights, src=self.rank)
            logger.info(f"Standalone client (rank {self.rank}) waiting for combined group barrier after broadcasting {name}.")
            self.pynccl_comm.group.barrier() # Barrier for the entire group
            logger.info(f"Combined group barrier passed for {name} after standalone client broadcast.")


    def update_model_params(self, model_state_dict):
        logger.info("Starting model parameter update process via VLLMClient.")
        if not self.pynccl_comm: # Ensure communicator is ready before iterating
            error_msg = "Communicator not initialized. Call init_communicator() first."
            logger.error(error_msg + " Cannot update model parameters.")
            raise RuntimeError(error_msg)

        for name, param_tensor in model_state_dict.items():
            # Determine target device for the tensor before broadcast
            # This should be the device associated with the communicator (e.g., CUDA device for NCCL)
            target_device = None
            if self.pynccl_comm and hasattr(self.pynccl_comm, 'device'):
                target_device = self.pynccl_comm.device
            elif self.accelerator: # Fallback to accelerator's device if comm device not clear
                target_device = self.accelerator.device
            elif torch.cuda.is_available(): # General fallback to current CUDA device
                target_device = torch.device(f"cuda:{torch.cuda.current_device()}")
            else: # CPU if no CUDA
                target_device = torch.device("cpu")

            # Ensure tensor is on the correct device and is contiguous for communication
            if param_tensor.device != target_device:
                logger.debug(f"Moving parameter {name} from {param_tensor.device} to {target_device} for broadcast.")
                param_tensor = param_tensor.to(target_device)
            
            if not param_tensor.is_contiguous():
                logger.debug(f"Parameter {name} is not contiguous. Making it contiguous for broadcast.")
                param_tensor = param_tensor.contiguous()

            self.update_named_param(name, param_tensor)

        logger.info("Finished all model parameter updates.")


    def generate(self, prompt: str, params: Optional[dict] = None): # Renamed from forward
        if not self.check_server(): # Use check_server
            logger.error("VLLM server is not running. Cannot process generation request.")
            raise ConnectionError("VLLM server is not running. Please ensure it's started and accessible.")

        request_payload = {
            "prompt": prompt,
            # Default VLLM params, can be overridden by 'params' dict
            "use_beam_search": params.get("use_beam_search", False) if params else False,
            "n": params.get("n", 1) if params else 1,
            "temperature": params.get("temperature", 1.0) if params else 1.0,
            "max_tokens": params.get("max_tokens", 128) if params else 128, # Increased default
            # Add other common VLLM parameters here if desired as defaults
        }
        if params: # Allow user to override any default or add more params
            request_payload.update(params)
        
        logger.debug(f"Sending generation request to vLLM server with payload: {request_payload}")

        try:
            response = self.session.post(
                f"http://{self.host}:{self.server_port}/generate", 
                json=request_payload,
                timeout=self.connection_timeout # Use connection_timeout for requests
            )
            response.raise_for_status()  # Raises HTTPError for bad responses (4XX or 5XX)
            return response.json()
        except requests.exceptions.Timeout as e:
            logger.error(f"Timeout during generation request to VLLM server: {e}")
            raise TimeoutError(f"VLLM server request timed out: {e}")
        except requests.exceptions.RequestException as e: # Catches other requests errors (ConnectionError, etc.)
            logger.error(f"Error communicating with VLLM server for generation: {e}")
            raise ConnectionError(f"Communication error with VLLM server: {e}")

    def reset_prefix_cache(self, prefix_id: int):
        """ Resets the prefix cache on the vLLM server for a given prefix_id. """
        if not self.check_server():
            logger.error("VLLM server not running. Cannot reset prefix cache.")
            raise ConnectionError("VLLM server not running.")
        
        reset_url = f"http://{self.host}:{self.server_port}/reset_prefix_cache/"
        try:
            response = self.session.post(reset_url, json={"prefix_id": prefix_id}, timeout=self.connection_timeout)
            response.raise_for_status()
            logger.info(f"Successfully requested prefix cache reset for prefix_id: {prefix_id}")
            return response.json() 
        except requests.exceptions.RequestException as e:
            logger.error(f"Error resetting prefix cache for prefix_id {prefix_id}: {e}")
            raise ConnectionError(f"Failed to reset prefix cache: {e}")


    def __del__(self):
        # atexit is responsible for close_communicator.
        # __del__ should only handle resources not managed by atexit, like the session.
        if hasattr(self, 'session') and self.session:
            try:
                self.session.close()
                logger.debug("Requests session closed in VLLMClient.__del__.")
            except Exception as e:
                logger.warning(f"Exception during session.close() in VLLMClient.__del__: {e}")
        
        # Do NOT call self.close_communicator() here if it's registered with atexit.
        # It can lead to issues, especially if __del__ is called during interpreter shutdown.
        # logger.debug("VLLMClient.__del__ finished.")


# Example Usage (Optional - for testing purposes)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO) # BasicConfig should be called once
    # Setup a more verbose logger for the example if needed
    example_logger = logging.getLogger(__name__) # Use the module's logger
    # Example: Set example_logger level to DEBUG if you want more verbose output from the client itself
    # example_logger.setLevel(logging.DEBUG) 
    # Ensure console handler is set for the root logger to see output
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    # Avoid adding handler if already added by basicConfig or previous runs in interactive session
    if not any(isinstance(h, logging.StreamHandler) for h in logging.getLogger().handlers):
        logging.getLogger().addHandler(console_handler)


    # --- Configuration for the example ---
    VLLM_SERVER_HOST = "localhost"
    VLLM_SERVER_PORT = 8000  # vLLM API server port
    VLLM_GROUP_PORT = 22222 # Shared port for the distributed communication group
    MODEL_NAME = "facebook/opt-125m" # Example model name
    # --- End Configuration ---

    example_logger.info("Starting VLLMClient example...")
    example_logger.info(f"Attempting to connect to vLLM server at {VLLM_SERVER_HOST}:{VLLM_SERVER_PORT}")
    example_logger.info(f"Distributed group port configured to: {VLLM_GROUP_PORT}")

    # Check if running under accelerate
    is_accelerate_launch = False
    try:
        if AcceleratorState._shared_state != {}:
             current_accelerator = Accelerator()
             if current_accelerator.state.distributed_type != 'NO':
                 is_accelerate_launch = True
             example_logger.info(f"Accelerate launch detected: {is_accelerate_launch}, "
                                 f"Num processes: {current_accelerator.num_processes}, "
                                 f"Process index: {current_accelerator.process_index}")
    except (ImportError, Exception): # Catch if accelerate not installed or other issues
        example_logger.info("Accelerator not found or not initialized by user script for this example run.")


    if not (is_vllm_available() and is_requests_available()):
        example_logger.warning("VLLM or Requests library is not available. Skipping VLLMClient example.")
    else:
        client = None # Initialize client to None for finally block
        try:
            example_logger.info(f"Creating VLLMClient instance for model '{MODEL_NAME}'.")
            client = VLLMClient(
                model=MODEL_NAME, # Model name used for reference
                host=VLLM_SERVER_HOST,
                server_port=VLLM_SERVER_PORT,
                group_port=VLLM_GROUP_PORT,
                connection_timeout=30 # Example timeout
            )

            if not client.check_server():
                example_logger.error(
                    f"vLLM server not found at http://{client.host}:{client.server_port}/health. "
                    "Please start the vLLM server with appropriate distributed settings if testing distributed features. "
                    "Example: python -m vllm.entrypoints.api_server --model facebook/opt-125m "
                    f"--host {VLLM_SERVER_HOST} --port {VLLM_SERVER_PORT} "
                    f"--distributed-init-method tcp://{VLLM_SERVER_HOST}:{VLLM_GROUP_PORT} --tensor-parallel-size 1"
                )
                # For this example, we'll exit if server is not running, as other operations will fail.
                # In a real application, you might have retry logic or different handling.
                raise ConnectionError("VLLM Server not running. Example cannot proceed.")

            example_logger.info("VLLM server is running. Initializing communicator...")
            client.init_communicator() # Explicitly initialize communicator
            example_logger.info("Communicator initialized successfully.")

            # Perform operations only on the main process if under accelerate, or always if standalone
            can_perform_main_ops = client.accelerator is None or client.accelerator.is_main_process

            if can_perform_main_ops:
                example_logger.info("Attempting generation task...")
                prompt = "The future of AI is"
                generation_params = {"n": 1, "temperature": 0.7, "max_tokens": 50}
                generated_text = client.generate(prompt, params=generation_params)
                example_logger.info(f"Generated text for '{prompt}': {generated_text}")

            # Distributed model parameter update example
            # All processes (if under accelerate) will participate in this part due to init_communicator
            example_logger.info("Attempting to update model parameters with dummy data (all participating client processes)...")
            
            # Determine device based on client's setup (via init_communicator)
            param_device = client.pynccl_comm.device if client.pynccl_comm else \
                           (client.accelerator.device if client.accelerator else \
                            (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")))

            dummy_state_dict = {
                "layer.weight": torch.randn(16, 16, device=param_device),
                "output.bias": torch.ones(16, device=param_device)
            }
            client.update_model_params(dummy_state_dict)
            example_logger.info("Dummy model parameters update cycle finished for this client process.")

            if can_perform_main_ops:
                example_logger.info("Attempting to reset prefix cache (example)...")
                # This is a dummy prefix_id; in real usage, it would be a valid ID from the server.
                client.reset_prefix_cache(prefix_id=12345) 
                example_logger.info("Prefix cache reset request sent.")


        except ImportError as e:
            example_logger.error(f"ImportError during VLLMClient example: {e}. Ensure all dependencies are installed.")
        except ConnectionError as e: # Catch connection errors specifically
            example_logger.error(f"ConnectionError during VLLMClient example: {e}")
        except RuntimeError as e: # Catch runtime errors (e.g., from communicator setup)
             example_logger.error(f"RuntimeError during VLLMClient example: {e}", exc_info=True)
        except Exception as e: # Catch any other exceptions
            example_logger.error(f"An unexpected error occurred during the VLLMClient example: {e}", exc_info=True)
        finally:
            example_logger.info("VLLMClient example finished for this process.")
            if client and client.accelerator: # Ensure all processes under accelerate sync before exiting
                example_logger.info(f"Client process (local rank {client.accelerator.process_index}) waiting for accelerate barrier before exit.")
                client.accelerator.wait_for_everyone()
            
            # Note: client.close_communicator() is registered with atexit, so it should be called automatically.
            # Explicitly deleting client triggers __del__ which closes the session.
            if client:
                del client 
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache() # Clean up CUDA memory if used
            
            example_logger.info("Exiting example script.")
