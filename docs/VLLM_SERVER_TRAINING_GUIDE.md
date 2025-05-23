# Training VINE-PPO with a VLLM Server

## Introduction

This guide explains how to use the VINE-PPO trainers (`VinePPOTrainer`, `VinePPOMCTSTrainer`) with an external VLLM (Very Large Language Model) server. Offloading generation to a dedicated VLLM server can be beneficial for:
-   Reducing memory footprint on training GPUs.
-   Potentially speeding up generation, especially when training with multiple processes.
-   Allowing the use of larger models for generation than might fit alongside the training process.

The `VLLMClient` utility (`verifiers/utils/vllm_client.py`) facilitates communication with this server.

## VLLM Server Setup

The project includes a compatible VLLM server script located at `verifiers/inference/vllm_serve.py`. This script is an adaptation of TRL's vLLM server and includes the necessary extensions for weight synchronization.

**1. Server Script:**
   `verifiers/inference/vllm_serve.py`

**2. Key Server-Side Components:**
   - It uses `vllm.LLM` for model serving.
   - It includes a `WeightSyncWorkerExtension` which handles:
     - Initializing a communication group (`StatelessProcessGroup`) that training client processes can join.
     - Receiving and applying model weight updates broadcast by the training client.
   - The server exposes HTTP endpoints that `VLLMClient` interacts with, including:
     - `/health`, `/generate`, `/get_world_size`, `/init_communicator`, `/update_named_param`, `/close_communicator`, `/reset_prefix_cache`.

**3. Starting the Server:**
   You need to run this script using Python, specifying the model and other configurations.
   Example:
   ```bash
   # Ensure your virtual environment with all dependencies is active
   # Adjust CUDA_VISIBLE_DEVICES to target the GPUs for the VLLM server
   CUDA_VISIBLE_DEVICES=0,1 python verifiers/inference/vllm_serve.py \
       --model "Qwen/Qwen2.5-7B-Instruct" \
       --host "0.0.0.0" \
       --port 8000 \
       --tensor_parallel_size 2 \
       --data_parallel_size 1 \
       --max_model_len 8192 \
       --gpu_memory_utilization 0.9 \
       --enable_prefix_caching True
   ```
   -   `--model`: The Hugging Face model identifier or path.
   -   `--host`: Host for the API server.
   -   `--port`: Port for the API server (e.g., 8000). This is what `--vllm_server_port` in the training script connects to for API calls.
   -   `--tensor_parallel_size` / `--data_parallel_size`: Configure vLLM's internal parallelism.
   -   **Important for Weight Synchronization Group**: The `WeightSyncWorkerExtension` within `verifiers/inference/vllm_serve.py` initializes its communication group. The `VLLMClient` will join this group. The server script currently defines the total world size for this group as `tensor_parallel_size * data_parallel_size + 1`. This means it expects its own (TP\*DP) processes plus **one** external client process to join for weight synchronization.

**4. Group Initialization Port (`--vllm_group_port` for the client):**
   The `VLLMClient` needs a `group_port` to initialize `StatelessProcessGroup`. This port is used for the rendezvous of the weight synchronization communication group. The `verifiers/inference/vllm_serve.py` script's `InitCommunicatorRequest` (within the FastAPI app) takes a `port` argument. **This `port` value specified by the client during the `/init_communicator` call is what the server's `WeightSyncWorkerExtension` will use to create its `StatelessProcessGroup`.** Therefore, the `--vllm_group_port` used when running your training script must be a free port that all vLLM server worker processes and the main training client process can bind to for setting up this distributed group. Example: `22222`.

## Trainer Configuration (Client-Side)

To use the VLLM server, configure your training script (e.g., `verifiers/examples/train_vineppo.py` or `verifiers/examples/train_vineppo_mcts.py`) with the following command-line arguments:

-   `--use_vllm_server`: A boolean flag to enable using the VLLM server.
-   `--vllm_host <server_host>`: Hostname or IP address of the machine running the VLLM API server (e.g., `localhost` if on the same machine).
-   `--vllm_server_port <server_api_port>`: The API port of the VLLM server (e.g., `8000`).
-   `--vllm_group_port <dist_group_port>`: The port to be used for initializing the `StatelessProcessGroup` for weight synchronization (e.g., `22222`). This must be a port accessible by the VLLM server workers and the main training client process.
-   `--vllm_connection_timeout <seconds>`: Connection timeout for server requests.

## Running Training with `accelerate`

You can (and generally should for multi-GPU training) launch your training script using `accelerate`:

```bash
# Example for VinePPOTrainer
# Adjust CUDA_VISIBLE_DEVICES for training GPUs, distinct from server GPUs if needed
CUDA_VISIBLE_DEVICES=2,3 accelerate launch --config_file configs/zero3.yaml --num_processes 2 \
    verifiers/examples/train_vineppo.py \
    --use_vllm_server \
    --vllm_host "localhost" \
    --vllm_server_port 8000 \
    --vllm_group_port 22222 \
    [...other arguments for the trainer...]
```

-   Ensure you have configured `accelerate` using `accelerate config` if you haven't already.
-   `--num_processes` will launch that many training client processes.

## Workflow and Multi-GPU Considerations

1.  **Initialization**:
    *   Each training process launched by `accelerate` will create a `VLLMClient` instance.
    *   **Crucially**, due to the current `verifiers/inference/vllm_serve.py` expecting only one external client for its weight synchronization group, the `vllm_client.init_communicator()` method (which sets up `pynccl_comm`) is called **only by the main training process** (where `accelerator.is_main_process` is true, or if not using `accelerate`).
    *   Other training processes will have a `VLLMClient` instance but won't initialize this specific communication channel for weight synchronization. They can still use the client for API calls like `generate`.

2.  **Monte Carlo Rollouts (Generation Offloading)**:
    *   All training processes (main or not) can use `vllm_client.generate()`. This method makes simple HTTP requests to the VLLM server's `/generate` endpoint.
    *   This allows generation to be offloaded in parallel by all training processes to the VLLM server, utilizing multiple training GPUs for handling other parts of the PPO algorithm while the server handles generation.

3.  **Model Weight Updates**:
    *   After local PPO updates, the model weights are synchronized across all training processes by `accelerate`.
    *   The `_update_vllm_model` method in the trainers is guarded to run only on the main training process.
    *   This main process then calls `vllm_client.update_model_params()`. The `VLLMClient` ensures only this main process sends the updated weights to the VLLM server (via HTTP call to `/update_named_param/`) and then initiates a broadcast *within the server's weight sync group* using its initialized `pynccl_comm`.

**In summary**: This setup enables multi-GPU training where generation is parallelized by offloading to the VLLM server. Weight synchronization back to the VLLM server is managed by the main training process, compatible with the current design of `verifiers/inference/vllm_serve.py`.
```
