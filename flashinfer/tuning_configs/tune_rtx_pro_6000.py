#!/usr/bin/env python3
"""
Autotuning script for FlashInfer CUTLASS MoE kernels on NVIDIA RTX PRO 6000.

Generates a tuning config file by profiling all available kernel tactics
across token-count buckets for the Qwen3-VL-30B-A3B-Instruct-NVFP4 model.

Usage:
    # Option 1: Standalone autotuning (requires GPU with CUDA)
    python tune_rtx_pro_6000.py

    # Option 2: Monkey-patch vLLM serving (see instructions at bottom)

The script will:
1. Profile all tile shape tactics for each batch-size bucket
2. Write the best config to the tuning_configs directory
3. Print instructions for activating the config

After running:
    export FLASHINFER_AUTOTUNER_LOAD_FROM_FILE=1
    # Then launch vLLM as usual
"""

import os
import sys
import time

import torch

# Ensure flashinfer is importable
try:
    from flashinfer.autotuner import AutoTuner, autotune, get_config_path
except ImportError:
    print("ERROR: flashinfer not found. Install it first.")
    sys.exit(1)


def get_gpu_name():
    """Get the GPU name formatted for config filename."""
    if not torch.cuda.is_available():
        print("ERROR: No CUDA GPU available.")
        sys.exit(1)
    name = torch.cuda.get_device_name(0)
    print(f"Detected GPU: {name}")
    return name


def run_autotuning_via_vllm():
    """
    Instructions for autotuning via vLLM monkey-patching.

    This approach loads the full model and exercises the MoE path:

    1. Start a Python session or script with:

        import torch
        from flashinfer.autotuner import AutoTuner

        # Enable tuning mode before vLLM starts processing
        AutoTuner.get().is_tuning_mode = True

    2. Launch vLLM with flashinfer MoE enabled:

        VLLM_USE_FLASHINFER_MOE_FP4=1 \\
        VLLM_FLASHINFER_MOE_BACKEND=throughput \\
        python -m vllm.entrypoints.openai.api_server \\
            --model Qwen/Qwen3-VL-30B-A3B-Instruct-NVFP4 \\
            --tensor-parallel-size 1

    3. Send requests at various batch sizes to trigger tuning across all buckets.
       Use a load generator like:

        for batch in 1 2 4 8 16 32 64 128 256 512 1024; do
            # Send $batch concurrent requests
            python -c "
            import openai, concurrent.futures
            client = openai.OpenAI(base_url='http://localhost:8000/v1', api_key='x')
            def req():
                return client.completions.create(model='Qwen/Qwen3-VL-30B-A3B-Instruct-NVFP4',
                    prompt='Hello world', max_tokens=64)
            with concurrent.futures.ThreadPoolExecutor($batch) as ex:
                list(ex.map(lambda _: req(), range($batch)))
            "
        done

    4. Dump the profiling cache:

        from flashinfer.autotuner import AutoTuner
        cache = AutoTuner.get().profiling_cache
        # Convert to config file format
        dump_profiling_cache(cache)
    """
    print(run_autotuning_via_vllm.__doc__)


def dump_profiling_cache(profiling_cache, output_path=None):
    """
    Convert the AutoTuner's profiling_cache to a config file.

    Args:
        profiling_cache: dict from AutoTuner.get().profiling_cache
        output_path: optional output file path. If None, uses the default
                     config path based on current GPU name.
    """
    if output_path is None:
        output_path = get_config_path(is_module=False)

    best_configs = {}
    for cache_key, (runner_id, tactic_id, profile) in profiling_cache.items():
        # cache_key = (custom_op, runner_class_name, runner_hash, nearest_profile)
        custom_op = cache_key[0]
        runner_name = cache_key[1]
        nearest_profile = cache_key[3]
        config_key = str((custom_op, runner_name, nearest_profile))
        best_configs[config_key] = (runner_id, tactic_id)

    # Write config file
    lines = ["best_configs = {\n"]
    for key in sorted(best_configs.keys()):
        runner_id, tactic_id = best_configs[key]
        lines.append(f"    {key!r}: (\n")
        lines.append(f"        {runner_id},\n")
        lines.append(f"        {tactic_id},\n")
        lines.append(f"    ),\n")
    lines.append("}\n")

    with open(output_path, "w") as f:
        f.writelines(lines)

    print(f"Config written to: {output_path}")
    print(f"Activate with: export FLASHINFER_AUTOTUNER_LOAD_FROM_FILE=1")
    return output_path


def run_standalone_autotuning():
    """
    Run standalone autotuning by directly calling the CUTLASS MoE kernels
    with synthetic inputs matching Qwen3-VL-30B-A3B model dimensions.

    This exercises all tile shape tactics for each batch-size bucket without
    needing the full model loaded.
    """
    try:
        from flashinfer.fused_moe.core import cutlass_fused_moe
    except ImportError:
        print("ERROR: Cannot import cutlass_fused_moe. Ensure flashinfer is installed with CUTLASS MoE support.")
        sys.exit(1)

    gpu_name = get_gpu_name()
    print(f"\nStarting standalone autotuning on {gpu_name}")
    print("This will profile all kernel tactics across batch-size buckets.\n")

    # Qwen3-VL-30B-A3B-Instruct-NVFP4 model dimensions
    hidden_size = 3584
    intermediate_size = 18944  # actual FFN intermediate size
    num_experts = 128
    top_k = 8  # number of experts selected per token

    # Token count buckets to profile
    buckets = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]

    # Enable tuning mode
    autotuner = AutoTuner.get()
    autotuner.is_tuning_mode = True
    print("Tuning mode enabled.\n")

    device = torch.device("cuda:0")

    for num_tokens in buckets:
        print(f"Profiling bucket: num_tokens={num_tokens} ...", end=" ", flush=True)
        start = time.time()

        try:
            # Create synthetic inputs
            input_tensor = torch.randn(
                num_tokens, hidden_size, dtype=torch.bfloat16, device=device
            )

            # Expert selection: each token selects top_k experts
            token_selected_experts = torch.randint(
                0, num_experts, (num_tokens, top_k), dtype=torch.int32, device=device
            )

            # Final scales (routing weights)
            token_final_scales = torch.rand(
                num_tokens, top_k, dtype=torch.bfloat16, device=device
            )

            # FP4 quantized weights (packed)
            # Gate+Up projection: hidden_size -> intermediate_size * 2
            # Down projection: intermediate_size -> hidden_size
            # For NVFP4, weights are packed as uint8 (2 values per byte)
            w1_packed = torch.randint(
                0, 256, (num_experts, intermediate_size * 2, hidden_size // 2),
                dtype=torch.uint8, device=device
            )
            w2_packed = torch.randint(
                0, 256, (num_experts, hidden_size, intermediate_size // 2),
                dtype=torch.uint8, device=device
            )

            # FP4 scale factors
            # Scale group size is typically 128 for NVFP4
            scale_group_size = 128
            w1_scale = torch.rand(
                num_experts, intermediate_size * 2, hidden_size // scale_group_size,
                dtype=torch.bfloat16, device=device
            )
            w2_scale = torch.rand(
                num_experts, hidden_size, intermediate_size // scale_group_size,
                dtype=torch.bfloat16, device=device
            )

            # Input scale for activation quantization
            a1_scale = torch.ones(1, dtype=torch.float32, device=device)
            a2_scale = torch.ones(1, dtype=torch.float32, device=device)

            # Run the fused MoE with autotuning enabled
            output = cutlass_fused_moe(
                input=input_tensor,
                token_selected_experts=token_selected_experts,
                token_final_scales=token_final_scales,
                w1=w1_packed,
                w2=w2_packed,
                w1_scale=w1_scale,
                w2_scale=w2_scale,
                a1_scale=a1_scale,
                a2_scale=a2_scale,
            )

            torch.cuda.synchronize()
            elapsed = time.time() - start
            print(f"done ({elapsed:.1f}s)")

        except Exception as e:
            print(f"FAILED: {e}")
            continue

    # Dump results
    print(f"\nAutotuning complete. Profiling cache has {len(autotuner.profiling_cache)} entries.")

    if autotuner.profiling_cache:
        output_path = dump_profiling_cache(autotuner.profiling_cache)
        print(f"\nConfig file written to: {output_path}")
    else:
        print("\nWARNING: No profiling results collected.")
        print("This may happen if the kernel inputs don't match expected dimensions.")
        print("Try the vLLM monkey-patch approach instead (pass --vllm-instructions).")

    # Disable tuning mode
    autotuner.is_tuning_mode = False


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Autotune FlashInfer CUTLASS MoE kernels for RTX PRO 6000"
    )
    parser.add_argument(
        "--vllm-instructions", action="store_true",
        help="Print instructions for autotuning via vLLM monkey-patching"
    )
    parser.add_argument(
        "--dump-cache", action="store_true",
        help="Dump the current AutoTuner profiling cache to a config file"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output path for the config file (default: auto-detect based on GPU)"
    )
    args = parser.parse_args()

    if args.vllm_instructions:
        run_autotuning_via_vllm()
    elif args.dump_cache:
        autotuner = AutoTuner.get()
        dump_profiling_cache(autotuner.profiling_cache, args.output)
    else:
        run_standalone_autotuning()


if __name__ == "__main__":
    main()
