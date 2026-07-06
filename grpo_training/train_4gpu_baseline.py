from common_grpo_gsm8k import train_variant


if __name__ == "__main__":
    train_variant(
        run_name="4gpu-baseline-lora-vllm",
        output_dir="/workspace/grpo_runs/4gpu_baseline_lora_vllm",
        use_lora=True,
        max_completion_length=768,
        mask_truncated_completions=False,
        vllm_max_model_length=1280,
    )

