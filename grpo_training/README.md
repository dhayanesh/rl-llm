# GRPO GSM8K Training Scripts

All scripts use vLLM colocated rollouts, 5 epochs, the HF XML reward format, held-out GSM8K test eval, and Trackio logging.

Single GPU:

```bash
cd /workspace/grpo_training
CUDA_VISIBLE_DEVICES=0 python train_1gpu_baseline.py
```

Four GPU examples:

```bash
cd /workspace/grpo_training
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 4 --main_process_port 29501 train_4gpu_baseline.py
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 4 --main_process_port 29502 train_4gpu_mask_truncated.py
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 4 --main_process_port 29503 train_4gpu_full_no_lora.py
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 4 --main_process_port 29504 train_4gpu_long_completion.py
```

If running two jobs on the same 8-GPU instance, split GPUs and use different ports:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 4 --main_process_port 29501 train_4gpu_baseline.py
CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --num_processes 4 --main_process_port 29502 train_4gpu_mask_truncated.py
```

The most useful Trackio metrics to compare:

- `train/reward` and `eval/reward`
- `train/answers/exact_rate` and `eval/answers/exact_rate`
- `train/completions/clipped_ratio` and `eval/completions/clipped_ratio`
- `train/reward_std`
- `train/frac_reward_zero_std`
- `train/entropy`

