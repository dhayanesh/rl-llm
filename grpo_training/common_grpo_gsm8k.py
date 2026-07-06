import datetime
import logging
import math
import re
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer


logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("gradio_client").setLevel(logging.WARNING)


MODEL_DIR = Path("/workspace/gemma-3-1b-it")
GSM8K_DIR = Path("/workspace/gsm8k-grade-school-math-8k-dataset/gsm8k/main")
PROJECT_NAME = "GRPO-Mathematical-Reasoning"

SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""


def extract_hash_answer(text):
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


def process_dataset_example(example):
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["question"]},
        ],
        "answer": extract_hash_answer(example["answer"]),
    }


def load_gsm8k_train_dataset():
    dataset = load_dataset(
        "parquet",
        data_files={"train": str(GSM8K_DIR / "train-00000-of-00001.parquet")},
        split="train",
    )
    return dataset.map(process_dataset_example)


def load_gsm8k_eval_dataset(max_examples=128, seed=3407):
    dataset = load_dataset(
        "parquet",
        data_files={"test": str(GSM8K_DIR / "test-00000-of-00001.parquet")},
        split="test",
    )
    dataset = dataset.map(process_dataset_example)
    if max_examples is not None and len(dataset) > max_examples:
        dataset = dataset.shuffle(seed=seed).select(range(max_examples))
    return dataset


def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


def completion_texts(completions):
    return [completion[0]["content"] for completion in completions]


match_format = re.compile(
    r"^\s*<reasoning>.*?</reasoning>\s*<answer>\s*(.*?)\s*</answer>\s*$",
    flags=re.DOTALL,
)


def correctness_reward_func(prompts, completions, answer, **kwargs):
    responses = completion_texts(completions)
    extracted_responses = [extract_xml_answer(response) for response in responses]
    return [2.0 if response == true_answer else 0.0 for response, true_answer in zip(extracted_responses, answer)]


def int_reward_func(completions, **kwargs):
    responses = completion_texts(completions)
    extracted_responses = [extract_xml_answer(response) for response in responses]
    return [0.5 if response.isdigit() else 0.0 for response in extracted_responses]


def strict_format_reward_func(completions, **kwargs):
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = completion_texts(completions)
    matches = [re.match(pattern, response, flags=re.DOTALL) for response in responses]
    return [0.5 if match else 0.0 for match in matches]


def soft_format_reward_func(completions, **kwargs):
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = completion_texts(completions)
    matches = [re.match(pattern, response, flags=re.DOTALL) for response in responses]
    return [0.5 if match else 0.0 for match in matches]


def count_xml(text):
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1]) * 0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1) * 0.001
    return count


def xmlcount_reward_func(completions, **kwargs):
    contents = completion_texts(completions)
    return [count_xml(content) for content in contents]


REWARD_FUNCS = [
    xmlcount_reward_func,
    soft_format_reward_func,
    strict_format_reward_func,
    int_reward_func,
    correctness_reward_func,
]


def load_model_and_tokenizer(use_lora=True, lora_rank=32):
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR),
        trust_remote_code=True,
        dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if use_lora:
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_rank,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_dropout=0.1,
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:
        model.print_trainable_parameters() if hasattr(model, "print_trainable_parameters") else None

    return model, tokenizer


def build_training_args(
    run_name,
    output_dir,
    *,
    num_train_epochs=5,
    max_completion_length=768,
    mask_truncated_completions=False,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    num_generations=8,
    num_generations_eval=4,
    eval_steps=250,
    vllm_gpu_memory_utilization=0.3,
    vllm_max_model_length=1280,
):
    return GRPOConfig(
        learning_rate=5e-6,
        max_grad_norm=0.1,
        bf16=True,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_generations=num_generations,
        num_generations_eval=num_generations_eval,
        max_completion_length=max_completion_length,
        mask_truncated_completions=mask_truncated_completions,
        num_train_epochs=num_train_epochs,
        eval_strategy="steps",
        eval_steps=eval_steps,
        eval_on_start=True,
        save_strategy="steps",
        save_steps=250,
        logging_steps=1,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_tensor_parallel_size=1,
        vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
        vllm_max_model_length=vllm_max_model_length,
        vllm_enable_sleep_mode=False,
        output_dir=output_dir,
        run_name=run_name,
        project=PROJECT_NAME,
        report_to="trackio",
        log_on_each_node=False,
    )


def attach_essential_metrics(trainer, tokenizer):
    import types

    def attach_reward_std_filter():
        if getattr(trainer, "_reward_func_std_filter_attached", False):
            return

        original_log = trainer.log

        def log_without_reward_func_std(self, logs, start_time=None):
            for split in ["train", "eval"]:
                metric_store = getattr(self, "_metrics", {}).get(split, {})
                for key in list(metric_store.keys()):
                    if key.startswith("rewards/") and key.endswith("/std"):
                        metric_store.pop(key, None)
            return original_log(logs, start_time)

        trainer.log = types.MethodType(log_without_reward_func_std, trainer)
        trainer._reward_func_std_filter_attached = True

    attach_reward_std_filter()

    if getattr(trainer, "_essential_grpo_metrics_attached", False):
        return trainer

    original_calculate_rewards = trainer._calculate_rewards

    def log_metric(store, name, value):
        if value is None:
            return
        if isinstance(value, torch.Tensor):
            value = value.detach().float().item()
        if isinstance(value, (int, float)) and math.isfinite(value):
            store[name].append(float(value))

    def completion_text(completion):
        if isinstance(completion, list) and completion and isinstance(completion[0], dict):
            return completion[0].get("content", "")
        return str(completion)

    def clean_number(text):
        if text is None:
            return None
        text = str(text).strip().replace(",", "")
        text = re.sub(r"\s+", "", text)
        return text or None

    def calculate_rewards_with_metrics(self, inputs, prompts, completions, completion_ids_list):
        rewards_per_func = original_calculate_rewards(inputs, prompts, completions, completion_ids_list)
        store = self._pending_metrics

        lengths = [len(ids) for ids in completion_ids_list]
        if lengths:
            log_metric(store, "completions/mean_length", sum(lengths) / len(lengths))
            log_metric(store, "completions/max_length", max(lengths))

            max_len = getattr(self.args, "max_completion_length", None)
            if max_len is not None:
                eos_or_pad = {tokenizer.eos_token_id, tokenizer.pad_token_id}
                clipped = []
                for ids in completion_ids_list:
                    hit_length_limit = len(ids) >= max_len
                    ended_cleanly = bool(ids) and ids[-1] in eos_or_pad
                    clipped.append(float(hit_length_limit and not ended_cleanly))
                log_metric(store, "completions/clipped_ratio", sum(clipped) / len(clipped))

        responses = [completion_text(completion) for completion in completions]
        if responses:
            exact_format = []
            extracted_answers = []
            exact_answers = []

            for response, example in zip(responses, inputs):
                format_match = match_format.search(response)
                exact_format.append(format_match is not None)

                predicted = format_match.group(1) if format_match else None
                predicted = clean_number(predicted)
                expected = clean_number(example.get("answer"))

                extracted_answers.append(predicted is not None)
                exact_answers.append(predicted is not None and predicted == expected)

            log_metric(store, "format/exact_rate", sum(exact_format) / len(exact_format))
            log_metric(store, "answers/extracted_rate", sum(extracted_answers) / len(extracted_answers))
            log_metric(store, "answers/exact_rate", sum(exact_answers) / len(exact_answers))

        return rewards_per_func

    trainer._calculate_rewards = types.MethodType(calculate_rewards_with_metrics, trainer)
    trainer._essential_grpo_metrics_attached = True
    return trainer


def train_variant(
    *,
    run_name,
    output_dir,
    use_lora=True,
    max_completion_length=768,
    mask_truncated_completions=False,
    vllm_max_model_length=1280,
):
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"{run_name}-{timestamp}"

    print(f"Run name: {run_name}")
    print(f"Model: {MODEL_DIR}")
    print(f"Output: {output_dir}")
    print(f"LoRA: {use_lora}")
    print(f"max_completion_length: {max_completion_length}")
    print(f"mask_truncated_completions: {mask_truncated_completions}")

    train_dataset = load_gsm8k_train_dataset()
    eval_dataset = load_gsm8k_eval_dataset(max_examples=128)
    model, tokenizer = load_model_and_tokenizer(use_lora=use_lora)

    args = build_training_args(
        run_name=run_name,
        output_dir=output_dir,
        max_completion_length=max_completion_length,
        mask_truncated_completions=mask_truncated_completions,
        vllm_max_model_length=vllm_max_model_length,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=REWARD_FUNCS,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer = attach_essential_metrics(trainer, tokenizer)
    trainer.train()

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    try:
        import trackio

        trackio.finish()
    except Exception as exc:
        print(f"Trackio finish skipped: {exc}")
