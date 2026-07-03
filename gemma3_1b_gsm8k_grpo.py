import datetime
import logging
import re

import torch
import trackio
from datasets import load_dataset
from huggingface_hub.errors import GatedRepoError
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import GRPOConfig, GRPOTrainer


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("gradio_client").setLevel(logging.WARNING)


# Gemma 3 1B settings adapted from the HF LLM course GRPO + Unsloth exercise.
model_name = "google/gemma-3-1b-it"
max_seq_length = 1024
max_prompt_length = 256
lora_rank = 32


# GSM8K structured reasoning format copied from the HF TRL GRPO cookbook.
reasoning_start = "<start_working_out>"
reasoning_end = "<end_working_out>"
solution_start = "<SOLUTION>"
solution_end = "</SOLUTION>"

system_prompt = f"""You are a mathematical reasoning assistant.
When given a math problem:
1. Show your step-by-step work between {reasoning_start} and {reasoning_end}
2. Provide your final numerical answer between {solution_start} and {solution_end}
3. Be precise and show all calculation steps clearly."""


def print_gpu_environment():
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"Current GPU: {torch.cuda.current_device()}")
        print(f"GPU name: {torch.cuda.get_device_name()}")
        print(
            f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )
    else:
        print("No GPU available. This script is intended for GPU GRPO training.")


def extract_hash_answer(text):
    """Extract numerical answer from GSM8K format (#### marker)."""
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


def process_dataset_example(example):
    """Convert GSM8K example to conversation format for GRPO training."""
    question = example["question"]
    answer = extract_hash_answer(example["answer"])
    prompt = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    return {
        "prompt": prompt,
        "answer": answer,
    }


match_format = re.compile(
    rf"^[\s]{{0,}}"
    rf"{reasoning_start}.+?{reasoning_end}.*?"
    rf"{solution_start}(.+?){solution_end}"
    rf"[\s]{{0,}}$",
    flags=re.MULTILINE | re.DOTALL,
)
match_numbers = re.compile(
    rf"{solution_start}.*?([\d\.]{{1,}})",
    flags=re.MULTILINE | re.DOTALL,
)


def match_format_exactly(completions, **kwargs):
    """High reward for perfect format adherence."""
    scores = []
    for completion in completions:
        response = completion[0]["content"]
        score = 3.0 if match_format.search(response) is not None else 0.0
        scores.append(score)
    return scores


def match_format_approximately(completions, **kwargs):
    """Graduated scoring for format elements."""
    scores = []
    for completion in completions:
        response = completion[0]["content"]
        score = 0
        score += 0.5 if response.count(reasoning_start) == 1 else -0.5
        score += 0.5 if response.count(reasoning_end) == 1 else -0.5
        score += 0.5 if response.count(solution_start) == 1 else -0.5
        score += 0.5 if response.count(solution_end) == 1 else -0.5
        scores.append(score)
    return scores


def check_answer_correctness(prompts, completions, answer, **kwargs):
    """Graduated scoring for mathematical accuracy."""
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [
        guess.group(1) if (guess := match_format.search(r)) is not None else None
        for r in responses
    ]

    scores = []
    for guess, true_answer in zip(extracted_responses, answer):
        if guess is None:
            scores.append(0)
            continue
        if guess.strip() == true_answer.strip():
            scores.append(3.0)
        else:
            try:
                ratio = float(guess) / float(true_answer)
                if 0.9 <= ratio <= 1.1:
                    scores.append(1.5)
                elif 0.8 <= ratio <= 1.2:
                    scores.append(0.5)
                else:
                    scores.append(-0.5)
            except (ValueError, ZeroDivisionError):
                scores.append(-0.5)
    return scores


def check_numbers_extraction(prompts, completions, answer, **kwargs):
    """Tests the model's ability to extract numerical values from solution sections."""
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [
        guess.group(1) if (guess := match_numbers.search(r)) is not None else None
        for r in responses
    ]

    scores = []
    for guess, true_answer in zip(extracted_responses, answer):
        if guess is None:
            scores.append(0)
            continue
        try:
            true_val = float(true_answer.strip())
            guess_val = float(guess.strip())
            scores.append(1.5 if guess_val == true_val else 0.0)
        except (ValueError, TypeError):
            scores.append(0)
    return scores


def load_gsm8k_dataset():
    print("Loading GSM8K mathematical reasoning dataset...")
    dataset = load_dataset("openai/gsm8k", "main", split="train")
    dataset = dataset.map(process_dataset_example)

    assert "prompt" in dataset.column_names
    assert "answer" in dataset.column_names
    assert len(dataset[0]["prompt"]) == 2
    assert dataset[0]["answer"] is not None

    print("Dataset loaded and processed.")
    print(f"Training examples: {len(dataset):,}")
    print(f"Sample question: {dataset[0]['prompt'][1]['content']}...")
    print(f"Sample answer: {dataset[0]['answer']}")
    return dataset


def load_model_and_tokenizer():
    print(f"Loading model: {model_name}")
    print(f"Max sequence length: {max_seq_length}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    print("4-bit quantization configured.")

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
    except (GatedRepoError, OSError) as exc:
        if "gated repo" not in str(exc).lower() and "access to model" not in str(exc).lower():
            raise
        raise SystemExit(
            "Gemma 3 1B Instruct is gated on Hugging Face. Log in with a token "
            "that has accepted access, then rerun: hf auth login"
        ) from exc

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    print("Applying LoRA adaptation to model...")
    model = get_peft_model(model, lora_config)
    print("LoRA Training Parameters Summary:")
    model.print_trainable_parameters()

    return model, tokenizer


def build_training_args():
    return GRPOConfig(
        learning_rate=5e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        logging_steps=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_generations=6,
        generation_batch_size=6,
        max_completion_length=max_seq_length - max_prompt_length,
        max_steps=250,
        save_steps=250,
        output_dir="./gemma3_1b_gsm8k_grpo_outputs",
        max_grad_norm=0.1,
        report_to="trackio",
    )


def test_model(model, tokenizer, question, max_length=512):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    print(f"Processing: {question}")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
            length_penalty=1.0,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_text = response[len(text) :].strip()
    return generated_text


def main():
    print_gpu_environment()
    dataset = load_gsm8k_dataset()
    model, tokenizer = load_model_and_tokenizer()
    training_args = build_training_args()

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"gemma3-1b-gsm8k-grpo-{timestamp}"
    trackio.init(
        project="GRPO-Mathematical-Reasoning",
        name=run_name,
        config={
            "model_name": model_name,
            "dataset": "GSM8K",
            "technique": "GRPO + LoRA + 4-bit",
            "learning_rate": training_args.learning_rate,
            "batch_size": training_args.per_device_train_batch_size,
            "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
            "effective_batch_size": (
                training_args.per_device_train_batch_size
                * training_args.gradient_accumulation_steps
            ),
            "max_steps": training_args.max_steps,
            "lora_r": lora_rank,
            "lora_alpha": lora_rank,
            "num_generations": training_args.num_generations,
            "max_prompt_length": max_prompt_length,
            "max_completion_length": training_args.max_completion_length,
            "num_reward_functions": 4,
        },
    )

    print("GRPO Configuration Summary:")
    print(f"Learning rate: {training_args.learning_rate}")
    print(
        "Effective batch size: "
        f"{training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}"
    )
    print(f"Training steps: {training_args.max_steps}")
    print(f"Generations per step: {training_args.num_generations}")
    print(f"Run name: {run_name}")

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            match_format_exactly,
            match_format_approximately,
            check_answer_correctness,
            check_numbers_extraction,
        ],
        args=training_args,
        train_dataset=dataset,
    )
    print("GRPO Trainer initialized successfully.")
    print(f"Training dataset: {len(dataset):,} examples")
    print(f"Reward functions: {len(trainer.reward_funcs)} active")

    print("Starting GRPO training...")
    trainer.train()
    try:
        trackio.finish()
    except RuntimeError as exc:
        print(f"Trackio finish skipped: {exc}")

    print("Training completed successfully.")
    print(f"Model saved to: {training_args.output_dir}")
    model.save_pretrained("grpo_saved_lora")
    tokenizer.save_pretrained("grpo_saved_lora")

    # Test model on GSM8K problem, matching the cookbook evaluation section.
    gsm8k_question = (
        "Natalia sold clips to 48 of her friends in April, and then she sold "
        "half as many clips in May. How many clips did Natalia sell altogether "
        "in April and May?"
    )
    expected_answer = "72"
    gsm8k_response = test_model(model, tokenizer, gsm8k_question, max_length=768)

    print(f"Question: {gsm8k_question}")
    print(f"Model Response:\n{gsm8k_response}")

    has_reasoning = reasoning_start in gsm8k_response and reasoning_end in gsm8k_response
    has_solution = solution_start in gsm8k_response and solution_end in gsm8k_response
    print("\nFormat Check:")
    print(f"Reasoning section: {has_reasoning}")
    print(f"Solution section: {has_solution}")
    if has_solution:
        try:
            solution_text = (
                gsm8k_response.split(solution_start)[1]
                .split(solution_end)[0]
                .strip()
            )
            extracted_number = "".join(filter(str.isdigit, solution_text))
            expected_number = "".join(filter(str.isdigit, expected_answer))
            is_correct = extracted_number == expected_number
            print(f"Extracted: {solution_text}")
            print(f"Expected: {expected_answer}")
            print(f"Correct: {is_correct}")
        except IndexError:
            print("Could not extract solution")


if __name__ == "__main__":
    main()
