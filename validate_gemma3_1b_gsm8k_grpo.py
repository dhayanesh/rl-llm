import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import gemma3_1b_gsm8k_grpo as train_ref


question = (
    "Natalia sold clips to 48 of her friends in April, and then she sold "
    "half as many clips in May. How many clips did Natalia sell altogether "
    "in April and May?"
)
expected_answer = "72"
trained_adapter_dir = "grpo_saved_lora"
checkpoint_adapter_dir = "gemma3_1b_gsm8k_grpo_outputs/checkpoint-250"


def score_response(response):
    completions = [[{"content": response}]]
    prompts = [
        [
            {"role": "system", "content": train_ref.system_prompt},
            {"role": "user", "content": question},
        ]
    ]
    answers = [expected_answer]
    scores = {
        "match_format_exactly": train_ref.match_format_exactly(completions)[0],
        "match_format_approximately": train_ref.match_format_approximately(completions)[0],
        "check_answer_correctness": train_ref.check_answer_correctness(
            prompts, completions, answers
        )[0],
        "check_numbers_extraction": train_ref.check_numbers_extraction(
            prompts, completions, answers
        )[0],
    }
    scores["total"] = sum(scores.values())
    return scores


def load_base_model_and_tokenizer():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        train_ref.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        train_ref.model_name,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def evaluate(label, model, tokenizer):
    model.eval()
    response = train_ref.test_model(model, tokenizer, question, max_length=768)
    scores = score_response(response)
    print(f"\n{label}")
    print(f"Response:\n{response}")
    print(f"Scores: {scores}")
    return scores


def main():
    model, tokenizer = load_base_model_and_tokenizer()
    base_scores = evaluate("Base model", model, tokenizer)

    adapter_dir = trained_adapter_dir
    if not os.path.isdir(adapter_dir) and os.path.isdir(checkpoint_adapter_dir):
        adapter_dir = checkpoint_adapter_dir
    if not os.path.isdir(adapter_dir):
        raise SystemExit(
            "No trained adapter found. Run gemma3_1b_gsm8k_grpo.py first."
        )

    trained_model = PeftModel.from_pretrained(model, adapter_dir)
    trained_scores = evaluate(f"Trained adapter: {adapter_dir}", trained_model, tokenizer)

    print("\nImprovement")
    print(f"Base total reward: {base_scores['total']}")
    print(f"Trained total reward: {trained_scores['total']}")
    print(f"Delta: {trained_scores['total'] - base_scores['total']}")


if __name__ == "__main__":
    main()
