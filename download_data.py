import kagglehub
from pathlib import Path

custom_dir = Path("./gsm8k-grade-school-math-8k-dataset")
custom_dir.mkdir(parents=True, exist_ok=True)

path = kagglehub.dataset_download(
    "johnsonhk88/gsm8k-grade-school-math-8k-dataset-for-llm",
    output_dir=str(custom_dir)
)

print("Path to dataset files:", path)