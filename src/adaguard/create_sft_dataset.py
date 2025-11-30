# build_s1_wildjail_dataset.py

import os
from pathlib import Path
from typing import Dict

from datasets import load_from_disk
from dotenv import load_dotenv

from .constants import DATA_DIR, ENV_FILE

RNG_SEED = 42
SYSTEM_MESSAGE = (
    "You are a safety-tuned assistant. You carefully reason about whether "
    "user requests are harmful or benign and always follow safety policies.\n"
    "If the request is harmful, you explain briefly why and refuse to help. "
    "If it is benign, you may answer helpfully."
)
DATASET_PATH = DATA_DIR / "wildjail_s1k"


def build_prompt_completion_row(example: Dict) -> Dict:
    """Convert one WildJailbreak+trace row into prompt–completion format."""
    user_prompt = example["adversarial"] or example["vanilla"]
    reasoning = (example["gpt_oss_20b_trace"] or "").strip()
    final_answer = (example["completion"] or "").strip()

    prompt = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": user_prompt},
    ]

    # S1-style: reasoning then answer, but within one assistant message, with <think> tags.
    completion = [
        {
            "role": "assistant",
            "content": f"<think>\n{reasoning}\n</think>\n{final_answer}",
        }
    ]

    label = "harmful" if example["data_type"].endswith("_harmful") else "benign"

    return {
        "prompt": prompt,
        "completion": completion,
        "label": label,
        "data_type": example["data_type"],
    }


def main(
    input_path: Path = DATASET_PATH,
    hf_repo_id: str = "msho/wildjailbreak-s1k-llama3",
):
    # 0. Load HF token
    load_dotenv(ENV_FILE)
    HF_TOKEN = os.getenv("HF_MBP_TOKEN")

    # 1. Load augmented subset
    ds = load_from_disk(input_path)
    ds = ds.shuffle(seed=RNG_SEED)

    # 2. Map to prompt–completion format
    pc_ds = ds.map(
        build_prompt_completion_row,
        remove_columns=ds.column_names,
        desc="Building prompt–completion dataset",
    )

    # 3. Push to Hub (you can set private=True in the Hub repo settings)
    print(f"Pushing to HF Hub: {hf_repo_id}")
    pc_ds.push_to_hub(hf_repo_id, token=HF_TOKEN)

    print("Done.")


if __name__ == "__main__":
    main()
