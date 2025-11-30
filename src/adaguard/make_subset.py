import os
import random

import numpy as np
from datasets import concatenate_datasets, load_dataset
from dotenv import load_dotenv

from .constants import ENV_FILE, WILDJAIL_SUBSET_PATH


def load_wildjailbreak_dataset():
    ds_dict = load_dataset(
        "allenai/wildjailbreak",
        "train",
        delimiter="\t",
        keep_default_na=False,
        token=os.getenv("HF_MBP_TOKEN"),
    )
    return ds_dict["train"]


def add_idx(example, idx):
    return {"row_idx": idx}


def main():
    # -----------------------------
    # Setup
    # -----------------------------
    # Load environment variables
    load_dotenv(ENV_FILE)

    # Set random seed
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)

    # -----------------------------
    # Load and sample from wildjailbreak dataset
    # -----------------------------
    # load wildjailbreak dataset
    ds = load_wildjailbreak_dataset()

    # 1. Attach original row indices
    ds = ds.map(add_idx, with_indices=True)

    # 2. Filter to adversarial rows and sample 1k benign / 1k harmful
    adv_benign = ds.filter(lambda ex: ex["data_type"] == "adversarial_benign")
    adv_harmful = ds.filter(lambda ex: ex["data_type"] == "adversarial_harmful")

    # Shuffle deterministically and sample
    adv_benign_sample = adv_benign.shuffle(seed=SEED).select(range(1000))
    adv_harmful_sample = adv_harmful.shuffle(seed=SEED).select(range(1000))

    sampled = concatenate_datasets([adv_benign_sample, adv_harmful_sample])

    print("Sample sizes:")
    print("  adversarial_benign:", len(adv_benign_sample))
    print("  adversarial_harmful:", len(adv_harmful_sample))

    # save sampled dataset to disk
    WILDJAIL_SUBSET_PATH.mkdir(parents=True, exist_ok=True)
    sampled.save_to_disk(WILDJAIL_SUBSET_PATH)
    print(f"Saved sampled dataset to {WILDJAIL_SUBSET_PATH}")


if __name__ == "__main__":
    main()
