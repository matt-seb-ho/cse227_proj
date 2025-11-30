# train/sft_llama3.py

import logging
import os
from dataclasses import asdict, dataclass, field
from typing import Optional

import transformers
import trl
from datasets import load_dataset
from dotenv import load_dotenv

from .constants import ENV_FILE

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


@dataclass
class TrainingConfig:
    # default to Llama 3 8B Instruct for your run
    model_name: str = field(default="meta-llama/Llama-3.1-8B-Instruct")
    # plenty for your ~1.5k token max sequences
    block_size: int = field(default=2048)

    wandb_project: Optional[str] = field(default="s1-wildjail")
    wandb_entity: Optional[str] = field(default=None)

    # this is our new HF dataset from the build script
    train_file_path: Optional[str] = field(default="msho/wildjailbreak-s1k-llama3")

    # not using dagger here, keep it for compatibility
    dagger: bool = field(default=False)

    hf_access_token_var: str = field(default="HF_ACCESS_TOKEN")

    def __post_init__(self):
        if self.wandb_project is not None:
            os.environ["WANDB_PROJECT"] = self.wandb_project
        if self.wandb_entity is not None:
            os.environ["WANDB_ENTITY"] = self.wandb_entity


def train():
    # Parse CLI into (our TrainingConfig, TRL's SFTConfig)
    parser = transformers.HfArgumentParser((TrainingConfig, trl.SFTConfig))
    config, args = parser.parse_args_into_dataclasses()

    log_config = {**asdict(config), **asdict(args)}
    logging.info(f"Training config: {log_config}")

    # get token
    load_dotenv(ENV_FILE)
    hf_token = os.getenv(config.hf_access_token_var)

    # -----------------------
    # 1. Load model
    # -----------------------
    # For 8B you can usually get away with default dtype; you can also pass
    # model_init_kwargs in args if you want bf16 explicitly.
    model = transformers.AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype="auto",
        token=hf_token,
    )

    # -----------------------
    # 2. Load dataset
    # -----------------------
    # Our dataset is prompt–completion, conversational:
    #   {"prompt": [...messages...], "completion": [...messages...]}
    dataset = load_dataset(config.train_file_path)

    train_ds = dataset["train"]
    eval_ds = dataset["test"] if "test" in dataset else train_ds

    # -----------------------
    # 3. Tokenizer setup
    # -----------------------
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        config.model_name,
        use_fast=True,
        token=hf_token,
    )

    # Llama-3 chat models typically don't have pad_token set; set to eos for training
    # NOTE: not necessary: https://huggingface.co/docs/trl/en/sft_trainer#trl.SFTConfig
    # if tokenizer.pad_token is None:
    #     tokenizer.pad_token = tokenizer.eos_token

    # use our desired max seq length
    # args.max_seq_length = config.block_size
    args.max_length = config.block_size

    # We are using a prompt–completion dataset, so:
    # - we do NOT set args.dataset_text_field
    # - by default, SFTTrainer will compute loss on completion only (completion_only_loss=True)
    #   and will apply the chat template for conversational data.

    # -----------------------
    # 4. SFTTrainer
    # -----------------------
    trainer = trl.SFTTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=args,
        # no custom collator: SFTTrainer will use DataCollatorForLanguageModeling
        # and set completion_only_loss based on the dataset format.
    )

    trainer.train()
    trainer.save_model(output_dir=args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    trainer.accelerator.wait_for_everyone()

    # push model to hub
    if config.push_to_hub:
        trainer.push_to_hub(token=hf_token)


if __name__ == "__main__":
    train()
