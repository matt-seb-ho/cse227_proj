import argparse
import os

from peft import AutoPeftModelForCausalLM, PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        help="Local dir with saved model/adapter",
        default="ckpts/s1_wildjb_",
    )
    parser.add_argument(
        "--repo_id",
        help="HF repo id: username/name or org/name",
        default="msho/s1_wjb_llama3_peft",
    )
    parser.add_argument(
        "--merge_and_push_full",
        action="store_true",
        help="If set and this is a PEFT model, merge LoRA into base and push full model.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir

    # Check if this looks like a PEFT adapter dir
    is_peft_adapter = os.path.exists(os.path.join(output_dir, "adapter_config.json"))

    if is_peft_adapter:
        print("Detected PEFT/LoRA adapter directory.")
        peft_config = PeftConfig.from_pretrained(output_dir)
        base_name = peft_config.base_model_name_or_path
        print(f"Base model: {base_name}")

        # Tokenizer comes from the base model
        tokenizer = AutoTokenizer.from_pretrained(base_name)

        if args.merge_and_push_full:
            # Load adapter as AutoPeftModel and merge into full weights
            print("Loading PEFT model and merging LoRA weights into base...")
            model = AutoPeftModelForCausalLM.from_pretrained(output_dir)
            model = model.merge_and_unload()  # this returns a plain transformers model
            print(f"Pushing FULL merged model to Hub as {args.repo_id}...")
            model.push_to_hub(args.repo_id)
            tokenizer.push_to_hub(args.repo_id)
        else:
            # Adapter-only: base + adapter, push as PEFT model
            print("Loading base model and attaching adapter (adapter-only push)...")
            base_model = AutoModelForCausalLM.from_pretrained(base_name)
            model = PeftModel.from_pretrained(base_model, output_dir)
            print(f"Pushing PEFT adapter model to Hub as {args.repo_id}...")
            model.push_to_hub(args.repo_id)
            tokenizer.push_to_hub(args.repo_id)

    else:
        # Normal non-PEFT case (just in case your output dir is a full standard model)
        print("No PEFT adapter detected; treating as standard HF model directory.")
        model = AutoModelForCausalLM.from_pretrained(output_dir)
        tokenizer = AutoTokenizer.from_pretrained(output_dir)
        print(f"Pushing standard model to Hub as {args.repo_id}...")
        model.push_to_hub(args.repo_id)
        tokenizer.push_to_hub(args.repo_id)

    print("Done.")


if __name__ == "__main__":
    main()
