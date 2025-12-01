from datasets import load_dataset
from openai import OpenAI

from .budget_force_inference import (
    BudgetForcingConfig,
    RemoteBudgetForcer,
)
from .create_sft_dataset import format_prompt_for_classification


def main():
    # prep model client and config
    client = OpenAI(
        # base_url="http://127.0.0.1:8000/v1",  # where you served vLLM
        base_url="http://0.0.0.0:8000/v1",
        api_key="NONE",
    )
    config = BudgetForcingConfig(
        system_message=None,
        max_tokens_thinking=2000,
        num_ignore=1,
        final_answer_tokens=48,
        ignore_str=" wait",
        stop_str="</think>",
    )
    budget_forcer = RemoteBudgetForcer(
        client=client,
        model_name="ckpts/s1_wildjb_20251201_024122/merged",
        config=config,
    )

    # prep data
    dataset = load_dataset("msho/wjb_eval_subset100")["train"]
    test_prompt = dataset[0]["adversarial"]
    formatted_prompt = format_prompt_for_classification(test_prompt)

    # run budget forced inference
    output = budget_forcer.budget_forced_generation(
        user_prompt=formatted_prompt,
    )
    print(output)


if __name__ == "__main__":
    main()
