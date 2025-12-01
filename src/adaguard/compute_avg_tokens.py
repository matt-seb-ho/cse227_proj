import argparse
import json
from pathlib import Path
from statistics import mean, stdev


def parse_args():
    p = argparse.ArgumentParser("Compute average token usage from JSONL eval results")
    p.add_argument(
        "--results",
        type=str,
        required=True,
        help="Path to JSONL file produced by the eval script.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    path = Path(args.results)

    thinking = []
    answer = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            thinking.append(ex.get("total_thinking_tokens", 0))
            answer.append(ex.get("total_answer_tokens", 0))

    if not thinking:
        print("No entries found in results file.")
        return

    avg_thinking = mean(thinking)
    avg_answer = mean(answer)
    avg_total = mean(t + a for t, a in zip(thinking, answer))

    print(f"n = {len(thinking)} examples")
    print(f"Average thinking tokens: {avg_thinking:.2f}")
    print(f"Average answer tokens:   {avg_answer:.2f}")
    print(f"Average total tokens:    {avg_total:.2f}")

    # Optional: spread
    if len(thinking) > 1:
        print()
        print("Spread / diagnostics:")
        print(
            f"  Thinking: min={min(thinking)}, max={max(thinking)}, stdev={stdev(thinking):.2f}"
        )
        print(
            f"  Answer:   min={min(answer)}, max={max(answer)}, stdev={stdev(answer):.2f}"
        )


if __name__ == "__main__":
    main()
