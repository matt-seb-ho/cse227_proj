from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from openai import OpenAI


@dataclass
class BudgetForcingConfig:
    """
    Configuration for budget-forced reasoning over a remote vLLM OpenAI server.
    """

    system_message: str | None = None
    max_tokens_thinking: int = 2_000
    num_ignore: int = 1
    final_answer_tokens: int = 48
    ignore_str: str = " wait"
    stop_str: str = "</think>"


@dataclass
class BudgetForcingResult:
    """
    Result of a single budget-forced generation run.
    """

    conversation: List[Dict[str, str]]
    thinking_trace: str
    final_answer_text: str
    total_thinking_tokens: int
    total_answer_tokens: int

    # Extra: token-level info for the *final answer* segment only
    answer_tokens: Optional[List[str]] = None
    answer_token_logprobs: Optional[List[float]] = None
    raw_answer_logprobs: Optional[Any] = None  # raw logprobs object from vLLM


class RemoteBudgetForcer:
    """
    Client-side budget forcing logic that talks to a vLLM OpenAI-compatible server.

    Assumes the server:
      - Exposes /v1/chat/completions
      - Supports extra_body params like:
          * continue_final_message
          * add_generation_prompt
          * min_tokens
          * include_stop_str_in_output
      - Supports logprobs/top_logprobs params on chat.completions.
    """

    def __init__(self, client: OpenAI, model_name: str, config: BudgetForcingConfig):
        self.client = client
        self.model_name = model_name
        self.config = config

    def _chat_once(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        stop: Optional[List[str]] = None,
        min_tokens: int = 0,
        want_logprobs: bool = False,
        top_logprobs: int = 0,
    ) -> tuple[str, int, Optional[Any]]:
        """
        Single call to /v1/chat/completions using continue_final_message.

        Returns:
          - full_assistant_message: the complete content of the final assistant message
          - completion_tokens: number of tokens generated in this call
          - logprobs_obj: raw logprobs object (or None if want_logprobs=False)
        """
        extra_body = {
            "continue_final_message": True,
            "add_generation_prompt": False,
            "min_tokens": min_tokens,
            "include_stop_str_in_output": False,
        }

        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.0,
            stop=stop,
            logprobs=want_logprobs or None,
            top_logprobs=top_logprobs if want_logprobs else None,
            extra_body=extra_body,
        )

        choice = completion.choices[0]
        full_text = choice.message.content or ""
        completion_tokens = completion.usage.completion_tokens or 0

        # vLLM currently uses a completion-style schema for logprobs in chat:
        #   logprobs.tokens: List[str]
        #   logprobs.token_logprobs: List[float]
        #   ...
        # (This may evolve; we just pass it through.)  [oai_citation:1‡GitHub](https://github.com/vllm-project/vllm/issues/3179?utm_source=chatgpt.com)
        logprobs_obj = choice.logprobs if want_logprobs else None

        return full_text, completion_tokens, logprobs_obj

    def budget_forced_generation(self, user_prompt: str) -> BudgetForcingResult:
        """
        Run budget-forced reasoning on a single user prompt.

        Pattern:
          1) Start with assistant "<think>".
          2) Generate until stop_str, but ignore that stop NUM_IGNORE times,
             each time appending chunk + ignore_str.
          3) On the last pass, append the final chunk (no ignore_str), then
             close with stop_str.
          4) Generate a short final answer (with logprobs).
        """
        cfg = self.config

        # system -> user -> assistant "<think>"
        if cfg.system_message is None:
            conversation: List[Dict[str, str]] = [
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": "<think>"},
            ]
        else:
            conversation: List[Dict[str, str]] = [
                {"role": "system", "content": cfg.system_message},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": "<think>"},
            ]

        remaining_budget = cfg.max_tokens_thinking
        total_thinking_tokens = 0

        # ----- First thinking pass -----
        assistant_prefix = conversation[-1]["content"]
        full_text, tokens, _ = self._chat_once(
            messages=conversation,
            max_tokens=remaining_budget,
            stop=[cfg.stop_str],
            min_tokens=0,
            want_logprobs=False,
        )
        total_thinking_tokens += tokens
        remaining_budget -= tokens

        assert full_text.startswith(assistant_prefix), (
            "Server returned assistant content that does not start with the "
            "local assistant_prefix; check continue_final_message handling."
        )
        chunk = full_text[len(assistant_prefix) :]

        # ----- Repeated "ignore" passes -----
        for _ in range(cfg.num_ignore):
            if remaining_budget <= 0:
                break

            conversation[-1]["content"] += chunk + cfg.ignore_str

            assistant_prefix = conversation[-1]["content"]
            full_text, tokens, _ = self._chat_once(
                messages=conversation,
                max_tokens=remaining_budget,
                stop=[cfg.stop_str],
                min_tokens=1,
                want_logprobs=False,
            )
            total_thinking_tokens += tokens
            remaining_budget -= tokens

            assert full_text.startswith(assistant_prefix), (
                "Server returned assistant content that does not start with the "
                "local assistant_prefix; check continue_final_message handling."
            )
            chunk = full_text[len(assistant_prefix) :]

        # After the loop, append the last thinking chunk without ignore_str
        conversation[-1]["content"] += chunk

        # Close the think tag
        conversation[-1]["content"] += cfg.stop_str
        thinking_trace = conversation[-1]["content"]

        # ----- Final answer phase (with logprobs) -----
        assistant_prefix = conversation[-1]["content"]

        full_text_answer, answer_tokens, logprobs_obj = self._chat_once(
            messages=conversation,
            max_tokens=cfg.final_answer_tokens,
            stop=None,
            min_tokens=0,
            want_logprobs=True,
            top_logprobs=0,  # just need chosen-token logprobs
        )

        # Ensure the prefix matches, then recover just the newly generated answer text
        assert full_text_answer.startswith(assistant_prefix), (
            "Server returned assistant content that does not start with the "
            "local assistant_prefix; check continue_final_message handling."
        )
        answer_chunk = full_text_answer[len(assistant_prefix) :]
        conversation[-1]["content"] = full_text_answer

        # Parse tokens/logprobs (vLLM completion-style schema)
        answer_tokens_list: Optional[List[str]] = None
        answer_token_logprobs: Optional[List[float]] = None
        if logprobs_obj is not None:
            # These names follow vLLM's current schema for logprobs in chat.  [oai_citation:2‡GitHub](https://github.com/vllm-project/vllm/issues/3179?utm_source=chatgpt.com)
            tokens_field = getattr(logprobs_obj, "tokens", None)
            token_lp_field = getattr(logprobs_obj, "token_logprobs", None)

            if tokens_field is not None and token_lp_field is not None:
                answer_tokens_list = list(tokens_field)
                answer_token_logprobs = list(token_lp_field)

        return BudgetForcingResult(
            conversation=conversation,
            thinking_trace=thinking_trace,
            final_answer_text=answer_chunk,
            total_thinking_tokens=total_thinking_tokens,
            total_answer_tokens=answer_tokens,
            answer_tokens=answer_tokens_list,
            answer_token_logprobs=answer_token_logprobs,
            raw_answer_logprobs=logprobs_obj,
        )
