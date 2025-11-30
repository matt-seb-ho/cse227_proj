import tiktoken

ENC_GPT4O = tiktoken.encoding_for_model("gpt-4o")


def count_tokens(text: str) -> int:
    """Count the number of tokens in the given text using tiktoken."""
    tokens = ENC_GPT4O.encode(text)
    return len(tokens)
