from __future__ import annotations


class PromptFuzzerNotImplemented(NotImplementedError):
    pass


def require_prompt_fuzzer() -> None:
    raise PromptFuzzerNotImplemented("Prompt-family fuzzing is planned for mi v0.5.")
