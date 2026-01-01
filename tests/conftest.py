import os

import pytest

TEST_MODEL = os.environ.get("AUTOFORM_TEST_MODEL", "ollama/llama3:8b")


def is_llm_available():
    try:
        import litellm

        resp = litellm.completion(
            model=TEST_MODEL,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=5,
        )
        return True
    except Exception:
        return bool(
            os.environ.get("OPENAI_API_KEY")
            or os.environ.get("ANTHROPIC_API_KEY")
            or os.environ.get("GEMINI_API_KEY")
        )


requires_llm = pytest.mark.skipif(not is_llm_available(), reason="No LLM backend available")
