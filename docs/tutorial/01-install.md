# Install

`autoform` requires Python 3.12 or newer.

Install from GitHub:

```bash
# install from github
pip install git+https://github.com/ASEM000/autoform.git
```

`autoform` uses LiteLLM for provider calls. For OpenAI, set `OPENAI_API_KEY` and use an OpenAI model name:

```bash
# set your provider key
export OPENAI_API_KEY="..."
```

Any LiteLLM-supported provider works if the provider credentials and model name are configured for that provider.

Run one direct LM call before starting the [tracing](../concepts/trace-ir-execute.md) tutorial:

```python
import autoform as af

# smoke test the provider before tracing
messages = [dict(role="user", content="Say hello in five words.")]
response = af.lm_call(messages, model="gpt-5.2")
print(response)
```

This is only a provider smoke test. It does not use tracing.

| Symptom | Fix |
| --- | --- |
| `ModuleNotFoundError: autoform` | Install into the same environment that runs Python. |
| Python version error | Use Python 3.12 or newer. |
| Provider authentication error | Set the provider API key expected by LiteLLM, such as `OPENAI_API_KEY` for OpenAI. |
| Provider model error | Use a model name supported by the configured provider or route through `litellm.Router`. |
