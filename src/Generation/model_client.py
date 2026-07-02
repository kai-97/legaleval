import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
import anthropic

from dotenv import load_dotenv

load_dotenv()  # reads .env once; SDK picks up ANTHROPIC_API_KEY


@dataclass(frozen=True)
class ModelResponse:
    text: str                  # raw model output; parsed upstream
    model: str                 # which model actually answered
    input_tokens: int | None   # None if provider can't report it
    output_tokens: int | None
    latency_ms: float
    stop_reason: str | None    # natural stop vs max_tokens hit


class ModelClient(ABC):
    @abstractmethod
    def complete(
        self,
        system: str,
        user: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> ModelResponse:
        """One turn: system + user prompt -> ModelResponse."""
        ...


class AnthropicClient(ModelClient):
    def __init__(self, model: str):
        self.client = anthropic.Anthropic()
        self.model = model

    def complete(
        self,
        system: str,
        user: str,
        max_tokens: int = 1024,
        temperature: float=0.0
    ) -> ModelResponse:
        start = time.perf_counter()

        response = self.client.messages.create(
            max_tokens=max_tokens,
            messages=[{
                "content": user,
                "role": "user",
            }],
            system=system,
            model=self.model,
            temperature=temperature
        )
        latency_ms = (time.perf_counter() - start) * 1000

        for message in response.content:
            if message.type == "text":
                break

        return ModelResponse(message.text,
                      response.model,
                      response.usage.input_tokens,
                      response.usage.output_tokens,
                      latency_ms,
                      response.stop_reason)


class OllamaClient(ModelClient):
    def __init__(self, model: str):
        # TODO: store self.model = model (no client yet — stub)
        ...

    def complete(
        self,
        system: str,
        user: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> ModelResponse:
        raise NotImplementedError(
            "Ollama stub — Day 4 leaves this for later"
        )


def get_model_client(provider: str, model: str) -> ModelClient:
    if provider == "anthropic":
        return AnthropicClient(model)
    elif provider == "ollama":
        return OllamaClient(model)
    else:
        raise ValueError(f"unknown provider: {provider}")
