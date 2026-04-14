import os
import json
import requests
from groq import Groq
from src.utils.logger import get_logger
from config.settings import (
    GROQ_API_KEY,
    HF_API_KEY,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
)

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Model registry — single source of truth for every model ID in the system
# ---------------------------------------------------------------------------
# HF models are called via the HuggingFace Router endpoint.
# Groq models are called via the Groq SDK.
HF_MODELS = {"zai-org/GLM-5.1", "Qwen/Qwen3.5-9B", "Qwen/Qwen2.5-Coder-7B-Instruct"}
GROQ_MODELS = {"llama-3.3-70b-versatile"}


class MultiModelClient:
    """
    Multi-model LLM client with strict linear fallback.

    Fallback order (never changes regardless of query content):
        1. zai-org/GLM-5.1          (HF — primary)
        2. Qwen/Qwen3.5-9B         (HF — first fallback)
        3. llama-3.3-70b-versatile  (Groq — second fallback)
        4. Qwen/Qwen2.5-Coder-7B-Instruct (HF — final fallback)
    """

    # Strict, ordered fallback chain — do NOT re-order at runtime
    MODEL_CHAIN = [
        "zai-org/GLM-5.1",
        "Qwen/Qwen3.5-9B",
        "llama-3.3-70b-versatile",
        "Qwen/Qwen2.5-Coder-7B-Instruct",
    ]

    def __init__(self):
        if GROQ_API_KEY:
            self.groq_client = Groq(api_key=GROQ_API_KEY)
        else:
            self.groq_client = None

        self.hf_api_key = HF_API_KEY

    # ------------------------------------------------------------------
    # Transport helpers
    # ------------------------------------------------------------------
    def _call_hf(self, model_id, messages, temperature, max_tokens, stream=False):
        if not self.hf_api_key:
            raise ValueError("HF_API_KEY not configured")

        url = f"https://router.huggingface.co/hf-inference/models/{model_id}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.hf_api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model_id,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream
        }

        response = requests.post(url, headers=headers, json=payload, stream=stream)
        if response.status_code == 429:
            raise Exception("Rate limit (HTTP 429)")
        if not response.ok:
            raise Exception(f"HF Error: {response.text}")

        if stream:
            def generator():
                for line in response.iter_lines():
                    if line:
                        line = line.decode('utf-8')
                        if line.startswith("data: "):
                            data_str = line[6:]
                            if data_str.strip() == "[DONE]":
                                break
                            try:
                                data = json.loads(data_str)
                                choices = data.get("choices", [])
                                if not choices:
                                    continue
                                token = choices[0].get("delta", {}).get("content", "")
                                if token:
                                    yield token
                            except:
                                pass
            return generator()
        else:
            return response.json()["choices"][0]["message"]["content"]

    def _call_groq(self, model_id, messages, temperature, max_tokens, stream=False):
        if not self.groq_client:
            raise ValueError("GROQ_API_KEY not configured")

        response = self.groq_client.chat.completions.create(
            model=model_id,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream
        )
        if stream:
            def generator():
                for chunk in response:
                    choices = chunk.choices
                    if not choices:
                        continue
                    token = choices[0].delta.content
                    if token:
                        yield token
            return generator()
        else:
            return response.choices[0].message.content

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        original_query: str = "",
        history: list = None,
        temperature: float = LLM_TEMPERATURE,
        max_tokens: int = LLM_MAX_TOKENS,
        stream: bool = False
    ):
        """
        Generate response trying models in strict fallback order.
        Returns a tuple of (result, model_used).
        If stream=True, result is a generator.
        Otherwise, result is a string.
        """
        messages = [{"role": "system", "content": system_prompt}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_prompt})

        for model in self.MODEL_CHAIN:
            try:
                is_hf = model in HF_MODELS
                logger.info(f"Attempting model: {model}")
                if is_hf:
                    out = self._call_hf(model, messages, temperature, max_tokens, stream)
                else:
                    out = self._call_groq(model, messages, temperature, max_tokens, stream)

                logger.info(f"Model {model} selected successfully.")
                return out, model
            except Exception as e:
                logger.warning(f"Model {model} failed: {e}")
                continue

        raise Exception("All models failed.")