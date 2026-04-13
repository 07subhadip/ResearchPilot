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

class MultiModelClient:
    """
    Multi-model LLM client with Qwen primary and Groq backup.
    Supports code routing based on keywords.
    """

    def __init__(self):
        if GROQ_API_KEY:
            self.groq_client = Groq(api_key=GROQ_API_KEY)
        else:
            self.groq_client = None

        self.hf_api_key = HF_API_KEY

        self.primary_model = "Qwen/Qwen2.5-72B-Instruct"
        self.secondary_model = "llama-3.3-70b-versatile"
        self.code_model = "Qwen/Qwen2.5-Coder-7B-Instruct"

        self.code_keywords = ["code", "implement", "function", "class", "python", "algorithm", "write a", "script"]

    def get_model_for_query(self, question: str):
        q_lower = question.lower()
        if any(kw in q_lower for kw in self.code_keywords):
            return [self.code_model, self.primary_model, self.secondary_model]
        return [self.primary_model, self.secondary_model]

    def _call_hf(self, model_id, messages, temperature, max_tokens, stream=False):
        if not self.hf_api_key:
            raise ValueError("HF_API_KEY not configured")

        url = f"https://api-inference.huggingface.co/models/{model_id}/v1/chat/completions"
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
        Generate response trying models in priority order.
        Returns a tuple of (result, model_used).
        If stream=True, result is a generator.
        Otherwise, result is a string.
        """
        models_to_try = self.get_model_for_query(original_query)
        messages = [{"role": "system", "content": system_prompt}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_prompt})

        for model in models_to_try:
            try:
                is_hf = "Qwen" in model
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