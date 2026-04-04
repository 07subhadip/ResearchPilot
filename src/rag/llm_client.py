"""
Groq API client for LLM inference.

WHY GROQ:
    - Free tier: 14,400 requests/day with Llama3
    - Speed: ~500 tokens/second (vs 10 tokens/second local CPU)
    - No GPU needed on our machine
    - Production-quality latency for demos

WHY LLAMA3-8B:
    - Free on Groq
    - 8B parameters: strong reasoning for research QA
    - 8192 token context window: fits our 5 retrieved chunks
    - Fast: ~1-2 seconds for a full response
"""

import os
from groq import Groq
from src.utils.logger import get_logger
from config.settings import (
    GROQ_API_KEY,
    LLM_MODEL_NAME,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
)

logger = get_logger(__name__)


class LLMClient:
    """
    Wrapper around Groq API for LLM inference.

    Designed as a simple interface so we can swap
    to any other LLM provider (OpenAI, Anthropic, local)
    by changing only this file.
    """


    def __init__(self):
        if not GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY not found. "
                "Add it to your .env file: GROQ_API_KEY=gsk_..."
            )
        self.client = Groq(api_key = GROQ_API_KEY)
        self.model  = LLM_MODEL_NAME
        logger.info(f"LLMClient initialized with model: {self.model}")


    def generate(
        self,
        system_prompt: str,
        user_prompt:   str,
        temperature:   float = LLM_TEMPERATURE,
        max_tokens:    int   = LLM_MAX_TOKENS,
    ) -> str:
        """
        Generate a response from the LLM.

        Args:
            system_prompt: Instructions for the LLM's behavior
            user_prompt:   The actual question + context
            temperature:   0.0 = deterministic, 1.0 = creative
                          We use 0.1 for factual research QA
            max_tokens:    Maximum response length

        Returns:
            Generated text string

        GROQ API STRUCTURE:
            Uses OpenAI-compatible chat format:
            [{"role": "system", "content": "..."}, 
             {"role": "user",   "content": "..."}]
        """

        try:
            response = self.client.chat.completions.create(
                model       = self.model,
                messages    = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature = temperature,
                max_tokens  = max_tokens,
            )

            answer = response.choices[0].message.content

            # Log token usage for monitoring
            usage = response.usage
            logger.debug(
                f"LLM usage - "
                f"prompt: {usage.prompt_tokens} tokens, "
                f"completion: {usage.completion_tokens} tokens, "
                f"total: {usage.total_tokens} tokens"
            )

            return answer

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise