"""
model_runner.py
Async model API caller supporting OpenAI-compatible and Anthropic endpoints.
"""

import asyncio
import logging
import os
from typing import Optional

import anthropic
import openai

logger = logging.getLogger(__name__)

ANTHROPIC_MODELS = {
    "claude-3-5-sonnet-20241022",
    "claude-3-haiku-20240307",
    "claude-3-opus-20240229",
}


class ModelRunner:
    def __init__(self, model_config: dict):
        self.config = model_config
        self.oai_client = openai.AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.ant_client = anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    async def generate(
        self,
        model_id: str,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Route to correct API based on model_id."""
        if model_id in ANTHROPIC_MODELS:
            return await self._call_anthropic(model_id, prompt, max_tokens, temperature, system_prompt)
        else:
            return await self._call_openai(model_id, prompt, max_tokens, temperature, system_prompt)

    async def _call_anthropic(
        self, model_id: str, prompt: str, max_tokens: int,
        temperature: float, system_prompt: Optional[str]
    ) -> str:
        kwargs = {
            "model":      model_id,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages":   [{"role": "user", "content": prompt}],
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        for attempt in range(3):
            try:
                response = await self.ant_client.messages.create(**kwargs)
                return response.content[0].text
            except anthropic.RateLimitError:
                wait = 2 ** attempt * 5
                logger.warning(f"Anthropic rate limit hit for {model_id}. Waiting {wait}s.")
                await asyncio.sleep(wait)
            except anthropic.APIError as e:
                logger.error(f"Anthropic API error for {model_id}: {e}")
                return f"[ERROR: {str(e)[:100]}]"
        return "[ERROR: Max retries exceeded]"

    async def _call_openai(
        self, model_id: str, prompt: str, max_tokens: int,
        temperature: float, system_prompt: Optional[str]
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        for attempt in range(3):
            try:
                response = await self.oai_client.chat.completions.create(
                    model=model_id,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                return response.choices[0].message.content
            except openai.RateLimitError:
                wait = 2 ** attempt * 5
                logger.warning(f"OpenAI rate limit hit for {model_id}. Waiting {wait}s.")
                await asyncio.sleep(wait)
            except openai.APIError as e:
                logger.error(f"OpenAI API error for {model_id}: {e}")
                return f"[ERROR: {str(e)[:100]}]"
        return "[ERROR: Max retries exceeded]"
