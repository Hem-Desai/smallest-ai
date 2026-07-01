"""
LLM client — works with OpenAI, Groq, or any OpenAI-compatible API.
Set OPENAI_API_KEY env var for deployment. Falls back to a local key for dev.
"""
import json
import logging
import os
import time
from typing import Dict, List

import aiohttp

logger = logging.getLogger("smallest.llm")

# -- backends ----------------------------------------------------------------
# Swap BASE_URL / MODEL to change provider. Set OPENAI_API_KEY in env.
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_MODEL = "gpt-4o-mini"
# Read from env, fall back to dev key if not set
OPENAI_KEY = os.environ.get("OPENAI_API_KEY") or os.environ.get(
    "_OPENAI_DEV_KEY",
    "",
)

VOICE_AGENT_SYSTEM_PROMPT = """You are a helpful, friendly voice assistant on a phone call.
Keep responses SHORT and CONVERSATIONAL — one or two sentences only.
Speak naturally, as if you're talking to a friend.
Never use markdown, lists, or formatting — plain speech only.

CRITICAL RULES:
- If the user's message seems garbled, unclear, or could be a transcription error,
  ask for clarification. Say things like "Sorry, I didn't quite catch that" or
  "Could you say that again?" instead of guessing what they meant.
- NEVER assume or invent facts about things you didn't hear clearly. If you're not
  sure what was asked, just say so and ask them to repeat.
- If you hear an ambiguous word (like "Cup" without context), ask which one they
  mean rather than guessing Cricket or Football.
- Only answer questions you understand clearly. It's better to ask twice than to
  give wrong information.

Stay in character as a real-time voice agent."""


class ElectronClient:
    """LLM client — works with Groq, Electron, or any OpenAI-compatible API."""

    def __init__(self, api_key: str = OPENAI_KEY,
                 model: str = OPENAI_MODEL,
                 base_url: str = OPENAI_URL,
                 system_prompt: str = VOICE_AGENT_SYSTEM_PROMPT):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.system_prompt = system_prompt
        self.conversation_history: List[Dict[str, str]] = []

    def add_to_history(self, role: str, content: str) -> None:
        self.conversation_history.append({"role": role, "content": content})
        total = sum(len(m["content"]) for m in self.conversation_history)
        while len(self.conversation_history) > 2 and total > 90000:
            removed = self.conversation_history.pop(0)
            total -= len(removed["content"])

    async def chat(self, user_message: str, max_tokens: int = 256) -> str:
        messages = [{"role": "system", "content": self.system_prompt}]
        recent = self.conversation_history[-16:]
        messages.extend(recent)
        messages.append({"role": "user", "content": user_message})

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.7,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        t0 = time.perf_counter()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.base_url, json=payload, headers=headers) as resp:
                    elapsed = time.perf_counter() - t0
                    body = await resp.text()

                    if resp.status != 200:
                        logger.error(
                            "LLM error | status=%d | duration=%.3fs | body=%s",
                            resp.status, elapsed, body[:300],
                        )
                        return ""

                    data = json.loads(body)
                    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    usage = data.get("usage", {})
                    logger.info(
                        "LLM response | model=%s | duration=%.3fs | "
                        "tokens: in=%d out=%d total=%d | content=%r",
                        self.model, elapsed,
                        usage.get("prompt_tokens", 0),
                        usage.get("completion_tokens", 0),
                        usage.get("total_tokens", 0),
                        content[:100],
                    )
                    return content.strip()

        except aiohttp.ClientError as e:
            elapsed = time.perf_counter() - t0
            logger.error("LLM connection error | duration=%.3fs | error=%s: %s",
                         elapsed, type(e).__name__, e)
            return ""
        except Exception:
            elapsed = time.perf_counter() - t0
            logger.exception("LLM unexpected error | duration=%.3fs", elapsed)
            return ""

    def reset(self) -> None:
        """Clear conversation history."""
        self.conversation_history.clear()
