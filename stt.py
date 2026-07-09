"""
STT service wrapper for smallest.ai Pulse.

Example:
    import asyncio
    from smallest_test.stt import transcribe

    with open("audio.wav", "rb") as f:
        audio_bytes = f.read()
    text = asyncio.run(transcribe(audio_bytes, language="en"))
    print(text)
"""

import logging
import os
import time

import aiohttp

logger = logging.getLogger("smallest.stt")

API_URL = "https://api.smallest.ai/waves/v1/stt"
API_KEY = os.environ.get("SMALLEST_API_KEY", "sk_ec6425e0db7a3e4222eb81f7ab57fe68")


async def transcribe(
    audio_bytes: bytes,
    language: str = "en",
) -> str:
    """
    Transcribe audio bytes to text using smallest.ai Pulse.

    Args:
        audio_bytes: Raw audio data (WAV, MP3, etc.).
        language: Language code (e.g. "en", "hi", "es", "multi").

    Returns:
        Transcribed text string.
    """
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "audio/wav",
    }
    params = {
        "language": language,
        "model": "pulse",
    }

    logger.info(
        "STT request | audio_size=%d bytes (%.1f KB) | lang=%s",
        len(audio_bytes), len(audio_bytes) / 1024, language,
    )

    t0 = time.perf_counter()
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                API_URL, params=params, data=audio_bytes, headers=headers
            ) as resp:
                elapsed = time.perf_counter() - t0
                raw = await resp.text()

                if resp.status != 200:
                    body_preview = raw[:500]
                    logger.error(
                        "STT HTTP %d | duration=%.3fs | body=%s",
                        resp.status, elapsed, body_preview,
                    )
                    resp.raise_for_status()

                result = await resp.json() if raw else {}
                transcription = result.get("transcription", "")

                duration = result.get("duration", "N/A")
                word_count = len(result.get("words", []))

                logger.info(
                    "STT success | status=%d | transcript=%r | "
                    "audio_duration=%s | words=%d | request_duration=%.3fs | "
                    "full_response_keys=%s",
                    resp.status, transcription, duration, word_count,
                    elapsed, list(result.keys()),
                )

                if not transcription:
                    logger.warning(
                        "STT returned empty transcription | "
                        "audio_size=%d | full_response=%s",
                        len(audio_bytes), raw[:300],
                    )

                return transcription

    except aiohttp.ClientError as e:
        elapsed = time.perf_counter() - t0
        logger.error(
            "STT connection error | duration=%.3fs | error=%s: %s",
            elapsed, type(e).__name__, e,
        )
        raise
    except Exception:
        elapsed = time.perf_counter() - t0
        logger.exception("STT unexpected error | duration=%.3fs", elapsed)
        raise
