"""
TTS service wrapper for smallest.ai Lightning v3.1.

Example:
    import asyncio
    from smallest_test.tts import synthesize

    audio_bytes = asyncio.run(synthesize("Hello world"))
    with open("output.wav", "wb") as f:
        f.write(audio_bytes)
"""

import logging
import os
import time

import aiohttp

logger = logging.getLogger("smallest.tts")

API_URL = "https://api.smallest.ai/waves/v1/lightning-v3.1/get_speech"
API_KEY = os.environ.get("SMALLEST_API_KEY", "sk_ec6425e0db7a3e4222eb81f7ab57fe68")


async def synthesize(
    text: str,
    voice_id: str = "sophia",
    output_format: str = "wav",
    sample_rate: int = 24000,
    speed: float = 1.0,
    language: str = "en",
) -> bytes:
    """
    Synthesize speech from text using smallest.ai Lightning v3.1.

    Args:
        text: Text to synthesize.
        voice_id: Voice identifier (e.g. "sophia", "magnus", "olivia").
        output_format: Audio format — "wav", "mp3", "pcm", "mulaw", "alaw".
        sample_rate: Sample rate — 8000, 16000, 24000, or 44100.
        speed: Speech rate multiplier (0.5 to 2.0).
        language: Language code (e.g. "en", "hi", "es", "auto").

    Returns:
        Raw audio bytes in the requested format.
    """
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "text": text,
        "voice_id": voice_id,
        "output_format": output_format,
        "sample_rate": sample_rate,
        "speed": speed,
        "language": language,
    }

    text_preview = text[:80] + ("..." if len(text) > 80 else "")
    logger.info(
        "TTS request | text=%r | voice=%s | format=%s | sample_rate=%d | speed=%.1f | lang=%s",
        text_preview, voice_id, output_format, sample_rate, speed, language,
    )

    t0 = time.perf_counter()
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(API_URL, json=payload, headers=headers) as resp:
                elapsed = time.perf_counter() - t0
                audio_bytes = await resp.read()

                if resp.status != 200:
                    body_preview = audio_bytes[:500].decode(errors="replace")
                    logger.error(
                        "TTS HTTP %d | duration=%.3fs | body=%s",
                        resp.status, elapsed, body_preview,
                    )
                    resp.raise_for_status()

                logger.info(
                    "TTS success | status=%d | size=%d bytes | duration=%.3fs | "
                    "header_4B=%r",
                    resp.status, len(audio_bytes), elapsed,
                    audio_bytes[:4],
                )

                if not audio_bytes:
                    logger.warning("TTS returned empty audio body for text=%r", text_preview)

                return audio_bytes

    except aiohttp.ClientError as e:
        elapsed = time.perf_counter() - t0
        logger.error(
            "TTS connection error | duration=%.3fs | error=%s: %s",
            elapsed, type(e).__name__, e,
        )
        raise
    except Exception:
        elapsed = time.perf_counter() - t0
        logger.exception("TTS unexpected error | duration=%.3fs", elapsed)
        raise
