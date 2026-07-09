"""
Real-time speech-to-speech bridge using smallest.ai Pulse (STT), Electron (LLM),
and Lightning (TTS).

Matches the interface from tests/gladia_bridge.py and backend/audio/deepgram_bridge_v2.py
so it can be monkeypatched into the existing Twilio WebSocket handler.

Architecture:
    Twilio WS ──mu-law──> decode ──PCM──> audio buffer
                                            │
                              silence detected (1.5s)
                                            │
                                            v
    Pulse STT REST API <── WAV bytes ───────┘
               │
         transcription
               │
               v
      Electron LLM API ────> response text
               │
               v
    Twilio WS <──mu-law── encode <──── Lightning TTS WS

STT uses the REST batch endpoint (proven working) with buffered audio + silence
detection. LLM uses Electron's OpenAI-compatible chat completions. TTS uses the
WebSocket streaming endpoint for low-latency playback.
"""

import asyncio
import base64
import io
import json
import logging
import os
import struct
import time
import uuid
import wave
from typing import Any, Dict, Optional

import aiohttp

try:
    import audioop  # type: ignore
except ModuleNotFoundError:
    audioop = None  # fallback for Python 3.13+ without audioop

logger = logging.getLogger("smallest.bridge")

from smallest_test.electron_llm import ElectronClient
from smallest_test.event_store import event_store

API_KEY = os.environ.get("SMALLEST_API_KEY", "sk_ec6425e0db7a3e4222eb81f7ab57fe68")
STT_REST_URL = "https://api.smallest.ai/waves/v1/stt"
LIGHTNING_WS_URL = "wss://api.smallest.ai/waves/v1/tts/live"

# Audio format constants
TWILIO_SAMPLE_RATE = 8000
PULSE_ENCODING = "linear16"

# Silence detection: if no media for this many seconds, flush the buffer to STT
SILENCE_TIMEOUT = 1.5

# Max buffer size before forced flush (bytes of PCM)
MAX_BUFFER_BYTES = 160_000  # ~10 seconds at 8kHz 16-bit mono

# Periodic summary interval (seconds)
_SUMMARY_INTERVAL = 10.0


def _ulaw_to_pcm(mulaw_bytes: bytes) -> bytes:
    """Convert mu-law encoded audio to 16-bit linear PCM."""
    if audioop is None:
        raise RuntimeError(
            "audioop module is not available (Python 3.13+ may have removed it). "
            "Install a replacement like 'pip install audioop-lts' or use a different approach."
        )
    return audioop.ulaw2lin(mulaw_bytes, 2)


def _pcm_to_mulaw(pcm_bytes: bytes) -> bytes:
    """Convert 16-bit linear PCM to mu-law."""
    if audioop is None:
        raise RuntimeError("audioop module is not available (Python 3.13+).")
    return audioop.lin2ulaw(pcm_bytes, 2)


def _pcm_to_wav(pcm_bytes: bytes, sample_rate: int = 8000) -> bytes:
    """Wrap raw 16-bit mono PCM in a WAV container."""
    buf = io.BytesIO()
    n_samples = len(pcm_bytes) // 2
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


class SmallestAiBridge:
    """
    Real-time bridge between Twilio Media Streams and smallest.ai.

    Flow:
    - Twilio sends mu-law audio (base64-encoded in JSON media events).
    - Bridge decodes mu-law → linear16 PCM → appends to audio buffer.
    - When silence is detected (no media for 1.5s) or buffer is full,
      buffer is flushed as a WAV to Pulse STT REST API.
    - When STT returns a transcript, a canned response is sent to Lightning TTS WS.
    - TTS returns mu-law audio bytes → base64-encoded → sent back to Twilio.
    """

    def __init__(self, execution_id: str, response_text: str | None = None, model: str = "lightning_v3.1"):
        self.execution_id = execution_id
        self.bridge_id = str(uuid.uuid4())[:8]
        self.tts_model = model
        # Only use canned response if no LLM is available (fallback)
        self._fallback_text = response_text or (
            "I received your message through the smallest AI real time pipeline."
        )

        self.twilio_ws: Any = None
        self.tts_ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self.stream_sid: Optional[str] = None

        self._tts_task: Optional[asyncio.Task] = None
        self._summary_task: Optional[asyncio.Task] = None
        self._flush_task: Optional[asyncio.Task] = None
        self._running = False
        self._tts_busy = False  # gate to prevent overlapping TTS requests
        self._tts_complete = False  # signal from TTS message handler to listener
        self._tts_session: Optional[aiohttp.ClientSession] = None

        # LLM client for conversational responses
        self.electron: ElectronClient = ElectronClient()

        # Audio buffer
        self._pcm_buffer: bytearray = bytearray()
        self._last_media_at: float = 0.0
        self._flushing: bool = False

        # Timing
        self._started_at: float = 0.0
        self._first_media_at: float = 0.0

        # Diagnostic counters
        self.media_count = 0
        self.media_bytes_total = 0
        self.transcript_count = 0
        self.stt_request_count = 0
        self.stt_error_count = 0
        self.tts_chunk_count = 0
        self.tts_bytes_total = 0
        self._tts_chunks_this_utterance = 0  # per-utterance counter for debug logging
        self.twilio_send_errors = 0

        # Per-interval snapshots for rate calculation
        self._prev_media_count = 0
        self._prev_media_bytes = 0
        self._prev_transcript_count = 0
        self._prev_tts_chunks = 0
        self._summary_at: float = 0.0

    # ---- lifecycle ----------------------------------------------------------

    async def initialize(self, twilio_websocket) -> None:
        self.twilio_ws = twilio_websocket
        logger.info(
            "[%s] bridge.initialize | execution_id=%s",
            self.bridge_id, self.execution_id,
        )

    async def start_bridge(self) -> None:
        self._running = True
        self._started_at = time.perf_counter()
        self._summary_at = self._started_at
        self._summary_task = asyncio.create_task(self._periodic_summary())
        logger.info("[%s] bridge.start_bridge | running", self.bridge_id)
        while self._running:
            await asyncio.sleep(5)

    async def _periodic_summary(self) -> None:
        while self._running:
            await asyncio.sleep(_SUMMARY_INTERVAL)
            if not self._running:
                break
            now = time.perf_counter()
            interval = now - self._summary_at
            self._summary_at = now

            media_delta = self.media_count - self._prev_media_count
            bytes_delta = self.media_bytes_total - self._prev_media_bytes
            transcript_delta = self.transcript_count - self._prev_transcript_count
            tts_delta = self.tts_chunk_count - self._prev_tts_chunks

            self._prev_media_count = self.media_count
            self._prev_media_bytes = self.media_bytes_total
            self._prev_transcript_count = self.transcript_count
            self._prev_tts_chunks = self.tts_chunk_count

            logger.info(
                "[%s] HEALTH | uptime=%.0fs | interval=%.0fs | "
                "media_events=%d (%d total) | bytes=%d (%d total) | "
                "transcripts=%d (%d total) | tts_chunks=%d (%d total) | "
                "stt_reqs=%d | stt_errs=%d | tts_busy=%s | buffer=%d bytes",
                self.bridge_id,
                now - self._started_at, interval,
                media_delta, self.media_count,
                bytes_delta, self.media_bytes_total,
                transcript_delta, self.transcript_count,
                tts_delta, self.tts_chunk_count,
                self.stt_request_count, self.stt_error_count,
                self._tts_busy,
                len(self._pcm_buffer),
            )

    # ---- silence detection and buffer flush ---------------------------------

    async def _start_flush_timer(self) -> None:
        """Start/restart a timer that flushes the buffer after SILENCE_TIMEOUT seconds."""
        if self._flush_task is not None and not self._flush_task.done():
            self._flush_task.cancel()
        self._flush_task = asyncio.create_task(self._flush_after_silence())

    async def _flush_after_silence(self) -> None:
        """Wait for silence, then flush the audio buffer to STT."""
        try:
            await asyncio.sleep(SILENCE_TIMEOUT)
            await self._flush_buffer()
        except asyncio.CancelledError:
            pass

    async def _flush_buffer(self) -> None:
        """Send buffered PCM audio to Pulse STT REST API."""
        if self._flushing:
            return
        buffered = bytes(self._pcm_buffer)
        self._pcm_buffer.clear()

        if len(buffered) < 1600:  # minimum 100ms at 8kHz 16-bit
            logger.debug(
                "[%s] buffer too small to flush (%d bytes), skipping",
                self.bridge_id, len(buffered),
            )
            # Put the bytes back if they were drained
            if buffered:
                self._pcm_buffer.extend(buffered)
            return

        self._flushing = True
        self.stt_request_count += 1
        t0 = time.perf_counter()

        # Convert PCM to WAV for the REST API
        wav_bytes = _pcm_to_wav(buffered, TWILIO_SAMPLE_RATE)
        buffer_duration_ms = (len(buffered) / 2) / TWILIO_SAMPLE_RATE * 1000

        logger.info(
            "[%s] STT flush | pcm_bytes=%d | duration=%.0fms | wav_bytes=%d | req=%d",
            self.bridge_id, len(buffered), buffer_duration_ms,
            len(wav_bytes), self.stt_request_count,
        )

        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "audio/wav",
        }
        params = {"language": "en", "model": "pulse"}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    STT_REST_URL, params=params, data=wav_bytes, headers=headers,
                ) as resp:
                    elapsed = time.perf_counter() - t0
                    body = await resp.text()

                    if resp.status != 200:
                        self.stt_error_count += 1
                        logger.error(
                            "[%s] STT REST error | status=%d | duration=%.3fs | body=%s",
                            self.bridge_id, resp.status, elapsed, body[:300],
                        )
                    else:
                        result = json.loads(body)
                        transcription = result.get("transcription", "")
                        logger.info(
                            "[%s] STT REST success | status=%d | duration=%.3fs | "
                            "transcript=%r | resp_keys=%s",
                            self.bridge_id, resp.status, elapsed,
                            transcription, list(result.keys()),
                        )

                        if transcription:
                            self.transcript_count += 1
                            asyncio.ensure_future(
                                event_store.push(
                                    self.execution_id, "user_transcript",
                                    text=transcription,
                                )
                            )
                            # Send transcript to LLM, then speak the response
                            if not self._tts_busy:
                                self._tts_busy = True
                                asyncio.create_task(self._think_and_speak(transcription))
                        else:
                            logger.info(
                                "[%s] STT REST: no transcription in response",
                                self.bridge_id,
                            )

        except aiohttp.ClientError as e:
            self.stt_error_count += 1
            elapsed = time.perf_counter() - t0
            logger.error(
                "[%s] STT REST connection error | duration=%.3fs | error=%s: %s",
                self.bridge_id, elapsed, type(e).__name__, e,
            )
        except Exception:
            self.stt_error_count += 1
            elapsed = time.perf_counter() - t0
            logger.exception(
                "[%s] STT REST unexpected error | duration=%.3fs", self.bridge_id, elapsed,
            )
        finally:
            self._flushing = False

    # ---- LLM integration -----------------------------------------------------

    async def _think_and_speak(self, user_text: str) -> None:
        """Send transcript to LLM, speak response via TTS, wait for TTS to finish."""
        llm_t0 = time.perf_counter()
        response = await self.electron.chat(user_text)

        if response:
            self.electron.add_to_history("user", user_text)
            self.electron.add_to_history("assistant", response)
            llm_elapsed = time.perf_counter() - llm_t0
            logger.info(
                "[%s] LLM turn complete | duration=%.3fs | response=%r",
                self.bridge_id, llm_elapsed, response[:80],
            )
            asyncio.ensure_future(
                event_store.push(self.execution_id, "agent_response", text=response)
            )
            tts_text = response
        else:
            logger.warning(
                "[%s] LLM returned empty, using fallback | user_text=%r",
                self.bridge_id, user_text[:80],
            )
            tts_text = self._fallback_text

        # Reset completion flag and start speaking
        self._tts_complete = False
        asyncio.ensure_future(
            event_store.push(self.execution_id, "agent_speaking", text=tts_text)
        )
        await self._send_to_tts(tts_text)

        # Wait for TTS playback to finish (TTS listener sets _tts_complete on "complete" msg)
        waited = 0.0
        while not self._tts_complete and waited < 30.0:
            await asyncio.sleep(0.1)
            waited += 0.1

        if not self._tts_complete:
            logger.warning(
                "[%s] TTS did not complete within 30s, releasing lock",
                self.bridge_id,
            )

        # Small extra pause so the caller doesn't start talking over the last syllable
        await asyncio.sleep(0.5)
        asyncio.ensure_future(
            event_store.push(self.execution_id, "agent_done")
        )
        self._tts_busy = False
        # Discard any audio accumulated during TTS playback (speaker echo, ambient)
        discarded = len(self._pcm_buffer)
        self._pcm_buffer.clear()
        self._last_media_at = 0.0
        if self._flush_task:
            self._flush_task.cancel()
            self._flush_task = None
        if discarded > 0:
            logger.debug("[%s] discarded %d bytes of echo after TTS", self.bridge_id, discarded)
        logger.debug("[%s] TTS playback complete, listening again", self.bridge_id)

    # ---- TTS WebSocket -------------------------------------------------------

    async def _connect_tts(self) -> None:
        logger.info(
            "[%s] TTS connecting | url=%s",
            self.bridge_id, LIGHTNING_WS_URL,
        )
        t0 = time.perf_counter()
        try:
            self._tts_session = aiohttp.ClientSession()
            self.tts_ws = await self._tts_session.ws_connect(
                LIGHTNING_WS_URL,
                headers={"Authorization": f"Bearer {API_KEY}"},
            )
            elapsed = time.perf_counter() - t0
            logger.info(
                "[%s] TTS connected | duration=%.3fs",
                self.bridge_id, elapsed,
            )
        except Exception:
            elapsed = time.perf_counter() - t0
            logger.exception(
                "[%s] TTS connection FAILED | duration=%.3fs",
                self.bridge_id, elapsed,
            )
            raise

    async def _send_to_tts(self, text: str) -> None:
        if self.tts_ws is None or self.tts_ws.closed:
            logger.warning(
                "[%s] TTS WS not connected, reconnecting | text=%r",
                self.bridge_id, text[:50],
            )
            await self._connect_tts()

        if self.tts_ws is None:
            logger.error("[%s] TTS WS still None after reconnect", self.bridge_id)
            self._tts_busy = False
            return

        payload = {
            "model": self.tts_model,
            "text": text,
            "voice_id": "sophia",
            "sample_rate": TWILIO_SAMPLE_RATE,
            "output_format": "ulaw",
            "language": "en",
            "speed": 1.0,
        }
        logger.info(
            "[%s] TTS request | text=%r | voice=%s | format=%s | rate=%d",
            self.bridge_id, text[:80], payload["voice_id"],
            payload["output_format"], payload["sample_rate"],
        )
        self._tts_chunks_this_utterance = 0
        await self.tts_ws.send_json(payload)

    async def _tts_cooldown(self, seconds: float = 4.0) -> None:
        await asyncio.sleep(seconds)
        self._tts_busy = False
        logger.debug("[%s] TTS cooldown ended", self.bridge_id)

    async def _listen_tts(self) -> None:
        if self.tts_ws is None:
            return
        self._tts_complete = False
        logger.info("[%s] TTS listener started", self.bridge_id)
        msg_idx = 0
        try:
            async for msg in self.tts_ws:
                msg_idx += 1
                if msg.type == aiohttp.WSMsgType.BINARY:
                    self.tts_chunk_count += 1
                    self.tts_bytes_total += len(msg.data)
                    asyncio.ensure_future(self._send_audio_to_twilio(msg.data))
                elif msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        self._handle_tts_text_message(msg.data, msg_idx)
                    except Exception:
                        logger.exception(
                            "[%s] TTS msg error | msg=%d", self.bridge_id, msg_idx,
                        )
                elif msg.type == aiohttp.WSMsgType.CLOSED:
                    logger.info(
                        "[%s] TTS WS closed | code=%s",
                        self.bridge_id,
                        msg.data if hasattr(msg, 'data') else '?',
                    )
                    break
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(
                        "[%s] TTS WS error | exception=%s",
                        self.bridge_id, self.tts_ws.exception(),
                    )
                    break
        except asyncio.CancelledError:
            logger.info("[%s] TTS listener cancelled", self.bridge_id)
        except Exception:
            logger.exception("[%s] TTS listener error", self.bridge_id)
        finally:
            logger.info(
                "[%s] TTS listener stopped | messages=%d | chunks=%d | bytes=%d",
                self.bridge_id, msg_idx, self.tts_chunk_count, self.tts_bytes_total,
            )

    def _handle_tts_text_message(self, raw_data: str, msg_idx: int) -> None:
        """Parse and handle a single TTS WS text message."""
        # smallest.ai TTS WS sends JSON text: {"status":"chunk", "data":{"audio":"<base64>"}}
        try:
            text_data = json.loads(raw_data)
        except json.JSONDecodeError:
            logger.info(
                "[%s] TTS text (non-JSON) | msg=%d | data=%s",
                self.bridge_id, msg_idx, raw_data[:200],
            )
            return
        status = text_data.get("status", "?")
        if status == "chunk":
            audio_b64 = text_data.get("data", {}).get("audio")
            if audio_b64:
                pcm_bytes = base64.b64decode(audio_b64)
                # TTS WS returns 16-bit PCM regardless of requested format.
                # Twilio expects mu-law at 8kHz; convert here.
                mulaw_bytes = _pcm_to_mulaw(pcm_bytes)
                self.tts_chunk_count += 1
                self._tts_chunks_this_utterance += 1
                self.tts_bytes_total += len(mulaw_bytes)
                asyncio.ensure_future(self._send_audio_to_twilio(mulaw_bytes))
                if self._tts_chunks_this_utterance <= 3 or self._tts_chunks_this_utterance % 20 == 0:
                    logger.debug(
                        "[%s] TTS chunk | msg=%d | pcm=%d -> mulaw=%d bytes",
                        self.bridge_id, msg_idx, len(pcm_bytes), len(mulaw_bytes),
                    )
            else:
                logger.info(
                    "[%s] TTS chunk (no audio) | msg=%d | keys=%s",
                    self.bridge_id, msg_idx, list(text_data.get("data", {}).keys()),
                )
        elif status == "complete":
            logger.info(
                "[%s] TTS complete | utterance_chunks=%d | total_chunks=%d | total_bytes=%d",
                self.bridge_id, self._tts_chunks_this_utterance,
                self.tts_chunk_count, self.tts_bytes_total,
            )
            self._tts_complete = True
        else:
            logger.info(
                "[%s] TTS status=%s | msg=%d | data=%s",
                self.bridge_id, status, msg_idx, raw_data[:200],
            )

    # ---- Twilio outbound ----------------------------------------------------

    async def _send_audio_to_twilio(self, audio_bytes: bytes) -> None:
        if self.twilio_ws is None:
            return
        encoded = base64.b64encode(audio_bytes).decode("utf-8")
        message = {
            "event": "media",
            "streamSid": self.stream_sid or "stream_sid_placeholder",
            "media": {"payload": encoded},
        }
        try:
            await self.twilio_ws.send_text(json.dumps(message))
        except Exception:
            self.twilio_send_errors += 1
            logger.exception(
                "[%s] Twilio send failed | chunk_size=%d",
                self.bridge_id, len(audio_bytes),
            )

    # ---- Twilio inbound -----------------------------------------------------

    async def handle_twilio_message(self, message_data: Dict[str, Any]) -> None:
        event = message_data.get("event", "")

        if event == "start":
            start_info = message_data.get("start", {})
            self.stream_sid = start_info.get("streamSid")
            logger.info(
                "[%s] TWILIO event=start | streamSid=%s | start_keys=%s",
                self.bridge_id, self.stream_sid, list(start_info.keys()),
            )
            asyncio.ensure_future(
                event_store.push(self.execution_id, "call_connected",
                                 stream_sid=self.stream_sid)
            )

            # Connect TTS (only TTS; STT uses REST, no WS needed)
            t0 = time.perf_counter()
            try:
                await self._connect_tts()
            except Exception:
                logger.exception(
                    "[%s] Failed to connect TTS", self.bridge_id,
                )
                raise

            connect_elapsed = time.perf_counter() - t0
            logger.info(
                "[%s] TTS connection ready | duration=%.3fs",
                self.bridge_id, connect_elapsed,
            )

            # Start TTS listener
            self._tts_task = asyncio.create_task(self._listen_tts())

        elif event == "media":
            media = message_data.get("media", {})
            payload_b64 = media.get("payload")
            if not payload_b64:
                return

            self.media_count += 1
            if self._first_media_at == 0:
                self._first_media_at = time.perf_counter()
            now = time.perf_counter()
            self._last_media_at = now

            # Decode base64 mu-law audio
            mulaw_bytes = base64.b64decode(payload_b64)
            mulaw_len = len(mulaw_bytes)
            self.media_bytes_total += mulaw_len

            # Log first few media events and periodic milestones
            if self.media_count <= 5:
                logger.debug(
                    "[%s] TWILIO event=media | seq=%d | mulaw=%d bytes",
                    self.bridge_id, self.media_count, mulaw_len,
                )
            elif self.media_count % 100 == 0:
                since_last = now - (self._last_media_at or now)
                logger.debug(
                    "[%s] TWILIO event=media | seq=%d | buffer=%d bytes",
                    self.bridge_id, self.media_count, len(self._pcm_buffer),
                )

            # Convert mu-law to linear16 PCM and append to buffer
            pcm_bytes = _ulaw_to_pcm(mulaw_bytes)
            self._pcm_buffer.extend(pcm_bytes)

            # Auto-flush if buffer exceeds max size
            if len(self._pcm_buffer) >= MAX_BUFFER_BYTES:
                logger.info(
                    "[%s] buffer full (%d bytes), forcing flush",
                    self.bridge_id, len(self._pcm_buffer),
                )
                await self._flush_buffer()

            # Restart silence timer
            await self._start_flush_timer()

        elif event == "stop":
            logger.info(
                "[%s] TWILIO event=stop | "
                "media_total=%d | bytes_total=%d | buffer=%d bytes | transcripts=%d",
                self.bridge_id, self.media_count,
                self.media_bytes_total, len(self._pcm_buffer),
                self.transcript_count,
            )
            # Flush remaining buffer
            await self._flush_buffer()
            asyncio.ensure_future(
                event_store.push(self.execution_id, "call_ended")
            )

        else:
            logger.debug(
                "[%s] TWILIO unhandled event=%s | keys=%s",
                self.bridge_id, event, list(message_data.keys()),
            )

    # ---- cleanup ------------------------------------------------------------

    async def cleanup(self) -> None:
        self._running = False
        uptime = time.perf_counter() - self._started_at if self._started_at else 0

        # Cancel flush timer
        if self._flush_task is not None and not self._flush_task.done():
            self._flush_task.cancel()

        # Flush any remaining audio
        if len(self._pcm_buffer) >= 1600:
            logger.info(
                "[%s] final flush | buffer=%d bytes",
                self.bridge_id, len(self._pcm_buffer),
            )
            await self._flush_buffer()

        logger.info(
            "[%s] CLEANUP START | uptime=%.1fs | "
            "media=%d | media_bytes=%d | transcripts=%d | "
            "tts_chunks=%d | tts_bytes=%d | "
            "stt_reqs=%d | stt_errs=%d | twilio_errs=%d | buffer=%d",
            self.bridge_id, uptime,
            self.media_count, self.media_bytes_total,
            self.transcript_count,
            self.tts_chunk_count, self.tts_bytes_total,
            self.stt_request_count, self.stt_error_count,
            self.twilio_send_errors, len(self._pcm_buffer),
        )

        # Cancel tasks
        for name, task in [
            ("tts", self._tts_task),
            ("summary", self._summary_task),
            ("flush", self._flush_task),
        ]:
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    logger.debug("[%s] %s task cancelled", self.bridge_id, name)
                except Exception:
                    logger.exception(
                        "[%s] %s task error during cancel", self.bridge_id, name,
                    )

        # Close TTS WebSocket
        if self.tts_ws is not None and not self.tts_ws.closed:
            try:
                await self.tts_ws.close()
                logger.info("[%s] TTS WS closed cleanly", self.bridge_id)
            except Exception:
                logger.exception("[%s] TTS WS close error", self.bridge_id)
        self.tts_ws = None

        # Close TTS session
        if self._tts_session is not None:
            try:
                await self._tts_session.close()
            except Exception:
                logger.exception("[%s] TTS session close error", self.bridge_id)
        self._tts_session = None

        logger.info("[%s] CLEANUP DONE", self.bridge_id)


async def create_smallest_ai_bridge(
    execution_id: str,
    twilio_websocket,
    template_metadata: Optional[dict] = None,
) -> SmallestAiBridge:
    """Factory function matching create_deepgram_bridge_v2 signature."""
    logger.info("create_smallest_ai_bridge | execution_id=%s", execution_id)
    bridge = SmallestAiBridge(execution_id)
    await bridge.initialize(twilio_websocket)
    return bridge
