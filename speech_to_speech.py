"""
Real-time speech-to-speech bridge using smallest.ai Pulse (STT), LLM,
and Lightning (TTS).

Architecture:
    Twilio WS --mu-law--> decode --PCM--> STT WS (streaming)
                                               |
                                         transcription (is_final)
                                               |
                                               v
                                    LLM API --response--> TTS WS (streaming)
                                                               |
                                          Twilio WS <--mu-law-- encode

STT uses the WebSocket streaming endpoint for real-time transcription
with server-side segmentation. LLM uses OpenAI-compatible chat completions.
TTS uses the WebSocket streaming endpoint for low-latency playback.
"""

import asyncio
import base64
import json
import logging
import os
import re
import time
import uuid
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
STT_WS_URL = "wss://api.smallest.ai/waves/v1/stt/live?model=pulse"
LIGHTNING_WS_URL = "wss://api.smallest.ai/waves/v1/tts/live"

# Audio format constants
TWILIO_SAMPLE_RATE = 8000

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


class SmallestAiBridge:
    """
    Real-time bridge between Twilio Media Streams and smallest.ai.

    Flow:
    - Twilio sends mu-law audio (base64-encoded in JSON media events).
    - Bridge decodes mu-law -> linear16 PCM -> forwards to STT WebSocket.
    - STT returns streaming transcripts (interim + final).
    - On final transcript, LLM generates a response.
    - Response text is sent to TTS WebSocket for streaming audio back to Twilio.
    """

    def __init__(self, execution_id: str, response_text: str | None = None, model: str = "lightning_v3.1"):
        self.execution_id = execution_id
        self.bridge_id = str(uuid.uuid4())[:8]
        self.tts_model = model
        self._fallback_text = response_text or (
            "I received your message through the smallest AI real time pipeline."
        )

        self.twilio_ws: Any = None
        self.tts_ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self.stt_ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self.stream_sid: Optional[str] = None

        self._stt_task: Optional[asyncio.Task] = None
        self._tts_task: Optional[asyncio.Task] = None
        self._summary_task: Optional[asyncio.Task] = None
        self._running = False
        self._tts_busy = False
        self._tts_done_event = asyncio.Event()
        self._tts_should_stop = False
        self._tts_session: Optional[aiohttp.ClientSession] = None
        self._stt_session: Optional[aiohttp.ClientSession] = None

        # LLM client for conversational responses
        self.electron: ElectronClient = ElectronClient()

        # Timing
        self._started_at: float = 0.0
        self._first_media_at: float = 0.0

        # Diagnostic counters
        self.media_count = 0
        self.media_bytes_total = 0
        self.transcript_count = 0
        self.stt_stream_bytes = 0
        self.stt_error_count = 0
        self.tts_chunk_count = 0
        self.tts_bytes_total = 0
        self._tts_chunks_this_utterance = 0
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
                "stt_stream=%d bytes | stt_errs=%d | tts_busy=%s",
                self.bridge_id,
                now - self._started_at, interval,
                media_delta, self.media_count,
                bytes_delta, self.media_bytes_total,
                transcript_delta, self.transcript_count,
                tts_delta, self.tts_chunk_count,
                self.stt_stream_bytes, self.stt_error_count,
                self._tts_busy,
            )

    # ---- STT WebSocket -------------------------------------------------------

    async def _connect_stt(self) -> None:
        logger.info(
            "[%s] STT connecting | url=%s",
            self.bridge_id, STT_WS_URL,
        )
        t0 = time.perf_counter()
        try:
            self._stt_session = aiohttp.ClientSession()
            self.stt_ws = await self._stt_session.ws_connect(
                STT_WS_URL,
                headers={"Authorization": f"Bearer {API_KEY}"},
            )
            elapsed = time.perf_counter() - t0
            logger.info(
                "[%s] STT connected | duration=%.3fs",
                self.bridge_id, elapsed,
            )
        except Exception:
            elapsed = time.perf_counter() - t0
            logger.exception(
                "[%s] STT connection FAILED | duration=%.3fs",
                self.bridge_id, elapsed,
            )
            raise

    async def _send_audio_to_stt(self, pcm_bytes: bytes) -> None:
        """Send raw 16-bit PCM audio to STT WebSocket as binary."""
        if self.stt_ws is None or self.stt_ws.closed:
            logger.warning("[%s] STT WS not connected, reconnecting", self.bridge_id)
            await self._connect_stt()
        if self.stt_ws is None:
            return

        try:
            await self.stt_ws.send_bytes(pcm_bytes)
            self.stt_stream_bytes += len(pcm_bytes)
        except Exception:
            self.stt_error_count += 1
            logger.exception("[%s] STT WS send failed", self.bridge_id)

    async def _listen_stt(self) -> None:
        """Listen for transcript messages from STT WebSocket."""
        if self.stt_ws is None:
            return
        logger.info("[%s] STT listener started", self.bridge_id)
        msg_idx = 0
        try:
            async for msg in self.stt_ws:
                msg_idx += 1
                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        await self._handle_stt_text_message(msg.data, msg_idx)
                    except Exception:
                        logger.exception(
                            "[%s] STT msg error | msg=%d", self.bridge_id, msg_idx,
                        )
                elif msg.type == aiohttp.WSMsgType.CLOSED:
                    logger.info(
                        "[%s] STT WS closed | code=%s",
                        self.bridge_id,
                        msg.data if hasattr(msg, 'data') else '?',
                    )
                    break
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(
                        "[%s] STT WS error | exception=%s",
                        self.bridge_id, self.stt_ws.exception(),
                    )
                    break
                else:
                    logger.debug(
                        "[%s] STT unknown msg type=%s | msg=%d",
                        self.bridge_id, msg.type, msg_idx,
                    )
        except asyncio.CancelledError:
            logger.info("[%s] STT listener cancelled", self.bridge_id)
        except Exception:
            logger.exception("[%s] STT listener error", self.bridge_id)
        finally:
            logger.info(
                "[%s] STT listener stopped | messages=%d | stream_bytes=%d",
                self.bridge_id, msg_idx, self.stt_stream_bytes,
            )

    async def _handle_stt_text_message(self, raw_data: str, msg_idx: int) -> None:
        """Parse and handle a single STT WS text message."""
        try:
            data = json.loads(raw_data)
        except json.JSONDecodeError:
            logger.info(
                "[%s] STT text (non-JSON) | msg=%d | data=%s",
                self.bridge_id, msg_idx, raw_data[:200],
            )
            return

        msg_type = data.get("type", "?")

        if msg_type == "transcription":
            # smallest.ai STT live sends flat JSON: {"type":"transcription", "transcript":"...", "is_final":true}
            transcription = data.get("transcript", "").strip()
            is_final = data.get("is_final", False)

            if is_final and transcription:
                self.transcript_count += 1
                logger.info(
                    "[%s] STT final transcript | msg=%d | text=%r",
                    self.bridge_id, msg_idx, transcription,
                )
                asyncio.ensure_future(
                    event_store.push(
                        self.execution_id, "user_transcript",
                        text=transcription,
                    )
                )
                if self._tts_busy:
                    logger.info(
                        "[%s] STT barge-in | text=%r",
                        self.bridge_id, transcription[:80],
                    )
                    await self._stop_current_tts()
                self._tts_should_stop = False
                self._tts_busy = True
                asyncio.create_task(self._think_and_speak(transcription))
            elif transcription:
                logger.debug(
                    "[%s] STT interim | msg=%d | text=%r",
                    self.bridge_id, msg_idx, transcription,
                )
            else:
                logger.debug(
                    "[%s] STT transcript (empty) | msg=%d", self.bridge_id, msg_idx,
                )
        else:
            logger.info(
                "[%s] STT type=%s | msg=%d | data=%s",
                self.bridge_id, msg_type, msg_idx, raw_data[:200],
            )

    # ---- LLM integration -----------------------------------------------------

    async def _think_and_speak(self, user_text: str) -> None:
        """Stream LLM response, speak each sentence as soon as it's ready."""
        llm_t0 = time.perf_counter()
        full_response = ""
        current = ""
        sentences: list[str] = []
        sent_end = re.compile(r'([.!?])\s')

        # Stream tokens from LLM, split into sentences
        async for chunk in self.electron.stream_chunks(user_text):
            if self._tts_should_stop:
                break
            full_response += chunk
            current += chunk

            # Extract complete sentences (.!? followed by whitespace)
            m = sent_end.search(current)
            while m:
                idx = m.end()
                sentence = current[:idx].strip()
                if sentence:
                    sentences.append(sentence)
                current = current[idx:].lstrip()
                m = sent_end.search(current)

        # Any remaining text is the last sentence
        if current.strip():
            sentences.append(current.strip())

        llm_elapsed = time.perf_counter() - llm_t0

        if full_response:
            self.electron.add_to_history("user", user_text)
            self.electron.add_to_history("assistant", full_response.strip())
            logger.info(
                "[%s] LLM streaming done | duration=%.3fs | response=%r | sentences=%d",
                self.bridge_id, llm_elapsed, full_response.strip()[:80], len(sentences),
            )
            asyncio.ensure_future(
                event_store.push(self.execution_id, "agent_response",
                                 text=full_response.strip())
            )
        else:
            logger.warning(
                "[%s] LLM returned empty, using fallback | user_text=%r",
                self.bridge_id, user_text[:80],
            )
            sentences = [self._fallback_text]

        # Speak each sentence one at a time, supporting barge-in between them
        for sentence in sentences:
            if self._tts_should_stop:
                logger.info("[%s] TTS interrupted by barge-in", self.bridge_id)
                break

            asyncio.ensure_future(
                event_store.push(self.execution_id, "agent_speaking", text=sentence)
            )
            self._tts_done_event.clear()
            await self._send_to_tts(sentence)

            # Wait for TTS sentence to complete, checking for barge-in
            waited = 0.0
            while not self._tts_done_event.is_set() and waited < 30.0:
                if self._tts_should_stop:
                    break
                await asyncio.sleep(0.05)
                waited += 0.05

        asyncio.ensure_future(
            event_store.push(self.execution_id, "agent_done")
        )
        self._tts_busy = False
        logger.debug("[%s] streaming TTS playback complete", self.bridge_id)

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

    async def _stop_current_tts(self) -> None:
        """Stop current TTS playback and reconnect for barge-in."""
        self._tts_should_stop = True
        logger.info("[%s] TTS barge-in: stopping current playback", self.bridge_id)

        # Cancel TTS listener (stops processing incoming audio chunks)
        if self._tts_task and not self._tts_task.done():
            self._tts_task.cancel()
            try:
                await self._tts_task
            except (asyncio.CancelledError, Exception):
                pass

        # Close TTS WS to cut audio mid-utterance
        if self.tts_ws and not self.tts_ws.closed:
            try:
                await self.tts_ws.close()
            except Exception:
                pass
        self.tts_ws = None

        # Close TTS session
        if self._tts_session:
            try:
                await self._tts_session.close()
            except Exception:
                pass
            self._tts_session = None

        # Reconnect TTS WS for the new response
        await self._connect_tts()
        self._tts_task = asyncio.create_task(self._listen_tts())

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

    async def _listen_tts(self) -> None:
        if self.tts_ws is None:
            return
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
            self._tts_done_event.set()
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

            # Connect STT and TTS WebSockets
            t0 = time.perf_counter()
            try:
                await self._connect_stt()
                await self._connect_tts()
            except Exception:
                logger.exception("[%s] Failed to connect STT/TTS", self.bridge_id)
                raise

            connect_elapsed = time.perf_counter() - t0
            logger.info(
                "[%s] STT/TTS connections ready | duration=%.3fs",
                self.bridge_id, connect_elapsed,
            )

            # Start STT and TTS listeners
            self._stt_task = asyncio.create_task(self._listen_stt())
            self._tts_task = asyncio.create_task(self._listen_tts())

        elif event == "media":
            media = message_data.get("media", {})
            payload_b64 = media.get("payload")
            if not payload_b64:
                return

            self.media_count += 1
            if self._first_media_at == 0:
                self._first_media_at = time.perf_counter()

            # Decode base64 mu-law audio and forward to STT WebSocket
            mulaw_bytes = base64.b64decode(payload_b64)
            self.media_bytes_total += len(mulaw_bytes)

            pcm_bytes = _ulaw_to_pcm(mulaw_bytes)
            asyncio.ensure_future(self._send_audio_to_stt(pcm_bytes))

            if self.media_count % 100 == 0:
                logger.debug(
                    "[%s] TWILIO event=media | seq=%d | stt_stream=%d bytes",
                    self.bridge_id, self.media_count, self.stt_stream_bytes,
                )

        elif event == "stop":
            logger.info(
                "[%s] TWILIO event=stop | "
                "media_total=%d | bytes_total=%d | transcripts=%d",
                self.bridge_id, self.media_count,
                self.media_bytes_total, self.transcript_count,
            )
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

        logger.info(
            "[%s] CLEANUP START | uptime=%.1fs | "
            "media=%d | media_bytes=%d | transcripts=%d | "
            "tts_chunks=%d | tts_bytes=%d | "
            "stt_stream=%d bytes | stt_errs=%d | twilio_errs=%d",
            self.bridge_id, uptime,
            self.media_count, self.media_bytes_total,
            self.transcript_count,
            self.tts_chunk_count, self.tts_bytes_total,
            self.stt_stream_bytes, self.stt_error_count,
            self.twilio_send_errors,
        )

        # Cancel tasks
        for name, task in [
            ("stt", self._stt_task),
            ("tts", self._tts_task),
            ("summary", self._summary_task),
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

        # Close STT WebSocket
        if self.stt_ws is not None and not self.stt_ws.closed:
            try:
                await self.stt_ws.close()
                logger.info("[%s] STT WS closed cleanly", self.bridge_id)
            except Exception:
                logger.exception("[%s] STT WS close error", self.bridge_id)
        self.stt_ws = None

        # Close STT session
        if self._stt_session is not None:
            try:
                await self._stt_session.close()
            except Exception:
                logger.exception("[%s] STT session close error", self.bridge_id)
        self._stt_session = None

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
