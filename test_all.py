#!/usr/bin/env python3
"""
End-to-end test runner for smallest.ai TTS, STT, and Twilio bridge simulation.

Usage:
    cd backend
    PYTHONPATH=backend python smallest-test/test_all.py

Sections:
    1. TTS: synthesize text, save WAV to output/
    2. STT: transcribe the WAV from (1), verify text matches
    3. Bridge: monkeypatch -> FastAPI TestClient WebSocket -> simulate Twilio events
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import sys
import time
import traceback
import uuid
from pathlib import Path

# Ensure backend modules are importable
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

# ---------------------------------------------------------------------------
# Logging setup (console + file)
# ---------------------------------------------------------------------------

_LOGS_DIR = Path(__file__).resolve().parent / "logs"
_LOGS_DIR.mkdir(exist_ok=True)
_LOG_FILE = _LOGS_DIR / "test_all.log"

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stderr),
        logging.FileHandler(str(_LOG_FILE)),
    ],
)
test_logger = logging.getLogger("smallest.test")

# ---------------------------------------------------------------------------
# Test result tracking
# ---------------------------------------------------------------------------

passed = 0
failed = 0


def ok(label: str) -> None:
    global passed
    passed += 1
    msg = f"PASS | {label}"
    test_logger.info(msg)


def fail(label: str, detail: str = "") -> None:
    global failed
    failed += 1
    msg = f"FAIL | {label}" + (f" | detail={detail}" if detail else "")
    test_logger.error(msg)


# ---------------------------------------------------------------------------
# Phase 1: TTS
# ---------------------------------------------------------------------------

def test_tts():
    test_logger.info("=" * 40)
    test_logger.info("Phase 1: TTS (smallest.ai Lightning v3.1)")
    test_logger.info("=" * 40)

    from smallest_test.tts import synthesize

    text = "Hello world. This is a test of the smallest AI text to speech system."
    voice = "sophia"
    test_logger.info("TTS test: text=%r voice=%s", text, voice)

    try:
        t0 = time.perf_counter()
        audio_bytes = asyncio.run(synthesize(text, voice_id=voice))
        elapsed = time.perf_counter() - t0
        test_logger.info("TTS call duration=%.3fs", elapsed)
    except Exception as e:
        test_logger.exception("synthesize() raised")
        fail("synthesize() raised", str(e))
        return None

    if not audio_bytes:
        fail("synthesize() returned empty audio")
        return None

    ok(f"synthesize() returned {len(audio_bytes)} bytes of audio")

    # Save to output/ directory
    output_dir = Path(__file__).resolve().parent / "output"
    output_dir.mkdir(exist_ok=True)
    wav_path = output_dir / f"tts_{uuid.uuid4().hex[:8]}.wav"
    wav_path.write_bytes(audio_bytes)

    ok(f"TTS audio saved to {wav_path}")

    # Quick sanity: WAV files start with "RIFF"
    if audio_bytes[:4] == b"RIFF":
        ok("Audio appears to be valid WAV (RIFF header)")
    else:
        fail(f"Audio missing RIFF header — first 4 bytes: {audio_bytes[:4]!r}")

    return wav_path


# ---------------------------------------------------------------------------
# Phase 2: STT
# ---------------------------------------------------------------------------

def test_stt(wav_path: Path | None):
    test_logger.info("=" * 40)
    test_logger.info("Phase 2: STT (smallest.ai Pulse)")
    test_logger.info("=" * 40)

    if wav_path is None or not wav_path.exists():
        fail("No WAV file from Phase 1 — skipping STT test")
        return

    from smallest_test.stt import transcribe

    audio_bytes = wav_path.read_bytes()
    test_logger.info("STT test: wav=%s size=%d bytes", wav_path.name, len(audio_bytes))

    try:
        t0 = time.perf_counter()
        transcription = asyncio.run(transcribe(audio_bytes, language="en"))
        elapsed = time.perf_counter() - t0
        test_logger.info("STT call duration=%.3fs", elapsed)
    except Exception as e:
        test_logger.exception("transcribe() raised")
        fail("transcribe() raised", str(e))
        return

    if not transcription:
        fail("transcribe() returned empty string")
        return

    ok(f"transcription: '{transcription}'")

    # Verify the transcription contains expected words (case-insensitive)
    lower = transcription.lower()
    expected_words = ["hello", "world", "test"]
    found = [w for w in expected_words if w in lower]

    if len(found) >= 2:
        ok(f"Transcription contains expected words: {found}")
    elif len(found) == 1:
        fail(f"Only 1/{len(expected_words)} expected words found: {found}")
    else:
        fail(f"No expected words found in: '{transcription}'", "STT may have misrecognized")


# ---------------------------------------------------------------------------
# Phase 3: Bridge simulation
# ---------------------------------------------------------------------------

class FakeSmallestAiBridge:
    """Stub bridge that records Twilio events without connecting to smallest.ai."""

    def __init__(self, execution_id: str):
        self.execution_id = execution_id
        self.twilio_ws = None
        self.received_start: dict | None = None
        self.media_payloads: list[str] = []
        self.started = False
        self.cleaned = False
        self.stream_sid: str | None = None

    async def initialize(self, twilio_websocket):
        self.twilio_ws = twilio_websocket

    async def start_bridge(self):
        self.started = True

    async def handle_twilio_message(self, message_data: dict):
        event = message_data.get("event", "")
        if event == "start":
            self.received_start = message_data.get("start") or {}
            self.stream_sid = self.received_start.get("streamSid")
        elif event == "media":
            media = message_data.get("media") or {}
            payload = media.get("payload")
            if payload:
                self.media_payloads.append(payload)

    async def cleanup(self):
        self.cleaned = True


def test_bridge_simulation():
    test_logger.info("=" * 40)
    test_logger.info("Phase 3: Bridge simulation")
    test_logger.info("=" * 40)

    # Lazy imports so we don't trigger side-effects if earlier phases fail
    from fastapi import FastAPI
    from starlette.testclient import TestClient

    import audio.deepgram_bridge_v2 as deepgram_bridge_v2
    from api.audio import router as audio_router

    fake_bridge = FakeSmallestAiBridge("exec-smallest-test")

    async def fake_get_execution(execution_id: str):
        assert execution_id == "exec-smallest-test"
        return {
            "execution_id": execution_id,
            "extra_data": {"language": "en"},
            "template_metadata": {},
        }

    async def fake_create_bridge(execution_id, twilio_websocket, template_metadata=None):
        assert execution_id == "exec-smallest-test"
        await fake_bridge.initialize(twilio_websocket)
        return fake_bridge

    # Apply monkeypatches via unittest.mock (stdlib)
    from unittest.mock import patch

    test_logger.info("Bridge simulation: setting up monkeypatches")
    t0 = time.perf_counter()

    with (
        patch(
            "services.executions.create.get_execution",
            side_effect=fake_get_execution,
        ),
        patch.object(
            deepgram_bridge_v2,
            "create_deepgram_bridge_v2",
            side_effect=fake_create_bridge,
        ),
    ):
        app = FastAPI()
        app.include_router(audio_router, prefix="/audio")

        media_payload_1 = base64.b64encode(b"twilio-mulaw-chunk-1\x00\x01").decode()
        media_payload_2 = base64.b64encode(b"twilio-mulaw-chunk-2\x02\x03").decode()

        with TestClient(app) as client:
            with client.websocket_connect("/audio/exec-smallest-test") as ws:
                ws.send_text(
                    json.dumps(
                        {
                            "event": "start",
                            "start": {"streamSid": "MZSMALLEST01"},
                        }
                    )
                )
                ws.send_text(
                    json.dumps(
                        {"event": "media", "media": {"payload": media_payload_1}}
                    )
                )
                ws.send_text(
                    json.dumps(
                        {"event": "media", "media": {"payload": media_payload_2}}
                    )
                )

    elapsed = time.perf_counter() - t0
    test_logger.info("Bridge simulation: WebSocket round-trip duration=%.3fs", elapsed)

    # Assertions
    if fake_bridge.started:
        ok("bridge.start_bridge() was called")
    else:
        fail("bridge.start_bridge() was NOT called")

    if fake_bridge.received_start == {"streamSid": "MZSMALLEST01"}:
        ok("bridge received start event with correct streamSid")
    else:
        fail(f"unexpected received_start: {fake_bridge.received_start}")

    if fake_bridge.media_payloads == [media_payload_1, media_payload_2]:
        ok("bridge received both media payloads")
    else:
        fail(
            f"expected 2 payloads, got {len(fake_bridge.media_payloads)}: "
            f"{fake_bridge.media_payloads}"
        )

    if fake_bridge.cleaned:
        ok("bridge.cleanup() was called after WebSocket close")
    else:
        fail("bridge.cleanup() was NOT called")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_start = time.perf_counter()
    test_logger.info("=" * 60)
    test_logger.info("smallest.ai Integration Test Suite START")
    test_logger.info("=" * 60)

    # Phase 1
    wav_path = test_tts()

    # Phase 2
    test_stt(wav_path)

    # Phase 3
    test_bridge_simulation()

    # Summary
    total = passed + failed
    total_time = time.perf_counter() - t_start
    test_logger.info("=" * 60)
    test_logger.info(
        "RESULTS: %d/%d passed, %d/%d failed | total_time=%.1fs | log=%s",
        passed, total, failed, total, total_time, _LOG_FILE,
    )
    test_logger.info("=" * 60)

    # Also print to stdout for direct visibility
    print()
    print("=" * 60)
    print(f"Results: {passed}/{total} passed, {failed}/{total} failed  (log: {_LOG_FILE})")
    if failed > 0:
        print("Some tests FAILED. Check output above for details.")
        return 1
    else:
        print("All tests PASSED.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
