"""
Emulate a multi-turn conversation without making a real call.
Tests: STT → LLM → TTS pipeline with busy-lock behavior.
"""
import asyncio
import base64
import json
import logging
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
BACKEND_ROOT = HERE.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    stream=sys.stderr,
)
logging.getLogger("smallest.bridge").setLevel(logging.DEBUG)
logging.getLogger("smallest.llm").setLevel(logging.INFO)
logger = logging.getLogger("smallest.test")


class FakeTwilioWs:
    """Mimics a Twilio WebSocket for testing."""
    def __init__(self):
        self.sent: list[dict] = []
        self.closed = False

    async def send_text(self, text: str):
        self.sent.append(json.loads(text))
        logger.info("  [Twilio] SENT media chunk: %d bytes base64", len(self.sent[-1].get("media", {}).get("payload", "")))

    async def close(self):
        self.closed = True

    async def accept(self):
        pass


async def main():
    from smallest_test.speech_to_speech import SmallestAiBridge

    print("=" * 60)
    print("Multi-turn conversation emulation test")
    print("=" * 60)

    bridge = SmallestAiBridge("emulated-conversation-test")
    ws = FakeTwilioWs()
    await bridge.initialize(ws)

    # ---- Turn 1: Simulate "start" + user speaks ----
    print("\n--- Turn 1: User says hello ---")
    t0 = time.perf_counter()

    # start event
    await bridge.handle_twilio_message({
        "event": "start",
        "start": {"streamSid": "MZ_EMULATED_01"},
    })
    # start the bridge loop
    bridge._running = True
    bridge._summary_task = asyncio.create_task(bridge._periodic_summary())

    # Simulate ~10s of audio (500 media events with mu-law silence + some fake audio)
    for i in range(500):
        awake = base64.b64encode(
            bytes([0xff] * 80 + [0x7f, 0xff, 0x55, 0xaa, 0x00, 0x7f] * 12)
        ).decode()
        await bridge.handle_twilio_message({
            "event": "media",
            "media": {"payload": awake},
        })
    # Let silence timer flush
    await asyncio.sleep(2.0)

    # Force a fresh STT flush with the buffer
    if len(bridge._pcm_buffer) > 0:
        await bridge._flush_buffer()
    await asyncio.sleep(0.5)

    logger.info("Turn 1: transcripts=%d, tts_chunks=%d, tts_busy=%s",
                bridge.transcript_count, bridge.tts_chunk_count, bridge._tts_busy)

    # ---- Wait for AI to finish speaking ----
    print("\n--- Waiting for AI to finish speaking... ---")
    patience = 30
    while bridge._tts_busy and patience > 0:
        await asyncio.sleep(0.5)
        patience -= 0.5
    logger.info("After Turn 1 wait: tts_busy=%s, tts_chunks=%d, transcript_count=%d",
                bridge._tts_busy, bridge.tts_chunk_count, bridge.transcript_count)

    if bridge._tts_busy:
        logger.error("TTS still BUSY after 30s — turn 2 would be dropped!")
    elif bridge.tts_chunk_count == 0:
        logger.error("No TTS chunks — AI never spoke!")
    else:
        logger.info("TTS finished, lock released — ready for turn 2")

    # ---- Turn 2: User asks a follow-up ----
    print("\n--- Turn 2: User asks follow-up ---")
    t1 = time.perf_counter()
    bridge._pcm_buffer.clear()
    bridge._flushing = False

    # simulate another ~10s of audio
    for i in range(500):
        awake = base64.b64encode(
            bytes([0xff] * 80 + [0x7f, 0xff, 0x55, 0xaa, 0x00, 0x7f] * 12)
        ).decode()
        await bridge.handle_twilio_message({
            "event": "media",
            "media": {"payload": awake},
        })
    await asyncio.sleep(2.0)

    if len(bridge._pcm_buffer) > 0:
        await bridge._flush_buffer()
    await asyncio.sleep(0.5)

    logger.info("Turn 2: transcripts=%d, tts_chunks=%d, tts_busy=%s",
                bridge.transcript_count, bridge.tts_chunk_count, bridge._tts_busy)

    # Wait for AI to finish speaking again
    patience = 30
    while bridge._tts_busy and patience > 0:
        await asyncio.sleep(0.5)
        patience -= 0.5

    # ---- Results ----
    total = time.perf_counter() - t0
    print("\n" + "=" * 60)
    print(f"RESULTS (total={total:.1f}s)")
    print(f"  Transcripts:  {bridge.transcript_count} (expected >= 2)")
    print(f"  TTS chunks:   {bridge.tts_chunk_count} (expected > 0)")
    print(f"  STT requests: {bridge.stt_request_count}")
    print(f"  STT errors:   {bridge.stt_error_count}")
    print(f"  TTS still busy: {bridge._tts_busy}")

    # Cleanup
    bridge._tts_complete = True
    await bridge.cleanup()

    # Verify
    ok = (
        bridge.transcript_count >= 2
        and bridge.tts_chunk_count > 0
        and not bridge._tts_busy
    )
    print(f"\n  OVERALL: {'PASS' if ok else 'FAIL'}")
    print("=" * 60)
    return ok


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
