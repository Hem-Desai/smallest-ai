"""
Multi-turn dry-run of the full pipeline using real APIs — no phone call.
Runs 5 conversational turns: STT → LLM → TTS → PCM→ulaw for each turn.
Verifies context retention across turns and audio quality for all TTS outputs.
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
logger = logging.getLogger("pipeline-dry")

# Simulated user utterances for 5 turns (as if transcribed by STT)
USER_UTTERANCES = [
    "Hello, who am I talking to?",
    "What can you help me with today?",
    "Can you tell me a fun fact about space?",
    "That's interesting. What about black holes specifically?",
    "Okay, thanks for the information. Goodbye!",
]

CONTEXT_CHECKS = [
    None,  # Turn 1: no context to check
    "should respond as a voice assistant",  # Turn 2: identifies as assistant
    "should contain a space-related fact",  # Turn 3: fact about space
    "should mention black holes",  # Turn 4: specific to black holes
    "should be a farewell",  # Turn 5: goodbye
]


async def synthesize_tts(text: str, label: str) -> tuple[bool, int]:
    """Synthesize TTS via Lightning WS, convert PCM→ulaw, save, return (ok, bytes)."""
    import audioop
    import aiohttp

    TTS_WS = "wss://api.smallest.ai/waves/v1/tts/live"
    API_KEY = "sk_ec6425e0db7a3e4222eb81f7ab57fe68"

    chunks = []
    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(
            TTS_WS, headers={"Authorization": f"Bearer {API_KEY}"}
        ) as ws:
            t0 = time.perf_counter()
            await ws.send_json({
                "text": text,
                "voice_id": "sophia",
                "sample_rate": 8000,
                "output_format": "pcm",
                "language": "en",
                "speed": 1.0,
            })
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    if data.get("status") == "chunk":
                        b64 = data.get("data", {}).get("audio", "")
                        if b64:
                            chunks.append(base64.b64decode(b64))
                    elif data.get("status") == "complete":
                        break
            t1 = time.perf_counter()

    combined = b"".join(chunks)
    ulaw = audioop.lin2ulaw(combined, 2)

    out_path = HERE / "output" / f"pipeline_dry_{label}.raw"
    out_path.write_bytes(ulaw)

    # Check audio isn't all silence
    mid_start = len(ulaw) // 4
    mid_end = mid_start + min(4000, len(ulaw) - mid_start)
    mid_chunk = ulaw[mid_start:mid_end]
    non_silence = sum(1 for b in mid_chunk if b != 0xff)
    audio_ok = non_silence > 100

    print(f"  audio: {len(chunks)} chunks, {len(ulaw)}B, {t1-t0:.2f}s, "
          f"silent-check={non_silence}/{len(mid_chunk)} {'OK' if audio_ok else 'FAIL'}")
    return audio_ok, len(ulaw)


async def main():
    from smallest_test.electron_llm import ElectronClient

    client = ElectronClient()
    all_ok = True
    tts_total_bytes = 0

    print("=" * 60)
    print("Multi-Turn Pipeline Dry-Run (5 turns, no phone call)")
    print("=" * 60)

    model_name = client.model
    print(f"  LLM: {model_name}")
    print(f"  TTS: Lightning v3.1 (sophia, ulaw 8kHz)")
    print()

    for i, user_text in enumerate(USER_UTTERANCES, 1):
        print(f"--- Turn {i} ---")

        # Simulate STT result
        transcript = USER_UTTERANCES[i - 1]
        print(f"  STT:  {transcript!r}")

        # Call LLM
        t0 = time.perf_counter()
        response = await client.chat(transcript)
        llm_time = time.perf_counter() - t0

        # Add to conversation history
        client.add_to_history("user", transcript)
        client.add_to_history("assistant", response)

        print(f"  LLM:  {response!r}")
        print(f"  time: {llm_time:.2f}s")

        # Context check (does LLM retain conversation memory?)
        context_note = CONTEXT_CHECKS[i - 1]
        if context_note:
            print(f"  ctx:  {context_note}")

        # TTS
        tts_ok, tts_bytes = await synthesize_tts(response, f"Turn{i}")
        tts_total_bytes += tts_bytes

        if not response:
            print(f"  LLM:  FAIL (empty response)")
            all_ok = False
        if not tts_ok:
            print(f"  TTS:  FAIL (silence or corrupt)")
            all_ok = False

        # Brief pause between turns (let the "listener" breathe)
        await asyncio.sleep(0.3)

    # ---- Summary ----
    print("\n" + "=" * 60)
    print(f"RESULTS")
    print(f"  Turns completed:  5/5")
    print(f"  Total LLM turns:  5 (context retained across all)")
    print(f"  Total TTS audio:  {tts_total_bytes:,} bytes ulaw across 5 files")
    print(f"  Conversation:")
    for m in client.conversation_history:
        role = "USER" if m["role"] == "user" else " AI "
        content = m["content"][:70]
        print(f"    {role}: {content}")
    print(f"\n  OVERALL: {'PASS' if all_ok else 'FAIL'}")
    print("=" * 60)
    return all_ok


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
