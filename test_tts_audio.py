"""
Standalone TTS WebSocket test — connects to smallest.ai Lightning WS,
requests speech, collects all audio chunks, saves to file for inspection.
"""
import asyncio
import base64
import json
import sys
from pathlib import Path

import aiohttp

API_KEY = "sk_ec6425e0db7a3e4222eb81f7ab57fe68"
WS_URL = "wss://api.smallest.ai/waves/v1/tts/live"
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


async def test_tts_ws(format_: str, sample_rate: int, label: str):
    print(f"\n{'='*60}")
    print(f"Testing TTS WS: format={format_}, rate={sample_rate}")
    print(f"{'='*60}")

    headers = {"Authorization": f"Bearer {API_KEY}"}
    payload = {
        "text": "Hello there, this is a test of the streaming text to speech engine. "
                "If you can hear this clearly, the audio format is correct.",
        "voice_id": "sophia",
        "sample_rate": sample_rate,
        "output_format": format_,
        "language": "en",
        "speed": 1.0,
    }

    chunks = []
    msg_count = 0
    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(WS_URL, headers=headers) as ws:
            print(f"  Connected. Sending: {json.dumps(payload, indent=2)[:200]}...")
            await ws.send_json(payload)

            async for msg in ws:
                msg_count += 1
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    status = data.get("status", "?")
                    if status == "chunk":
                        audio_b64 = data.get("data", {}).get("audio", "")
                        if audio_b64:
                            chunk = base64.b64decode(audio_b64)
                            chunks.append(chunk)
                            if len(chunks) <= 3 or len(chunks) % 10 == 0:
                                print(f"  chunk #{len(chunks)}: {len(chunk)} bytes, first 8 hex: {chunk[:8].hex()}")
                    elif status == "complete":
                        print(f"  complete | total_msgs={msg_count} | total_chunks={len(chunks)}")
                        break
                    else:
                        print(f"  status={status} | data={msg.data[:120]}")
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    print(f"  ERROR: {ws.exception()}")
                    break
                elif msg.type == aiohttp.WSMsgType.CLOSED:
                    print(f"  CLOSED: {msg.data}")
                    break

    if chunks:
        combined = b"".join(chunks)
        outfile = OUTPUT_DIR / f"tts_ws_{label}.raw"
        outfile.write_bytes(combined)
        print(f"  SAVED: {outfile} ({len(combined)} bytes, {len(chunks)} chunks)")
        print(f"  First 32 bytes hex: {combined[:32].hex()}")
        print(f"  Audio starts with silence: {combined[:100].hex().count('00') / 2} of first 100 bytes are zero")
    else:
        print(f"  NO AUDIO received!")


async def test_tts_rest(label: str):
    """Test REST endpoint as baseline."""
    print(f"\n{'='*60}")
    print(f"Testing TTS REST: ulaw 8000")
    print(f"{'='*60}")

    url = "https://api.smallest.ai/waves/v1/tts"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "text": "Hello there, this is a test of streaming text to speech. Can you hear me clearly?",
        "voice_id": "sophia",
        "sample_rate": 8000,
        "output_format": "ulaw",
        "language": "en",
        "speed": 1.0,
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload, headers=headers) as resp:
            audio = await resp.read()
            print(f"  status={resp.status} | size={len(audio)} bytes")
            if resp.status == 200:
                outfile = OUTPUT_DIR / f"tts_rest_{label}.raw"
                outfile.write_bytes(audio)
                print(f"  SAVED: {outfile}")
                print(f"  First 32 bytes hex: {audio[:32].hex()}")


async def main():
    # Test REST baseline
    await test_tts_rest("ulaw8k")

    # Test WS with different formats
    for fmt, rate, label in [
        ("ulaw", 8000, "ulaw8k"),
        ("pcm", 8000, "pcm8k"),
        ("wav", 8000, "wav8k"),
    ]:
        await test_tts_ws(fmt, rate, label)

    print("\nDone! Check the output/ directory for .raw files.")


if __name__ == "__main__":
    asyncio.run(main())
