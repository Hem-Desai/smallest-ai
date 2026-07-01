#!/usr/bin/env python3
"""
Standalone Twilio calling server using smallest.ai TTS/STT.

Start the server:
    cd backend
    python smallest_test/standalone_server.py

Then open http://localhost:8002 in a browser to make calls.

The server auto-starts an ngrok tunnel so Twilio can reach it.
To use without ngrok, set PUBLIC_URL env var or run behind a reverse proxy.

Flow:
    Browser -> POST /call            (initiates Twilio outbound call)
    Twilio  -> GET  /twiml/{id}      (fetches call instructions)
    Twilio  -> WSS  /media/{id}      (media stream -> SmallestAiBridge)
"""

import asyncio
import json
import logging
import os
import sys
import time
import uuid
from pathlib import Path

import uvicorn

# -- path setup --------------------------------------------------------------

HERE = Path(__file__).resolve().parent
BACKEND_ROOT = HERE.parent
REPO_ROOT = BACKEND_ROOT.parent

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

_DOTENV_PATH = REPO_ROOT / ".env"
if _DOTENV_PATH.exists():
    from dotenv import load_dotenv
    load_dotenv(_DOTENV_PATH)

# -- logging -----------------------------------------------------------------
# Configure root logger so all smallest.* loggers inherit the format.
# Also add a file handler for persistent logs.

_LOGS_DIR = HERE / "logs"
_LOGS_DIR.mkdir(exist_ok=True)
_LOG_FILE = _LOGS_DIR / "smallest_standalone.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stderr),
        logging.FileHandler(str(_LOG_FILE)),
    ],
)
logger = logging.getLogger("smallest-standalone")

# Ensure bridge logger is also at INFO even if root was set differently
logging.getLogger("smallest.bridge").setLevel(logging.DEBUG)
logging.getLogger("smallest.tts").setLevel(logging.INFO)
logging.getLogger("smallest.stt").setLevel(logging.INFO)

# -- Twilio config -----------------------------------------------------------

TWILIO_SID = os.environ.get("TWILIO_ACCOUNT_SID", "")
TWILIO_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN", "")
TWILIO_PHONE = os.environ.get("TWILIO_PHONE_NUMBER", "")

if not all([TWILIO_SID, TWILIO_TOKEN, TWILIO_PHONE]):
    logger.error(
        "Missing Twilio credentials. Set TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, "
        "TWILIO_PHONE_NUMBER in .env"
    )
    sys.exit(1)

# -- FastAPI app -------------------------------------------------------------

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from twilio.rest import Client as TwilioClient
from twilio.twiml.voice_response import VoiceResponse, Connect, Stream

app = FastAPI(title="smallest.ai Standalone Calling Server")

# Serve static frontend
STATIC_DIR = HERE / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# -- ngrok -------------------------------------------------------------------

_ngrok_url: str | None = None


def _start_ngrok(port: int) -> str | None:
    """Try to start an ngrok tunnel for the given port. Returns the public URL or None."""
    try:
        from pyngrok import ngrok
        ngrok_token = os.environ.get("NGROK_AUTHTOKEN", "")
        if ngrok_token:
            ngrok.set_auth_token(ngrok_token)
        tunnel = ngrok.connect(port, "http")
        url = tunnel.public_url
        logger.info(f"ngrok tunnel active: {url} -> http://localhost:{port}")
        return url
    except Exception as e:
        logger.warning(f"Could not start ngrok: {e}")
        return None


def get_public_url() -> str:
    """Return the server's public URL (ngrok or manual override)."""
    global _ngrok_url
    if _ngrok_url:
        return _ngrok_url
    manual = os.environ.get("PUBLIC_URL", "")
    if manual:
        return manual.rstrip("/")
    return f"http://localhost:{PORT}"


@app.get("/", response_class=HTMLResponse)
async def index():
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return HTMLResponse("<h1>Frontend not found</h1>", status_code=404)


# -- API models --------------------------------------------------------------

class CallRequest(BaseModel):
    phone_number: str = Field(..., pattern=r"^\+[1-9]\d{1,14}$")


class CallResponse(BaseModel):
    call_sid: str
    execution_id: str
    status: str
    twiml_url: str
    media_url: str


# -- /call -------------------------------------------------------------------

@app.post("/call", response_model=CallResponse)
async def make_call(req: CallRequest):
    """Initiate an outbound Twilio call."""
    execution_id = str(uuid.uuid4())
    public_url = get_public_url()

    twiml_url = f"{public_url}/twiml/{execution_id}"
    media_ws = public_url.replace("https://", "wss://").replace("http://", "ws://")
    media_url = f"{media_ws}/media/{execution_id}"

    logger.info(
        "CALL INITIATE | to=%s | from=%s | execution_id=%s | twiml=%s | "
        "status_callback=%s/webhook/status/%s",
        req.phone_number, TWILIO_PHONE, execution_id,
        twiml_url, public_url, execution_id,
    )

    client = TwilioClient(TWILIO_SID, TWILIO_TOKEN)
    t0 = time.perf_counter()
    try:
        call = client.calls.create(
            to=req.phone_number,
            from_=TWILIO_PHONE,
            url=twiml_url,
            status_callback=f"{public_url}/webhook/status/{execution_id}",
            status_callback_event=["initiated", "ringing", "answered", "completed"],
            timeout=30,
        )
    except Exception as e:
        elapsed = time.perf_counter() - t0
        logger.error(
            "CALL FAILED | to=%s | execution_id=%s | duration=%.3fs | error=%s: %s",
            req.phone_number, execution_id, elapsed, type(e).__name__, e,
        )
        raise HTTPException(status_code=400, detail=str(e))

    elapsed = time.perf_counter() - t0
    logger.info(
        "CALL CREATED | sid=%s | to=%s | status=%s | duration=%.3fs",
        call.sid, req.phone_number, call.status, elapsed,
    )

    return CallResponse(
        call_sid=call.sid,
        execution_id=execution_id,
        status=call.status,
        twiml_url=twiml_url,
        media_url=media_url,
    )


# -- /twiml ------------------------------------------------------------------

@app.api_route("/twiml/{execution_id}", methods=["GET", "POST"])
async def twiml(execution_id: str, request: Request):
    """
    TwiML webhook — called by Twilio when the outbound call is answered.
    Returns instructions to connect the call audio via WebSocket Media Stream.
    """
    public_url = get_public_url()
    ws_url = public_url.replace("https://", "wss://").replace("http://", "ws://")
    stream_url = f"{ws_url}/media/{execution_id}"

    logger.info(
        "TWIML REQUEST | execution_id=%s | method=%s | stream=%s",
        execution_id, request.method, stream_url,
    )

    resp = VoiceResponse()
    resp.say("Hello, this is the smallest AI pipeline demo. Speak after the tone.")
    resp.pause(length=1)
    connect = Connect()
    connect.stream(url=stream_url, track="inbound_track")
    resp.append(connect)

    twiml_str = str(resp)
    logger.info(
        "TWIML RESPONSE | execution_id=%s | length=%d",
        execution_id, len(twiml_str),
    )
    return PlainTextResponse(twiml_str, media_type="application/xml")


# -- /media WebSocket --------------------------------------------------------

@app.websocket("/media/{execution_id}")
async def media_stream(websocket: WebSocket, execution_id: str):
    """
    Twilio Media Stream WebSocket endpoint.
    Forwards audio through the SmallestAiBridge (Pulse STT + Lightning TTS).
    """
    logger.info("MEDIA WS CONNECTING | execution_id=%s | client=%s", execution_id, websocket.client)
    await websocket.accept()
    logger.info("MEDIA WS ACCEPTED | execution_id=%s", execution_id)

    bridge = None
    bridge_task = None
    msg_count = 0
    t_connected = time.perf_counter()

    try:
        from smallest_test.speech_to_speech import SmallestAiBridge

        while True:
            try:
                message = await websocket.receive_text()
                msg_count += 1
                # Parse once for event type (avoid double parsing for performance)
                event_start = message.find('"event"')
                event_str = "unknown"
                if event_start != -1:
                    # Quick partial parse: find "event":"<type>"
                    colon = message.find(':', event_start)
                    if colon != -1:
                        val_start = message.find('"', colon) + 1
                        val_end = message.find('"', val_start)
                        if val_start > 0 and val_end > val_start:
                            event_str = message[val_start:val_end]

                if event_str == "start" and bridge is None:
                    logger.info(
                        "MEDIA WS start event | execution_id=%s | msg=%d",
                        execution_id, msg_count,
                    )
                    message_data = json.loads(message)
                    bridge = SmallestAiBridge(execution_id)
                    await bridge.initialize(websocket)
                    bridge_task = asyncio.create_task(bridge.start_bridge())
                    logger.info(
                        "MEDIA WS bridge created | execution_id=%s | bridge_id=%s",
                        execution_id, bridge.bridge_id,
                    )
                    # CRITICAL: route the start event so bridge opens STT/TTS connections
                    await bridge.handle_twilio_message(message_data)
                elif bridge:
                    message_data = json.loads(message)
                    await bridge.handle_twilio_message(message_data)
                else:
                    logger.debug(
                        "MEDIA WS event before bridge | execution_id=%s | event=%s | msg=%d",
                        execution_id, event_str, msg_count,
                    )

            except WebSocketDisconnect:
                uptime = time.perf_counter() - t_connected
                logger.info(
                    "MEDIA WS DISCONNECTED | execution_id=%s | messages=%d | uptime=%.1fs",
                    execution_id, msg_count, uptime,
                )
                break

        if bridge_task:
            await bridge_task

    except Exception:
        uptime = time.perf_counter() - t_connected
        logger.exception(
            "MEDIA WS ERROR | execution_id=%s | messages=%d | uptime=%.1fs",
            execution_id, msg_count, uptime,
        )
    finally:
        if bridge:
            await bridge.cleanup()
        try:
            await websocket.close()
        except Exception:
            pass
        total_time = time.perf_counter() - t_connected
        logger.info(
            "MEDIA WS FINISHED | execution_id=%s | messages=%d | total_time=%.1fs",
            execution_id, msg_count, total_time,
        )


# -- webhook for call status -------------------------------------------------

@app.post("/webhook/status/{execution_id}")
async def call_status(execution_id: str, request: Request):
    """Receive Twilio call status updates."""
    form = await request.form()
    call_status_val = form.get("CallStatus", "unknown")
    sid = form.get("CallSid", "")
    duration = form.get("CallDuration", "0")
    direction = form.get("Direction", "?")
    caller = form.get("Caller", "?")
    called = form.get("Called", "?")

    logger.info(
        "CALL STATUS | execution_id=%s | sid=%s | status=%s | "
        "duration=%s | direction=%s | from=%s | to=%s",
        execution_id, sid, call_status_val, duration, direction, caller, called,
    )

    # Push status update to event store for SSE subscribers
    from smallest_test.event_store import event_store as _ev_store
    asyncio.ensure_future(
        _ev_store.push(execution_id, "call_status",
                       status=call_status_val, duration=duration)
    )

    return PlainTextResponse("OK")


# -- /events SSE -------------------------------------------------------------

@app.get("/events/{execution_id}")
async def event_stream(execution_id: str, request: Request):
    """
    Server-Sent Events endpoint. The frontend subscribes to this to get
    real-time transcript and status updates for a specific call.
    """
    from smallest_test.event_store import event_store as _ev_store

    async def generate():
        # Send any historical events first
        for event in _ev_store.get_history(execution_id):
            yield _fmt_sse(event)

        # Stream new events as they arrive
        try:
            async for event in _ev_store.subscribe(execution_id):
                yield _fmt_sse(event)
                if event.type == "call_ended":
                    break
        except asyncio.CancelledError:
            pass
        finally:
            _ev_store.cleanup(execution_id)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _fmt_sse(event) -> str:
    """Format a CallEvent as an SSE message."""
    import json as _json
    data = {"type": event.type, **event.payload}
    return f"data: {_json.dumps(data)}\n\n"


# -- startup -----------------------------------------------------------------

PORT = int(os.environ.get("PORT", "8002"))


def main():
    global _ngrok_url
    print("=" * 60)
    print("  smallest.ai Standalone Calling Server")
    print("=" * 60)
    print()
    print(f"  Twilio phone: {TWILIO_PHONE}")
    print(f"  Local:        http://localhost:{PORT}")
    print()

    # Start ngrok
    _ngrok_url = _start_ngrok(PORT)
    if _ngrok_url:
        print(f"  Ngrok:        {_ngrok_url}")
        print(f"  TwiML URL:    {_ngrok_url}/twiml/{{execution_id}}")
        print(f"  Media WSS:    {_ngrok_url.replace('https://', 'wss://')}/media/{{execution_id}}")
    else:
        print("  WARNING: No ngrok tunnel. Set PUBLIC_URL env var or run ngrok manually:")
        print(f"    ngrok http {PORT}")
        print()

    print()
    print("  Open http://localhost:{} in a browser to make calls.".format(PORT))
    print()

    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")


if __name__ == "__main__":
    main()
