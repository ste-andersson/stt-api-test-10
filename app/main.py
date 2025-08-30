# app/main.py
import os
import json
import base64
import asyncio
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import httpx

# Optional dependency: websockets for connecting to OpenAI Realtime
try:
    import websockets
except Exception:  # pragma: no cover
    websockets = None  # We'll gracefully fall back to non-realtime mode

PORT = int(os.getenv("PORT", "8000"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
REALTIME_URL = os.getenv("REALTIME_URL", "")
REALTIME_TRANSCRIBE_MODEL = os.getenv("REALTIME_TRANSCRIBE_MODEL", "gpt-4o-mini-transcribe")
INPUT_LANGUAGE = os.getenv("INPUT_LANGUAGE", "sv")
OPENAI_ADD_BETA_HEADER = os.getenv("OPENAI_ADD_BETA_HEADER", "1") == "1"

# CORS setup
CORS_ORIGINS = os.getenv(
    "CORS_ORIGINS",
    "*.lovable.app,http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173"
)
origins = [o.strip() for o in CORS_ORIGINS.split(",") if o.strip()]

app = FastAPI(title="stt-api-test-10", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def index():
    return {"ok": True, "name": "stt-api-test-10"}

@app.get("/healthz")
def healthz():
    return {"ok": True}

# We expose both /ws and /ws/transcribe for maximum compatibility with the existing frontend(s).
@app.websocket("/ws")
async def ws(ws: WebSocket):
    await _handle_ws(ws)

@app.websocket("/ws/transcribe")
async def ws_transcribe(ws: WebSocket):
    await _handle_ws(ws)

async def _handle_ws(client_ws: WebSocket):
    await client_ws.accept()
    # Frontend expects this on handshake (based on previous project tooling)
    await client_ws.send_text(json.dumps({"type": "ready"}))

    # If we can, set up an upstream OpenAI Realtime WS, and proxy between them.
    # If not, we'll buffer input locally and use the HTTP transcription endpoint on demand.
    upstream = None
    upstream_reader_task: Optional[asyncio.Task] = None
    buffered_bytes = bytearray()
    httpx_client = httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=20.0))

    async def close_upstream():
        try:
            if upstream is not None:
                await upstream.close()
        except Exception:
            pass

    # Try to connect to OpenAI Realtime if REALTIME_URL + websockets are available.
    if REALTIME_URL and websockets is not None and OPENAI_API_KEY:
        try:
            headers = {
                "Authorization": f"Bearer {OPENAI_API_KEY}",
            }
            if OPENAI_ADD_BETA_HEADER:
                headers["OpenAI-Beta"] = "realtime=v1"

            upstream = await websockets.connect(REALTIME_URL, extra_headers=headers)

            # Configure session so it will transcribe appended audio buffers.
            session_update = {
                "type": "session.update",
                "session": {
                    # Let the server know we'll push audio chunks and we want transcription.
                    "input_audio_transcription": {
                        "model": REALTIME_TRANSCRIBE_MODEL,
                        "language": INPUT_LANGUAGE,
                    },
                },
            }
            await upstream.send(json.dumps(session_update))

            # Pipe upstream -> client (for deltas and finals)
            async def upstream_reader():
                accumulated = ""  # collect text during a response
                async for raw in upstream:
                    try:
                        msg = json.loads(raw)
                    except Exception:
                        continue

                    mtype = msg.get("type")
                    # Common Realtime events for text output
                    if mtype == "response.output_text.delta":
                        delta = msg.get("delta", "")
                        accumulated += delta
                        await safe_send_json(client_ws, {"type": "stt.delta", "text": delta})
                    elif mtype == "response.completed":
                        # Finalize the transcript for this response
                        if accumulated.strip():
                            await safe_send_json(client_ws, {"type": "stt.final", "text": accumulated.strip()})
                        accumulated = ""
                    elif mtype == "error":
                        await safe_send_json(client_ws, {"type": "error", "message": msg.get("error", {}).get("message", "upstream error")})
            upstream_reader_task = asyncio.create_task(upstream_reader())

        except Exception:
            # If anything goes wrong, we fall back to local HTTP transcription path.
            upstream = None

    try:
        while True:
            data = await client_ws.receive()
            if data["type"] == "websocket.disconnect":
                # On disconnect, try to finalize if we only buffered (HTTP mode)
                break

            if "bytes" in data and data["bytes"] is not None:
                chunk = data["bytes"]

                if upstream:
                    # Forward to OpenAI Realtime as an input_audio_buffer chunk
                    event = {
                        "type": "input_audio_buffer.append",
                        # Base64 encode raw chunk (can be WAV or PCM — Realtime API will decode)
                        "audio": base64.b64encode(chunk).decode("ascii"),
                    }
                    try:
                        await upstream.send(json.dumps(event))
                    except Exception:
                        # Upstream broken: downgrade to HTTP mode and start buffering locally
                        await close_upstream()
                        upstream = None
                        buffered_bytes.extend(chunk)
                else:
                    # HTTP fallback mode: buffer locally
                    buffered_bytes.extend(chunk)

            elif "text" in data and data["text"] is not None:
                text = data["text"].strip()
                # Frontend may send simple signals; we accept a few variants.
                if text.lower() in {"flush", "commit", "end", "stop"}:
                    if upstream:
                        # Tell Realtime to finalize the input buffer and create a response
                        try:
                            await upstream.send(json.dumps({"type": "input_audio_buffer.commit"}))
                            await upstream.send(json.dumps({"type": "response.create"}))
                        except Exception:
                            # If upstream fails at this point, drop to HTTP mode
                            await close_upstream()
                            upstream = None
                    else:
                        # HTTP transcription of the buffered audio
                        if buffered_bytes:
                            transcript = await transcribe_bytes(httpx_client, bytes(buffered_bytes))
                            await safe_send_json(client_ws, {"type": "stt.final", "text": transcript})
                            buffered_bytes.clear()
                        else:
                            await safe_send_json(client_ws, {"type": "stt.final", "text": ""})
                elif text.lower() == "ping":
                    await client_ws.send_text("pong")
                else:
                    # Unknown text messages are ignored to keep the server minimal and robust
                    pass

    except WebSocketDisconnect:
        pass
    finally:
        await close_upstream()
        if upstream_reader_task:
            upstream_reader_task.cancel()
        await httpx_client.aclose()
        try:
            await client_ws.close()
        except Exception:
            pass

async def transcribe_bytes(client: httpx.AsyncClient, audio_bytes: bytes) -> str:
    """
    HTTP fallback: send the entire audio blob to OpenAI's /audio/transcriptions.
    We assume the bytes form a valid audio file (e.g., WAV). Minimal by design.
    """
    if not OPENAI_API_KEY:
        return ""
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    files = {
        "file": ("audio.wav", audio_bytes, "audio/wav"),
        "model": (None, REALTIME_TRANSCRIBE_MODEL),
        "language": (None, INPUT_LANGUAGE),
        "response_format": (None, "json"),
        "temperature": (None, "0"),
    }
    try:
        r = await client.post("https://api.openai.com/v1/audio/transcriptions", headers=headers, files=files)
        r.raise_for_status()
        data = r.json()
        # OpenAI returns {"text": "..."} for json format
        return data.get("text", "") or ""
    except Exception:
        return ""

async def safe_send_json(ws: WebSocket, payload: dict):
    try:
        await ws.send_text(json.dumps(payload, ensure_ascii=False))
    except Exception:
        pass
