# app/main.py
import os
import json
import base64
import asyncio
import time
import uuid
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import httpx

try:
    import websockets
except Exception:
    websockets = None

# ---- Settings (match 7-backend defaults where practical) ----
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
REALTIME_URL = os.getenv("REALTIME_URL", os.getenv("REALTIME_WS_URL", ""))  # tolerate alt names
REALTIME_TRANSCRIBE_MODEL = os.getenv("REALTIME_TRANSCRIBE_MODEL", "gpt-4o-mini-transcribe")
INPUT_LANGUAGE = os.getenv("INPUT_LANGUAGE", "sv")
OPENAI_ADD_BETA_HEADER = os.getenv("OPENAI_ADD_BETA_HEADER", "1") not in ("0", "", "false", "False")
COMMIT_INTERVAL_MS = int(os.getenv("COMMIT_INTERVAL_MS", "500"))  # 7-backend default was 500
CORS_ORIGINS = os.getenv(
    "CORS_ORIGINS",
    "*.lovable.app,http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173"
)
origins = [o.strip() for o in CORS_ORIGINS.split(",") if o.strip()]

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", os.getenv("APP_PORT", "8000")))

app = FastAPI(title="stt-api-test-10", version="1.0.7-compat7")
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
async def healthz():
    return {"ok": True, "ts": time.time()}

# ---- WS Routes (compat) ----
@app.websocket("/ws")
async def ws(ws: WebSocket):
    await _handle_ws(ws)

@app.websocket("/ws/transcribe")
async def ws_transcribe(ws: WebSocket):
    await _handle_ws(ws)

@app.websocket("/transcribe")  # extra legacy alias
async def ws_compat(ws: WebSocket):
    await _handle_ws(ws)

async def _handle_ws(ws: WebSocket):
    await ws.accept()
    # Frontend expects JSON handshake with audio_in (7-backend behavior)
    await ws.send_text(json.dumps({
        "type": "ready",
        "audio_in": {"encoding": "pcm16", "sample_rate_hz": 16000, "channels": 1},
        "audio_out": {"mimetype": "audio/mpeg"},
    }))
    # Optional: send a session.started notification (7-backend did this)
    session_id = str(uuid.uuid4())
    await ws.send_text(json.dumps({"type": "session.started", "session_id": session_id}))

    upstream = None
    httpx_client = httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=20.0))

    # Realtime accumulators (so we can send partials as full text)
    rt_text_accum = ""         # for response.output_text.* path
    vad_text_accum = ""        # for conversation.item.input_audio_transcription.* path

    # Auto-commit helpers
    has_audio = False
    last_audio_time = 0.0

    # Connect to Realtime if configured
    if REALTIME_URL and OPENAI_API_KEY and websockets is not None:
        try:
            headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
            if OPENAI_ADD_BETA_HEADER:
                headers["OpenAI-Beta"] = "realtime=v1"
            upstream = await websockets.connect(REALTIME_URL, extra_headers=headers)

            # Configure transcription model/language (minimal; 7-backend also supported server VAD)
            session_update = {
                "type": "session.update",
                "session": {
                    "input_audio_transcription": {
                        "model": REALTIME_TRANSCRIBE_MODEL,
                        "language": INPUT_LANGUAGE,
                    },
                },
            }
            await upstream.send(json.dumps(session_update))
        except Exception:
            upstream = None

    # Spawn reader and committer tasks if Realtime is active
    upstream_reader_task: Optional[asyncio.Task] = None
    committer_task: Optional[asyncio.Task] = None

    async def close_upstream():
        try:
            if upstream is not None:
                await upstream.close()
        except Exception:
            pass

    async def upstream_reader():
        nonlocal rt_text_accum, vad_text_accum
        try:
            async for raw in upstream:
                try:
                    msg = json.loads(raw)
                except Exception:
                    continue

                t = msg.get("type")

                # Map several event variants to stt.partial/final
                if t == "response.output_text.delta":
                    delta = msg.get("delta", "")
                    if delta:
                        rt_text_accum += delta
                        await safe_send_json(ws, {"type": "stt.partial", "text": rt_text_accum})
                elif t == "response.completed":
                    if rt_text_accum.strip():
                        await safe_send_json(ws, {"type": "stt.final", "text": rt_text_accum.strip()})
                        rt_text_accum = ""

                # VAD-style transcription events (some Realtime variants emit these)
                elif t == "conversation.item.input_audio_transcription.delta":
                    part = msg.get("delta") or msg.get("text") or ""
                    if part:
                        vad_text_accum += part
                        await safe_send_json(ws, {"type": "stt.partial", "text": vad_text_accum})
                elif t == "conversation.item.input_audio_transcription.completed":
                    text = (
                        msg.get("text")
                        or (msg.get("input_audio_transcription") or {}).get("text")
                        or vad_text_accum
                        or ""
                    ).strip()
                    if text:
                        await safe_send_json(ws, {"type": "stt.final", "text": text})
                    vad_text_accum = ""

                # Older event type seen in 7-backend code
                elif t == "response.audio_transcript.delta":
                    delta = msg.get("delta") or msg.get("text") or ""
                    if delta:
                        rt_text_accum += delta
                        await safe_send_json(ws, {"type": "stt.partial", "text": rt_text_accum})
                elif t == "response.audio_transcript.completed":
                    text = msg.get("text") or rt_text_accum
                    if text:
                        await safe_send_json(ws, {"type": "stt.final", "text": text})
                    rt_text_accum = ""

                elif t == "error":
                    await safe_send_json(ws, {"type": "error", "message": (msg.get("error") or {}).get("message", "upstream error")})

        except asyncio.CancelledError:
            pass
        except Exception:
            # swallow
            pass

    async def periodic_committer():
        # Periodically commit only if we have recent audio (<= 2s ago)
        if COMMIT_INTERVAL_MS <= 0:
            return
        try:
            while True:
                await asyncio.sleep(max(0.05, COMMIT_INTERVAL_MS / 1000.0))
                if upstream is None:
                    continue
                if not has_audio:
                    continue
                if (time.time() - last_audio_time) > 2.0:
                    # stale, stop auto committing until audio resumes
                    continue
                try:
                    await upstream.send(json.dumps({"type": "input_audio_buffer.commit"}))
                    await upstream.send(json.dumps({"type": "response.create"}))
                except Exception:
                    # drop realtime and rely on HTTP fallback
                    await close_upstream()
                    break
        except asyncio.CancelledError:
            pass

    if upstream is not None:
        upstream_reader_task = asyncio.create_task(upstream_reader())
        if COMMIT_INTERVAL_MS > 0:
            committer_task = asyncio.create_task(periodic_committer())

    # Fallback buffer for HTTP transcription
    buffered_bytes = bytearray()

    try:
        while True:
            data = await ws.receive()
            if data["type"] == "websocket.disconnect":
                break

            if "bytes" in data and data["bytes"] is not None:
                chunk = data["bytes"]
                if upstream is not None:
                    try:
                        event = {
                            "type": "input_audio_buffer.append",
                            "audio": base64.b64encode(chunk).decode("ascii"),
                        }
                        await upstream.send(json.dumps(event))
                        has_audio = True
                        last_audio_time = time.time()
                    except Exception:
                        await close_upstream()
                        upstream = None
                        buffered_bytes.extend(chunk)
                else:
                    buffered_bytes.extend(chunk)

            elif "text" in data and data["text"] is not None:
                text = (data["text"] or "").strip().lower()
                if text in {"flush", "commit", "end", "stop"}:
                    if upstream is not None:
                        try:
                            await upstream.send(json.dumps({"type": "input_audio_buffer.commit"}))
                            await upstream.send(json.dumps({"type": "response.create"}))
                        except Exception:
                            await close_upstream()
                            upstream = None
                    else:
                        if buffered_bytes:
                            transcript = await transcribe_bytes(httpx_client, bytes(buffered_bytes))
                            await safe_send_json(ws, {"type": "stt.final", "text": transcript})
                            buffered_bytes.clear()
                        else:
                            await safe_send_json(ws, {"type": "stt.final", "text": ""})
                elif text == "ping":
                    await ws.send_text("pong")
                # ignore others
    except WebSocketDisconnect:
        pass
    finally:
        # cleanup
        try:
            await close_upstream()
        except Exception:
            pass
        if upstream_reader_task:
            upstream_reader_task.cancel()
        if committer_task:
            committer_task.cancel()
        await httpx_client.aclose()
        try:
            await ws.close()
        except Exception:
            pass

async def transcribe_bytes(client: httpx.AsyncClient, audio_bytes: bytes) -> str:
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
        return data.get("text", "") or ""
    except Exception:
        return ""

async def safe_send_json(ws: WebSocket, payload: dict):
    try:
        await ws.send_text(json.dumps(payload, ensure_ascii=False))
    except Exception:
        pass
