# app/main.py
import os
import json
import base64
import asyncio
import time
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import httpx

try:
    import websockets
except Exception:
    websockets = None

PORT = int(os.getenv("PORT", "8000"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
REALTIME_URL = os.getenv("REALTIME_URL", "")
REALTIME_TRANSCRIBE_MODEL = os.getenv("REALTIME_TRANSCRIBE_MODEL", "gpt-4o-mini-transcribe")
INPUT_LANGUAGE = os.getenv("INPUT_LANGUAGE", "sv")
OPENAI_ADD_BETA_HEADER = os.getenv("OPENAI_ADD_BETA_HEADER", "1") == "1"
COMMIT_INTERVAL_MS = int(os.getenv("COMMIT_INTERVAL_MS", "0"))  # 0 = av

CORS_ORIGINS = os.getenv(
    "CORS_ORIGINS",
    "*.lovable.app,http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173"
)
origins = [o.strip() for o in CORS_ORIGINS.split(",") if o.strip()]

app = FastAPI(title="stt-api-test-10", version="1.0.1")
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

# WS-endpoints för kompatibilitet
@app.websocket("/ws")
async def ws(ws: WebSocket):
    await _handle_ws(ws)

@app.websocket("/ws/transcribe")
async def ws_transcribe(ws: WebSocket):
    await _handle_ws(ws)

# Full bakåtkompatibilitet om frontend pekar på /transcribe
@app.websocket("/transcribe")
async def ws_compat(ws: WebSocket):
    await _handle_ws(ws)

async def _handle_ws(client_ws: WebSocket):
    await client_ws.accept()
    # Skicka JSON direkt vid anslutning så klienter kan JSON.parse: {"type":"ready"}
    await client_ws.send_text(json.dumps({"type": "ready"}))

    upstream = None
    upstream_reader_task: Optional[asyncio.Task] = None
    committer_task: Optional[asyncio.Task] = None
    buffered_bytes = bytearray()
    audio_since_commit = False
    last_commit_ns = time.time_ns()
    httpx_client = httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=20.0))

    async def close_upstream():
        try:
            if upstream is not None:
                await upstream.close()
        except Exception:
            pass

    # Försök Realtime-bridge om REALTIME_URL + websockets + key finns
    if REALTIME_URL and websockets is not None and OPENAI_API_KEY:
        try:
            headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
            if OPENAI_ADD_BETA_HEADER:
                headers["OpenAI-Beta"] = "realtime=v1"

            upstream = await websockets.connect(REALTIME_URL, extra_headers=headers)

            # Konfigurera transkription
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

            # Upstream -> client (deltas och finals)
            async def upstream_reader():
                accumulated = ""
                async for raw in upstream:
                    try:
                        msg = json.loads(raw)
                    except Exception:
                        continue
                    mtype = msg.get("type")
                    if mtype == "response.output_text.delta":
                        delta = msg.get("delta", "")
                        accumulated += delta
                        await safe_send_json(client_ws, {"type": "stt.delta", "text": delta})
                    elif mtype == "response.completed":
                        if accumulated.strip():
                            await safe_send_json(client_ws, {"type": "stt.final", "text": accumulated.strip()})
                        accumulated = ""
                    elif mtype == "error":
                        await safe_send_json(client_ws, {"type": "error", "message": msg.get("error", {}).get("message", "upstream error")})
            upstream_reader_task = asyncio.create_task(upstream_reader())

            # Periodisk auto-commit om påslaget
            async def periodic_committer():
                nonlocal audio_since_commit, last_commit_ns, upstream
                if COMMIT_INTERVAL_MS <= 0:
                    return
                try:
                    while True:
                        await asyncio.sleep(max(0.05, COMMIT_INTERVAL_MS / 1000.0))
                        if upstream is None:
                            continue
                        now_ns = time.time_ns()
                        if audio_since_commit and (now_ns - last_commit_ns) / 1e6 >= COMMIT_INTERVAL_MS:
                            try:
                                await upstream.send(json.dumps({"type": "input_audio_buffer.commit"}))
                                await upstream.send(json.dumps({"type": "response.create"}))
                                audio_since_commit = False
                                last_commit_ns = now_ns
                            except Exception:
                                await close_upstream()
                                upstream = None
                                break
                except asyncio.CancelledError:
                    pass

            if COMMIT_INTERVAL_MS > 0:
                committer_task = asyncio.create_task(periodic_committer())

        except Exception:
            upstream = None

    try:
        while True:
            data = await client_ws.receive()
            if data["type"] == "websocket.disconnect":
                break

            if "bytes" in data and data["bytes"] is not None:
                chunk = data["bytes"]
                if upstream:
                    event = {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(chunk).decode("ascii"),
                    }
                    try:
                        await upstream.send(json.dumps(event))
                        # markera att vi har nytt ljud för auto-commit
                        audio_since_commit = True
                    except Exception:
                        await close_upstream()
                        upstream = None
                        buffered_bytes.extend(chunk)
                else:
                    buffered_bytes.extend(chunk)

            elif "text" in data and data["text"] is not None:
                text = data["text"].strip().lower()
                if text in {"flush", "commit", "end", "stop"}:
                    if upstream:
                        try:
                            await upstream.send(json.dumps({"type": "input_audio_buffer.commit"}))
                            await upstream.send(json.dumps({"type": "response.create"}))
                        except Exception:
                            await close_upstream()
                            upstream = None
                    else:
                        if buffered_bytes:
                            transcript = await transcribe_bytes(httpx_client, bytes(buffered_bytes))
                            await safe_send_json(client_ws, {"type": "stt.final", "text": transcript})
                            buffered_bytes.clear()
                        else:
                            await safe_send_json(client_ws, {"type": "stt.final", "text": ""})
                elif text == "ping":
                    await client_ws.send_text("pong")

    except WebSocketDisconnect:
        pass
    finally:
        await close_upstream()
        if upstream_reader_task:
            upstream_reader_task.cancel()
        if committer_task:
            try:
                committer_task.cancel()
            except Exception:
                pass
        await httpx_client.aclose()
        try:
            await client_ws.close()
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
