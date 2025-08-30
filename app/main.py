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

# Läs .env automatiskt (som i 7-backend)
from dotenv import load_dotenv
load_dotenv()

# Valfri dependency: websockets för Realtime-bridge
try:
    import websockets
except Exception:
    websockets = None

# -------------------- Config --------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
REALTIME_URL = os.getenv(
    "REALTIME_URL",
    os.getenv(
        "REALTIME_WS_URL",
        "wss://api.openai.com/v1/realtime?model=gpt-4o-mini-realtime-preview-2024-12-17",
    ),
)
REALTIME_TRANSCRIBE_MODEL = os.getenv("REALTIME_TRANSCRIBE_MODEL", "gpt-4o-mini-transcribe")
INPUT_LANGUAGE = os.getenv("INPUT_LANGUAGE", "sv")
OPENAI_ADD_BETA_HEADER = os.getenv("OPENAI_ADD_BETA_HEADER", "1") not in ("0", "", "false", "False")
COMMIT_INTERVAL_MS = int(os.getenv("COMMIT_INTERVAL_MS", "500"))
VAD_ENABLED = os.getenv("VAD", "1") not in ("0", "", "false", "False")
SILENCE_MS = int(os.getenv("SILENCE_MS", "600"))
PREFIX_MS = int(os.getenv("PREFIX_MS", "200"))

CORS_ORIGINS = os.getenv(
    "CORS_ORIGINS",
    "*.lovable.app,http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173",
)
origins = [o.strip() for o in CORS_ORIGINS.split(",") if o.strip()]

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", os.getenv("APP_PORT", "8000")))

app = FastAPI(title="stt-api-test-10", version="1.1.0-compat7")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- Debug state --------------------
_debug = {
    "frontend_chunks": [],   # list of chunk lengths
    "commits": 0,
    "rt_events": [],         # last ~200 event types
    "last_error": None,
    "last_partial": "",
    "last_final": "",
}

def _dbg_set(key, value):
    try:
        _debug[key] = value
    except Exception:
        pass

def _dbg_inc(key, inc=1):
    try:
        _debug[key] = int(_debug.get(key, 0)) + inc
    except Exception:
        pass

# -------------------- HTTP endpoints --------------------
@app.get("/")
def index():
    return {"ok": True, "name": "stt-api-test-10"}

@app.get("/healthz")
async def healthz():
    return {"ok": True, "ts": time.time()}

@app.get("/config")
async def get_config():
    return {
        "realtime_url": REALTIME_URL,
        "transcribe_model": REALTIME_TRANSCRIBE_MODEL,
        "input_language": INPUT_LANGUAGE,
        "commit_interval_ms": COMMIT_INTERVAL_MS,
        "cors_origins": origins,
        "openai_beta_header": OPENAI_ADD_BETA_HEADER,
        "ws_paths": ["/ws", "/ws/transcribe", "/transcribe"],
        "vad_enabled": VAD_ENABLED,
        "silence_ms": SILENCE_MS,
        "prefix_ms": PREFIX_MS,
    }

# Debug (hjälper felsökning)
@app.get("/debug/counters")
async def debug_counters():
    return {
        "frontend_chunks": len(_debug["frontend_chunks"]),
        "bytes_total": sum(_debug["frontend_chunks"]) if _debug["frontend_chunks"] else 0,
        "commits": _debug["commits"],
        "rt_events_last": _debug["rt_events"][-20:],
        "last_error": _debug["last_error"],
        "last_partial": _debug["last_partial"],
        "last_final": _debug["last_final"],
    }

@app.get("/debug/rt-events")
async def debug_rt_events(limit: int = 100):
    return _debug["rt_events"][-limit:]

@app.get("/debug/frontend-chunks")
async def debug_frontend_chunks(limit: int = 20):
    return _debug["frontend_chunks"][-limit:]

# -------------------- WebSocket endpoints --------------------
@app.websocket("/ws")
async def ws(ws: WebSocket):
    await _handle_ws(ws)

@app.websocket("/ws/transcribe")
async def ws_transcribe(ws: WebSocket):
    await _handle_ws(ws)

# Extra alias för 100% kompatibilitet
@app.websocket("/transcribe")
async def ws_compat(ws: WebSocket):
    await _handle_ws(ws)

# -------------------- Core WS handler --------------------
async def _handle_ws(ws: WebSocket):
    await ws.accept()

    # Handshake (som 7-backend): JSON med audio_in + session.started
    await ws.send_text(json.dumps({
        "type": "ready",
        "audio_in": {"encoding": "pcm16", "sample_rate_hz": 16000, "channels": 1},
        "audio_out": {"mimetype": "audio/mpeg"},
    }))
    session_id = str(uuid.uuid4())
    await ws.send_text(json.dumps({"type": "session.started", "session_id": session_id}))

    upstream = None
    httpx_client = httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=20.0))

    # Ackumulatorer för text
    rt_text_accum = ""    # response.output_text.*
    vad_text_accum = ""   # conversation.item.input_audio_transcription.*

    # Auto-commit state
    has_audio = False
    last_audio_time = 0.0

    # Försök ansluta till OpenAI Realtime
    if REALTIME_URL and OPENAI_API_KEY and websockets is not None:
        try:
            headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
            if OPENAI_ADD_BETA_HEADER:
                headers["OpenAI-Beta"] = "realtime=v1"

            upstream = await websockets.connect(
                REALTIME_URL,
                extra_headers=headers,
                subprotocols=["realtime"],  # <- Viktigt för Realtime
            )

            # Konfigurera sessionen
            session_update = {
                "type": "session.update",
                "session": {
                    "input_audio_transcription": {
                        "model": REALTIME_TRANSCRIBE_MODEL,
                        "language": INPUT_LANGUAGE,
                    },
                },
            }
            if VAD_ENABLED:
                session_update["session"]["turn_detection"] = {
                    "type": "server_vad",
                    "silence_duration_ms": SILENCE_MS,
                    "prefix_padding_ms": PREFIX_MS,
                }

            await upstream.send(json.dumps(session_update))

        except Exception as e:
            _dbg_set("last_error", f"realtime_connect_failed: {e}")
            upstream = None

    # Upstream reader & committer
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
                # Spara senaste eventtyper för debug
                try:
                    _debug["rt_events"].append(t)
                    if len(_debug["rt_events"]) > 200:
                        _debug["rt_events"] = _debug["rt_events"][-100:]
                except Exception:
                    pass

                # (A) Klassisk response-text
                if t == "response.output_text.delta":
                    delta = msg.get("delta", "")
                    if delta:
                        rt_text_accum += delta
                        _dbg_set("last_partial", rt_text_accum)
                        await safe_send_json(ws, {"type": "stt.partial", "text": rt_text_accum})

                elif t == "response.completed":
                    if rt_text_accum.strip():
                        _dbg_set("last_final", rt_text_accum.strip())
                        await safe_send_json(ws, {"type": "stt.final", "text": rt_text_accum.strip()})
                    rt_text_accum = ""

                # (B) VAD-transkript events
                elif t == "conversation.item.input_audio_transcription.delta":
                    part = msg.get("delta") or msg.get("text") or ""
                    if part:
                        vad_text_accum += part
                        _dbg_set("last_partial", vad_text_accum)
                        await safe_send_json(ws, {"type": "stt.partial", "text": vad_text_accum})

                elif t == "conversation.item.input_audio_transcription.completed":
                    text = (
                        msg.get("text")
                        or (msg.get("input_audio_transcription") or {}).get("text")
                        or vad_text_accum
                        or ""
                    ).strip()
                    if text:
                        _dbg_set("last_final", text)
                        await safe_send_json(ws, {"type": "stt.final", "text": text})
                    vad_text_accum = ""

                # (C) Äldre varianter (bakåtkomp)
                elif t == "response.audio_transcript.delta":
                    delta = msg.get("delta") or msg.get("text") or ""
                    if delta:
                        rt_text_accum += delta
                        _dbg_set("last_partial", rt_text_accum)
                        await safe_send_json(ws, {"type": "stt.partial", "text": rt_text_accum})

                elif t == "response.audio_transcript.completed":
                    text = msg.get("text") or rt_text_accum
                    if text:
                        _dbg_set("last_final", text)
                        await safe_send_json(ws, {"type": "stt.final", "text": text})
                    rt_text_accum = ""

                elif t == "error":
                    _dbg_set("last_error", (msg.get("error") or {}).get("message"))
                    await safe_send_json(ws, {"type": "error", "message": (msg.get("error") or {}).get("message", "upstream error")})

        except asyncio.CancelledError:
            pass
        except Exception as e:
            _dbg_set("last_error", f"upstream_reader_error: {e}")

    async def periodic_committer():
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
                    # för länge sedan senaste ljud → vänta tills nytt ljud kommer
                    continue
                try:
                    # Commit och explicit response.create med instruktioner
                    await upstream.send(json.dumps({"type": "input_audio_buffer.commit"}))
                    _dbg_inc("commits", 1)
                    await upstream.send(json.dumps({
                        "type": "response.create",
                        "response": {
                            "instructions": "Transcribe the user speech from the most recent audio buffer. Output only the transcript; no punctuation if unsure.",
                            "modalities": ["text"],
                        },
                    }))
                except Exception as e:
                    _dbg_set("last_error", f"commit_failed: {e}")
                    await close_upstream()
                    break
        except asyncio.CancelledError:
            pass

    if upstream is not None:
        upstream_reader_task = asyncio.create_task(upstream_reader())
        if COMMIT_INTERVAL_MS > 0:
            committer_task = asyncio.create_task(periodic_committer())

    # Fallback-buffert för HTTP-transcribe
    buffered_bytes = bytearray()

    try:
        while True:
            data = await ws.receive()
            if data["type"] == "websocket.disconnect":
                break

            if "bytes" in data and data["bytes"] is not None:
                chunk = data["bytes"]

                # Debug: registrera chunk-längd
                try:
                    _debug["frontend_chunks"].append(len(chunk))
                    if len(_debug["frontend_chunks"]) > 2000:
                        _debug["frontend_chunks"] = _debug["frontend_chunks"][-1000:]
                except Exception:
                    pass

                if upstream is not None:
                    try:
                        event = {
                            "type": "input_audio_buffer.append",
                            "audio": base64.b64encode(chunk).decode("ascii"),
                        }
                        await upstream.send(json.dumps(event))
                        has_audio = True
                        last_audio_time = time.time()
                    except Exception as e:
                        _dbg_set("last_error", f"append_failed: {e}")
                        await close_upstream()
                        upstream = None
                        buffered_bytes.extend(chunk)
                else:
                    buffered_bytes.extend(chunk)

            elif "text" in data and data["text"] is not None:
                text_msg = (data["text"] or "").strip().lower()
                if text_msg in {"flush", "commit", "end", "stop"}:
                    if upstream is not None:
                        try:
                            await upstream.send(json.dumps({"type": "input_audio_buffer.commit"}))
                            _dbg_inc("commits", 1)
                            await upstream.send(json.dumps({
                                "type": "response.create",
                                "response": {
                                    "instructions": "Transcribe the user speech from the most recent audio buffer. Output only the transcript; no punctuation if unsure.",
                                    "modalities": ["text"],
                                },
                            }))
                        except Exception as e:
                            _dbg_set("last_error", f"manual_commit_failed: {e}")
                            await close_upstream()
                            upstream = None
                    else:
                        # HTTP-fallback: transkribera buffrat ljud
                        if buffered_bytes:
                            transcript = await transcribe_bytes(httpx_client, bytes(buffered_bytes))
                            _dbg_set("last_final", transcript)
                            await safe_send_json(ws, {"type": "stt.final", "text": transcript})
                            buffered_bytes.clear()
                        else:
                            await safe_send_json(ws, {"type": "stt.final", "text": ""})
                elif text_msg == "ping":
                    await ws.send_text("pong")
                # andra textmeddelanden ignoreras

    except WebSocketDisconnect:
        pass
    finally:
        await close_upstream()
        if upstream_reader_task:
            upstream_reader_task.cancel()
        if committer_task:
            committer_task.cancel()
        await httpx_client.aclose()
        try:
            await ws.close()
        except Exception:
            pass

# -------------------- Helpers --------------------
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
    except Exception as e:
        _dbg_set("last_error", f"http_fallback_failed: {e}")
        return ""

async def safe_send_json(ws: WebSocket, payload: dict):
    try:
        await ws.send_text(json.dumps(payload, ensure_ascii=False))
    except Exception:
        pass
