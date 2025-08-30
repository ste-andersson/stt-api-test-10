# stt-api-test-10

Minimal FastAPI-baserad backend för röst → text via OpenAI.
Kompatibel med nuvarande frontend som skickar ljud-chunks över WebSocket.

## TL;DR

```bash
cp .env.example .env  # fyll i din OPENAI_API_KEY
make install
make run
# WS endpoint: ws://localhost:8000/ws  (alternativt /ws/transcribe)
```

## WS-protokoll (enkelt)

- Server skickar strängen `"ready"` direkt vid anslutning.
- Klienten skickar binära ljud-chunks (t.ex. WAV/PCM).
- För att få transkript:
  - Realtime-läge: skicka `"flush"` (eller `"commit"`) för att begära svar.
  - HTTP-fallback: skicka `"flush"` (eller `"commit"`) för att transkribera de buffrade byten.
- Servern svarar löpande med:
  - `{"type":"stt.delta","text":"..."}` (endast när Realtime är aktivt)
  - `{"type":"stt.final","text":"..."}`

## Miljövariabler

Se **.env.example** för full lista. Kortfattat:
- `OPENAI_API_KEY` – din API-nyckel
- `REALTIME_URL` – om satt används OpenAI Realtime WS (proxy/bridge)
- `REALTIME_TRANSCRIBE_MODEL` – t.ex. `gpt-4o-mini-transcribe` eller `whisper-1`
- `INPUT_LANGUAGE` – t.ex. `sv`
- `OPENAI_ADD_BETA_HEADER=1` om Realtime kräver beta-headern

## Deploy på Render

- Start command: `uvicorn app.main:app --host 0.0.0.0 --port ${PORT}`
- Exponera port 8000 (eller använd variabeln `PORT` som Render sätter).

## Kompatibilitetstips

Om din frontend redan använder `/ws/transcribe` behåll den URL:en – denna backend har båda (`/ws` och `/ws/transcribe`).

## Licens

MIT (valfritt att justera).
