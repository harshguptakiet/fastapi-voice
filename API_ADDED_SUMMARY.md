# API Added Summary

This file summarizes the FastAPI routes currently registered in `app/main.py`.

## Core

- `GET /health` — basic service health check (`{"status": "ok"}`)

## Users (`/users`)

- `GET /users/` — list users
- `POST /users/` — create a user (201)

## Interactions (`/interactions`)

- `POST /interactions` — normalize and return a text interaction payload

## Voice (`/voice`)

- `GET /voice/health` — speech provider health (can return `disabled`)
- `POST /voice/transcribe` — speech-to-text (base64 audio payload) -> normalized transcript
- `GET /voice/voices` — list available voices
- `POST /voice/synthesize` — text-to-speech -> base64 audio response
- `WS /voice/stream` — websocket bridge to realtime voice upstream (gated by config)

## Sessions (`/sessions`)

- `POST /sessions` — create a new session (UUID)
- `GET /sessions/{session_id}` — fetch session state
- `DELETE /sessions/{session_id}` — delete/reset a session
- `GET /sessions/{session_id}/messages` — list session messages
- `POST /sessions/{session_id}/messages` — add a message to a session

## LLM (`/llm`)

- `GET /llm/health` — provider resolution health
- `GET /llm/models` — list available/configured models
- `POST /llm/generate` — generate text from a prompt
- `POST /llm/stream` — server-sent events (SSE) token stream

## Transcripts (`/transcripts`)

- `POST /transcripts/normalize` — normalize transcript payloads into a common shape

## Status (`/status`)

- `GET /status` — system/dependency status (db/llm/voice + environment)

## Metrics

- `GET /metrics` — Prometheus-compatible text endpoint
