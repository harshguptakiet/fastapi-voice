from __future__ import annotations

import asyncio
import base64
import os
import sys
from pathlib import Path

import httpx


async def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

    # Set safe defaults for local smoke tests.
    os.environ.setdefault("DATABASE_URL", "sqlite:///./local_test.db")
    os.environ.setdefault("ENV", "dev")
    os.environ.setdefault("USE_MICROSOFT_VOICE_LIVE", "false")
    os.environ.setdefault("LLM_PROVIDER", "dummy")

    from app.main import app

    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        health = await client.get("/health")
        assert health.status_code == 200, health.text
        assert health.json().get("status") == "ok", health.json()

        status = await client.get("/status")
        assert status.status_code == 200, status.text
        assert status.json().get("status") in {"ok", "degraded", "unhealthy"}

        llm_health = await client.get("/llm/health")
        assert llm_health.status_code == 200, llm_health.text
        assert llm_health.json().get("status") == "ok"

        llm_generate = await client.post("/llm/generate", json={"prompt": "ping"})
        assert llm_generate.status_code == 200, llm_generate.text
        assert isinstance(llm_generate.json().get("text"), str)

        sess = await client.post("/sessions")
        assert sess.status_code == 200, sess.text
        session_id = sess.json()["session_id"]

        add_msg = await client.post(
            f"/sessions/{session_id}/messages",
            json={"role": "user", "content": "hello"},
        )
        assert add_msg.status_code == 200, add_msg.text
        assert len(add_msg.json().get("messages") or []) >= 1

        interaction = await client.post(
            "/interactions",
            json={"session_id": "s1", "text": " hello ", "language": "en-US"},
        )
        assert interaction.status_code == 200, interaction.text
        interaction_data = interaction.json()
        assert interaction_data["session_id"] == "s1"
        assert interaction_data["input_type"] == "text"
        assert interaction_data["normalized_text"] == "hello"

        voice_health = await client.get("/voice/health")
        assert voice_health.status_code == 200, voice_health.text
        assert voice_health.json().get("status") in {"disabled", "ok", "unhealthy"}, voice_health.json()

        voices = await client.get("/voice/voices")
        assert voices.status_code == 200, voices.text
        assert isinstance(voices.json(), list)

        synth = await client.post("/voice/synthesize", json={"text": "hello"})
        assert synth.status_code in {200, 502, 501}, synth.text
        if synth.status_code == 200:
            assert synth.json().get("audio_b64")

        raw_pcm = b"\x00\x00" * 160
        payload = {
            "audio": {
                "audio_b64": base64.b64encode(raw_pcm).decode("ascii"),
                "sample_rate_hz": 16000,
            },
            "language": "en-US",
        }
        transcribe = await client.post("/voice/transcribe", json=payload)
        assert transcribe.status_code in {200, 502}, transcribe.text

        norm = await client.post(
            "/transcripts/normalize",
            json={"provider": "microsoft", "raw": {"DisplayText": " hello "}},
        )
        assert norm.status_code == 200, norm.text
        assert norm.json()["transcript"]["text"] == "hello"

    print("smoke_test: OK")


if __name__ == "__main__":
    asyncio.run(main())
