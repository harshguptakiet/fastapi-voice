from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path


def _add_repo_root_to_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))


def _make_silence_wav(*, sample_rate_hz: int, duration_ms: int) -> bytes:
    import io
    import wave

    frames = int(sample_rate_hz * (duration_ms / 1000.0))
    raw_pcm = b"\x00\x00" * frames

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate_hz)
        wf.writeframes(raw_pcm)

    return buf.getvalue()


async def main() -> None:
    _add_repo_root_to_path()

    os.environ.setdefault("DATABASE_URL", "sqlite:///./local_test.db")

    from app.core import config
    from app.providers.deepgram_elevenlabs_provider import (
        DeepgramElevenLabsError,
        DeepgramElevenLabsProvider,
    )

    print("Deepgram + ElevenLabs check")
    print(f"- DEEPGRAM_API_KEY set: {bool(config.DEEPGRAM_API_KEY)}")
    print(f"- ELEVENLABS_API_KEY set: {bool(config.ELEVENLABS_API_KEY)}")
    print(f"- ELEVENLABS_VOICE_ID set: {bool(config.ELEVENLABS_VOICE_ID)}")

    if not config.DEEPGRAM_API_KEY or not config.ELEVENLABS_API_KEY or not config.ELEVENLABS_VOICE_ID:
        print("Missing config: set DEEPGRAM_API_KEY, ELEVENLABS_API_KEY, ELEVENLABS_VOICE_ID in your .env")
        return

    provider = DeepgramElevenLabsProvider()

    try:
        ok = await provider.health_check()
        print(f"- health_check: {ok}")

        wav_bytes = _make_silence_wav(sample_rate_hz=16000, duration_ms=600)
        transcript = await provider.transcribe_wav(
            wav_bytes=wav_bytes,
            sample_rate_hz=16000,
            language="en-US",
        )

        print("- transcribe_wav: success")
        print(f"  provider: {transcript.provider}")
        print(f"  request_id: {transcript.request_id}")
        print(f"  language: {transcript.language}")
        print(f"  text_length: {len(transcript.text)}")

        audio_bytes, mime, voice_used, request_id = await provider.synthesize_text(text="hello")
        print("- synthesize_text: success")
        print(f"  request_id: {request_id}")
        print(f"  mime: {mime}")
        print(f"  voice: {voice_used}")
        print(f"  bytes: {len(audio_bytes)}")

    except DeepgramElevenLabsError as exc:
        print(f"DeepgramElevenLabsError: {exc}")
    except Exception as exc:
        print(f"Unexpected error: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    asyncio.run(main())
