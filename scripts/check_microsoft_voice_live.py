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
    raw_pcm = b"\x00\x00" * frames  # PCM16 mono silence

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate_hz)
        wf.writeframes(raw_pcm)

    return buf.getvalue()


async def main() -> None:
    _add_repo_root_to_path()

    # Ensure config is loadable.
    os.environ.setdefault("DATABASE_URL", "sqlite:///./local_test.db")

    from app.core import config
    from app.providers.microsoft_voice_live_provider import (
        MicrosoftVoiceLiveError,
        MicrosoftVoiceLiveProvider,
    )

    print("Microsoft Voice Live check")
    print(f"- STT_URL set: {bool(config.MICROSOFT_VOICE_LIVE_STT_URL)}")
    print(f"- BASE_URL set: {bool(config.MICROSOFT_VOICE_LIVE_BASE_URL)}")
    print(f"- API_KEY set: {bool(config.MICROSOFT_VOICE_LIVE_API_KEY)}")

    if not config.MICROSOFT_VOICE_LIVE_API_KEY or not config.MICROSOFT_VOICE_LIVE_STT_URL:
        print("Missing config: set MICROSOFT_VOICE_LIVE_API_KEY and MICROSOFT_VOICE_LIVE_STT_URL in your .env")
        return

    provider = MicrosoftVoiceLiveProvider()

    try:
        ok = await provider.health_check()
        print(f"- health_check: {ok}")

        wav_bytes = _make_silence_wav(sample_rate_hz=16000, duration_ms=800)
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
        if transcript.text:
            print(f"  text_preview: {transcript.text[:120]}")

    except MicrosoftVoiceLiveError as exc:
        print(f"MicrosoftVoiceLiveError: {exc}")
    except Exception as exc:
        print(f"Unexpected error: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    asyncio.run(main())
