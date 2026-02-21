from __future__ import annotations

import uuid
from typing import Any, Optional

import httpx

from app.core import config
from app.providers.speech_provider import SpeechProvider
from app.schemas.voice import NormalizedTranscript


class MicrosoftVoiceLiveError(RuntimeError):
    """Raised when a Microsoft Voice Live API call fails."""


class MicrosoftVoiceLiveProvider(SpeechProvider):
    """SpeechProvider backed by Microsoft Speech Services."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        region: Optional[str] = None,
        base_url: Optional[str] = None,
        stt_url: Optional[str] = None,
        tts_url: Optional[str] = None,
    ) -> None:
        # Configuration comes from app.core.config
        self.api_key = api_key or config.MICROSOFT_VOICE_LIVE_API_KEY
        self.region = region or config.MICROSOFT_VOICE_LIVE_REGION
        self.base_url = base_url or config.MICROSOFT_VOICE_LIVE_BASE_URL
        self.stt_url = stt_url or config.MICROSOFT_VOICE_LIVE_STT_URL
        self.tts_url = tts_url or config.MICROSOFT_VOICE_LIVE_TTS_URL

        if not self.api_key:
            raise MicrosoftVoiceLiveError("MICROSOFT_VOICE_LIVE_API_KEY is not set")

        # base_url is used for realtime websocket bridging.

    @property
    def name(self) -> str:
        return "microsoft"

    async def health_check(self) -> bool:
        url = (self.stt_url or self.base_url or "").rstrip("/")
        if not url:
            return False

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(url, headers=self._auth_headers())
        except httpx.HTTPError as exc:
            raise MicrosoftVoiceLiveError(f"Health check failed: {exc}") from exc

        return resp.status_code < 500

    async def transcribe_wav(
        self,
        *,
        wav_bytes: bytes,
        sample_rate_hz: int,
        language: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> NormalizedTranscript:
        if not self.stt_url:
            raise MicrosoftVoiceLiveError("MICROSOFT_VOICE_LIVE_STT_URL is not set")

        lang = (language or "en-US").strip() or "en-US"
        rid = request_id or str(uuid.uuid4())

        stt_base = self.stt_url.rstrip("/")
        url = (
            f"{stt_base}/speech/recognition/conversation/cognitiveservices/v1"
            f"?language={lang}&format=detailed"
        )

        headers = {
            "Accept": "application/json",
            "Ocp-Apim-Subscription-Key": self.api_key,
            "Content-Type": f"audio/wav; codecs=audio/pcm; samplerate={sample_rate_hz}",
        }

        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(url, headers=headers, content=wav_bytes)

        if resp.status_code >= 400:
            raise MicrosoftVoiceLiveError(
                f"STT request failed (status={resp.status_code})"
            )

        try:
            data: dict[str, Any] = resp.json()
        except Exception as exc:
            raise MicrosoftVoiceLiveError("Invalid STT response") from exc

        text = (
            data.get("DisplayText")
            or data.get("displayText")
            or data.get("Text")
            or data.get("text")
        )

        confidence: float | None = None
        if isinstance(data.get("NBest"), list) and data["NBest"]:
            first = data["NBest"][0] or {}
            if not isinstance(text, str) or not text.strip():
                text = (
                    first.get("Display")
                    or first.get("display")
                    or first.get("Lexical")
                    or first.get("lexical")
                    or text
                )

            c = first.get("Confidence") or first.get("confidence")
            if isinstance(c, (int, float)):
                confidence = float(c)

        final_text = text.strip() if isinstance(text, str) else ""

        return NormalizedTranscript(
            request_id=rid,
            provider=self.name,
            text=final_text,
            language=lang,
            confidence=confidence,
            segments=[],
            raw=None,
        )

    async def list_voices(self) -> list[dict]:
        # Azure voices list endpoint lives under the TTS host.
        if not self.tts_url:
            raise MicrosoftVoiceLiveError("MICROSOFT_VOICE_LIVE_TTS_URL is not set")

        base = self.tts_url.rstrip("/")
        # If configured as /cognitiveservices/v1, the list endpoint is a sibling.
        if base.endswith("/cognitiveservices/v1"):
            list_url = base.replace("/cognitiveservices/v1", "/cognitiveservices/voices/list")
        else:
            list_url = f"{base}/cognitiveservices/voices/list"

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(list_url, headers=self._auth_headers())

        if resp.status_code >= 400:
            raise MicrosoftVoiceLiveError(f"Voices list failed (status={resp.status_code})")

        try:
            data = resp.json()
        except Exception as exc:
            raise MicrosoftVoiceLiveError("Invalid voices list response") from exc

        if not isinstance(data, list):
            raise MicrosoftVoiceLiveError("Unexpected voices list format")

        # Normalize a small subset of fields.
        voices: list[dict] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            voices.append(
                {
                    "name": item.get("ShortName") or item.get("shortName") or item.get("Name") or item.get("name"),
                    "locale": item.get("Locale") or item.get("locale"),
                    "gender": item.get("Gender") or item.get("gender"),
                    "provider": self.name,
                }
            )
        return [v for v in voices if v.get("name")]

    async def synthesize_text(
        self,
        *,
        text: str,
        language: Optional[str] = None,
        voice: Optional[str] = None,
        request_id: Optional[str] = None,
        output_format: Optional[str] = None,
    ) -> tuple[bytes, str, str | None, str]:
        if not self.tts_url:
            raise MicrosoftVoiceLiveError("MICROSOFT_VOICE_LIVE_TTS_URL is not set")

        rid = request_id or str(uuid.uuid4())
        voice_name = voice or config.MICROSOFT_VOICE_LIVE_VOICE
        if not voice_name:
            # Sensible default if none configured.
            voice_name = "en-US-JennyNeural"

        lang = (language or "en-US").strip() or "en-US"

        # Default to mp3 for compactness.
        fmt = (output_format or "audio-16khz-32kbitrate-mono-mp3").strip()
        mime = "audio/mpeg" if "mp3" in fmt.lower() else "audio/wav"

        ssml = (
            f"<speak version='1.0' xml:lang='{lang}'>"
            f"<voice name='{voice_name}'>{text}</voice>"
            f"</speak>"
        )

        headers = {
            **self._auth_headers(),
            "Content-Type": "application/ssml+xml",
            "X-Microsoft-OutputFormat": fmt,
            "User-Agent": "bot-backend",
        }

        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(self.tts_url.rstrip("/"), headers=headers, content=ssml.encode("utf-8"))

        if resp.status_code >= 400:
            raise MicrosoftVoiceLiveError(f"TTS request failed (status={resp.status_code}): {resp.text}")

        return (resp.content, mime, voice_name, rid)

    def _auth_headers(self) -> dict[str, str]:
        return {
            "Ocp-Apim-Subscription-Key": self.api_key,
        }

    def build_realtime_ws_url(self, *, model: str, api_version: str | None = None) -> str:
        if not self.base_url:
            raise MicrosoftVoiceLiveError(
                "MICROSOFT_VOICE_LIVE_BASE_URL is not set; configure it in .env"
            )

        version = api_version or config.MICROSOFT_VOICE_LIVE_REALTIME_API_VERSION
        if not version:
            raise MicrosoftVoiceLiveError(
                "MICROSOFT_VOICE_LIVE_REALTIME_API_VERSION is not set"
            )

        host = self.base_url.replace("https://", "").replace("http://", "").rstrip("/")
        return f"wss://{host}/voice-live/realtime?api-version={version}&model={model}"
