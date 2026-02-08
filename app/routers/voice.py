import asyncio
import base64
import io
import uuid
import wave

import websockets
from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from websockets.exceptions import ConnectionClosed

from app.core import config
from app.dependencies import get_speech_provider
from app.providers.microsoft_voice_live_provider import MicrosoftVoiceLiveError
from app.providers.disabled_speech_provider import DisabledSpeechProvider
from app.providers.speech_provider import SpeechProvider
from app.schemas.voice import (
    NormalizedTranscript,
    SynthesizeRequest,
    SynthesizeResponse,
    TranscribeAudioRequest,
    VoiceInfo,
)


router = APIRouter(prefix="/voice", tags=["voice"])


@router.get("/health")
async def voice_health(provider: SpeechProvider = Depends(get_speech_provider)) -> dict[str, str]:
    if isinstance(provider, DisabledSpeechProvider):
        return {"status": "disabled"}

    try:
        ok = await provider.health_check()
    except Exception:
        return {"status": "unhealthy"}

    return {"status": "ok" if ok else "unhealthy"}


@router.post("/transcribe", response_model=NormalizedTranscript)
async def transcribe_audio(
    body: TranscribeAudioRequest,
    provider: SpeechProvider = Depends(get_speech_provider),
) -> NormalizedTranscript:
    if isinstance(provider, DisabledSpeechProvider):
        return await provider.transcribe_wav(
            wav_bytes=b"",
            sample_rate_hz=body.audio.sample_rate_hz,
            language=body.language,
            request_id=body.request_id,
        )

    raw_pcm = base64.b64decode(body.audio.audio_b64)

    sample_rate = body.audio.sample_rate_hz
    request_id = body.request_id or str(uuid.uuid4())
    language = body.language

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(raw_pcm)

    try:
        return await provider.transcribe_wav(
            wav_bytes=buf.getvalue(),
            sample_rate_hz=sample_rate,
            language=language,
            request_id=request_id,
        )
    except MicrosoftVoiceLiveError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail="STT request failed") from exc


@router.get("/voices", response_model=list[VoiceInfo])
async def list_voices(
    provider: SpeechProvider = Depends(get_speech_provider),
) -> list[VoiceInfo]:
    try:
        voices = await provider.list_voices()
    except NotImplementedError:
        return []
    except Exception:
        return []

    out: list[VoiceInfo] = []
    for v in voices:
        if isinstance(v, dict) and v.get("name"):
            out.append(VoiceInfo(**v))
    return out


@router.post("/synthesize", response_model=SynthesizeResponse)
async def synthesize(
    body: SynthesizeRequest,
    provider: SpeechProvider = Depends(get_speech_provider),
) -> SynthesizeResponse:
    rid = body.request_id or str(uuid.uuid4())
    try:
        audio_bytes, mime, voice_used, final_rid = await provider.synthesize_text(
            text=body.text,
            language=body.language,
            voice=body.voice,
            request_id=rid,
            output_format=body.output_format,
        )
    except MicrosoftVoiceLiveError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail="TTS request failed") from exc

    return SynthesizeResponse(
        request_id=final_rid,
        provider=getattr(provider, "name", provider.__class__.__name__),
        voice=voice_used,
        mime_type=mime,
        audio_b64=base64.b64encode(audio_bytes).decode("ascii"),
        raw=None if not isinstance(provider, DisabledSpeechProvider) else {"status": "disabled"},
    )


@router.websocket("/stream")
async def voice_stream(
    websocket: WebSocket,
    provider: SpeechProvider = Depends(get_speech_provider),
) -> None:
    """Websocket bridge to the upstream realtime voice service.

    Disabled by default unless ENABLE_VOICE_STREAM_WS=true.
    """

    await websocket.accept()

    if not config.ENABLE_VOICE_STREAM_WS:
        await websocket.close(code=1013)
        return

    # Streaming requires a provider that exposes a realtime websocket endpoint.
    if isinstance(provider, DisabledSpeechProvider):
        await websocket.close(code=1013)
        return

    if not hasattr(provider, "build_realtime_ws_url"):
        await websocket.close(code=1013)
        return

    api_key = config.MICROSOFT_VOICE_LIVE_API_KEY
    if not api_key:
        await websocket.close(code=1011)
        return

    model = config.MICROSOFT_VOICE_LIVE_REALTIME_MODEL

    # Prefer header auth; allow query auth only when enabled.
    ws_url = provider.build_realtime_ws_url(model=model)  # type: ignore[attr-defined]
    ws_url_with_query_key = f"{ws_url}&api-key={api_key}"

    try:
        # Attempt header-based auth first; fall back to query-string auth for compatibility.
        try:
            connect_ctx = websockets.connect(
                ws_url,
                max_size=None,
                subprotocols=["realtime"],
                extra_headers={"api-key": api_key},
            )
            ms_ws_cm = connect_ctx
            if config.MICROSOFT_VOICE_LIVE_AUTH_IN_QUERY:
                raise RuntimeError("Forced query auth")
        except Exception:
            ms_ws_cm = websockets.connect(
                ws_url_with_query_key,
                max_size=None,
                subprotocols=["realtime"],
            )

        async with ms_ws_cm as ms_ws:

            async def client_to_ms() -> None:
                while True:
                    try:
                        message = await websocket.receive()
                    except (WebSocketDisconnect, RuntimeError):
                        await ms_ws.close()
                        break

                    try:
                        if message.get("text") is not None:
                            await ms_ws.send(message["text"])
                        elif message.get("bytes") is not None:
                            await ms_ws.send(message["bytes"])
                    except ConnectionClosed:
                        break

            async def ms_to_client() -> None:
                try:
                    async for msg in ms_ws:
                        if isinstance(msg, str):
                            await websocket.send_text(msg)
                        else:
                            await websocket.send_bytes(msg)
                except ConnectionClosed:
                    await websocket.close(code=1011)

            await asyncio.wait(
                [
                    asyncio.create_task(client_to_ms()),
                    asyncio.create_task(ms_to_client()),
                ],
                return_when=asyncio.FIRST_EXCEPTION,
            )

    except WebSocketDisconnect:
        return
    except Exception:
        await websocket.close(code=1011)
