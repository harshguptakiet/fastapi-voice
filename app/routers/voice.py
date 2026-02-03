import asyncio
import base64
import io
import wave

import httpx
import websockets
from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from websockets.exceptions import ConnectionClosed

from app.core import config
from app.dependencies import get_llm_provider
from app.providers.llm_provider import LLMProvider
from app.providers.microsoft_voice_live_provider import MicrosoftVoiceLiveProvider


router = APIRouter(prefix="/voice", tags=["voice"])


class TranscribeRequest(BaseModel):
    # Base64-encoded PCM16 mono audio; optional sample rate in Hz
    audio: str
    sample_rate: int | None = None


@router.get("/health")
async def voice_health(provider: LLMProvider = Depends(get_llm_provider)) -> dict[str, str]:
    if not isinstance(provider, MicrosoftVoiceLiveProvider):
        return {"status": "disabled"}

    ok = await provider._simple_health_check()  # type: ignore[attr-defined]
    return {"status": "ok" if ok else "unhealthy"}


@router.post("/transcribe")
async def transcribe_audio(body: TranscribeRequest) -> dict[str, str]:
    stt_base = config.MICROSOFT_VOICE_LIVE_STT_URL
    api_key = config.MICROSOFT_VOICE_LIVE_API_KEY
    if not stt_base or not api_key:
        raise HTTPException(status_code=500, detail="STT not configured")

    try:
        raw_pcm = base64.b64decode(body.audio)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid audio payload")

    sample_rate = body.sample_rate or 24000

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(raw_pcm)

    stt_base = stt_base.rstrip("/")
    url = f"{stt_base}/speech/recognition/conversation/cognitiveservices/v1?language=en-US&format=detailed"
    headers = {
        "Accept": "application/json",
        "Ocp-Apim-Subscription-Key": api_key,
        "Content-Type": f"audio/wav; codecs=audio/pcm; samplerate={sample_rate}",
    }

    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(url, headers=headers, content=buf.getvalue())

    if resp.status_code >= 400:
        raise HTTPException(status_code=502, detail="STT request failed")

    try:
        data = resp.json()
    except Exception:
        raise HTTPException(status_code=502, detail="Invalid STT response")

    text = (
        data.get("DisplayText")
        or data.get("displayText")
        or data.get("Text")
        or data.get("text")
    )

    if (not isinstance(text, str) or not text.strip()) and isinstance(
        data.get("NBest"), list
    ) and data["NBest"]:
        first = data["NBest"][0] or {}
        text = (
            first.get("Display")
            or first.get("display")
            or first.get("Lexical")
            or first.get("lexical")
            or text
        )
    if not isinstance(text, str):
        return {"text": ""}

    if not text.strip():
        return {"text": ""}

    return {"text": text.strip()}


@router.websocket("/stream")
async def voice_stream(
    websocket: WebSocket,
    provider: LLMProvider = Depends(get_llm_provider),
) -> None:
    await websocket.accept()

    if not isinstance(provider, MicrosoftVoiceLiveProvider):
        await websocket.close(code=1013)
        return

    base_url = config.MICROSOFT_VOICE_LIVE_BASE_URL
    api_key = config.MICROSOFT_VOICE_LIVE_API_KEY
    if not base_url or not api_key:
        await websocket.close(code=1011)
        return

    host = base_url.replace("https://", "").replace("http://", "").rstrip("/")
    ws_url = (
        f"wss://{host}/voice-live/realtime?api-version=2025-10-01"
        f"&model=gpt-4.1&api-key={api_key}"
    )

    try:
        async with websockets.connect(ws_url, max_size=None, subprotocols=["realtime"]) as ms_ws:

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
