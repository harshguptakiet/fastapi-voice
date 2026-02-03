// Element references
const logEl = document.getElementById("log");
const textInput = document.getElementById("textInput");
const agentStyleEl = document.getElementById("agentStyle");
const responseLengthEl = document.getElementById("responseLength");
const speedControlEl = document.getElementById("speedControl");
const speedValueEl = document.getElementById("speedValue");
const startVoiceBtn = document.getElementById("startVoice");
const stopVoiceBtn = document.getElementById("stopVoice");
const replayVoiceBtn = document.getElementById("replayVoice");
const sendVoiceBtn = document.getElementById("sendVoice");
const voiceStatusEl = document.getElementById("voiceStatus");

let ws;

// Playback state
let audioCtx;
let playbackSpeed = 1.0;
let nextPlayTime = 0;

// Recording state
let mediaStream = null;
let mediaSource = null;
let processorNode = null;
let isRecording = false;
let recordedPcmChunks = [];

function log(msg) {
  logEl.textContent += msg + "\n";
  logEl.scrollTop = logEl.scrollHeight;
}

function ensureAudioContext() {
  if (!audioCtx) {
    audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 24000 });
    nextPlayTime = audioCtx.currentTime;
  }
  if (audioCtx.state === "suspended") {
    audioCtx.resume();
  }
}

// Base64 <-> PCM helpers
function base64ToPCM16(base64) {
  const binary = atob(base64);
  const len = binary.length;
  const buffer = new ArrayBuffer(len);
  const view = new Uint8Array(buffer);
  for (let i = 0; i < len; i++) view[i] = binary.charCodeAt(i);
  return new Int16Array(buffer);
}

function floatToPCM16(float32Array) {
  const pcm16 = new Int16Array(float32Array.length);
  for (let i = 0; i < float32Array.length; i++) {
    let s = float32Array[i];
    s = Math.max(-1, Math.min(1, s));
    pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return pcm16;
}

function pcm16ToBase64(pcm16) {
  const bytes = new Uint8Array(pcm16.buffer);
  let binary = "";
  for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i]);
  return btoa(binary);
}

function playPCM16(pcm16) {
  ensureAudioContext();
  const sampleRate = 24000;
  const audioBuffer = audioCtx.createBuffer(1, pcm16.length, sampleRate);
  const channel = audioBuffer.getChannelData(0);
  for (let i = 0; i < pcm16.length; i++) channel[i] = pcm16[i] / 32768;

  const source = audioCtx.createBufferSource();
  source.buffer = audioBuffer;
  source.playbackRate.value = playbackSpeed;
  source.connect(audioCtx.destination);

  const now = audioCtx.currentTime;
  if (nextPlayTime < now) nextPlayTime = now;
  source.start(nextPlayTime);
  nextPlayTime += audioBuffer.duration / playbackSpeed;
}

function buildBaseInstructions() {
  const style = agentStyleEl.value;
  const length = responseLengthEl.value;

  let persona = "You are a friendly, helpful assistant having a natural conversation with the user.";
  if (style === "teacher") {
    persona = "You are a patient teacher who explains concepts step by step.";
  } else if (style === "coach") {
    persona = "You are an encouraging motivational coach who is positive and supportive.";
  }

  let lengthHint = "Give short answers.";
  if (length === "medium") {
    lengthHint = "Give medium length answers with a bit more detail.";
  } else if (length === "long") {
    lengthHint = "Give detailed answers and include explanations.";
  }

  const speaking = "Speak in a calm, clear voice at a slightly slower than normal pace, but do not keep repeating instructions.";
  return `${persona} ${lengthHint} ${speaking}`;
}

function buildResponseInstructions() {
  return buildBaseInstructions() + " Prioritize clarity over speed.";
}

// UI wiring
speedControlEl.addEventListener("input", () => {
  playbackSpeed = parseFloat(speedControlEl.value) || 1.0;
  speedValueEl.textContent = playbackSpeed.toFixed(1) + "x";
});

// Recording
async function startVoiceInput() {
  if (!ws || ws.readyState !== WebSocket.OPEN || isRecording) return;

  ensureAudioContext();
  recordedPcmChunks = [];

  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (err) {
    log("ERROR: Failed to access microphone: " + err);
    return;
  }

  mediaSource = audioCtx.createMediaStreamSource(mediaStream);
  processorNode = audioCtx.createScriptProcessor(4096, 1, 1);
  mediaSource.connect(processorNode);

  const silentGain = audioCtx.createGain();
  silentGain.gain.value = 0;
  processorNode.connect(silentGain);
  silentGain.connect(audioCtx.destination);

  isRecording = true;
  voiceStatusEl.textContent = "Listening...";
  startVoiceBtn.disabled = true;
  stopVoiceBtn.disabled = false;
  replayVoiceBtn.disabled = true;
  sendVoiceBtn.disabled = true;

  processorNode.onaudioprocess = (event) => {
    if (!isRecording || !ws || ws.readyState !== WebSocket.OPEN) return;
    const inputBuffer = event.inputBuffer.getChannelData(0);
    recordedPcmChunks.push(floatToPCM16(inputBuffer));
  };
}

function stopVoiceInput() {
  if (!isRecording) return;
  isRecording = false;

  if (processorNode) {
    processorNode.disconnect();
    processorNode.onaudioprocess = null;
    processorNode = null;
  }
  if (mediaSource) {
    mediaSource.disconnect();
    mediaSource = null;
  }
  if (mediaStream) {
    mediaStream.getTracks().forEach((t) => t.stop());
    mediaStream = null;
  }

  if (!ws || ws.readyState !== WebSocket.OPEN) {
    voiceStatusEl.textContent = "Idle";
    startVoiceBtn.disabled = false;
    stopVoiceBtn.disabled = true;
    replayVoiceBtn.disabled = true;
    sendVoiceBtn.disabled = true;
    return;
  }

  startVoiceBtn.disabled = false;
  stopVoiceBtn.disabled = true;
  if (recordedPcmChunks.length) {
    voiceStatusEl.textContent = "Recorded  click Send";
    replayVoiceBtn.disabled = false;
    sendVoiceBtn.disabled = false;
  } else {
    voiceStatusEl.textContent = "Idle";
    replayVoiceBtn.disabled = true;
    sendVoiceBtn.disabled = true;
  }
}

function replayVoiceRecording() {
  if (!recordedPcmChunks.length) return;

  let totalLength = 0;
  for (const chunk of recordedPcmChunks) totalLength += chunk.length;
  const combined = new Int16Array(totalLength);
  let offset = 0;
  for (const chunk of recordedPcmChunks) {
    combined.set(chunk, offset);
    offset += chunk.length;
  }

  log("Replaying recorded audio...");
  playPCM16(combined);
}

function sendVoiceRecording() {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  if (!recordedPcmChunks.length) return;

  voiceStatusEl.textContent = "Transcribing...";
  sendVoiceBtn.disabled = true;

  let totalLength = 0;
  for (const chunk of recordedPcmChunks) totalLength += chunk.length;
  const combined = new Int16Array(totalLength);
  let offset = 0;
  for (const chunk of recordedPcmChunks) {
    combined.set(chunk, offset);
    offset += chunk.length;
  }

  const base64Audio = pcm16ToBase64(combined);

  const sr = audioCtx && typeof audioCtx.sampleRate === "number" ? Math.round(audioCtx.sampleRate) : 24000;

  fetch("http://127.0.0.1:8000/voice/transcribe", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ audio: base64Audio, sample_rate: sr }),
  })
    .then(async (res) => {
      if (!res.ok) {
        throw new Error("HTTP " + res.status);
      }

      let payload = null;
      try {
        payload = await res.json();
      } catch (e) {
        // ignore JSON parse errors
      }

      const text = payload && typeof payload.text === "string" ? payload.text.trim() : "";
      if (!text) {
        const fallback = "Sorry, I couldn't hear you. Please try speaking again.";
        log("No speech recognized by STT; sending fallback: " + fallback);
        voiceStatusEl.textContent = "Idle";
        sendTextAsUser(fallback);
        return;
      }
      log("TRANSCRIBED: " + text);
      voiceStatusEl.textContent = "Sending to assistant...";
      sendTextAsUser(text);
      voiceStatusEl.textContent = "Idle";
    })
    .catch((err) => {
      log("ERROR: STT failed  " + err);
      voiceStatusEl.textContent = "Idle";
    });
}

startVoiceBtn.onclick = startVoiceInput;
stopVoiceBtn.onclick = stopVoiceInput;
replayVoiceBtn.onclick = replayVoiceRecording;
sendVoiceBtn.onclick = sendVoiceRecording;

// WebSocket connection
document.getElementById("connect").onclick = () => {
  ws = new WebSocket("ws://127.0.0.1:8000/voice/stream");

  ws.onopen = () => {
    log("WS connected to /voice/stream");
    document.getElementById("sendSession").disabled = false;
    document.getElementById("sendText").disabled = false;
    startVoiceBtn.disabled = false;
    stopVoiceBtn.disabled = true;
    sendVoiceBtn.disabled = true;
  };

  ws.onmessage = (event) => {
    if (typeof event.data === "string") {
      try {
        const msg = JSON.parse(event.data);
        const type = msg.type;

        if (type === "session.created") {
          log("SESSION: created");
        } else if (type === "session.updated") {
          log("SESSION: updated");
        } else if (type === "response.audio.delta") {
          const pcm16 = base64ToPCM16(msg.delta);
          playPCM16(pcm16);
        } else if (type === "response.text.delta") {
          log("TEXT : " + msg.delta);
        } else if (type === "error") {
          log("ERROR: " + JSON.stringify(msg.error));
        } else if (type) {
          log("EVT: " + type);
        }
      } catch {
        log("RECV (raw): " + event.data);
      }
    } else {
      log("RECV binary (" + event.data.size + " bytes)");
    }
  };

  ws.onclose = (e) => {
    log("WS closed: " + e.code + " " + e.reason);
    document.getElementById("sendSession").disabled = true;
    document.getElementById("sendText").disabled = true;
    startVoiceBtn.disabled = true;
    stopVoiceBtn.disabled = true;
    replayVoiceBtn.disabled = true;
    sendVoiceBtn.disabled = true;
    voiceStatusEl.textContent = "Idle";
  };

  ws.onerror = (err) => {
    log("WS error: " + err);
  };
};

// Session configuration
document.getElementById("sendSession").onclick = () => {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;

  const sessionUpdate = {
    type: "session.update",
    session: {
      modalities: ["text", "audio"],
      instructions: buildBaseInstructions(),
      voice: {
        name: "en-US-JennyNeural",
        type: "azure-standard",
      },
      input_audio_format: "pcm16",
      input_audio_transcription: {
        model: "gpt-4o-mini-transcribe",
      },
      output_audio_format: "pcm16",
    },
  };

  ws.send(JSON.stringify(sessionUpdate));
  log("SEND: session.update");
};

function sendTextAsUser(text) {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  const trimmed = (text || "").trim();
  if (!trimmed) return;

  if (audioCtx) {
    audioCtx.close();
    audioCtx = null;
  }
  nextPlayTime = 0;

  const createEvent = {
    type: "conversation.item.create",
    item: {
      type: "message",
      role: "user",
      content: [
        {
          type: "input_text",
          text: trimmed,
        },
      ],
    },
  };
  ws.send(JSON.stringify(createEvent));
  log("SEND: conversation.item.create (" + trimmed + ")");

  const respEvent = {
    type: "response.create",
    response: {
      instructions: buildResponseInstructions(),
      modalities: ["text", "audio"],
    },
  };
  ws.send(JSON.stringify(respEvent));
  log("SEND: response.create");
}

document.getElementById("sendText").onclick = () => {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  sendTextAsUser(textInput.value);
};
