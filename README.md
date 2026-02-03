## Bot Backend – Quick Setup

This project is a FastAPI backend with MySQL and Microsoft Voice Live (Azure Speech + Realtime) support.

---

### 1. Requirements

- Python 3.12 (recommended)
- MySQL running locally (default: `mysql://root@localhost:3306/botdb`)
- Azure Speech / Voice Live resource (key + region)

---

### 2. Install Dependencies

Using pip (simple):

```bash
pip install -r requirements.txt
```

Using Poetry (if you prefer):

```bash
poetry install
```

Optional virtualenv (pip):

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

---

### 3. Configure Environment (.env)

Create a `.env` file in the project root (same folder as `app/`, `requirements.txt`) with at least:

```env
APP_NAME=Bot Backend
ENV=dev

# Database
DATABASE_URL=mysql+pymysql://root:@localhost:3306/botdb

# Microsoft Voice Live / Azure Speech
MICROSOFT_VOICE_LIVE_API_KEY="<your-azure-speech-key>"
MICROSOFT_VOICE_LIVE_REGION="eastus"
MICROSOFT_VOICE_LIVE_BASE_URL="https://<your-resource-name>.services.ai.azure.com"
USE_MICROSOFT_VOICE_LIVE="true"
MICROSOFT_VOICE_LIVE_VOICE="en-US-JennyNeural"
MICROSOFT_VOICE_LIVE_LANGUAGE="en-US"
MICROSOFT_VOICE_LIVE_TTS_URL="https://eastus.tts.speech.microsoft.com"
MICROSOFT_VOICE_LIVE_STT_URL="https://eastus.stt.speech.microsoft.com"
```

Make sure the key/region and base URL match your Azure Speech resource.

---

### 4. Run the Server

Dev (auto-reload):

```bash
uvicorn app.main:app --reload
```

Prod-style:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Health check:

- `GET http://127.0.0.1:8000/health`
- `GET http://127.0.0.1:8000/voice/health`

---

### 5. Voice Test Page

There is a simple HTML client for testing text and voice:

- File: `voice-test.html` (in the project root)

Usage:

1. Start the FastAPI server.
2. Open `voice-test.html` in a browser (double-click or serve it statically).
3. Click **Connect** to open the WebSocket.
4. Click **Send session.update** once to configure the Realtime session.
5. Use:
	- Text input → **Send text** for text-to-voice.
	- **Start recording / Stop / Replay / Send recording** for voice-to-voice.

The page talks to:

- `ws://127.0.0.1:8000/voice/stream` for Realtime text+audio
- `POST http://127.0.0.1:8000/voice/transcribe` for STT

---

### 6. (Optional) Migrations with Alembic

In dev, tables are auto-created when `ENV=dev`. For production, use Alembic:

1. Install Alembic
	```bash
	pip install alembic
	```
2. Init Alembic (once)
	```bash
	alembic init alembic
	```
3. Point Alembic to your models
	- In `alembic/env.py`:
	  ```python
	  from app.core.database import Base
	  target_metadata = Base.metadata
	  ```
4. Generate a migration
	```bash
	alembic revision --autogenerate -m "create tables"
	```
5. Apply migrations
	```bash
	alembic upgrade head
	```

Run this before starting the app in production to ensure the schema is up to date.

