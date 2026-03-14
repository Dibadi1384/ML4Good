# English ↔ Rohingya Live Companion (Demo)

Simple end-to-end demo with:

- FastAPI backend that calls **Gemini** (via REST)
- Single-page HTML/JS frontend with:
  - Camera preview
  - Live speech recognition (Chrome/Edge)
  - English ↔ Rohingya (Latin script) translation
  - Suggested next phrases, clickable and spoken aloud

## 1. Prerequisites

- Python 3.9+ installed
- A Gemini API key from [Google AI Studio](https://aistudio.google.com)
- Modern Chromium browser (Chrome / Edge) for `SpeechRecognition`

## 2. Setup

From the project root (`ML4Good`):

```bash
python -m venv .venv
.\.venv\Scripts\activate   # On PowerShell (Windows)

pip install -r requirements.txt
```

The backend needs `python-multipart` for the audio upload endpoint. If you see *"Form data requires python-multipart"*, run:

```bash
pip install python-multipart
```

Create a `.env` file (you can copy `.env.example`):

```bash
copy .env.example .env
```

Edit `.env` and set:

```bash
GEMINI_API_KEY=your_real_key_here
GEMINI_MODEL=gemini-2.5-flash
```

## 3. Run the backend

From the project root:

```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

You should see FastAPI starting on `http://localhost:8000`.

Check health:

```bash
curl http://localhost:8000/health
```

You should get JSON like:

```json
{ "status": "ok", "model": "gemini-2.5-flash", "has_api_key": true }
```

## 4. Run the frontend

The frontend is a static `index.html` file in `frontend/`.

On Windows with Python:

```bash
cd ..\frontend
python -m http.server 5500
```

When you see **"Serving HTTP on :: port 5500"**, the server is running. Open this in your browser:

**→ [http://localhost:5500](http://localhost:5500)** or **[http://localhost:5500/index.html](http://localhost:5500/index.html)**

> Note: `[::]` is IPv6 notation; use `localhost` in the browser. A simple HTTP server avoids mic/camera issues that can occur with `file://` URLs.

## 5. Using the demo

- Ensure the backend is running on `http://localhost:8000`.
- Open the frontend in Chrome/Edge and allow camera + mic.
- At the top-right, choose a mode:
  - **English → Rohingya**: Speak English; you get Rohingya in Latin script.
  - **Rohingya → English**: Speak Rohingya (typed / pronounced via Latin script); you get English meaning.
- Click the **mic button** to start/stop listening.
- After each segment, the backend:
  - Calls Gemini to translate.
  - Calls Gemini again to generate 3 **suggested replies** (English + Rohingya Latin).
- Click a suggestion card to hear it spoken via `speechSynthesis`.

## 6. Files overview

- `requirements.txt` – Python dependencies (FastAPI, uvicorn, httpx, python-dotenv).
- `backend/main.py` – FastAPI app with:
  - `GET /health` health check
  - `POST /api/translate` for segment translation
  - `POST /api/suggest` for suggested dialogue phrases
- `frontend/index.html` – Single-page UI with camera, captions, and suggestions.
- `.env.example` – Example environment file for Gemini configuration.

