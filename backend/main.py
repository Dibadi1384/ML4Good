import os
import json
import base64
from typing import List, Literal

import httpx
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv


load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_AUDIO_TIMEOUT = float(os.getenv("GEMINI_AUDIO_TIMEOUT", "25"))  # seconds for audio transcription
MAX_AUDIO_BYTES = int(os.getenv("MAX_AUDIO_BYTES", "3_000_000"))  # 3MB max upload
GEMINI_BASE_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
)

if not GEMINI_API_KEY:
    # We still start the app, but will return 500 on requests until key is configured.
    print("[WARN] GEMINI_API_KEY is not set. Set it in your environment or .env file.")


class TranslateRequest(BaseModel):
    text: str
    mode: Literal["english", "rohingya"]


class TranslateResponse(BaseModel):
    translation: str


class ConversationTurn(BaseModel):
    mode: Literal["english", "rohingya"]
    source: str
    target: str


class SuggestRequest(BaseModel):
    history: List[ConversationTurn]
    mode: Literal["english", "rohingya"]


class SuggestionItem(BaseModel):
    english: str
    rohingya: str


class SuggestResponse(BaseModel):
    suggestions: List[SuggestionItem]


class TranscribeResponse(BaseModel):
    transcript: str
    translation: str


app = FastAPI(title="English ↔ Rohingya Demo Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def call_gemini(prompt: str) -> str:
    if not GEMINI_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY is not configured on the server.",
        )

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}],
            }
        ]
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            GEMINI_BASE_URL,
            params={"key": GEMINI_API_KEY},
            json=payload,
        )

    if resp.status_code != 200:
        try:
            msg = resp.text
        except Exception:
            msg = "<no body>"
        raise HTTPException(
            status_code=502,
            detail=f"Gemini API error {resp.status_code}: {msg}",
        )

    data = resp.json()
    candidates = data.get("candidates") or []
    if not candidates:
        return ""
    parts = candidates[0].get("content", {}).get("parts") or []
    text = " ".join(part.get("text", "") for part in parts)
    return text.strip()


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "model": GEMINI_MODEL, "has_api_key": bool(GEMINI_API_KEY)}


@app.post("/api/translate", response_model=TranslateResponse)
async def translate(req: TranslateRequest) -> TranslateResponse:
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text must not be empty.")

    if req.mode == "english":
        prompt = (
            "You are assisting with live conversation between an English speaker and a Rohingya speaker.\n\n"
            "Task: Translate the following English sentence into Rohingya, but write it ONLY in Latin script "
            "(Roman letters) as Rohingya people commonly type it on phones.\n\n"
            "Rules:\n"
            "- Output ONLY the Rohingya sentence in Latin script.\n"
            "- Do NOT include any explanation.\n"
            "- Keep punctuation simple.\n\n"
            f"English:\n\"{text}\""
        )
    else:
        prompt = (
            "You are assisting with live conversation between an English speaker and a Rohingya speaker.\n\n"
            "The user will speak Rohingya using phonetic Latin script (Roman letters), as is common on phones.\n\n"
            "Task: Interpret the following Rohingya Latin-script text and provide the closest natural English meaning.\n\n"
            "Rules:\n"
            "- Output ONLY the English sentence.\n"
            "- Do NOT include any explanation.\n\n"
            f"Rohingya (Latin script):\n\"{text}\""
        )

    raw = await call_gemini(prompt)
    cleaned = raw.replace("```", "").strip().strip("'\"`")
    return TranslateResponse(translation=cleaned)


@app.post("/api/suggest", response_model=SuggestResponse)
async def suggest(req: SuggestRequest) -> SuggestResponse:
    if not req.history:
        return SuggestResponse(suggestions=[])

    # Limit context size
    last_turns = req.history[-12:]
    formatted_turns = []
    for turn in last_turns:
        direction = "English→Rohingya" if turn.mode == "english" else "Rohingya→English"
        formatted_turns.append(
            f"Mode: {direction}\nSource: {turn.source}\nTarget: {turn.target}"
        )
    history_block = "\n\n".join(formatted_turns)

    mode_desc = (
        "The English speaker is speaking and we respond in Rohingya (Latin script)."
        if req.mode == "english"
        else "The Rohingya speaker is speaking (Latin script) and we respond primarily in English."
    )

    prompt = (
        "You are helping facilitate a live conversation between an English speaker and a Rohingya speaker.\n\n"
        f"Current interaction mode:\n{mode_desc}\n\n"
        "Recent conversation turns (chronological):\n"
        f"{history_block}\n\n"
        "Based on this context, suggest 3 short and useful phrases that the person might want to say NEXT.\n\n"
        "For each suggestion, return:\n"
        '- A concise, natural English sentence.\n'
        "- The Rohingya version written ONLY in Latin script.\n\n"
        "Output format MUST be strict JSON, with no extra text or commentary:\n"
        "{\n"
        '  "suggestions": [\n'
        '    { "english": "...", "rohingya": "..." },\n'
        '    { "english": "...", "rohingya": "..." },\n'
        '    { "english": "...", "rohingya": "..." }\n'
        "  ]\n"
        "}\n\n"
        "Keep each phrase short (max 15 words)."
    )

    raw = await call_gemini(prompt)
    text = raw.strip()

    if text.startswith("```"):
        first = text.find("```")
        last = text.rfind("```")
        if last > first:
            text = text[first + 3 : last].strip()
            # Remove optional `json` after fence
            if text.lower().startswith("json"):
                text = text[4:].strip()

    suggestions: List[SuggestionItem] = []
    try:
        parsed = json.loads(text)
        items = parsed.get("suggestions") or []
        for item in items:
            eng = str(item.get("english", "")).strip()
            roh = str(item.get("rohingya", "")).strip()
            if eng or roh:
                suggestions.append(SuggestionItem(english=eng, rohingya=roh))
            if len(suggestions) >= 3:
                break
    except Exception:
        # Fallback: naive line-based suggestions
        lines = [ln.strip("-• \t") for ln in text.splitlines() if ln.strip()]
        for ln in lines[:3]:
            suggestions.append(SuggestionItem(english=ln, rohingya=""))

    return SuggestResponse(suggestions=suggestions)


async def call_gemini_audio(audio_bytes: bytes, mime_type: str, prompt: str) -> str:
    """Call Gemini with an inline audio blob + text prompt. Fails fast on timeout."""
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY is not configured.")

    audio_b64 = base64.b64encode(audio_bytes).decode()
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"inline_data": {"mime_type": mime_type, "data": audio_b64}},
                    {"text": prompt},
                ],
            }
        ]
    }

    timeout = httpx.Timeout(10.0, read=GEMINI_AUDIO_TIMEOUT)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                GEMINI_BASE_URL,
                params={"key": GEMINI_API_KEY},
                json=payload,
            )
    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail=f"Gemini did not respond within {GEMINI_AUDIO_TIMEOUT}s. Check your connection and try again.",
        )

    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Gemini error {resp.status_code}: {resp.text}")

    data = resp.json()
    candidates = data.get("candidates") or []
    if not candidates:
        return ""
    parts = candidates[0].get("content", {}).get("parts") or []
    return " ".join(p.get("text", "") for p in parts).strip()


def _extract_json(raw: str) -> dict:
    """Strip markdown fences and parse JSON from a Gemini response."""
    text = raw.strip()
    if text.startswith("```"):
        first = text.find("```")
        last  = text.rfind("```")
        if last > first:
            text = text[first + 3 : last].strip()
            if text.lower().startswith("json"):
                text = text[4:].strip()
    return json.loads(text)


@app.post("/api/transcribe", response_model=TranscribeResponse)
async def transcribe(
    file: UploadFile = File(...),
    mode: str = Form("english"),
) -> TranscribeResponse:
    try:
        audio_bytes = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read upload: {e!s}")

    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file.")
    if len(audio_bytes) > MAX_AUDIO_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Audio too large (max {MAX_AUDIO_BYTES // 1_000_000}MB). Record a shorter clip.",
        )

    mime_type = (file.content_type or "audio/webm").split(";")[0].strip()
    if mime_type not in ("audio/webm", "audio/ogg", "audio/mpeg", "audio/mp4", "audio/wav"):
        mime_type = "audio/webm"

    if mode == "english":
        prompt = (
            "You are a transcription and translation assistant.\n\n"
            "The attached audio contains someone speaking in ENGLISH.\n\n"
            "Tasks:\n"
            "1. Transcribe exactly what was said in English.\n"
            "2. Translate the English text into Rohingya, written ONLY in Latin script "
            "(Roman letters), as Rohingya people commonly type on phones.\n\n"
            "Return ONLY valid JSON — no explanation, no markdown:\n"
            '{"transcript": "...", "translation": "..."}'
        )
    else:
        prompt = (
            "You are a transcription and translation assistant.\n\n"
            "The attached audio contains someone speaking ROHINGYA using Latin-script phonetics.\n\n"
            "Tasks:\n"
            "1. Write out the Rohingya speech in Latin script as best you can.\n"
            "2. Translate the meaning into clear, natural English.\n\n"
            "Return ONLY valid JSON — no explanation, no markdown:\n"
            '{"transcript": "...", "translation": "..."}'
        )

    try:
        raw = await call_gemini_audio(audio_bytes, mime_type, prompt)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Gemini/network error: {e!s}")

    try:
        parsed = _extract_json(raw)
        return TranscribeResponse(
            transcript=str(parsed.get("transcript", "")).strip(),
            translation=str(parsed.get("translation", "")).strip(),
        )
    except Exception:
        return TranscribeResponse(transcript=raw.strip(), translation="")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

