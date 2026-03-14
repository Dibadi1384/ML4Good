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
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
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
    """Call Gemini with an inline audio blob + text prompt."""
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

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            GEMINI_BASE_URL,
            params={"key": GEMINI_API_KEY},
            json=payload,
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
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file.")

    mime_type = file.content_type or "audio/webm"

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

    raw = await call_gemini_audio(audio_bytes, mime_type, prompt)

    try:
        parsed = _extract_json(raw)
        return TranscribeResponse(
            transcript=str(parsed.get("transcript", "")).strip(),
            translation=str(parsed.get("translation", "")).strip(),
        )
    except Exception:
        # Fallback: treat the entire response as the transcript with no translation
        return TranscribeResponse(transcript=raw.strip(), translation="")


PRACTICE_SCENARIOS = {
    "doctor":        "a doctor at a hospital",
    "grocery":       "a grocery store cashier",
    "job_interview": "a hiring manager conducting a job interview",
    "restaurant":    "a waiter at a restaurant",
    "pharmacy":      "a pharmacist at a pharmacy",
}


class PracticeStartRequest(BaseModel):
    scenario: str


class PracticeStartResponse(BaseModel):
    message: str


class PracticeTurn(BaseModel):
    role: Literal["ai", "user"]
    message: str


class PracticeChatRequest(BaseModel):
    scenario: str
    history: List[PracticeTurn]
    user_message: str


class PracticeChatResponse(BaseModel):
    message: str


class PracticeCheckRequest(BaseModel):
    target: str
    attempt: str


class PracticeCheckResponse(BaseModel):
    understood: bool
    feedback: str


@app.post("/api/practice/start", response_model=PracticeStartResponse)
async def practice_start(req: PracticeStartRequest) -> PracticeStartResponse:
    role = PRACTICE_SCENARIOS.get(req.scenario, f"someone in a {req.scenario} situation")
    prompt = (
        f"You are {role}. A Rohingya person who is learning English has just approached you.\n\n"
        "Start the conversation by:\n"
        "1. Briefly stating your role (e.g. 'I am a doctor at this hospital.')\n"
        "2. Asking a simple, natural opening question.\n\n"
        "Rules:\n"
        "- Keep it to 1-2 sentences total.\n"
        "- Use simple, clear English.\n"
        "- Output ONLY your opening line, no extra text."
    )
    message = await call_gemini(prompt)
    return PracticeStartResponse(message=message.strip())


@app.post("/api/practice/chat", response_model=PracticeChatResponse)
async def practice_chat(req: PracticeChatRequest) -> PracticeChatResponse:
    role = PRACTICE_SCENARIOS.get(req.scenario, f"someone in a {req.scenario} situation")
    history_text = "\n".join(
        f"{'You (AI)' if t.role == 'ai' else 'Learner'}: {t.message}"
        for t in req.history[-10:]
    )
    prompt = (
        f"You are {role} having a conversation with a Rohingya person learning English.\n\n"
        f"Conversation so far:\n{history_text}\n"
        f"Learner: {req.user_message}\n\n"
        "Respond naturally as your character. Keep it simple (1-2 sentences). "
        "Output ONLY your response."
    )
    message = await call_gemini(prompt)
    return PracticeChatResponse(message=message.strip())


@app.post("/api/practice/check", response_model=PracticeCheckResponse)
async def practice_check(req: PracticeCheckRequest) -> PracticeCheckResponse:
    prompt = (
        "A language learner is trying to express a specific meaning in English.\n\n"
        f"Intended meaning: \"{req.target}\"\n"
        f"What the learner said: \"{req.attempt}\"\n\n"
        "Did they communicate the same general meaning? Be lenient — small grammar mistakes are fine, "
        "focus on whether the core meaning is conveyed.\n\n"
        "Return ONLY valid JSON:\n"
        '{"understood": true/false, "feedback": "one short encouraging sentence"}'
    )
    raw = await call_gemini(prompt)
    try:
        parsed = _extract_json(raw)
        return PracticeCheckResponse(
            understood=bool(parsed.get("understood", False)),
            feedback=str(parsed.get("feedback", "Keep trying!")).strip(),
        )
    except Exception:
        return PracticeCheckResponse(understood=False, feedback="Keep trying!")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

