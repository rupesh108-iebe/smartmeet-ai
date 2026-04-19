import os
import tempfile
from typing import List, Dict, Any

import numpy as np
import requests
import soundfile as sf
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
from textblob import TextBlob

# If NUMBA gives trouble with some environments, uncomment this:
# os.environ["NUMBA_DISABLE_JIT"] = "1"

app = FastAPI()

# Allow requests from the local React dev server.
# If you later deploy frontend and backend separately, update these origins.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the Whisper ASR model once at startup.
# For now we fix it to "base" on CPU with int8 for a good speed/quality trade‑off.
# If you have a stronger GPU, you can experiment with "small" or higher.
whisper_model = WhisperModel("base", device="cpu", compute_type="int8")


def save_upload_to_wav(upload_file: UploadFile) -> str:
    """
    Store the uploaded audio file on disk and return the temporary path.

    For this version we just write the raw upload to a temp file and let
    faster-whisper handle supported formats (e.g. WAV, MP3).
    """
    suffix = os.path.splitext(upload_file.filename)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(upload_file.file.read())
        tmp_path = tmp.name
    return tmp_path


def transcribe_with_whisper(
    wav_path: str,
    language: str = "auto",
    max_audio_seconds: int = 0,
) -> List[Dict[str, Any]]:
    """
    Run faster-whisper and return a list of segments.

    Each segment looks like:
    {
      "start": float,
      "end": float,
      "text": str,
    }

    language:
        "auto" or a language code such as "en", "hi", "mr".
    max_audio_seconds:
        If > 0, only the first N seconds are transcribed.
        This is useful for quick demos on long recordings.
    """
    audio_path = wav_path

    # Optional fast‑demo path: trim audio to the first N seconds.
    if max_audio_seconds > 0:
        y, sr = sf.read(wav_path)
        if y.ndim > 1:
            # Mix stereo down to mono by averaging channels.
            y = y.mean(axis=1)
        max_samples = int(sr * max_audio_seconds)
        y_trim = y[:max_samples]
        tmp_trim = wav_path.replace(".wav", "_trim.wav")
        sf.write(tmp_trim, y_trim, sr)
        audio_path = tmp_trim

    options: Dict[str, Any] = {"beam_size": 5, "best_of": 5}
    if language != "auto":
        options["language"] = language

    segments_iter, info = whisper_model.transcribe(audio_path, **options)

    result: List[Dict[str, Any]] = []
    for seg in segments_iter:
        result.append(
            {
                "start": float(seg.start),
                "end": float(seg.end),
                "text": seg.text.strip(),
            }
        )

    # Clean up the trimmed file if we created one.
    if audio_path != wav_path and os.path.exists(audio_path):
        os.remove(audio_path)

    return result


def assign_speakers_heuristic(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Assign rough speaker labels by grouping segments into "turns".

    Idea:
    - Sort segments by start time.
    - If the silence gap between two segments is small, treat them as the same turn.
    - Alternate speakers per turn: Speaker 1, Speaker 2, Speaker 1, ...

    This is a lightweight substitute for real diarization.
    It works surprisingly well for small meetings but will fail on very
    overlapping or noisy conversations.
    """
    if not segments:
        return []

    # Make sure segments are in time order.
    segments_sorted = sorted(segments, key=lambda s: s["start"])

    # If the gap between segments is larger than this, we treat it as a new turn.
    GAP_THRESHOLD = 1.5  # seconds

    turns: List[List[Dict[str, Any]]] = []
    current_turn: List[Dict[str, Any]] = [segments_sorted[0]]

    for prev, curr in zip(segments_sorted, segments_sorted[1:]):
        gap = curr["start"] - prev["end"]

        if gap <= GAP_THRESHOLD:
            current_turn.append(curr)
        else:
            turns.append(current_turn)
            current_turn = [curr]

    if current_turn:
        turns.append(current_turn)

    # Alternate speaker labels between turns.
    speaker_segments: List[Dict[str, Any]] = []
    speaker_id = 1

    for turn in turns:
        speaker_label = f"Speaker {speaker_id}"
        for seg in turn:
            speaker_segments.append(
                {
                    "start": seg["start"],
                    "end": seg["end"],
                    "speaker": speaker_label,
                    "text": seg["text"],
                }
            )
        speaker_id = 2 if speaker_id == 1 else 1

    return speaker_segments


def build_meeting_text_for_llm(speaker_segments: List[Dict[str, Any]]) -> str:
    """
    Flatten speaker segments into a single transcript string for the LLM.

    Example line:
        "Speaker 1: Hello everyone, thanks for joining..."
    """
    lines = []
    for seg in speaker_segments:
        line = f"{seg['speaker']}: {seg['text']}"
        lines.append(line)
    return "\n".join(lines)


def compute_sentiment(meeting_text: str) -> Dict[str, Any]:
    """
    Estimate overall sentiment of the meeting using TextBlob.

    We take the polarity in [-1, 1] and map it to a coarse label:
    "Negative", "Neutral", or "Positive".
    """
    blob = TextBlob(meeting_text)
    polarity = float(blob.sentiment.polarity)  # -1.0 (neg) to 1.0 (pos)

    # Simple thresholds: keep a small neutral band around zero.
    if polarity > 0.1:
        label = "Positive"
    elif polarity < -0.1:
        label = "Negative"
    else:
        label = "Neutral"

    return {
        "score": round(polarity, 3),
        "label": label,
    }


def compute_engagement_scores(
    speaker_segments: List[Dict[str, Any]],
    audio_path: str,
) -> List[Dict[str, Any]]:
    """
    Attach a simple engagement score to each segment based on audio + text.

    Rough scoring recipe:
    - Longer speaking segments get higher scores.
    - Louder segments (higher RMS energy) get higher scores.
    - Faster speech (more words per second) gets higher scores.

    The final score is scaled to [0, 100] for easy display in the UI.
    This is heuristic by design and not a trained model (yet).
    """
    if not speaker_segments:
        return speaker_segments

    # Load full audio once in mono so we can slice it per segment.
    y, sr = sf.read(audio_path)
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = np.asarray(y, dtype=np.float32)

    durations = []
    energies = []
    speech_rates = []

    for seg in speaker_segments:
        start_sample = int(seg["start"] * sr)
        end_sample = int(seg["end"] * sr)
        seg_audio = y[start_sample:end_sample]

        duration = seg["end"] - seg["start"]
        durations.append(max(duration, 1e-3))

        if seg_audio.size > 0:
            rms = float(np.sqrt(np.mean(seg_audio**2)))
        else:
            rms = 0.0
        energies.append(rms)

        words = len(seg["text"].split())
        speech_rate = words / max(duration, 1e-3)
        speech_rates.append(speech_rate)

    durations = np.array(durations)
    energies = np.array(energies)
    speech_rates = np.array(speech_rates)

    def norm(x: np.ndarray) -> np.ndarray:
        """Normalize an array to [0, 1] with safe handling for constant values."""
        xmin, xmax = x.min(), x.max()
        if xmax - xmin < 1e-9:
            return np.zeros_like(x)
        return (x - xmin) / (xmax - xmin)

    dur_norm = norm(durations)
    eng_norm = norm(energies)
    rate_norm = norm(speech_rates)

    # Weighted combination of the three normalized features.
    scores = 0.4 * dur_norm + 0.3 * eng_norm + 0.3 * rate_norm
    scores = (scores * 100).tolist()

    for seg, score in zip(speaker_segments, scores):
        seg["engagement_score"] = round(float(score), 1)

    return speaker_segments


def aggregate_engagement_per_speaker(
    speaker_segments: List[Dict[str, Any]],
) -> Dict[str, float]:
    """
    Compute the average engagement score for each speaker.

    Returns a dictionary like:
        {
          "Speaker 1": 72.3,
          "Speaker 2": 55.1,
        }
    """
    scores_by_speaker: Dict[str, List[float]] = {}

    for seg in speaker_segments:
        speaker = seg.get("speaker", "Unknown")
        score = seg.get("engagement_score")
        if score is None:
            continue
        scores_by_speaker.setdefault(speaker, []).append(float(score))

    avg_by_speaker: Dict[str, float] = {}
    for speaker, scores in scores_by_speaker.items():
        if scores:
            avg_by_speaker[speaker] = round(sum(scores) / len(scores), 1)

    return avg_by_speaker


def call_ollama(
    meeting_text: str,
    mode: str = "both",
    temperature: float = 0.2,
    max_tokens: int = 512,
) -> Dict[str, Any]:
    """
    Ask the local Llama model (via Ollama) for a JSON summary and action items.

    The model is instructed to always return JSON of the form:
    {
      "summary": [...],
      "action_items": [
        {"owner": "...", "description": "...", "due_date": "... or null"}
      ]
    }

    We still defensively parse the response in case it wraps the JSON in extra text.
    """
    url = "http://localhost:11434/api/chat"

    system_prompt = """
You are an AI meeting assistant.
You MUST respond with JSON only. No explanations, no markdown.

JSON schema:
{
  "summary": [
    "bullet point 1",
    "bullet point 2"
  ],
  "action_items": [
    {
      "owner": "Speaker 1 or a name if clearly mentioned",
      "description": "what needs to be done",
      "due_date": "YYYY-MM-DD if explicitly mentioned, otherwise null"
    }
  ]
}
"""

    if mode == "summary":
        task_instructions = """
Generate ONLY the summary array (5-7 bullet points).
Set "action_items" to an empty array [].
"""
    elif mode == "actions":
        task_instructions = """
Generate ONLY the action_items array.
Set "summary" to an empty array [].
"""
    else:
        task_instructions = """
Generate BOTH a concise summary (5-7 bullet points) and a list of action items.
"""

    user_prompt = f"""
Here is the meeting transcript with speaker labels:

{meeting_text}

{task_instructions}
Follow the JSON schema exactly.
"""

    payload = {
        "model": "llama3.1:8b",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }

    resp = requests.post(url, json=payload, timeout=300)
    resp.raise_for_status()
    data = resp.json()
    content = data["message"]["content"].strip()

    import json

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        # If the model wrapped the JSON in text, try to slice out the JSON block.
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            json_str = content[start : end + 1]
            parsed = json.loads(json_str)
        else:
            # If we really can't parse it, return the raw text for debugging.
            return {"raw_response": content}

    summary = parsed.get("summary", [])
    action_items = parsed.get("action_items", [])

    return {
        "summary": summary,
        "action_items": action_items,
        "raw_response": content,
    }


@app.post("/process-meeting")
async def process_meeting(
    file: UploadFile = File(...),
    llm_mode: str = Query("both", regex="^(summary|actions|both)$"),
    temperature: float = Query(0.2, ge=0.0, le=1.0),
    max_tokens: int = Query(512, ge=64, le=2048),
    asr_model: str = Query("base", regex="^(tiny|base|small)$"),
    asr_language: str = Query("auto"),
    max_audio_seconds: int = Query(0, ge=0, le=3600),
):
    """
    Main endpoint: take an uploaded meeting recording and return:

    - Transcribed segments with heuristic speaker labels.
    - Engagement scores per segment and per speaker.
    - Overall sentiment of the meeting.
    - LLM-generated summary and action items.
    """
    # 1) Save upload to a local temp file so Whisper can read it.
    wav_path = save_upload_to_wav(file)

    # 2) Transcribe speech to text (with optional language and time limit).
    segments = transcribe_with_whisper(
        wav_path,
        language=asr_language,
        max_audio_seconds=max_audio_seconds,
    )

    # 3) Assign rough speaker labels based on timing gaps.
    speaker_segments = assign_speakers_heuristic(segments)

    # 4) Compute acoustic/text engagement features per segment.
    speaker_segments = compute_engagement_scores(speaker_segments, wav_path)

    # 5) Prepare a single transcript string for the LLM.
    meeting_text = build_meeting_text_for_llm(speaker_segments)

    # 6) Estimate overall sentiment from the transcript.
    sentiment = compute_sentiment(meeting_text)

    # 7) Ask the local LLM for summary and action items.
    ollama_result = call_ollama(
        meeting_text,
        mode=llm_mode,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    # 8) Aggregate engagement at the speaker level for the dashboard.
    engagement_by_speaker = aggregate_engagement_per_speaker(speaker_segments)

    # 9) Clean up the temporary audio file.
    if os.path.exists(wav_path):
        os.remove(wav_path)

    return {
        "segments": speaker_segments,
        "engagement_by_speaker": engagement_by_speaker,
        "sentiment": sentiment,
        "llm_output": ollama_result,
    }