# server.py — XTTS FastAPI server
# Features: in-memory speaker cache, auto-split (≤240 chars), CPU thread tuning, per-run /metrics
import io, os, re, json, hashlib, tempfile, textwrap, time
import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, UploadFile, Form, Response
from fastapi.responses import JSONResponse
from TTS.api import TTS
import uvicorn

# ---------- CPU thread tuning ----------
def apply_thread_tuning() -> int:
    cpu = os.cpu_count() or 4
    threads = int(os.getenv("TTS_THREADS", max(1, cpu - 2)))
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[k] = str(threads)
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    return threads

ACTIVE_THREADS = apply_thread_tuning()

# ---------- model config ----------
MODEL = os.getenv("MODEL", "tts_models/multilingual/multi-dataset/xtts_v2")
LANG_DEFAULT = os.getenv("LANG", "en")
MAX_CHARS = 240  # safe cap to avoid XTTS truncation warnings

app = FastAPI()
tts = TTS(model_name=MODEL, gpu=False)

# ---------- speaker cache (in-memory) ----------
VOICE_CACHE: dict[str, str] = {}  # sha1(bytes) -> speaker_name

def get_or_register_speaker(voice_bytes: bytes) -> str:
    h = hashlib.sha1(voice_bytes).hexdigest()
    if h in VOICE_CACHE:
        return VOICE_CACHE[h]
    try:
        data = torch.load(io.BytesIO(voice_bytes), map_location="cpu")
        gpt, spk = data["gpt"], data["spk"]
    except Exception as e:
        raise ValueError("invalid voice .pt (missing gpt/spk)") from e

    sm = tts.synthesizer.tts_model.speaker_manager
    name = f"spk_{h[:8]}"
    sm.speakers[name] = {"gpt_cond_latent": gpt, "speaker_embedding": spk}
    VOICE_CACHE[h] = name
    return name

# ---------- text splitting (≤ MAX_CHARS) ----------
def split_for_xtts(s: str, limit: int = MAX_CHARS) -> list[str]:
    s = " ".join((s or "").split())
    if not s:
        return []
    if len(s) <= limit:
        return [s]
    out = []
    for seg in re.split(r'(?<=[.!?])\s+', s):
        seg = seg.strip()
        if not seg:
            continue
        if len(seg) <= limit:
            out.append(seg)
        else:
            out.extend(textwrap.wrap(seg, limit))
    return out

# ---------- metrics storage ----------
LAST_METRICS = None  # filled after each /speak-batch

# ---------- routes ----------
@app.get("/health")
def health():
    return {
        "ok": True,
        "model": MODEL,
        "cache_entries": len(VOICE_CACHE),
        "threads": ACTIVE_THREADS,
    }

@app.post("/speak")
async def speak(
    voice: UploadFile,
    text: str = Form(...),
    lang: str = Form(LANG_DEFAULT),
    temp: float = Form(0.1),
    top_p: float = Form(0.0),
):
    try:
        speaker_name = get_or_register_speaker(await voice.read())
        tts.tts_to_file(
            text=text,
            speaker=speaker_name,
            language=lang,
            file_path="out.wav",
            temperature=temp,
            top_p=top_p,
        )
        with open("out.wav", "rb") as f:
            wav = f.read()
        return Response(
            content=wav,
            media_type="audio/wav",
            headers={"Content-Disposition": 'inline; filename="speech.wav"'},
        )
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/speak-batch")
async def speak_batch(
    voice: UploadFile,
    chunks: str = Form(...),  # JSON array of strings
    lang: str = Form(LANG_DEFAULT),
    temp: float = Form(0.1),
    top_p: float = Form(0.0),
):
    try:
        parts = json.loads(chunks)
        if not isinstance(parts, list):
            return JSONResponse(status_code=400, content={"error": "chunks must be a JSON array"})
        expanded: list[str] = []
        for s in parts:
            if isinstance(s, str) and s.strip():
                expanded.extend(split_for_xtts(s))
        if not expanded:
            return JSONResponse(status_code=400, content={"error": "no non-empty text"})

        speaker_name = get_or_register_speaker(await voice.read())

        sr = 24000
        all_samples = []
        timings = []  # per-subchunk metrics

        for idx, txt in enumerate(expanded, 1):
            start = time.time()
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
                tts.tts_to_file(
                    text=txt,
                    speaker=speaker_name,
                    language=lang,
                    file_path=tmp.name,
                    temperature=temp,
                    top_p=top_p,
                )
                wall = time.time() - start

                audio, got_sr = sf.read(tmp.name, dtype="int16")
                if got_sr != sr:
                    return JSONResponse(status_code=500, content={"error": f"unexpected sample rate {got_sr}"})
                if audio.ndim > 1:
                    audio = audio[:, 0]
                all_samples.append(audio)

                audio_sec = float(len(audio)) / sr
                rtf = wall / audio_sec if audio_sec > 0 else None
                timings.append({
                    "i": idx,
                    "chars": len(txt),
                    "wall_s": round(wall, 3),
                    "audio_s": round(audio_sec, 3),
                    "rtf": round(rtf, 3) if rtf is not None else None,
                })

        # concat & write response
        cat = np.concatenate(all_samples, axis=0)
        buf = io.BytesIO()
        sf.write(buf, cat, sr, format="WAV", subtype="PCM_16")
        buf.seek(0)

        total_wall = round(sum(t["wall_s"] for t in timings), 3)
        total_audio = round(sum(t["audio_s"] for t in timings), 3)
        overall_rtf = round(total_wall / total_audio, 3) if total_audio > 0 else None

        global LAST_METRICS
        LAST_METRICS = {
            "subchunks": timings,
            "total_wall_s": total_wall,
            "total_audio_s": total_audio,
            "overall_rtf": overall_rtf,
            "n_subchunks": len(timings),
            "threads": ACTIVE_THREADS,
        }

        headers = {
            "Content-Disposition": 'inline; filename="batch.wav"',
            "X-Subchunks": str(len(expanded)),
            "X-Total-Wall": str(total_wall),
            "X-Total-Audio": str(total_audio),
            "X-RTF": str(overall_rtf),
        }

        return Response(content=buf.getvalue(), media_type="audio/wav", headers=headers)

    except json.JSONDecodeError:
        return JSONResponse(status_code=400, content={"error": "chunks must be valid JSON"})
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/metrics")
def metrics():
    if LAST_METRICS is None:
        return JSONResponse(status_code=404, content={"error": "no run yet"})
    return LAST_METRICS

if __name__ == "__main__":
    # dev: uvicorn server:app --host 127.0.0.1 --port 5055 --reload
    uvicorn.run(app, host="127.0.0.1", port=5055)