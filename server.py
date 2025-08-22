# server.py — XTTS FastAPI server (clean in-memory synth)
import io, os, re, json, hashlib, textwrap, time
import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, UploadFile, Form, Response
from fastapi.responses import JSONResponse, FileResponse
from TTS.api import TTS
import uvicorn

# Accept Coqui TOS if not set (you should still review CPML/commercial terms)
os.environ.setdefault("COQUI_TOS_ACCEPT", "1")
os.environ.setdefault("COQUI_TOS_AGREED", "1")

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
SR = 24000

app = FastAPI()
tts = TTS(model_name=MODEL, gpu=False)  # set gpu=True if you later move to a CUDA box

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
@app.get("/")
def root():
    return {"ok": True, "endpoints": ["/health", "/speak", "/speak-batch", "/metrics", "/download/last"]}

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

        start = time.time()
        wav = tts.tts(
            text=text,
            speaker=speaker_name,
            language=lang,
            temperature=temp,
            top_p=top_p,
        )  # numpy float32, mono
        wall = time.time() - start

        # write to memory as 16-bit PCM WAV
        buf = io.BytesIO()
        sf.write(buf, wav, SR, format="WAV", subtype="PCM_16")
        data = buf.getvalue()

        # keep a copy for download
        with open("out_last.wav", "wb") as f:
            f.write(data)

        headers = {
            "Content-Disposition": 'inline; filename="speech.wav"',
            "X-Wall": f"{wall:.3f}",
            "X-AudioSec": f"{len(wav)/SR:.3f}",
            "X-RTF": f"{(wall/(len(wav)/SR)):.3f}" if len(wav) else "nan",
        }
        return Response(content=data, media_type="audio/wav", headers=headers)

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

        all_wavs = []
        timings = []
        for idx, txt in enumerate(expanded, 1):
            start = time.time()
            
            wav = tts.tts(
                text=txt,
                speaker=speaker_name,
                language=lang,
                temperature=temp,
                top_p=top_p,
            )

            # normalize to float32 NumPy mono
            wav_np = np.asarray(wav, dtype=np.float32).reshape(-1)

            wall = time.time() - start
            audio_sec = float(wav_np.size) / SR if wav_np.size else 0.0
            rtf = wall / audio_sec if audio_sec > 0 else None

            timings.append({
                "i": idx,
                "chars": len(txt),
                "wall_s": round(wall, 3),
                "audio_s": round(audio_sec, 3),
                "rtf": round(rtf, 3) if rtf is not None else None,
            })

            all_wavs.append(wav_np)

        # concat & write
        if not all_wavs:
            return JSONResponse(status_code=400, content={"error": "no chunks to synthesize"})
        cat = np.concatenate(all_wavs, axis=0)

        buf = io.BytesIO()
        sf.write(buf, cat, SR, format="WAV", subtype="PCM_16")
        data = buf.getvalue()

        # save a copy for download
        with open("out_last.wav", "wb") as f:
            f.write(data)

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
        return Response(content=data, media_type="audio/wav", headers=headers)

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

@app.get("/ping")
def ping():
    return {"status": "ok"}

@app.get("/download/last")
def download_last():
    path = "out_last.wav"
    if not os.path.exists(path):
        return JSONResponse(status_code=404, content={"error": "no out_last.wav yet"})
    return FileResponse(path, media_type="audio/wav", filename="out_last.wav")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)