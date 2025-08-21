import io
import torch
from fastapi import FastAPI, UploadFile, Form, Response
from fastapi.responses import JSONResponse
from TTS.api import TTS
import soundfile as sf
import numpy as np
import tempfile
import json

app = FastAPI()

MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
LANG_DEFAULT = "en"

# load once at startup
tts = TTS(MODEL)

@app.get("/health")
def health():
    return {"ok": True, "model": MODEL}

@app.post("/speak")
async def speak(
    voice: UploadFile,
    text: str = Form(...),
    lang: str = Form(LANG_DEFAULT),
    temp: float = Form(0.1),
    top_p: float = Form(0.0),
):
    try:
        data = torch.load(io.BytesIO(await voice.read()), map_location="cpu")
        gpt, spk = data["gpt"], data["spk"]

        sm = tts.synthesizer.tts_model.speaker_manager
        speaker_name = f"req_{id(data)}"
        sm.speakers[speaker_name] = {
            "gpt_cond_latent": gpt,
            "speaker_embedding": spk,
        }

        buf = io.BytesIO()
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
        sm.speakers.pop(speaker_name, None)

        return Response(
            content=wav,
            media_type="audio/wav",
            headers={"Content-Disposition": 'inline; filename="speech.wav"'},
        )
    except KeyError:
        return JSONResponse(status_code=400, content={"error": "invalid voice .pt (missing gpt/spk)"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/speak-batch")
async def speak_batch(
    voice: UploadFile,
    chunks: str = Form(...),  # JSON array string
    lang: str = Form(LANG_DEFAULT),
    temp: float = Form(0.1),
    top_p: float = Form(0.0),
):
    try:
        parts = json.loads(chunks)
        if not isinstance(parts, list) or not all(isinstance(x, str) and x.strip() for x in parts):
            return JSONResponse(status_code=400, content={"error": "chunks must be a JSON array of non-empty strings"})

        data = torch.load(io.BytesIO(await voice.read()), map_location="cpu")
        gpt, spk = data["gpt"], data["spk"]

        sm = tts.synthesizer.tts_model.speaker_manager
        speaker_name = f"req_{id(data)}"
        sm.speakers[speaker_name] = {
            "gpt_cond_latent": gpt,
            "speaker_embedding": spk,
        }

        sr = 24000
        all_samples = []
        for txt in parts:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
                tts.tts_to_file(
                    text=txt,
                    speaker=speaker_name,
                    language=lang,
                    file_path=tmp.name,
                    temperature=temp,
                    top_p=top_p,
                )
                audio, got_sr = sf.read(tmp.name, dtype="int16")
                if got_sr != sr:
                    return JSONResponse(status_code=500, content={"error": f"unexpected sample rate {got_sr}"})
                if audio.ndim > 1:
                    audio = audio[:, 0]
                all_samples.append(audio)

        sm.speakers.pop(speaker_name, None)

        if not all_samples:
            return JSONResponse(status_code=400, content={"error": "no chunks to synthesize"})

        cat = np.concatenate(all_samples, axis=0)

        buf = io.BytesIO()
        sf.write(buf, cat, sr, format="WAV", subtype="PCM_16")
        buf.seek(0)

        return Response(
            content=buf.getvalue(),
            media_type="audio/wav",
            headers={"Content-Disposition": 'inline; filename="batch.wav"'},
        )

    except json.JSONDecodeError:
        return JSONResponse(status_code=400, content={"error": "chunks must be valid JSON"})
    except KeyError:
        return JSONResponse(status_code=400, content={"error": "invalid voice .pt (missing gpt/spk keys)"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})