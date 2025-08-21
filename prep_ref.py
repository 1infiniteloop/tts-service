# prep_ref.py
import os, torch, soundfile as sf, tempfile
from dotenv import load_dotenv
from TTS.api import TTS
import librosa

load_dotenv()
REF   = os.environ["REF"]          # from .env
ART   = os.environ["ART"]          # e.g. conditioning/conditioning.pt
MODEL = os.environ.get("MODEL","tts_models/multilingual/multi-dataset/xtts_v2")

os.makedirs(os.path.dirname(ART), exist_ok=True)

tts = TTS(model_name=MODEL, gpu=False)
xtts = getattr(getattr(tts, "synthesizer", None), "tts_model", None) or getattr(tts, "tts_model", None)

# (optional) write a temp 22.05k mono copy to be safe
y, _sr = librosa.load(REF, sr=None, mono=True)   # load as mono, native rate
y = librosa.resample(y, orig_sr=_sr, target_sr=22050)
with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
    sf.write(tmp.name, y, 22050, subtype="PCM_16")
    ref = tmp.name

gpt, spk = xtts.get_conditioning_latents(
    audio_path=ref, gpt_cond_len=30, gpt_cond_chunk_len=6, max_ref_length=60, sound_norm_refs=False
)
torch.save({"gpt": gpt.cpu(), "spk": spk.cpu()}, ART)
print(f"[ok] saved conditioning → {ART}")