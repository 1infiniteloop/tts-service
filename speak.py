import os, torch
from TTS.api import TTS
import random, numpy as np
import argparse, glob, os.path

parser = argparse.ArgumentParser(description="XTTS synth with cached speaker.")
parser.add_argument("--text", required=True, help="Text to speak.")
parser.add_argument("--out", default="test.wav", help="Output wav path.")
parser.add_argument("--speaker", default="conditioning/conditioning.pt",
                    help="Path to cached conditioning .pt file (gpt+spk).")
parser.add_argument("--temp", type=float, default=0.1,
                    help="Sampling temperature (0.0=greedy, recommended 0.0–0.2).")
parser.add_argument("--top_p", type=float, default=0.0,
                    help="Nucleus sampling (0.0=greedy).")
args = parser.parse_args()

# show available voices in conditioning/
voices = glob.glob("conditioning/*.pt")
print("👉 Available voices:")
for v in voices:
    print("  -", v)

# set deterministic-ish randomness
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.use_deterministic_algorithms(False)

MODEL = os.getenv("MODEL", "tts_models/multilingual/multi-dataset/xtts_v2")
LANG = os.getenv("LANG", "en")

tts = TTS(model_name=MODEL, gpu=False)

# load cached conditioning
if not os.path.exists(args.speaker):
    raise SystemExit(f"❌ Missing speaker file: {args.speaker}")

data = torch.load(args.speaker, map_location="cpu")
gpt, spk = data["gpt"], data["spk"]

# derive clean speaker name (e.g., conditioning/alice.pt -> alice)
speaker_name = os.path.splitext(os.path.basename(args.speaker))[0]

# register speaker
sm = tts.synthesizer.tts_model.speaker_manager
sm.speakers[speaker_name] = {
    "gpt_cond_latent": gpt,
    "speaker_embedding": spk,
}

# synthesize
tts.tts_to_file(
    text=args.text,
    speaker=speaker_name,
    language=LANG,
    file_path=args.out,
    temperature=args.temp,
    top_p=args.top_p,
)

print(f"[✅] Wrote {args.out}")