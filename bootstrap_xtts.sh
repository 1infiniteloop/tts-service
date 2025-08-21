#!/usr/bin/env bash
set -euo pipefail

# ---- settings you can tweak ----
PYVER="${PYVER:-3.11}"                   # requires Python >=3.10
VENV_DIR="${VENV_DIR:-.venv}"

# PIP_PKGS=(
#   "pip>=24.0" "setuptools>=70.0" "wheel>=0.43"
#   "torch==2.5.1" "torchaudio==2.5.1"
#   "transformers==4.40.2"
#   "TTS==0.22.0"
#   "librosa" "soundfile" "python-dotenv"
# )
# --------------------------------

echo "==> creating venv (${VENV_DIR}) with python ${PYVER} ..."
if command -v pyenv >/dev/null 2>&1; then
  # prefer pyenv python if available
  PYBIN="$(pyenv which python${PYVER} 2>/dev/null || true)"
  if [[ -z "${PYBIN}" ]]; then
    echo "pyenv doesn't have python${PYVER}. install with: pyenv install ${PYVER}.x"
    exit 1
  fi
else
  PYBIN="$(command -v python${PYVER} || command -v python3 || true)"
fi
if [[ -z "${PYBIN}" ]]; then
  echo "Could not find python ${PYVER}. Please install Python ${PYVER}.x"
  exit 1
fi

"${PYBIN}" -m venv "${VENV_DIR}"
# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"

echo "==> upgrading base tooling ..."
python -V
pip install -U -r requirements.txt

echo "==> verifying versions ..."
python - <<'PY'
import torch, transformers, TTS, sys
print("python:", sys.version.split()[0])
print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("TTS:", TTS.__version__)
PY

# optional: ensure project skeleton exists
mkdir -p conditioning samples out

# optional: seed a .env if missing
if [[ ! -f .env ]]; then
  cat > .env <<'ENV'
# absolute or ${PWD}-relative paths:
REF=${PWD}/samples/voice.wav
ART=${PWD}/conditioning/conditioning.pt
MODEL=tts_models/multilingual/multi-dataset/xtts_v2
LANG=en
ENV
  echo "==> wrote .env (edit REF to your reference audio path)"
fi

echo "==> quick model check (lists XTTS entries) ..."
tts --list_models | grep -i xtts || true

echo "==> smoke synth (built-in speaker index) -> out/test.wav ..."
tts --text "XTTS environment ready." \
    --model_name tts_models/multilingual/multi-dataset/xtts_v2 \
    --speaker_idx 0 \
    --language_idx en \
    --out_path out/test.wav || true

echo "==> done. activate with:  source ${VENV_DIR}/bin/activate"