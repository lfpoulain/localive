#!/usr/bin/env bash
# Installe Localive sous Linux (x86-64) : Python, llama.cpp, les modèles, et en option un
# service systemd utilisateur.
#
#   scripts/install-linux.sh                    # inclut d3-mini Q8_0 pour trier les outils
#   scripts/install-linux.sh --device cpu --llama vulkan  # Radeon, transcription sur CPU
#   scripts/install-linux.sh --llama cuda       # cerveau sur llama.cpp CUDA (NVIDIA)
#   scripts/install-linux.sh --systemd          # + service « localive » au démarrage
#   scripts/install-linux.sh --skip-voice       # Phonon plus tard (accès Hugging Face)
#   scripts/install-linux.sh --no-tool-router   # omet le téléchargement du routeur (5,16 Go)
#   scripts/install-linux.sh --models gemma-e4b # autres modèles de raisonnement (ou all)
#
# Prérequis : uv (https://docs.astral.sh/uv/), et pour Vulkan le chargeur Vulkan et le
# pilote de la carte (Debian/Ubuntu : sudo apt install libvulkan1 mesa-vulkan-drivers).
# Relancer le script est sans risque : ce qui est déjà vérifié est conservé.
set -euo pipefail
cd "$(dirname "$0")/.."
root="$(pwd)"
export UV_CACHE_DIR="$root/.runtime/uv-cache"

device="auto"; llama="auto"; skip_voice=""; models=""; systemd=""; router=""
while [ $# -gt 0 ]; do
  case "$1" in
    --device) device="$2"; shift 2 ;;
    --llama) llama="$2"; shift 2 ;;
    --skip-voice) skip_voice="--skip-voice"; shift ;;
    --tool-router|--no-tool-router) router="$1"; shift ;;
    --models) models="--models=$2"; shift 2 ;;
    --systemd) systemd="yes"; shift ;;
    -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
    *) echo "Option inconnue : $1" >&2; exit 2 ;;
  esac
done

echo "1/3 Environnement Python"
command -v uv >/dev/null || { echo "uv est requis : curl -LsSf https://astral.sh/uv/install.sh | sh" >&2; exit 1; }
python=".venv/bin/python"
[ -x "$python" ] || uv venv --python 3.12 .venv
if [ "$device" = "auto" ]; then
  device="cpu"
  if command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name --format=csv,noheader >/dev/null 2>&1; then
    device="cuda"
  fi
fi
if [ "$device" = "cpu" ] && ! "$python" -c "from importlib.util import find_spec; import sys; sys.exit(find_spec('torch') is None)"; then
  uv pip install --python "$python" --index https://download.pytorch.org/whl/cpu 'torch>=2.8'
fi
uv pip install --python "$python" -e .
if [ "$device" = "cuda" ]; then
  if ! "$python" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
    uv pip install --python "$python" torch --index-url https://download.pytorch.org/whl/cu128 --reinstall-package torch
  fi
  "$python" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 'PyTorch ne voit pas la carte NVIDIA')"
fi

echo "2/3 Moteurs et modèles"
"$python" -m localive.install --device "$device" --llama "$llama" $skip_voice $models $router

echo "3/3 Lancement"
chmod +x scripts/start.sh
if [ -n "$systemd" ]; then
  unit="$HOME/.config/systemd/user/localive.service"
  mkdir -p "$(dirname "$unit")"
  cat > "$unit" <<UNIT
[Unit]
Description=Localive, la voix locale de Cornelius
After=network-online.target

[Service]
WorkingDirectory=$root
ExecStart=$root/.venv/bin/python -m localive
# The admin page's "Restart" button exits with 75: start again at once.
Restart=on-failure
RestartForceExitStatus=75
RestartSec=2

[Install]
WantedBy=default.target
UNIT
  systemctl --user daemon-reload
  systemctl --user enable --now localive.service
  echo "Service installé : systemctl --user status localive"
  echo "Pour démarrer sans session ouverte : sudo loginctl enable-linger $USER"
else
  echo "Démarrer : scripts/start.sh"
fi
echo "Administration : http://127.0.0.1:8787"
