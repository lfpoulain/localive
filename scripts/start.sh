#!/usr/bin/env bash
# Démarre Localive. Le bouton « Redémarrer » de l'administration arrête Localive avec le
# code 75 : ce script le relance aussitôt. Toute autre fin arrête la boucle.
#   scripts/start.sh            # sans ouvrir de navigateur
#   scripts/start.sh --open     # ouvre l'administration une fois prêt
set -u
cd "$(dirname "$0")/.."
python=".venv/bin/python"
if [ ! -x "$python" ]; then
  echo "Installation manquante : scripts/install-linux.sh" >&2
  exit 1
fi
args=("$@")
restart_args=()
for arg in "$@"; do
  if [ "$arg" != "--open" ]; then restart_args+=("$arg"); fi
done
while true; do
  "$python" -m localive "${args[@]}"
  code=$?
  [ "$code" -eq 75 ] || exit "$code"
  echo "Redémarrage demandé depuis l'administration…"
  args=("${restart_args[@]}")
done
