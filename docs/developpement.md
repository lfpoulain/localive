# Développement

## Tests

Dans le dépôt, ajoutez les dépendances de développement à l’environnement créé par
l’installeur. Sous Linux, remplacez `.venv\Scripts\python.exe` par `.venv/bin/python`.

```powershell
uv pip install --python .venv\Scripts\python.exe -e '.[dev]'
.venv\Scripts\python.exe -m pytest
.venv\Scripts\python.exe -m ruff check src tests
node --test tests/admin-ui.test.cjs
```

Les tests n'utilisent ni carte graphique ni modèle : transcription, cerveau et voix sont simulés,
sauf Silero (fourni) et le transport WebRTC, testé de bout en bout avec aiortc dans le rôle du
navigateur. `docs/reglages.md` est généré à partir du code :
`python -m localive.settings > docs/reglages.md` (un test vérifie qu'il est à jour).

Pour tester les intentions de raccrochage avec les poids réels, chargez d3-mini
dans Localive puis indiquez le port du décideur (8085 par défaut).

```powershell
$env:LOCALIVE_TEST_D3_PORT = "8085"
.venv\Scripts\python.exe -m pytest -q -k test_native_d3_distinguishes
```

## Bancs de mesure

| Banc | Mesure |
|---|---|
| `bench/prompts/run.py` | La vraie configuration de Cornelius dans la vraie chaîne de réponse : outils, mémoire, corrections, interruptions, accueil, recherche web (Localive ou ses llama-server lancés) |
| `bench/prompts/route_tools.py` | Sélection des outils sur 34 demandes françaises avec un modèle SystemOne. Mesure la latence, les outils oubliés, les sélections exactes et le volume de schémas transmis. |
| `bench/prompts/tool_prefill.py` | Temps de Qwen avant le premier événement, avec catalogue complet ou sélection fixée, à froid puis à chaud. |
| `bench/prompts/live_tool_latency.py` | Temps entre la transcription d’une question d’heure ou de date et l’appel d’horloge complet. Accepte un routeur GGUF ou le catalogue complet. |
| `bench/decision/compare_models.py` | Décideur : 93 situations et 24 plans, latence |
| `bench/decision/run.py`, `calibrate.py`, `live_scenarios.py` | Variantes de consignes, calibration, scénarios audio avec écho |
| `bench/turn/` | Smart Turn contre la règle du silence, sur des extraits complets et coupés |
| `bench/echo/run.py` | Annulation d'écho (AEC3) sur écho simulé |
| `bench/compare_chat.py`, `compare_live.py`, `tune_qwen.py` | Comparaison de cerveaux, conversation complète, réglages llama.cpp |

`bench/prompts/cornelius_session.json` est produit par le code de Cornelius
(`liveConfiguration` de `server/live/configuration.ts`) : le régénérer quand ses consignes
changent.

## Structure

```
src/localive/
  server.py     HTTP, WebSocket Live, WebRTC, administration (API)
  webrtc.py     transport WebRTC compatible OpenAI (aiortc)
  live.py       une session : écoute, fin de tour, interruptions, réponses, outils, voix
  decision.py   décideur d'interruption et plans
  searxng.py    SearXNG dans Docker, géré par Localive
  settings.py   réglages expliqués de l'administration, localive.toml
  install.py    installeur commun Windows / Linux
  asr.py        Parakeet (Photon)        tts.py    Phonon (ptts)
  turn.py       Smart Turn               vad.py    Silero v5
  echo.py       WebRTC AEC3              engines.py  llama-server
  static/       administration (HTML, CSS, JS, polices)
scripts/        setup.ps1, start.ps1, install-linux.sh, start.sh, live_client.py
```

`scripts/live_client.py` joue des fichiers WAV dans une vraie session Live et enregistre la
réponse : pratique pour un essai sans Cornelius.
