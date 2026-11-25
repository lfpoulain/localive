"""Reasoning models Localive can download for its own llama.cpp (installer and page Cerveau).

Q4_K_M quantizations (MXFP4 for gpt-oss, its native format) from the makers' own GGUF
repositories when they publish one (Mistral, Liquid, ggml-org for OpenAI), otherwise from
Unsloth. Sizes are the files' (the model also needs its context in video memory).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Model:
    id: str
    family: str
    label: str
    size_gb: float
    note: str
    repo: str
    file: str
    folder: str

    @property
    def path(self) -> str:
        return f"{self.folder}/{self.file}"


def _m(id_: str, family: str, label: str, size_gb: float, note: str, repo: str, file: str,
       folder: str = "") -> Model:  # fmt: skip
    return Model(id_, family, label, size_gb, note, repo, file,
                 folder or "models/" + repo.split("/")[1].removesuffix("-GGUF"))  # fmt: skip


FAMILIES = ("Qwen", "Gemma", "Mistral", "OpenAI", "Liquid (LFM)", "MiniCPM")

QWEN_4B_MTP = _m(
    "qwen3.5-4b-mtp", "Qwen", "Qwen3.5-4B MTP (UD-Q4_K_XL)", 3.0,
    "Profil optionnel. Qwen propose jusqu’à trois jetons d’avance pour accélérer la génération.",
    "unsloth/Qwen3.5-4B-MTP-GGUF", "Qwen3.5-4B-UD-Q4_K_XL.gguf",
)

CATALOG: tuple[Model, ...] = (
    # Qwen (Alibaba)
    _m("qwen3-4b", "Qwen", "Qwen3-4B Instruct 2507", 2.5,
       "Par défaut : le plus juste et le plus rapide dans nos mesures.",
       "unsloth/Qwen3-4B-Instruct-2507-GGUF", "Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
       "models/Qwen3-4B-Instruct-2507"),
    _m("qwen3.5-2b", "Qwen", "Qwen3.5-2B", 1.3, "Petit et rapide, moins fiable avec les outils.",
       "unsloth/Qwen3.5-2B-GGUF", "Qwen3.5-2B-Q4_K_M.gguf"),
    _m("qwen3.5-4b", "Qwen", "Qwen3.5-4B", 2.7,
       "Même taille, plus récent ; un peu moins sûr pour les outils dans nos mesures.",
       "unsloth/Qwen3.5-4B-GGUF", "Qwen3.5-4B-Q4_K_M.gguf"),
    QWEN_4B_MTP,
    _m("qwen3.5-9b", "Qwen", "Qwen3.5-9B", 5.7,
       "Plus de connaissances ; environ deux fois plus lent que le 4B.",
       "unsloth/Qwen3.5-9B-GGUF", "Qwen3.5-9B-Q4_K_M.gguf"),
    _m("qwen3.5-27b", "Qwen", "Qwen3.5-27B", 16.7, "Grand modèle : carte graphique de 24 Go.",
       "unsloth/Qwen3.5-27B-GGUF", "Qwen3.5-27B-Q4_K_M.gguf"),
    _m("qwen3.5-35b-a3b", "Qwen", "Qwen3.5-35B-A3B", 22.0,
       "Mélange d'experts : rapide pour sa taille, mais 24 Go de mémoire vidéo ou plus.",
       "unsloth/Qwen3.5-35B-A3B-GGUF", "Qwen3.5-35B-A3B-Q4_K_M.gguf"),
    # Gemma (Google)
    _m("gemma-e2b", "Gemma", "Gemma 4 E2B", 3.1, "Léger, bon en français.",
       "unsloth/gemma-4-E2B-it-GGUF", "gemma-4-E2B-it-Q4_K_M.gguf"),
    _m("gemma-e4b", "Gemma", "Gemma 4 E4B", 5.0,
       "Bon avec les outils, mais deux fois plus gros que Qwen3-4B : plus lent.",
       "unsloth/gemma-4-E4B-it-GGUF", "gemma-4-E4B-it-Q4_K_M.gguf", "models/gemma-4-E4B-it"),
    _m("gemma-12b", "Gemma", "Gemma 4 12B", 7.1, "Intermédiaire : carte graphique dédiée.",
       "unsloth/gemma-4-12b-it-GGUF", "gemma-4-12b-it-Q4_K_M.gguf"),
    _m("gemma-26b", "Gemma", "Gemma 4 26B-A4B", 13.6,
       "Mélange d'experts : 16 Go de mémoire vidéo ou plus.",
       "unsloth/gemma-4-26B-A4B-it-GGUF", "gemma-4-26B-A4B-it-UD-IQ4_XS.gguf",
       "models/gemma-4-26B-A4B-it"),
    _m("gemma-31b", "Gemma", "Gemma 4 31B", 18.3, "Grand modèle : carte graphique de 24 Go.",
       "unsloth/gemma-4-31B-it-GGUF", "gemma-4-31B-it-Q4_K_M.gguf"),
    # Mistral
    _m("ministral-3b", "Mistral", "Ministral 3 3B", 2.2, "Petit modèle français de Mistral.",
       "mistralai/Ministral-3-3B-Instruct-2512-GGUF", "Ministral-3-3B-Instruct-2512-Q4_K_M.gguf"),
    _m("ministral-8b", "Mistral", "Ministral 3 8B", 5.2, "Taille moyenne, bon en français.",
       "mistralai/Ministral-3-8B-Instruct-2512-GGUF", "Ministral-3-8B-Instruct-2512-Q4_K_M.gguf"),
    _m("ministral-14b", "Mistral", "Ministral 3 14B", 8.2, "Plus juste : carte graphique dédiée.",
       "mistralai/Ministral-3-14B-Instruct-2512-GGUF",
       "Ministral-3-14B-Instruct-2512-Q4_K_M.gguf"),
    _m("mistral-small-24b", "Mistral", "Mistral Small 3.2 24B", 14.3,
       "Grand modèle : 16 Go de mémoire vidéo ou plus.",
       "unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF",
       "Mistral-Small-3.2-24B-Instruct-2506-Q4_K_M.gguf"),
    # OpenAI
    _m("gpt-oss-20b", "OpenAI", "gpt-oss 20B", 12.1,
       "Mélange d'experts qui raisonne toujours avant de répondre : plus lent à démarrer "
       "chaque réponse. 16 Go de mémoire vidéo.",
       "ggml-org/gpt-oss-20b-GGUF", "gpt-oss-20b-MXFP4.gguf"),
    # Liquid AI
    _m("lfm2.5-1.2b", "Liquid (LFM)", "LFM2.5 1.2B Instruct", 0.7, "Très petit et très rapide.",
       "LiquidAI/LFM2.5-1.2B-Instruct-GGUF", "LFM2.5-1.2B-Instruct-Q4_K_M.gguf"),
    _m("lfm3-1.7b", "Liquid (LFM)", "LFM3 1.7B", 1.1, "Petit et rapide.",
       "LiquidAI/LFM3-1.7B-GGUF", "LFM3-1.7B-Q4_K_M.gguf"),
    _m("lfm2.5-2.6b", "Liquid (LFM)", "LFM2.5 2.6B", 1.7, "Petit, un peu plus juste.",
       "LiquidAI/LFM2.5-2.6B-GGUF", "LFM2.5-2.6B-Q4_K_M.gguf"),
    _m("lfm2.5-8b-a1b", "Liquid (LFM)", "LFM2.5 8B-A1B", 5.2,
       "Mélange d'experts : rapide comme un petit modèle, plus de connaissances.",
       "LiquidAI/LFM2.5-8B-A1B-GGUF", "LFM2.5-8B-A1B-Q4_K_M.gguf"),
    _m("lfm2.5-24b-a2b", "Liquid (LFM)", "LFM2.5 24B-A2B", 14.5,
       "Mélange d'experts : rapide, mais 16 Go de mémoire vidéo ou plus.",
       "LiquidAI/LFM2.5-24B-A2B-GGUF", "LFM2.5-24B-A2B-Q4_K_M.gguf"),
    # MiniCPM (OpenBMB)
    _m("minicpm5-1b", "MiniCPM", "MiniCPM5-1B", 0.7, "Très petit : pour les essais.",
       "openbmb/MiniCPM5-1B-GGUF", "MiniCPM5-1B-Q4_K_M.gguf", "models/MiniCPM5-1B"),
    _m("minicpm5-2b", "MiniCPM", "MiniCPM5-2B", 1.6, "Petit et rapide, moins fiable.",
       "openbmb/MiniCPM5-2B-GGUF", "MiniCPM5-2B-Q4_K_M.gguf", "models/MiniCPM5-2B"),
)  # fmt: skip
BY_ID = {m.id: m for m in CATALOG}
DEFAULT_ID = "qwen3-4b"
