"""Settings of the admin page: schema, explanations, live application and ``localive.toml``.

Values marked ``restart`` are saved and take effect at the next start (models, ports,
engines); the others apply at once or at the next conversation. Secrets (tokens) are never
sent back to the page: it only learns whether one is set.
"""

from __future__ import annotations

import copy
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .catalog import QWEN_4B_MTP
from .config import BRAIN_SERVERS, GOODBYE_MODES, Config, merge


@dataclass(frozen=True)
class Setting:
    key: str  # "section.name"
    label: str
    help: str = ""
    choices: tuple[str, ...] = ()
    restart: bool = False
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    unit: str = ""
    advanced: bool = False  # hidden until "Réglages avancés" is opened
    secret: bool = False  # never returned to the page
    dynamic: str = ""  # "gguf": the models found on disk; "gguf_inherit": same, or inherit;
    # "external_models": listed by the page from the chosen Ollama / LM Studio server
    # Shown only when another setting matches, e.g. "brain.backend=localive".
    only: str = ""


@dataclass(frozen=True)
class Section:
    id: str
    title: str
    icon: str  # name of an icon of the admin page
    intro: str
    settings: tuple[Setting, ...]


SECTIONS: tuple[Section, ...] = (
    Section("voice", "Voix", "voice", (
        "La voix de Cornelius, synthétisée sur le processeur par Phonon. Cornelius choisit une "
        "voix OpenAI (Cedar, Marin…) : Localive la traduit en voix masculine ou féminine."), (
        Setting("voice.default", "Voix quand le client n'en choisit pas",
                "Les clients comme Cornelius imposent généralement la leur.",
                ("female", "male")),
        Setting("voice.female", "Voix féminine",
                "Ninon (par défaut), Tiphaine, Hélène, Noémie, Coralie, Maëlys, Orianne, Solène, "
                "Lilou… ou un fichier .safetensors.", restart=True),
        Setting("voice.male", "Voix masculine",
                "Lucien (par défaut), Arnaud, Augustin, Jules, Baptiste, Marius…", restart=True),
        Setting("voice.wait_notice", "Parler pendant une attente",
                "Après deux secondes sans réponse, la voix prononce une courte phrase. "
                "Si un outil attend encore son résultat, elle le signale. Une fois par demande."),
        Setting("voice.threads", "Cœurs de calcul de la voix",
                "4 est le plus rapide sur la plupart des processeurs ; davantage ralentit.",
                minimum=1, maximum=32, step=1, restart=True, advanced=True),
        Setting("voice.model", "Modèle Phonon",
                "hf://gradium/phonon-40m-fr ou un dossier local.", restart=True, advanced=True),
    )),
    Section("listening", "Écoute", "ear", (
        "Comment Localive entend : détection de la voix (Silero), transcription (Parakeet) et "
        "réaction quand on parle pendant que Cornelius répond."), (
        Setting("vad.threshold", "Sensibilité du micro",
                "Plus bas : capte les voix lointaines ou faibles, mais aussi plus de bruit. "
                "0,5 convient à la plupart des pièces.", minimum=0.2, maximum=0.9, step=0.05),
        Setting("vad.silence_ms", "Silence de fin de phrase (sans fin de tour intelligente)",
                "Utilisé seulement si la fin de tour intelligente est désactivée.",
                minimum=250, maximum=2000, step=50, unit="ms", advanced=True),
        Setting("vad.min_speech_ms", "Parole minimale",
                "En dessous, un son est ignoré (clic, toux). Une phrase déjà transcrite est "
                "toujours gardée.", minimum=60, maximum=600, step=20, unit="ms", advanced=True),
        Setting("vad.barge_in_ms", "Parole avant de baisser la voix",
                "Quand quelqu'un parle pendant que Cornelius répond : durée avant la première "
                "vérification et la baisse de volume.", minimum=150, maximum=1500, step=50,
                unit="ms"),
        Setting("vad.duck_early_level", "Volume dès qu'une voix est entendue",
                "Pendant que Cornelius parle (1 = aucune baisse).", minimum=0.0, maximum=1.0,
                step=0.05),
        Setting("vad.duck_level", "Volume quand une personne lui parle",
                "Une fois la voix confirmée (et l'écho écarté), le temps de décider.",
                minimum=0.0, maximum=1.0, step=0.05),
        Setting("asr.device_vocabulary", "Vocabulaire des appareils après transcription",
                "Dans les nouvelles conversations, aide à reconnaître les noms et alias des "
                "appareils Home Assistant déclarés par le client. Un nom partagé demande une "
                "précision. Les sous-titres conservent les mots entendus."),
        Setting("asr.device", "Processeur de la transcription",
                "auto : la carte NVIDIA si elle est disponible, sinon le processeur.",
                ("auto", "cuda", "cpu"), restart=True, advanced=True),
        Setting("asr.model", "Modèle de transcription",
                "moondream/parakeet-ultra (par défaut).", restart=True, advanced=True),
    )),
    Section("turn", "Fin de tour", "timer", (
        "Savoir quand vous avez fini de parler. Smart Turn écoute l'intonation à chaque pause : "
        "Cornelius répond dès 200 ms si la phrase est finie, et attend si elle ne l'est pas "
        "(« mets un minuteur de… »)."), (
        Setting("turn.enabled", "Fin de tour intelligente",
                "Désactivée : Localive attend un silence fixe, plus lent.", restart=True),
        Setting("turn.end_on_goodbye", "Raccrocher sur une formule de fin",
                "Dans les nouvelles conversations, « au revoir » ou « c’est bon, merci » "
                "ferme la connexion. Un simple « merci » permet de continuer."),
        Setting("turn.goodbye_mode", "Détection du raccrochage",
                "Les formules prédéfinies sont instantanées. Le décideur local d3-mini peut "
                "aussi reconnaître une fin de conversation exprimée autrement. En cas de doute "
                "ou de modèle indisponible, la conversation reste ouverte.",
                GOODBYE_MODES, only="turn.end_on_goodbye=true"),
        Setting("turn.check_ms", "Pause avant de vérifier",
                "Plus court : répond plus vite, mais vérifie plus souvent.", minimum=100,
                maximum=800, step=20, unit="ms"),
        Setting("turn.threshold", "Seuil « phrase finie »",
                "Plus haut : attend davantage d'être sûr que vous avez fini.", minimum=0.1,
                maximum=0.95, step=0.05),
        Setting("turn.max_silence_ms", "Attente maximale si la phrase semble inachevée",
                "Ensuite, Cornelius vous invite à continuer.", minimum=600, maximum=6000,
                step=100, unit="ms"),
        Setting("turn.over_silence_ms", "Pause de fin quand vous lui coupez la parole",
                "Pendant qu'il parle, la fin de votre phrase se décide sur les mots après cette "
                "pause. Montez-la si vos phrases sont coupées en deux.", minimum=300,
                maximum=2000, step=50, unit="ms"),
        Setting("turn.threads", "Cœurs de calcul de Smart Turn", minimum=1, maximum=16,
                step=1, restart=True, advanced=True),
    )),
    Section("decision", "Interruptions", "split", (
        "Quand quelqu'un parle pendant que Cornelius réfléchit ou répond, un décideur choisit : "
        "ignorer (« mm-hm », télé, quelqu'un d'autre), compléter la demande, s'arrêter ou "
        "passer à une nouvelle demande. Puis il choisit de finir d'abord, de corriger ou de "
        "répondre tout de suite."), (
        Setting("decision.enabled", "Mode naturel (décideur d'interruption)",
                "Activé : un petit modèle juge ce que vous dites pendant que Cornelius réfléchit "
                "ou parle : ignorer un « mm-hm » ou la télé, compléter, s'arrêter, changer de "
                "sujet. Désactivé, mode classique : aucun modèle de décision n'est chargé "
                "(environ 3 Go de mémoire vidéo et du calcul en moins) ; « stop » ou « chut » "
                "coupent la réponse, et toute autre phrase dite pendant qu'il parle devient une "
                "nouvelle demande, à la fin de sa phrase en cours.", restart=True),
        Setting("decision.assistant_name", "Nom de l'assistant",
                "« Julie, viens voir » s'adresse à quelqu'un d'autre, « Cornelius, … » non."),
        Setting("decision.timeout_ms", "Délai maximal d'une décision",
                "Au-delà, les règles simples s'appliquent.", minimum=100, maximum=2000, step=50,
                unit="ms"),
        Setting("decision.dedicated", "Serveur dédié au décideur",
                "Oui (conseillé) : décisions en environ 50 ms, environ 3 Go de mémoire vidéo. "
                "Non : il partage le cerveau principal et attend parfois qu'il ait fini de lire "
                "(0,3 à 5 s).", restart=True),
        Setting("decision.model", "Modèle du décideur",
                "Qwen3-4B par défaut, quel que soit le cerveau : c'est le décideur mesuré, et un "
                "gros cerveau ne charge pas un deuxième gros modèle.",
                restart=True, advanced=True, dynamic="gguf_inherit"),
        Setting("decision.llama_dir", "Dossier llama.cpp du décideur",
                "Vide : le même moteur que le cerveau principal.", restart=True, advanced=True),
        Setting("decision.port", "Port local du décideur", minimum=1024, maximum=65535, step=1,
                restart=True, advanced=True),
        Setting("decision.context", "Contexte du décideur", minimum=2048, maximum=32768,
                step=1024, unit="jetons", restart=True, advanced=True),
    )),
    Section("brain", "Cerveau", "brain", (
        "Le modèle de langage qui répond et appelle les outils de Cornelius : Qwen3-4B par "
        "défaut, sur la carte graphique via le llama.cpp de Localive, ou un modèle servi par "
        "Ollama ou LM Studio."), (
        Setting("brain.backend", "Moteur du cerveau",
                "Localive (conseillé) : llama.cpp intégré, réglé et mesuré pour la voix. Ollama "
                "ou LM Studio : le modèle de votre choix, servi par ce logiciel (qui doit être "
                "lancé, avec son serveur activé). Le décideur reste sur le llama.cpp de Localive.",
                ("localive", "ollama", "lmstudio", "openai"), restart=True),
        Setting("brain.external_model", "Modèle du serveur externe",
                "La liste vient du serveur choisi ci-dessus. Ignoré avec le moteur Localive.",
                restart=True, dynamic="external_models", only="brain.backend!=localive"),
        Setting("brain.external_url", "Adresse du serveur externe",
                "Vide : l'adresse habituelle (Ollama http://127.0.0.1:11434/v1, LM Studio "
                "http://127.0.0.1:1234/v1). Obligatoire pour « Autre serveur ».", restart=True,
                only="brain.backend!=localive"),
        Setting("brain.ollama_context", "Contexte demandé à Ollama",
                "Ollama garde sinon son contexte par défaut (souvent 4 096 jetons) et coupe sans "
                "le dire le début des consignes de Cornelius (environ 8 000 jetons). 16 384 "
                "suffit ; plus demande de la mémoire vidéo et peut beaucoup ralentir un gros "
                "modèle. 0 : celui du serveur. Changer cette valeur recharge le modèle dans "
                "Ollama (jusqu'à deux minutes).", minimum=0, maximum=131072, step=1024,
                unit="jetons", restart=True, only="brain.backend=ollama"),
        Setting("brain.model", "Modèle de raisonnement",
                "Les modèles GGUF trouvés dans le dossier models. Qwen3-4B (par défaut) est le "
                "plus juste et le plus rapide dans nos mesures ; un modèle plus gros demande plus "
                "de mémoire vidéo. Le décideur garde Qwen3-4B (page Interruptions).",
                restart=True, dynamic="gguf", only="brain.backend=localive"),
        Setting("llm.temperature", "Créativité",
                "Bas (0,2) : réponses précises et stables. Plus haut : plus variées.", minimum=0,
                maximum=1.5, step=0.05),
        Setting("llm.max_tokens", "Longueur maximale d'une réponse",
                "Environ 400 jetons : une histoire courte. Plus : réponses plus longues possibles.",
                minimum=64, maximum=4096, step=16, unit="jetons"),
        Setting("llm.history_turns", "Échanges gardés en mémoire",
                "Le contexte est de toute façon réduit pour tenir dans le modèle.", minimum=1,
                maximum=50, step=1),
        Setting("routing.enabled", "Choisir les outils après la transcription",
                "Qwen reçoit directement les petits catalogues. d3-mini trie à partir de "
                "6 outils, ou dès 2 si leurs schémas totalisent au moins 3 072 octets.",
                restart=True),
        Setting("routing.model", "Fichier GGUF du décideur d’outils",
                "Modèle SystemOne partagé avec le raccrochage vocal en mode d3. "
                "d3-mini Q8_0 par défaut.", restart=True, advanced=True),
        Setting("routing.port", "Port local du décideur d’outils",
                minimum=1024, maximum=65535, restart=True, advanced=True),
        Setting("routing.llama_dir", "Dossier llama.cpp du décideur d’outils",
                "Le moteur Vulkan est utilisé par défaut, même avec un cerveau sur CPU. "
                "Laisser vide pour utiliser le moteur du cerveau.", restart=True, advanced=True),
        Setting("routing.gpu_layers", "Couches du décideur d’outils sur la carte graphique",
                "La valeur par défaut de 99 place tout d3-mini sur le GPU. "
                "À -1, reprend le nombre de couches du cerveau si le moteur est partagé, "
                "ou utilise 99 couches avec un moteur distinct. À 0, le décideur tourne sur CPU.",
                minimum=-1, maximum=999, step=1, restart=True, advanced=True),
        Setting("routing.timeout_ms", "Temps maximal pour choisir les outils",
                "À 0, Localive adapte le délai aux décisions mesurées (400 à 8 000 ms). "
                "Passé ce délai, le cerveau reçoit tous les outils.", minimum=0,
                maximum=10000, step=50, unit="ms", advanced=True),
        Setting("llm.tool_timeout_s", "Délai maximal d'un outil de Cornelius",
                "Au-delà, Cornelius répond que l'outil n'a pas répondu (agent, Home Assistant "
                "lent…).", minimum=5, maximum=300, step=5, unit="s"),
        Setting("brain.context", "Contexte",
                "Cornelius envoie de longues consignes et des outils : 24 576 jetons conseillés.",
                minimum=4096, maximum=131072, step=1024, unit="jetons", restart=True,
                only="brain.backend=localive"),
        Setting("llm.model", "Nom publié",
                "Le nom sous lequel Cornelius voit ce cerveau (/v1/models).", restart=True,
                advanced=True),
        Setting("brain.llama_dir", "Dossier llama.cpp",
                "Le moteur : Vulkan (toutes cartes), CUDA (NVIDIA) ou CPU.", restart=True,
                advanced=True, only="brain.backend=localive"),
        Setting("brain.gpu_layers", "Couches sur la carte graphique",
                "99 : tout le modèle. Moins : une partie reste en mémoire vive (plus lent).",
                minimum=0, maximum=999, step=1, restart=True, advanced=True,
                only="brain.backend=localive"),
        Setting("brain.extra_args", "Options llama-server",
                "Options supplémentaires du cerveau principal (pas du décideur).", restart=True,
                advanced=True, only="brain.backend=localive"),
        Setting("llm.api_key", "Clé du serveur externe", "Seulement pour un serveur externe "
                "qui en demande une.", restart=True, advanced=True, secret=True,
                only="brain.backend!=localive"),
    )),
    Section("search", "Recherche web", "globe", (
        "L'outil web_search de l'API Live (module Web de Cornelius), assuré par SearXNG. "
        "Localive installe et lance SearXNG dans Docker, sur cette machine uniquement."), (
        Setting("search.engine", "Recherche web",
                "off : Cornelius dit simplement qu'il ne peut pas vérifier.",
                ("searxng", "off")),
        Setting("search.managed", "Gérer SearXNG dans Docker",
                "Non : Localive utilise l'adresse ci-dessous telle quelle (votre propre SearXNG).",
                restart=True),
        Setting("search.url", "Adresse de SearXNG", "Le format JSON doit être activé.",
                restart=True),
        Setting("search.language", "Langue des résultats", "fr-FR, en-US…"),
        Setting("search.max_results", "Résultats lus par le cerveau",
                "Plus : réponses mieux informées, mais lecture plus longue.", minimum=1,
                maximum=10, step=1),
        Setting("search.start_docker", "Démarrer Docker Desktop si besoin",
                "Windows : lance Docker Desktop s'il est installé mais arrêté.", restart=True,
                advanced=True),
        Setting("search.image", "Image Docker", restart=True, advanced=True),
        Setting("search.container", "Nom du conteneur Docker",
                "Choisissez un nouveau nom et un port disponible pour créer une autre instance.",
                restart=True, advanced=True),
        Setting("search.timeout_s", "Délai maximal d'une recherche", minimum=1, maximum=30,
                step=1, unit="s", advanced=True),
    )),
    Section("echo", "Écho", "waves", (
        "Annulation d'écho côté serveur (WebRTC AEC3) : retire du micro la voix que Localive "
        "envoie au haut-parleur. Désactivée par défaut : la baisse de volume et le filtre "
        "d'écho textuel ont donné de meilleures décisions dans nos essais. À essayer dans "
        "votre pièce."), (
        Setting("echo.enabled", "Annulation d'écho côté serveur", "Prise en compte à la "
                "prochaine conversation."),
        Setting("echo.auto_gain_control", "Gain automatique du micro"),
        Setting("echo.high_pass_filter", "Filtre des basses (souffle, vibrations)",
                advanced=True),
        Setting("echo.noise_suppression", "Réduction de bruit",
                "Peut dégrader la transcription : à mesurer chez vous.", advanced=True),
        Setting("echo.clean_transcription", "Transcrire le signal nettoyé",
                "Non (conseillé) : le signal nettoyé sert à détecter la voix, la transcription "
                "entend le micro intact.", advanced=True),
        Setting("echo.delay_hint_ms", "Délai estimé haut-parleur → micro",
                "Point de départ ; l'annulation mesure ensuite le vrai délai.", minimum=0,
                maximum=500, step=10, unit="ms", advanced=True),
    )),
    Section("server", "Serveur et sécurité", "shield", (
        "Où Localive écoute et qui peut l'utiliser. Pour un Cornelius sur une autre machine, "
        "écoutez sur 0.0.0.0 et définissez un jeton."), (
        Setting("server.host", "Adresse d'écoute",
                "127.0.0.1 : cette machine seulement. 0.0.0.0 : tout le réseau local.",
                restart=True),
        Setting("server.port", "Port", "8787 par défaut ; il fait partie de l'adresse donnée à "
                "Cornelius.", minimum=1, maximum=65535, step=1, restart=True),
        Setting("server.api_key", "Jeton d'accès",
                "Vide : aucun jeton (déconseillé sur le réseau). Sinon, Cornelius doit l'envoyer "
                "comme clé API.", restart=True, secret=True),
        Setting("huggingface.token", "Jeton Hugging Face",
                "Pour télécharger la voix Phonon (modèle à conditions). Vide : variable "
                "HF_TOKEN, puis `hf auth login`.", restart=True, secret=True),
        Setting("huggingface.offline", "Hors ligne",
                "Ne plus contacter Hugging Face : tous les modèles doivent être téléchargés.",
                restart=True, advanced=True),
    )),
)  # fmt: skip
BY_KEY = {s.key: s for section in SECTIONS for s in section.settings}
# Compatibility with the former (title, settings) pairs.
GROUPS = [(section.title, list(section.settings)) for section in SECTIONS]


def _routing_weights(path: Path, config: Config) -> bool:
    # Recognize standard decision filenames as well as the configured, possibly renamed GGUF.
    return (path.name.lower().startswith(("d1-", "vllm-d3-", "d3-"))
            or path == config.path(config.routing.model))


def gguf_models(config: Config) -> list[str]:
    """Language models found under models/ (not Hugging Face's cache, not vision projectors),
    as paths relative to the Localive folder."""
    folder = config.path("models")
    found = []
    for path in folder.rglob("*.gguf") if folder.is_dir() else []:
        relative = path.relative_to(folder)
        if (relative.parts[0] == "hf" or "mmproj" in path.name.lower()
                or _routing_weights(path, config)):
            continue
        found.append("models/" + relative.as_posix())
    return sorted(found, key=lambda f: Path(f).name.lower())


def choices_for(setting: Setting, config: Config) -> tuple[list[str], dict[str, str]]:
    """Allowed values and their labels (models on disk are listed when the page asks)."""
    if not setting.dynamic or setting.dynamic == "external_models":
        return list(setting.choices), {}  # external models: listed by the page
    current = _get(config, setting.key)
    files = gguf_models(config)
    if current and current not in files and not _routing_weights(config.path(current), config):
        files.append(current)  # keep the configured file selectable, even if missing
    labels = {f: Path(f).stem + ("" if config.path(f).is_file() else " (introuvable)")
              for f in files}  # fmt: skip
    if (setting.key == "brain.model" and QWEN_4B_MTP.path in files
            and config.path(QWEN_4B_MTP.path).is_file()):
        labels[QWEN_4B_MTP.path] = QWEN_4B_MTP.label
    if setting.dynamic == "gguf_inherit":
        return ["", *files], {"": "Le même que le cerveau", **labels}
    return files, labels


def _get(target: Any, key: str) -> Any:
    for part in key.split("."):
        target = getattr(target, part)
    return target


def _nest(key: str, value: Any) -> dict[str, Any]:
    """'a.b.c', v -> {'a': {'b': {'c': v}}}"""
    out: dict[str, Any] = {}
    *parents, last = key.split(".")
    node = out
    for part in parents:
        node = node.setdefault(part, {})
    node[last] = value
    return out


def _deep_update(target: dict[str, Any], values: dict[str, Any]) -> None:
    for key, value in values.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value


def schema(config: Config, restart_pending: set[str] | None = None) -> dict[str, Any]:
    defaults = Config()
    groups = []
    for section in SECTIONS:
        entries = []
        for s in section.settings:
            value, default = _get(config, s.key), _get(defaults, s.key)
            if restart_pending and s.key in restart_pending:
                section_name, key = s.key.split(".")
                value = config.file_values.get(section_name, {}).get(key, default)
            choices, labels = choices_for(s, config)
            entry = {
                "key": s.key, "label": s.label, "help": s.help, "choices": choices,
                "choice_labels": labels,
                "restart": s.restart, "min": s.minimum, "max": s.maximum, "step": s.step,
                "unit": s.unit, "advanced": s.advanced, "secret": s.secret, "dynamic": s.dynamic,
                "only": s.only,
                "type": type(default).__name__, "value": value, "default": default,
            }  # fmt: skip
            if s.secret:  # the page only learns whether a secret is set
                entry.update(value="", default="", is_set=bool(value))
            entries.append(entry)
        groups.append({"id": section.id, "title": section.title, "icon": section.icon,
                       "intro": section.intro, "settings": entries})  # fmt: skip
    return {"file": str(config.file) if config.file else None, "groups": groups}


def apply(config: Config, changes: dict[str, Any]) -> dict[str, Any]:
    """Validate, apply what can be applied live, save everything. Returns a summary.

    A secret sent as an empty string is cleared: the page only sends secrets it changed."""
    unknown = [key for key in changes if key not in BY_KEY]
    if unknown:
        raise ValueError(f"Réglage inconnu : {', '.join(unknown)}")
    values = copy.deepcopy(config.file_values)
    updates = {}
    for key, value in changes.items():
        setting = BY_KEY[key]
        checked = setting.choices or setting.dynamic in ("gguf", "gguf_inherit")
        if checked and value not in choices_for(setting, config)[0]:
            raise ValueError(f"{setting.label} : valeur invalide")
        current = _get(config, key)
        if isinstance(current, float) and isinstance(value, int) and not isinstance(value, bool):
            value = float(value)
        if isinstance(value, str):
            value = value.strip()
        if isinstance(value, int | float) and not isinstance(value, bool):
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"{setting.label} : nombre fini attendu")
            if setting.minimum is not None and value < setting.minimum:
                raise ValueError(f"{setting.label} : minimum {setting.minimum:g}")
            if setting.maximum is not None and value > setting.maximum:
                raise ValueError(f"{setting.label} : maximum {setting.maximum:g}")
        _deep_update(values, _nest(key, value))
        updates[key] = value
    return _commit(config, values, updates)


def reset(config: Config, keys: list[str]) -> dict[str, Any]:
    """Back to the default: the key is removed from localive.toml (not written as a value)."""
    unknown = [key for key in keys if key not in BY_KEY]
    if unknown:
        raise ValueError(f"Réglage inconnu : {', '.join(unknown)}")
    defaults = Config()
    values = copy.deepcopy(config.file_values)
    updates = {}
    for key in keys:
        section, *rest = key.split(".")
        node = values.get(section)
        for part in rest[:-1]:
            node = node.get(part) if isinstance(node, dict) else None
        if isinstance(node, dict):
            node.pop(rest[-1], None)
        updates[key] = _get(defaults, key)
    return _commit(config, values, updates)


def external_model_url(backend: str, address: str) -> str:
    url = (address.strip() or BRAIN_SERVERS.get(backend, "")).rstrip("/")
    parsed = urlsplit(url)
    if (backend not in BRAIN_SERVERS or parsed.scheme not in {"http", "https"}
            or not parsed.hostname or parsed.username is not None or parsed.query
            or parsed.fragment or parsed.port == 0):
        raise ValueError("Cerveau : donnez l'adresse HTTP(S) du serveur de modèles.")
    return url


def _commit(config: Config, values: dict, updates: dict) -> dict[str, Any]:
    """Validate the complete next start, then publish to disk before changing live state."""
    candidate = Config(root=config.root, file=config.file)
    merge(candidate, values)
    if {"brain.model", "decision.model", "routing.model"} & updates.keys():
        for model in (candidate.brain.model, candidate.decision.model):
            if model and _routing_weights(candidate.path(model), candidate):
                raise ValueError("Ces poids servent au tri des outils et au raccrochage vocal.")
    backend = candidate.brain.backend
    if backend != "localive":
        external_model_url(backend, candidate.brain.external_url)
    elif config.brain.backend == "localive" and not config.brain.managed:
        candidate.brain.managed = False  # preserve the launcher's --external-brain override
    try:
        candidate.finalize()
    except FileNotFoundError as exc:
        raise ValueError(str(exc)) from exc

    # An unedited key belongs to the provider previously saved in the file.
    if candidate.llm.api_key and "llm.api_key" not in updates:
        previous = config.file_values.get("brain", {})
        previous_backend = previous.get("backend", "localive")
        previous_url = previous.get("external_url") or BRAIN_SERVERS.get(previous_backend, "")
        next_url = candidate.brain.external_url or BRAIN_SERVERS.get(backend, "")
        if (previous_backend, previous_url.rstrip("/")) != (backend, next_url.rstrip("/")):
            values["llm"].pop("api_key", None)
            updates["llm.api_key"] = ""
    values = {k: v for k, v in values.items() if v != {}}
    if config.file:
        path = config.file
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp",
                                         delete=False) as tmp:
            temp_path = Path(tmp.name)
        try:
            temp_path.write_text(to_toml(values), encoding="utf-8")
            os.replace(temp_path, path)
        finally:
            temp_path.unlink(missing_ok=True)
    config.file_values = values
    live = {k: v for k, v in updates.items() if not BY_KEY[k].restart}
    for key, value in live.items():
        merge(config, _nest(key, value))
    pending = [k for k, v in updates.items() if BY_KEY[k].restart and _get(config, k) != v]
    return {"applied": sorted(live), "restart_required": sorted(pending)}


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return repr(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    return json.dumps(str(value), ensure_ascii=False)  # JSON escapes are valid TOML


def to_toml(values: dict[str, Any]) -> str:
    lines = ["# Écrit par Localive (page d'administration). Voir localive.toml.example.", ""]

    def table(prefix: str, data: dict[str, Any]) -> None:
        scalars = {k: v for k, v in data.items() if not isinstance(v, dict)}
        if scalars:
            lines.append(f"[{prefix}]")
            lines.extend(f"{k} = {_toml_value(v)}" for k, v in scalars.items())
            lines.append("")
        for key, value in data.items():
            if isinstance(value, dict):
                table(f"{prefix}.{key}", value)

    for section, data in values.items():
        if isinstance(data, dict):
            table(section, data)
    return "\n".join(lines)


def reference_markdown() -> str:
    """docs/reglages.md: every setting of the admin page, as a reference (kept in sync by a
    test; regenerate with ``python -m localive.settings > docs/reglages.md``)."""
    defaults = Config()

    def shown(setting: Setting) -> str:
        if setting.secret:
            return "*(secret)*"
        value = _get(defaults, setting.key)
        if isinstance(value, bool):
            return "oui" if value else "non"
        if value == "":
            return "*(vide)*"
        return f"`{value}`" + (f" {setting.unit}" if setting.unit else "")

    lines = [
        "# Réglages",
        "",
        "Tous les réglages de la page d'administration de Localive "
        "(<http://127.0.0.1:8787>), avec leur valeur par défaut. Ils sont enregistrés dans "
        "`localive.toml` (section et nom indiqués), que l'on peut aussi modifier à la main. "
        "« Redémarrage » : la valeur s'applique au prochain démarrage de Localive.",
        "",
        "*Page générée à partir du code : `python -m localive.settings > docs/reglages.md`.*",
    ]
    for section in SECTIONS:
        lines += ["", f"## {section.title}", "", section.intro, "",
                  "| Réglage | Clé | Par défaut | Description |", "|---|---|---|---|"]
        for setting in section.settings:
            notes = [setting.help] if setting.help else []
            if setting.choices:
                notes.append("Choix : " + ", ".join(f"`{c}`" for c in setting.choices) + ".")
            if setting.minimum is not None and setting.maximum is not None:
                notes.append(f"De {setting.minimum:g} à {setting.maximum:g}"
                             + (f" {setting.unit}" if setting.unit else "") + ".")
            if setting.restart:
                notes.append("Redémarrage.")
            if setting.advanced:
                notes.append("Avancé.")
            description = " ".join(notes).replace("|", "\\|")
            lines.append(f"| {setting.label} | `{setting.key}` | {shown(setting)} | "
                         f"{description} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    sys.stdout.write(reference_markdown())
