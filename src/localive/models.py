"""Reasoning models that the admin page can download (page Cerveau).

The catalogue (``catalog.py``) lists official GGUF files by family; any other GGUF of a
public Hugging Face repository can be added by its name. Files are checked like the
installer's (size and SHA-256, resumable). Once a file is in models/, it appears in the
"Modèle de raisonnement" dropdown.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import urllib.error
import urllib.request

from . import install
from .catalog import BY_ID, CATALOG, Model
from .config import Config

log = logging.getLogger(__name__)

REPO = re.compile(r"^[A-Za-z0-9][\w.-]*/[\w.-]+$")
FILE = re.compile(r"^[\w.-]+\.gguf$", re.IGNORECASE)
SHARD = re.compile(r"-\d{5}-of-\d{5}\.gguf$", re.IGNORECASE)


def _repository(repo: str) -> str:
    repo = repo.strip()
    if not REPO.fullmatch(repo) or ".." in repo or repo.endswith("/."):
        raise ValueError("Dépôt attendu sous la forme auteur/nom (ex. unsloth/Qwen3.5-4B-GGUF).")
    return repo


def _language_model(file: str) -> bool:
    return bool(FILE.fullmatch(file) and "mmproj" not in file.lower() and not SHARD.search(file))


def custom_model(repo: str, file: str) -> Model:
    """A GGUF of any public repository, stored in models/<repository name>/."""
    repo, file = _repository(repo), file.strip()
    if not _language_model(file):
        if SHARD.search(file):
            raise ValueError("Choisissez un modèle GGUF en un seul fichier.")
        if "mmproj" in file.lower():
            raise ValueError("Ce fichier est un projecteur d'images, pas un modèle de langage.")
        raise ValueError("Choisissez un fichier .gguf de ce dépôt.")
    name = repo.split("/")[1].removesuffix("-GGUF").removesuffix("-gguf")
    return Model(f"custom:{repo}/{file}", "Autres", file.removesuffix(".gguf"), 0.0, repo,
                 repo, file, f"models/{name}")  # fmt: skip


def repo_files(repo: str) -> list[dict]:
    """Single-file language models at the repository root, sorted by filename."""
    url = f"https://huggingface.co/api/models/{_repository(repo)}/tree/main"
    request = urllib.request.Request(url, headers={"User-Agent": install.USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            tree = json.load(response)
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            raise ValueError("Dépôt privé ou à conditions : non pris en charge.") from exc
        raise ValueError(f"Dépôt introuvable sur Hugging Face ({exc.code}).") from exc
    except OSError as exc:
        raise ValueError(f"Hugging Face injoignable : {exc}") from exc
    files = []
    for entry in tree if isinstance(tree, list) else []:
        path = str(entry.get("path", ""))
        if _language_model(path):
            size = (entry.get("lfs") or {}).get("size") or entry.get("size") or 0
            files.append({"file": path, "size_gb": round(size / 1e9, 2)})
    if not files:
        raise ValueError("Aucun fichier GGUF à la racine de ce dépôt (les modèles découpés en "
                         "plusieurs fichiers ne sont pas pris en charge).")
    return sorted(files, key=lambda f: f["file"].lower())


class Downloads:
    """One download at a time, in a thread; its progress is read by the admin page."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.progress: dict[str, dict] = {}
        self.custom: dict[str, Model] = {}  # added by repository name, this run
        self._task: asyncio.Task | None = None

    def busy(self) -> bool:
        return self._task is not None and not self._task.done()

    def _row(self, model: Model, current: str) -> dict:
        return {
            "id": model.id, "family": model.family, "label": model.label,
            "size_gb": model.size_gb, "note": model.note, "path": model.path,
            "installed": self.config.path(model.path).is_file(),
            "selected": model.path == current,
            **self.progress.get(model.id, {"state": "idle"}),
        }  # fmt: skip

    def listing(self) -> list[dict]:
        current = self.config.brain.model
        return [self._row(m, current) for m in (*CATALOG, *self.custom.values())]

    def start(self, model_id: str = "", repo: str = "", file: str = "") -> None:
        if repo or file:
            model = custom_model(repo, file)
        else:
            model = BY_ID.get(model_id)
            if model is None:
                raise ValueError(f"Modèle inconnu : {model_id}")
        if self.busy():
            raise RuntimeError("Un téléchargement est déjà en cours.")
        if model.id not in BY_ID:
            self.custom[model.id] = model
        self.progress[model.id] = {"state": "downloading", "done": 0, "total": 0}
        self._task = asyncio.create_task(self._run(model))

    async def _run(self, model: Model) -> None:
        status = self.progress[model.id]

        def advance(done: int, total: int) -> None:  # called from the download thread
            status.update(done=done, total=total)
            if done >= total:
                status["state"] = "verifying"

        log.info("Téléchargement du modèle %s (%s)", model.label, model.file)
        try:
            await asyncio.to_thread(install.hub_file, model.repo, model.file,
                                    self.config.path(model.folder), advance)  # fmt: skip
        except Exception as exc:  # noqa: BLE001 - shown on the page, the user can retry
            log.warning("Téléchargement de %s échoué : %s", model.label, exc)
            status.update(state="error", error=str(exc))
            return
        status.update(state="done")
        log.info("Modèle %s téléchargé : il est dans la liste de la page Cerveau", model.label)
