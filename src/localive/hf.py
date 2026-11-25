"""Hugging Face access: one token, one cache, set up before any model library is imported.

Token, first match wins: ``HF_TOKEN`` / ``HUGGING_FACE_HUB_TOKEN`` environment variables (or a
Docker secret file named by ``HF_TOKEN_FILE``), ``[huggingface] token`` in localive.toml, then
the token saved by ``hf auth login``. Gated models (Phonon 40m) also need their terms accepted
once on huggingface.co with the token's account; after the first download, set
``[huggingface] offline = true`` and no token is needed any more.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from .config import Config

log = logging.getLogger(__name__)


class GatedModelError(RuntimeError):
    """``kind``: "terms" (conditions not accepted), "token" (missing or invalid token) or
    "download" (anything else: network, disk). The admin page guides the first two."""

    def __init__(self, message: str, kind: str = "download", repo: str = "") -> None:
        super().__init__(message)
        self.kind = kind
        self.repo = repo


def token(config: Config) -> str:
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if os.environ.get(name):
            return os.environ[name].strip()
    secret = os.environ.get("HF_TOKEN_FILE")
    if secret and Path(secret).is_file():
        return Path(secret).read_text(encoding="utf-8").strip()
    if config.huggingface.token:
        return config.huggingface.token.strip()
    saved = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "token"
    saved = saved if saved.is_file() else Path.home() / ".cache" / "huggingface" / "token"
    return saved.read_text(encoding="utf-8").strip() if saved.is_file() else ""


def configure(config: Config) -> str:
    """Export the token and cache location for huggingface_hub, ptts and friends."""
    value = token(config)  # read before HF_HOME moves: the saved login lives in the old one
    cache = config.path(config.huggingface.cache)
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache)
    if value:
        os.environ["HF_TOKEN"] = value
    if config.huggingface.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
    source = "aucun" if not value else "fourni"
    log.info("Hugging Face : cache %s, jeton %s%s", cache, source,
             ", hors ligne" if config.huggingface.offline else "")  # fmt: skip
    return value


def explain(error: Exception, repo: str) -> GatedModelError:
    """Turn a download failure into an actionable French message."""
    text = str(error)
    name = repo.removeprefix("hf://").split("@")[0]
    page = "https://huggingface.co/" + name
    authenticated = bool(os.environ.get("HF_TOKEN"))
    lowered = text.lower()
    if "401" in text or "authenticated" in lowered or "token" in lowered:
        if authenticated and ("gated" in lowered or "restricted" in lowered):
            return GatedModelError(
                f"Le compte Hugging Face du jeton n'a pas encore accepté les conditions de "
                f"{name} : acceptez-les sur {page}, puis redémarrez Localive.", "terms", name)
        return GatedModelError(
            f"La voix {name} demande un jeton Hugging Face : renseignez-le dans "
            f"l'administration de Localive (page d'accueil). Modèle : {page}",
            "token", name)
    if "gated" in lowered or "restricted" in lowered or "403" in text:
        return GatedModelError(
            f"Accès refusé à {name} : acceptez ses conditions sur {page} avec le compte du "
            "jeton Hugging Face, puis redémarrez Localive.", "terms", name)
    return GatedModelError(f"Téléchargement de {repo} impossible : {text}", "download", name)


async def check_access(http, token: str, repo: str) -> tuple[str, str]:
    """Can ``token`` download the gated ``repo``? ("ok" | "token" | "terms" | "unreachable",
    message in French)."""
    import aiohttp

    name = repo.removeprefix("hf://").split("@")[0]
    page = "https://huggingface.co/" + name
    headers = {"Authorization": f"Bearer {token.strip()}"} if token.strip() else {}
    try:
        # A gated repo lists its files publicly; reading one is what needs access.
        async with http.head(f"https://huggingface.co/{name}/resolve/main/config.json",
                             headers=headers, allow_redirects=False,
                             timeout=aiohttp.ClientTimeout(total=15)) as reply:
            status = reply.status
    except (aiohttp.ClientError, TimeoutError, OSError) as exc:
        return "unreachable", f"Hugging Face injoignable : {exc}"
    if status in (200, 302, 307):  # the file, or its redirect to the storage
        return "ok", "Jeton valide, conditions acceptées."
    if status == 401:
        return "token", "Jeton refusé par Hugging Face : vérifiez-le (il commence par hf_)."
    if status == 403:
        return "terms", (f"Jeton valide, mais les conditions de {name} ne sont pas "
                          f"acceptées : {page}")
    return "unreachable", f"Réponse inattendue de Hugging Face (HTTP {status})."
