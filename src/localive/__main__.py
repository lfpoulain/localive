"""Command line: ``python -m localive`` (or ``localive``)."""

from __future__ import annotations

import argparse
import logging
import logging.handlers
from pathlib import Path

from . import __version__
from .config import load_config
from .server import LOG_BUFFER, RESTART_EXIT_CODE, run

LOG_FILE = Path(".runtime/logs/localive.log")


def setup_logging(verbose: bool) -> None:
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S")
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    root.addHandler(console)
    root.addHandler(LOG_BUFFER)  # the admin page's "Journal"
    try:  # kept after the console window closes: what to send when something goes wrong
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        file = logging.handlers.RotatingFileHandler(LOG_FILE, maxBytes=2_000_000,
                                                    backupCount=3, encoding="utf-8")  # fmt: skip
        file.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                                            "%Y-%m-%d %H:%M:%S"))  # fmt: skip
        root.addHandler(file)
    except OSError:
        pass
    logging.getLogger("aiohttp.access").setLevel(logging.WARNING)
    for noisy in ("httpx", "httpx2", "httpcore", "huggingface_hub", "aioice", "aiortc"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="localive", description="Serveur vocal local pour Cornelius et vos clients Live."
    )
    parser.add_argument("-c", "--config", help="fichier TOML (défaut : ./localive.toml)")
    parser.add_argument("--host", help="adresse d'écoute")
    parser.add_argument("--port", type=int, help="port HTTP")
    parser.add_argument("--open", action="store_true", help="ouvrir le navigateur une fois prêt")
    parser.add_argument(
        "--external-brain",
        action="store_true",
        help="ne pas lancer llama-server : utiliser llm.url tel quel",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--version", action="version", version=__version__)
    args = parser.parse_args()
    setup_logging(args.verbose)
    log = logging.getLogger("localive")
    log.info("Localive %s", __version__)

    try:
        config = load_config(args.config)
        if args.host:
            config.server.host = args.host
        if args.port:
            config.server.port = args.port
        if args.external_brain:
            config.brain.managed = False
        restart = run(config.finalize(), open_browser=args.open)
    except (FileNotFoundError, ValueError, OSError) as exc:
        # Configuration or installation problem: one clear line, the detail in the log file.
        log.error("Localive ne peut pas démarrer : %s", exc)
        log.debug("Détail", exc_info=True)
        raise SystemExit(1) from exc
    if restart:
        # The admin page asked for a restart: the launch scripts start Localive again.
        raise SystemExit(RESTART_EXIT_CODE)


if __name__ == "__main__":
    main()
