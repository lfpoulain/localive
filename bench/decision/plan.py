"""While it talks and has more to say: finish then answer (AFTER), correct now (FIX), or switch
now (NOW)? Localive's own prompt against the running Qwen (:8081).

    .venv\\Scripts\\python.exe bench\\decision\\plan.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import aiohttp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from localive.config import Config  # noqa: E402
from localive.decision import Arbiter, Situation  # noqa: E402

WEEK = ("Lundi à Bordeaux, beau temps et 18 degrés.", "Mardi, nuageux, 16 degrés. Mercredi, "
        "averses. Jeudi, retour du soleil. Vendredi et samedi, doux. Dimanche, orages.")  # fmt: skip
CASES = [
    # (request, said, remaining, heard, accepted)
    ("La météo de Bordeaux sur sept jours ?", *WEEK, "Et Nantes ?", {"AFTER"}),
    ("La météo de Bordeaux sur sept jours ?", *WEEK, "Et pour Toulouse aussi.", {"AFTER"}),
    ("La météo de Bordeaux sur sept jours ?", *WEEK, "Non, je voulais dire Brest.", {"FIX"}),
    ("La météo de Bordeaux sur sept jours ?", *WEEK, "Juste le week-end.", {"FIX"}),
    ("La météo de Bordeaux sur sept jours ?", *WEEK, "Allume la lumière du salon.", {"NOW"}),
    ("Raconte-moi une histoire.", "Il était une fois un dragon qui vivait dans une montagne.",
     "Un jour, un chevalier arriva au village et décida de partir à sa recherche.",
     "Plus courte, s'il te plaît.", {"FIX"}),
    ("Raconte-moi une histoire.", "Il était une fois un dragon qui vivait dans une montagne.",
     "Un jour, un chevalier arriva au village et décida de partir à sa recherche.",
     "Il y a quelqu'un à la porte, montre-moi la caméra.", {"NOW"}),
    ("Donne-moi la recette des crêpes.", "Il te faut 250 grammes de farine et quatre œufs.",
     "Ajoute un demi-litre de lait petit à petit, puis laisse reposer une heure.",
     "Et c'est pour combien de personnes ?", {"AFTER"}),
    ("Donne-moi la recette des crêpes.", "Il te faut 250 grammes de farine et quatre œufs.",
     "Ajoute un demi-litre de lait petit à petit, puis laisse reposer une heure.",
     "Pour huit personnes en fait.", {"FIX"}),
    ("Donne-moi la recette des crêpes.", "Il te faut 250 grammes de farine et quatre œufs.",
     "Ajoute un demi-litre de lait petit à petit, puis laisse reposer une heure.",
     "Mets un minuteur de dix minutes.", {"NOW", "AFTER"}),
    ("Les horaires des trains pour Lyon demain ?", "Le premier part à 6 h 12, le suivant à 7 h 04.",
     "Ensuite il y en a un toutes les heures jusqu'à 21 h, sauf entre 13 h et 15 h.",
     "Et pour le retour ?", {"AFTER"}),
    ("Explique-moi la relativité.", "La relativité générale décrit la gravitation comme une "
     "courbure de l'espace-temps.", "Une masse déforme l'espace autour d'elle, et cette "
     "déformation guide le mouvement des objets.", "Laisse tomber, quelle heure est-il ?", {"NOW"}),
    ("Les nouvelles du jour ?", "Le gouvernement a présenté son budget ce matin.",
     "Dans le sport, le PSG a gagné hier soir, et côté culture, un nouveau festival s'ouvre.",
     "Et la météo aussi après.", {"AFTER"}),
    ("Les nouvelles du jour ?", "Le gouvernement a présenté son budget ce matin.",
     "Dans le sport, le PSG a gagné hier soir, et côté culture, un nouveau festival s'ouvre.",
     "Plutôt les nouvelles locales.", {"FIX"}),
]  # fmt: skip
# Not used to write the prompt (the cases above were): an honest check of generalisation.
TRAIN = (
    "Le premier part à 6 h 12, le suivant à 7 h 04.",
    "Ensuite il y en a un toutes les heures jusqu'à 21 h, sauf entre 13 h et 15 h.",
)
STORY = (
    "Il était une fois un dragon qui vivait dans une montagne.",
    "Un jour, un chevalier arriva au village et décida de partir à sa recherche.",
)
HELD_OUT = [
    ("La météo de Bordeaux sur sept jours ?", *WEEK, "Et à Rennes ?", {"AFTER"}),
    ("La météo de Bordeaux sur sept jours ?", *WEEK, "Seulement mercredi.", {"FIX"}),
    ("La météo de Bordeaux sur sept jours ?", *WEEK, "Ferme les volets de la chambre.", {"NOW"}),
    ("La météo de Bordeaux sur sept jours ?", *WEEK, "Donne-les en Fahrenheit.", {"FIX"}),
    ("Raconte-moi une histoire.", *STORY, "Avec une princesse plutôt.", {"FIX"}),
    ("Raconte-moi une histoire.", *STORY, "Le chat s'est échappé, ouvre la porte du jardin.", {"NOW"}),
    ("Raconte-moi une histoire.", *STORY, "Et après tu me racontes une blague.", {"AFTER"}),
    ("Les horaires des trains pour Lyon demain ?", *TRAIN, "Plutôt vendredi.", {"FIX"}),
    ("Les horaires des trains pour Lyon demain ?", *TRAIN, "Et le prix du billet ?", {"AFTER"}),
    ("Les horaires des trains pour Lyon demain ?", *TRAIN, "Seulement ceux de l'après-midi.", {"FIX"}),
]  # fmt: skip


async def main() -> None:
    config = Config()
    async with aiohttp.ClientSession() as http:
        arbiter = Arbiter(http, config)
        for title, cases in (
            ("Cas ayant servi à écrire la consigne", CASES),
            ("Cas nouveaux (jamais vus)", HELD_OUT),
        ):
            print(f"\n== {title}")
            ok = 0
            for request, said, remaining, heard, accepted in cases:
                situation = Situation(request=request, heard=heard, speaking=True, said=said)
                how, ms = await arbiter.plan(situation, remaining, False)
                ok += how in accepted
                print(f"{'OK' if how in accepted else '!!'} {how:5s} ({ms:3.0f} ms) attendu "
                      f"{'/'.join(sorted(accepted)):9s} « {heard} »")  # fmt: skip
                if how == "AFTER":
                    print(f"      annonce : « {await arbiter.acknowledge(heard)} »")
            print(f"{ok}/{len(cases)} justes")


if __name__ == "__main__":
    asyncio.run(main())
