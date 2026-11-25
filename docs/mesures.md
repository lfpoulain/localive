# Mesures

## Combien coûte le tri des outils sur le HX 470 ?

Avec 18 outils et d3-mini Q8_0 déjà chargé, **le tri prend de 500 à 649 ms** selon la longueur de l’historique. Le tableau donne la médiane de trois demandes par case (lumière, carburant, conversation), rejouées sur la Radeon 890M le 25 novembre 2026.

| Texte du tour précédent | Avant optimisation | Localive 0.3.1 |
|---|---:|---:|
| Aucun | 593 ms | 500 ms |
| 200 caractères | 669 ms | 572 ms |
| 450 caractères | 765 ms | 649 ms |

Le chronomètre couvre la première décision de d3 après transcription. La génération de Qwen, l’exécution des outils, la voix et une éventuelle confirmation de raccrochage ajoutent leur propre délai. Sur un autre catalogue de 18 outils, la médiane est passée de 593 à 492 ms sur 62 formulations. Qwen a reçu tous les outils nécessaires dans les 52 demandes qui en exigeaient.

## Quand envoyer directement les outils à Qwen ?

Sur une RTX 4090 dont la mémoire dédiée était déjà très occupée, une demande d’heure avec trois petits outils a donné son premier audio en **1,01 s en accès direct**, contre **2,28 s avec tri** (médianes de trois sessions, depuis la fin de parole). Localive transmet donc les petits catalogues directement. Il trie à partir de six outils, ou dès deux si leurs schémas totalisent au moins 3 072 octets.

## Mesurer votre installation

Avec l’environnement Python activé et le serveur prêt, envoyez le même WAV pour chaque réglage.

```sh
python scripts/live_client.py --url ws://127.0.0.1:8787/live --report mesures.json --save reponse.wav tests/fixtures/wake_calculation.wav
```

Dans le rapport, `audio_ready_latency_ms` va de la fin de parole estimée au premier audio prêt côté serveur. `tool_routing_ms` isole le tri. Pour comparer deux réglages de routage, utilisez le même catalogue d’outils et mesurez séparément le premier tour et les suivants.

[Profil matériel du HX 470](installation.md#radeon-890m-du-ryzen-ai-9-hx-470) · [Bancs de test](developpement.md) · [Rapport détaillé de la version 0.3.1](https://github.com/lfpoulain/localive/blob/81d1bde0e5535129cfd4e70f1d365e6a6c46241f/docs/mesures.md)
