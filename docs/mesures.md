# Mesures

Campagnes de mesures qui ont guidé les choix par défaut de Localive (modèles, décideur, fin de tour, réglages llama.cpp). Elles sont datées et liées à une machine : refaites-les chez vous avec les bancs de `bench/` (voir [Développement](developpement.md)).

## Tri conjoint des outils et des adieux sur HX 470, 25 novembre 2026

Sur le catalogue de 18 outils d’une session Cornelius, trois demandes (lumière, carburant,
conversation) ont été rejouées avec le même d3-mini Q8_0 chaud. Chaque demande a été envoyée
aux deux versions dans un ordre mélangé. Seule la longueur du texte du tour précédent varie
entre les lignes. Les valeurs sont les médianes de trois appels au routeur par cellule.

| Historique transmis | Format précédent | Libellés compacts |
|---|---:|---:|
| Aucun | 593 ms | 500 ms |
| 200 caractères | 669 ms | 572 ms |
| 450 caractères | 765 ms | 649 ms |

Ces durées couvrent la première décision de tri après transcription. Une éventuelle
confirmation de fin d’appel, Qwen, les outils et l’audio sont chronométrés séparément.
Les deux versions ont retenu les mêmes outils dans les neuf comparaisons.

Sur un second catalogue de 18 outils, les 62 formulations du banc historique donnent
593 ms de médiane avec le format précédent et 492 ms avec les libellés compacts.
Les outils requis sont restés disponibles dans les 52 demandes qui en nécessitaient.
Les deux formats ont choisi le calendrier pour la même demande ambiguë.

## Audit du parcours vocal `/live`, 23 novembre 2026

Ces mesures précèdent l’ajout des seuils automatiques de taille et de nombre d’outils.
Le banc utilise un i9-13900K, une RTX 4090 avec llama.cpp Vulkan, Parakeet et Phonon
sur CPU. Il envoie trois fois le même WAV de calcul dans une session sans outil, avant et
après les changements. Les deux versions répondent « 7 fois 6 font 42 » aux trois tours.
Les délais ci-dessous sont mesurés par le serveur.

| Même demande de calcul | Avant | Après |
|---|---:|---:|
| Premier audio après la fin de parole estimée, par tour | 713 / 871 / 1 302 ms | 870 / 1 366 / 837 ms |
| Médiane de ce délai | 871 ms | 870 ms |
| Fin de parole → transcription, médiane | 480 ms | 448 ms |
| Transcription → premier audio, médiane | 391 ms | 438 ms |
| Jetons du message système avec la même consigne client, selon Qwen | 242 | 94 |
| Routeur chargé pendant la session | d3-mini | aucun |

La dispersion sur trois tours dépasse largement l’écart entre les médianes. Le gain
vérifié porte sur le préfixe et la mémoire. En démarrant un serveur utilisé avec zéro ou
un outil par session, les allocations Vulkan des deux moteurs Qwen représentent environ
7,21 Gio sur cette carte. Charger d3-mini ajoute 4,66 Gio. Après son premier chargement,
le routeur reste en mémoire jusqu’à l’arrêt du serveur. La mémoire totale du processus,
du système et des services vocaux est plus élevée.

Silero partage désormais ses deux graphes ONNX entre les connexions. Sur 24 paires de
constructions du détecteur, la médiane passe de 3,25 ms à moins de 0,01 ms après le
chargement initial. L’état audio reste propre à chaque session. Parakeet peut aussi
réutiliser sa première transcription si une phrase jugée inachevée atteint la limite de
silence avec exactement le même extrait audio. Une reprise de parole change cet extrait.
Sur six WAV, limiter le silence final présenté à Parakeet à 128 ms au lieu de 224 ms a
donné les mêmes mots six fois, avec deux différences de ponctuation.

Le client d’exemple neutre a reçu une réponse parlée via `/live`. Le profil de démonstration
Cornelius a appelé `clock__read` et dit l’heure renvoyée par le client via `/v1/live/sessions`.
Sur le premier appel, effectué pendant le chargement de d3-mini, Qwen a reçu les trois
outils disponibles.

Pour comparer le catalogue complet et le tri déjà chargé, trois sessions Cornelius
indépendantes par mode ont ensuite demandé l’heure, avec la même consigne et trois
outils déclarés. Chacune a appelé l’horloge et lu son résultat.

| Médiane des trois premières demandes | Catalogue complet | d3-mini déjà chargé |
|---|---:|---:|
| Schémas transmis à Qwen | 3 | 1 |
| Tri des outils | 0 ms | 1 235 ms |
| Fin de parole → transcription | 384 ms | 416 ms |
| Transcription → premier audio prêt | 625 ms | 1 875 ms |
| Fin de parole → premier audio prêt | **1 009 ms** | **2 275 ms** |

La RTX 4090 avait environ 15,6 Gio de mémoire dédiée déjà occupés hors du banc. Avec
la pile complète, Windows attribuait au routeur environ 585 Mio de mémoire GPU dédiée
et 5 009 Mio de mémoire GPU partagée. Dans ces conditions, transmettre directement trois
petits schémas a été plus rapide.

Lorsque la même session a reposé la question deux fois, Qwen a réutilisé l’heure du
premier appel malgré la consigne de la relire. Les trois réponses de chaque série ont
eu lieu dans la même minute et étaient exactes. La sélection d’un schéma ne garantit
donc pas une nouvelle exécution de sa fonction à chaque tour.

Le tri a aussi été mesuré sur un catalogue de serre inédit. Parmi 11 demandes, d3-mini
a fait 10 sélections exactes (32 ms de médiane, 126 ms au 95ᵉ percentile). Un
`session.update` a remplacé l’éclairage par la ventilation, puis le ventilateur a été
sélectionné dès la demande suivante. Le routeur a omis le capteur de température de
l’air dans une demande combinant température et humidité du terreau. Sur les 20 contrôles
Cornelius, il a fait 14 sélections exactes et conservé chaque outil nécessaire aux 15
demandes qui en exigeaient (37 ms de médiane, 165 ms au 95ᵉ percentile). Ces délais
mesurent uniquement le routage après transcription.

### Effet des seuils sur les catalogues du dépôt

| Catalogue | Outils disponibles | Taille de la liste normalisée, JSON compact UTF-8 | Chemin automatique |
|---|---:|---:|---|
| Client WAV de démonstration, recherche désactivée | 3 | 777 octets | Qwen directement |
| Même client avec recherche web | 4 | 1 156 octets | Qwen directement |
| Configuration Cornelius de `bench/prompts/cornelius_session.json` | 5 | 3 197 octets | d3-mini |
| Même configuration avec recherche web | 6 | 3 576 octets | d3-mini |

Les seuils de 6 outils ou 3 072 octets, à partir de 2 outils, sont des repères
pour éviter le coût du tri sur les petits catalogues. Le gain de latence dépend
de la machine, de la disponibilité de la mémoire et de l’état du cache de Qwen.

Quatre tours vocaux dans la même session `/live` ont vérifié les deux critères sur
RTX 4090. Avec 3 outils et 777 octets, Qwen a appelé l’horloge sans charger d3-mini.
Avec 16 outils et 2 568 octets, d3-mini a retenu uniquement la météo pour Paris.
Avec 3 outils et 13 026 octets, il a écarté les outils pour « 7 fois 6 » et Qwen
a répondu 42. Au retour au catalogue initial, les trois outils ont été transmis
directement à Qwen alors que d3-mini était toujours chargé.

### Refaire la mesure sur un HX 470

Utilisez le [profil Radeon 890M](installation.md#radeon-890m-du-ryzen-ai-9-hx-470).
Pour reproduire le catalogue de trois fonctions du banc, réglez `search.engine = "off"`.
Le profil de démonstration propose également `web_search` quand la recherche est activée.
Dans les deux cas il reste sous les seuils de tri. Pour comparer le tri et le catalogue
complet, utilisez votre application avec au moins six outils ou au moins 3 072 octets
de schémas répartis entre plusieurs outils.
Depuis la machine cliente, avec l’environnement Python activé et le serveur prêt,
remplacez l’hôte de ces commandes par l’adresse de votre serveur si nécessaire.

```sh
python scripts/live_client.py --url ws://127.0.0.1:8787/live --report public.json --save calcul.wav tests/fixtures/wake_calculation.wav
python scripts/live_client.py --url ws://127.0.0.1:8787/live --profile cornelius --report outils.json --save heure.wav tests/fixtures/wake_clock.wav
```

Passez plusieurs WAV à la suite pour mesurer les tours suivants dans la même session.
Pour comparer deux réglages, gardez les mêmes WAV et le même catalogue, relevez le premier
tour séparément, puis comparez médiane et 95ᵉ percentile sur plusieurs dizaines de tours.
Vérifiez aussi que chaque demande appelle le bon outil. Dans le rapport, `endpoint_ms`
mesure le temps entre fin de parole estimée et transcription, `first_audio_ms` le temps
suivant jusqu’au premier audio prêt, et `audio_ready_latency_ms` le délai total côté
serveur. `tool_routing_ms` isole le tri des outils. Le transport et la lecture par le
haut-parleur s’ajoutent côté client. Les mesures Radeon historiques du dépôt utilisent
un HX 370. La latence du HX 470 reste à mesurer sur ce processeur.

## d3-mini et Aplomb 2 sur les requêtes vocales, 23 novembre 2026

Les candidats reprennent les 34 transcriptions françaises et les six outils de la comparaison ci-dessous.
Les 20 contrôles comprennent 15 demandes avec outil, trois conversations et deux ambiguïtés.
Le délai mesuré commence après la transcription. Le serveur de test est un i9-13900K limité à six threads par moteur.

### Routage sur CPU

| Routeur et format | Poids sur disque | Médiane de routage | P95 | Sélections exactes /20 | Outil requis oublié /15 | Catalogue complet /20 | Octets de schémas économisés |
|---|---:|---:|---:|---:|---:|---:|---:|
| d1-Omni-2B Q8_0 (référence) | 1,85 Go | 1,70 s | 2,04 s | 9 | 0 | 9 | 42,5 % |
| d3-lite Q8_0 | 1,08 Go | 0,78 s | 0,84 s | 8 | 1 | 11 | 37,3 % |
| d3-mini Q8_0 | 5,16 Go | 5,20 s | 8,00 s | 14 | 0 | 6 | 59,5 % |
| Aplomb 2 FP32, requêtes espacées après expiration | 9,10 Go | 8,00 s | 8,00 s | 6 | 0 | 14 | 25,6 % |
| Aplomb 2 FP32 en rafale | 9,10 Go | 8,00 s | 8,00 s | 3 | 0 | 17 | 13,1 % |

d3-mini a retenu uniquement l’horloge sur les deux formulations de l’heure et de la date
testées avec Qwen. Il a aussi écarté les six outils pour les trois demandes conversationnelles.
Les six retours au catalogue complet se répartissent entre les deux demandes de carte,
les deux ambiguïtés et les deux combinaisons horloge/bus. Pour ces deux dernières,
Localive a atteint le délai de 8 s et envoyé les six outils. Sur les 14 demandes de mise
au point, d3-mini a obtenu 12 sélections exactes, avec un dépassement du délai sur la
combinaison horloge/météo. [Poids d3-mini mesurés](https://huggingface.co/ggml-org/vllm-d3-mini-GGUF/tree/8b29fd3b87373ea85fa59cc2f26251c1591d771a).

Avec Qwen3-4B Q4_K_M sur CPU, un cache neuf et un passage par question, d3-mini a
permis d’émettre `clock__read` en 38,29 s sur « Quelle heure est-il ? », dont 4,64 s
de routage. Le même banc donnait 57,24 s avec les six outils, 34,20 s avec Omni
et 33,19 s avec d3-lite. Sur « On est le combien, déjà ? », d3-mini a sélectionné
l’horloge en 4,99 s. Qwen a répondu en 41,32 s avec la date exacte fournie par le
message système, sans appeler l’outil.

Aplomb 2 standard utilise ici le chemin texte court publié par EmpirioLabs avec
PyTorch 2.13 et Transformers 5.19. Le checkpoint et les têtes de décision occupent
9,10 Go sur disque (10,65 Go pour le dossier local avec les fichiers audio).
Localive envoie au plus huit questions par appel. Le lanceur de
mesure utilise le chemin prévu pour cette taille et laisse de côté l’encodeur audio et
le second checkpoint, destiné aux groupes d’au moins 17 questions. Avec les poids
BF16 sur six threads CPU, Aplomb a sélectionné l’horloge seule en 67,97 s sur
« Quelle heure est-il ? » et en 64,27 s sur « On est quel jour de la semaine ? ».
Ces deux mesures après préchauffage utilisent un délai prolongé pour observer la
décision du modèle. [Révision d’Aplomb 2 mesurée](https://huggingface.co/empiriolabsai/aplomb-2/tree/b16bcad26f5464615b8110d2d5b675b436393af5).

Avec le délai automatique de Localive, le parcours réel BF16 sur « Quelle heure est-il ? »
a consacré 8,00 s au routeur, transmis les six outils et émis l’appel d’horloge en
78,86 s au total. Le serveur Python synchrone peut continuer à calculer après
l’expiration de la requête HTTP. Le chronométrage a commencé après la fin de la
requête de préchauffage.

Sur ce CPU, le calcul FP32 recommandé par l’éditeur a ramené les deux décisions
simples à 5,58 et 5,59 s. Les fichiers BF16 sont convertis en mémoire pour ce calcul.
En enchaînant les 34 demandes, l’expiration d’un appel a également retardé des demandes
suivantes. Sur les 20 contrôles en rafale, 15 appels au routeur ont échoué.
Un second passage a attendu la fin effective de chaque calcul expiré avant le tour
suivant. Cette attente est hors chronométrage. Même dans ces conditions, 13 des
20 contrôles ont dépassé le délai. Les sept décisions arrivées à temps ont pris
entre 6,56 et 7,98 s. Sur le parcours complet « Quelle heure est-il ? », Aplomb
FP32 a sélectionné l’horloge seule en 7,23 s et Qwen a émis l’appel en 38,40 s.

### Routage sur RTX 4090

| Routeur | Médiane | P95 | Sélections exactes /20 | Outil requis oublié /15 | Catalogue complet /20 | Octets de schémas économisés |
|---|---:|---:|---:|---:|---:|---:|
| d3-mini Q8_0, llama.cpp Vulkan | 38 ms | 171 ms | 14 | 0 | 4 | 62,3 % |
| Aplomb 2 BF16, PyTorch CUDA | 120 ms | 285 ms | 14 | 0 | 5 | 62,3 % |

Les deux moteurs ont servi les 34 demandes sans expiration avec le délai automatique.
Les 33 couches de d3-mini étaient chargées sur le GPU. Sur les deux demandes qui
combinent horloge et bus, d3-mini a retenu les deux outils requis ainsi que la recherche
web et la carte. Aplomb a renvoyé exactement les deux outils pour la date et les
prochains bus. Il a aussi proposé la recherche et la carte pour l’autre formulation.
Le banc conserve les mêmes instructions et seuils de confiance sur CPU et GPU.

Avec Qwen sur CPU, d3-mini sur GPU a retenu l’horloge seule en 31 ms et l’appel a été
émis en 31,28 s. Avec Aplomb sur le même GPU, ces durées ont été de 125 ms et 32,46 s.
Chaque parcours a été chronométré une fois. Pour « On est le combien, déjà ? », Aplomb a aussi retenu
l’horloge seule (141 ms). Qwen a utilisé la date exacte inscrite dans le message
système et répondu en 32,87 s sans exécuter d’appel.

### Empreinte du profil GPU par défaut

Les trois processus llama.cpp b11960 ont démarré ensemble avec le profil Vulkan
de Localive sur RTX 4090. Le tableau additionne les buffers Vulkan déclarés dans
le journal de chaque moteur (poids, cache et calcul).

| Moteur | Contexte | Couches sur GPU | Buffers Vulkan |
|---|---:|---:|---:|
| Qwen3-4B pour la conversation | 24 576 | 37/37 | 4,21 Gio |
| Qwen3-4B pour les interruptions | 8 192 | 37/37 | 3,00 Gio |
| d3-mini Q8_0 pour les outils | 4 096 | 33/33 | 4,66 Gio |
| **Total** | | | **11,88 Gio** |

« Quelle heure est-il ? » a sélectionné uniquement `clock__read` parmi les six outils.
Sur un GPU intégré, la RAM doit aussi accueillir le système, les allocations CPU,
la transcription et la voix. Les mesures de Qwen sur Radeon 890M figurent dans
la campagne du 5 novembre ci-dessous.

### Reproduire les mesures d’Aplomb

Pour reproduire le routage texte avec Aplomb sous Windows, lancer le serveur puis le
banc dans deux terminaux. Le premier utilise un environnement Python séparé avec
PyTorch 2.13.0+cpu, torchvision 0.28.0+cpu, Transformers 5.19.0, Accelerate 1.15.0
et safetensors 0.8.0. Le second utilise l’environnement de Localive.
Pour la mesure GPU, utiliser les roues PyTorch et torchvision CUDA 13.0
dans l’environnement du serveur, puis passer `--device cuda --dtype bfloat16`
au lanceur. Qwen conserve son moteur CPU à six threads.

```powershell
.runtime\aplomb-bench-env\Scripts\python.exe bench/prompts/serve_aplomb_text.py --model models/decision/aplomb-2 --device cpu --dtype float32 --threads 6 --port 18857
.venv\Scripts\python.exe bench/prompts/route_tools.py --existing-router aplomb-2 --port 18857 --timeout-ms 0 --output bench/results/aplomb-2-cpu.json
.venv\Scripts\python.exe bench/prompts/route_tools.py --existing-router aplomb-2 --port 18857 --timeout-ms 0 --wait-for-idle --output bench/results/aplomb-2-cpu-isolated.json
.venv\Scripts\python.exe bench/prompts/live_tool_latency.py --profile routed --existing-router aplomb-2 --port 18175 --router-port 18857 --timeout-ms 0 --details
```

## Comparaison CPU de d3-lite, d3-nano et des deux d1-2B, 23 novembre 2026

Les quatre modèles ont classé les mêmes 34 demandes françaises avec les six outils de Cornelius.
Le tableau porte sur les 20 contrôles, composés des dix contrôles existants et de dix nouvelles
formulations fixées avant le premier passage. Quinze demandes nécessitent un ou plusieurs
outils, trois autorisent une réponse conversationnelle directe et deux appellent une clarification.
Les 14 demandes qui avaient guidé la mise au point de d1 sont comptabilisées séparément.

| Routeur Q8_0 | Fichier (Go) | Médiane | 95ᵉ percentile | Sélections exactes /20 | Demandes couvertes /20 | Outil requis oublié /15 | Catalogue complet /20 | Octets de schémas économisés |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| d3-lite | 1,08 | 779 ms | 844 ms | 8 | 18 | 1 | 11 | 37,3 % |
| d3-nano | 2,55 | 1 741 ms | 1 900 ms | 11 | 18 | 1 | 6 | 56,7 % |
| d1-2B standard | 1,85 | 1 540 ms | 1 697 ms | 11 | 17 | 1 | 6 | 55,0 % |
| d1-Omni-2B | 1,85 | 1 697 ms | 2 039 ms | 9 | 16 | 0 | 9 | 42,5 % |
| Six outils systématiquement | — | — | — | 0 | 17 | 0 | 20 | 0 % |

Une sélection exacte contient uniquement les outils attendus. Pour une conversation ou une
clarification, la sélection attendue est vide. Une demande avec outil est couverte dès que
tous les outils nécessaires sont disponibles. Pour une ambiguïté, le critère de couverture
accepte également le catalogue complet. Les économies comparent les octets de schémas JSON
transmis au volume des six outils à chaque tour. Le 95ᵉ percentile est la 19ᵉ valeur des
vingt durées triées. La valeur maximale de d3-nano atteint 5,67 s sur une demande composée.

Sur « Donne-moi la date du jour et les deux prochains bus de mon arrêt », les trois candidats
ont sélectionné uniquement le bus. Omni a gardé les six outils, dont l’horloge. En revanche,
sur les trois demandes conversationnelles, d3-nano a écarté tous les outils à chaque fois.
Omni a transmis le catalogue complet les trois fois. Sur « On est le combien, déjà ? »,
d3-nano a retenu uniquement l’horloge et les trois autres ont gardé le catalogue entier.

Sur les 14 demandes de mise au point, d3-lite et d1 standard en couvrent 13, d3-nano et
Omni en couvrent 14. Aucun appel au routeur n’a échoué sur les 136 demandes chronométrées.
L’évaluation utilise un i9-13900K, le moteur CPU llama.cpp b11960, six threads, des poids
Q8_0 et un serveur dédié par modèle. Chaque série suit le même ordre, après une question
de préchauffage distincte. Les seuils de sélection et le délai automatique de Localive
sont identiques pour tous. Chaque durée porte sur une décision ou, si nécessaire, plusieurs
décisions pour traiter une demande composée.

Les trois nouveaux fichiers sont les GGUF texte Q8_0, vérifiés par SHA-256. Les révisions
figurent dans les dépôts de [d3-lite](https://huggingface.co/ggml-org/vllm-d3-lite-GGUF/tree/0d544132bc35075d6346013f56454ea6dc19e640),
[d3-nano](https://huggingface.co/ggml-org/vllm-d3-nano-GGUF/tree/f2256dfaff5b8337e8bf67ca064aeef18ee14a79)
et [d1-2B](https://huggingface.co/LiquidAI/d1-2B-GGUF/tree/4a181094d1b6addfaa3a0956f78e7ab2aa406a0c).

### Jusqu’à l’appel d’horloge complet

Chaque cellule donne le temps écoulé depuis la transcription jusqu’à l’émission de
`clock__read` par Qwen3-4B-Instruct-2507 Q4_K_M. Le nombre d’outils présentés
à Qwen figure entre parenthèses. La synthèse vocale et l’exécution de l’horloge
sont hors mesure.

| Routeur | « Quelle heure est-il ? » | « On est le combien, déjà ? » |
|---|---:|---:|
| Catalogue complet | 57,24 s (6) | 54,25 s (6) |
| d3-lite | 33,19 s (1) | 61,33 s (6) |
| d3-nano | 38,95 s (1) | Aucun appel, réponse en 35,59 s (1) |
| d1-2B standard | 33,85 s (1) | 56,56 s (6) |
| d1-Omni-2B | 34,20 s (1) | 57,59 s (6) |

Qwen a démarré avec un cache de conversation neuf à chaque passage. Sa température
était nulle et les deux moteurs utilisaient chacun six threads CPU. Chaque couple
modèle/question a été mesuré une fois dans ce tableau et l’ordre des modèles a été
inversé pour la deuxième question. Les écarts entre deux parcours comprennent aussi
la variation du temps de Qwen. Sur la première phrase, d3-lite gagne 24,06 s face au
catalogue complet et 1,02 s face à Omni. Sur la deuxième, les trois routeurs qui
gardent tout le catalogue mettent plus de temps que le témoin.

Un onzième passage a permis de vérifier le cas d3-nano sur la date. Le seul outil
présenté à Qwen était bien `clock__read`. Qwen a terminé en 35,44 s sans l’appeler
et a donné le lundi 23 novembre 2026. Le message système lui fournissait déjà cette
date pour interpréter les recherches. La réponse était exacte et le critère d’appel
à l’horloge a échoué sur les deux passages.

### Reproduire la comparaison

Le banc peut redémarrer un moteur CPU privé et écrire la sélection et le temps de chaque
requête. Les exemples ci-dessous utilisent d3-lite depuis la racine du dépôt sous PowerShell.

```powershell
.venv\Scripts\python.exe bench/prompts/route_tools.py --port 18162 --model models/decision/vllm-d3-lite-Q8_0.gguf --timeout-ms 0 --output bench/results/routing-2026-11-23/vllm-d3-lite.json
.venv\Scripts\python.exe bench/prompts/live_tool_latency.py --profile full --port 18171 --question "On est le combien, déjà ?"
.venv\Scripts\python.exe bench/prompts/live_tool_latency.py --profile routed --router-device cpu --router-model models/decision/vllm-d3-lite-Q8_0.gguf --timeout-ms 0 --port 18171 --router-port 18172 --question "On est le combien, déjà ?"
```

## Sélection des outils de Cornelius, 23 novembre 2026

**Sur la question « Quelle heure est-il ? », sélectionner l’outil ramène le délai de
57,67 s à 35,24 s sur CPU.** Avec le classement sur GPU, le délai atteint 34,06 s.
Qwen3-4B-Instruct-2507 Q4_K_M tournait sur un Core i9-13900K limité à six threads.
Liquid d1-Omni-2B Q8_0 classait les outils sur le même CPU ou sur une RTX 4090 avec
Vulkan. La session Cornelius du dépôt proposait six outils. Chaque parcours utilisait
llama.cpp b11960 et commençait avec les modèles chargés et le prompt de conversation
absent du cache de Qwen.

### De la transcription à l’appel d’outil

| Parcours | Outils présentés à Qwen | Sélection | Premier fragment de texte ou d’appel | Appel complet reçu par le client |
|---|---:|---:|---:|---:|
| Catalogue complet | 6 | — | 57,08 s | 57,67 s |
| Sélection d1 sur GPU | 1 | 78 ms | 33,52 s | 34,06 s |
| Sélection d1 sur CPU, six threads | 1 | 1 672 ms | 34,71 s | 35,24 s |

Le chronomètre démarre à l’entrée de la transcription dans la session Live. Le premier
fragment contient du texte, un nom de fonction ou des arguments. L’événement suivant
mesuré contient l’appel `clock__read` complet dans les trois parcours. Le gain atteint
**23,61 s** avec d1 sur GPU et **22,43 s** avec d1 sur CPU, sélection comprise.
Le passage entièrement CPU utilisait un budget de 8 000 ms et le même témoin sans routeur.
Le tableau donne un passage par configuration. L’appel est intercepté avant son exécution.
La reconnaissance vocale, le résultat de l’horloge et la synthèse restent hors chronométrage.

Un second passage entièrement sur CPU, avec le délai automatique livré par défaut, a donné
**54,01 s avec six outils et 33,68 s avec l’horloge seule**. Le classement a pris 1,56 s,
compris dans les 33,68 s. Sur les deux comparaisons CPU, le gain est de 20,33 à 22,43 s.

### Coût des schémas pour Qwen

Le banc fournit directement la sélection attendue pour mesurer son effet sur Qwen.
Avec les consignes compactes, la question d’heure occupe **3 061 jetons avec six outils
et 2 183 avec la seule horloge**, soit 878 jetons de moins. Avec les anciennes consignes
et le catalogue complet, la même question occupait 3 438 jetons.

| Machine et état du cache | Catalogue complet | Sélection fournie | Premier événement mesuré |
|---|---:|---:|---|
| CPU, préfixe froid | 54,38–57,14 s | 31,98–32,10 s | Heure |
| CPU, préfixe chaud | 1,31–1,36 s | 1,03–1,04 s | Reformulation sur l’heure |
| RTX 4090 Vulkan, préfixe froid | 293 ms | 213 ms | Heure |
| RTX 4090 Vulkan, préfixe chaud | 66 ms | 62 ms | Reformulation sur l’heure |
| RTX 4090 Vulkan, changement heure → météo | 68 ms | 74 ms | Météo |
| RTX 4090 Vulkan, changement météo → remerciement | 19 ms | 17 ms | Texte |

Les intervalles CPU donnent les deux passages, avec l’ordre des profils inversé.
Qwen a émis `clock__read` dans les huit demandes d’heure. Le GPU a fait un passage par
profil. Changer le schéma d’horloge pour le schéma météo a fait relire 159 jetons à Qwen,
contre 17 en conservant le catalogue. À chaud sur cette carte, le classement peut donc
allonger le tour. Ces demandes isolent la réutilisation du préfixe avec les mêmes consignes,
sans accumulation d’historique. Le délai jusqu’au premier son dépend aussi de la suite
de la génération, de la réponse de l’outil et de la synthèse vocale.

### Fiabilité du classement d1 sur GPU

Le décideur reçoit la transcription, une description courte des outils et, pour un
suivi, le contexte précédent. Sur la RTX 4090, avec un budget de 400 ms, sa durée
médiane était de **17 ms**. Une demande avec outil réussit si la sélection garde tous
les outils nécessaires. Un catalogue complet compte donc comme utilisable dans ces cas.
Pour une conversation ordinaire, le critère est une sélection vide. Pour la demande
ambiguë du contrôle, une sélection vide ou le catalogue complet sont admis.

| Jeu de demandes françaises | Critère satisfait | Catalogue complet retenu | Économie sur les octets de schémas |
|---|---:|---:|---:|
| Développement | 14/14 | 3/14 | 63 % |
| Formulations indépendantes | 8/10 | 5/10 | 44 % |

Deux demandes indépendantes illustrent des erreurs différentes. « Invente une devinette
pour un enfant sur un escargot. » conserve les six outils, alors que Qwen pouvait répondre
directement. « Ça commence à quelle heure ? » sélectionne `clock__read`, qui donne
l’heure actuelle. Il fallait d’abord demander de quel événement il s’agissait.

Les pourcentages comparent le volume UTF-8 des schémas JSON transmis sur chaque jeu
avec le volume qu’auraient pris les six outils à chaque demande. La qualité de la
réponse finale et la taille totale du prompt relèvent de mesures distinctes. Ces
24 demandes emploient un seul catalogue.

### Classement d1 sur CPU

Sur le même i9-13900K limité à six threads, le budget de 400 ms a expiré pour les
24 demandes. Localive a alors conservé le catalogue complet. Avec un budget de
8 000 ms, d1 a retrouvé les mêmes sélections que sur GPU. La médiane atteint
**1 801 ms** sur les 14 demandes de développement et **1 682 ms** sur les dix demandes
indépendantes. Le classement d’une demande composée a pris **4 316 ms**. Sur cette
machine, ce coût pèse davantage sur les tours dont le préfixe Qwen est déjà en cache.
Avec le délai automatique livré par défaut, les 24 demandes ont donné les mêmes sélections.
Les deux médianes étaient respectivement de 1 584 et 1 835 ms sur ce passage.

### Reproduire

Depuis le dépôt et son environnement Python, les commandes suivantes mesurent Qwen
sur un port libre et arrêtent leur propre processus à la fin. Le banc CPU demande un
binaire llama.cpp pour CPU dans le répertoire indiqué.

```powershell
.venv\Scripts\python.exe bench/prompts/tool_prefill.py --llama-dir .runtime/llama/b11960-cpu --gpu-layers 0 --threads 6 --clock-only
.venv\Scripts\python.exe bench/prompts/tool_prefill.py --llama-dir .runtime/llama/b11960-cpu --gpu-layers 0 --threads 6 --clock-only --selected-first
.venv\Scripts\python.exe bench/prompts/tool_prefill.py --llama-dir .runtime/llama/b11960-vulkan --gpu-layers 99 --threads 6
```

Pour le classement sur GPU, démarrer Localive avec d1 installé et la sélection activée,
puis lancer le banc sur le port du décideur. Chaque ligne indique les outils retenus.
[Installation et réglages du décideur](installation.md)

```powershell
.venv\Scripts\python.exe bench/prompts/route_tools.py --port 8085 --timeout-ms 400
```

Pour le contrôle CPU, lancer d1 dans un terminal sur un port libre, puis la seconde
commande dans un autre. Arrêter l’instance de mesure avec Ctrl+C après le passage.

```powershell
.runtime\llama\b11960-cpu\llama-server.exe -m models/decision/d1-Omni-2B-Q8_0.gguf --host 127.0.0.1 --port 18092 -c 4096 -b 2048 -ub 2048 -ngl 0 --parallel 1 --threads 6 --threads-batch 6 -fa auto -ctk f16 -ctv f16 --no-webui
.venv\Scripts\python.exe bench/prompts/route_tools.py --port 18092 --timeout-ms 8000
```

Pour reproduire le parcours Live, exécuter la première commande pour un A/B entièrement
sur CPU, ou la seconde pour placer d1 sur Vulkan. Le banc démarre ses propres moteurs
sur les ports locaux 18091 et 18092. Il charge la session Cornelius et redémarre Qwen
entre la mesure avec sélection et la mesure témoin. Les poids sont préchauffés, le
prompt Cornelius est froid au début de chaque mesure. Chaque ligne JSON donne le nombre
d’outils retenus, le temps de classement et le délai entre la transcription fournie et
l’appel complet `clock__read`, intercepté en mémoire. Les ports se règlent avec
`--port` et `--router-port`.

```powershell
.venv\Scripts\python.exe bench/prompts/live_tool_latency.py --router-model models/decision/d1-Omni-2B-Q8_0.gguf --router-device cpu --timeout-ms 8000
.venv\Scripts\python.exe bench/prompts/live_tool_latency.py --router-model models/decision/d1-Omni-2B-Q8_0.gguf --router-device vulkan --timeout-ms 400
```

## Machine de référence et latences, 5 novembre 2026


| Rôle | Modèle | Mesuré sur Ryzen AI 9 HX 370, le 5 novembre 2026 |
|---|---|---|
| Oreilles | `moondream/parakeet-ultra` via Photon, CPU | 119–176 ms ; 3 extraits de 2,4 à 3,8 s, 3 passages chacun |
| Cerveau | Qwen3-4B-Instruct-2507 Q4_K_M, Vulkan | premier jeton 125 ms sur deux demandes simples ; 750 ms sur une demande avec outil |
| Voix | `gradium/phonon-40m-fr` via `ptts`, CPU | premier bloc 20–28 ms ; calcul ≈ 10 % de la durée audio |
| Fin de tour | `pipecat-ai/smart-turn-v3` (v3.2), FP32 CPU | médiane 32 ms avec 2 threads, contre 59 ms avec 1 |

Machine mesurée : **Ryzen AI 9 HX 370, 12 cœurs / 24 threads, Radeon 890M Vulkan,
23,6 Gio de RAM utilisable**. Ces mesures sont distinctes des essais RTX 4090 du 6 novembre ci-dessous.
Un test Live complet (heure avec outil, calcul, puis « ajoute 2 ») produit les trois bonnes
réponses : audio prêt côté serveur **2,17 s / 0,92 s / 0,95 s après la fin de parole estimée**.
Cela inclut 352–384 ms de détection de fin de tour. Ce n'est pas une mesure acoustique :
transport, tampon du navigateur et périphérique audio ajoutent leur délai.


## Choix vérifiés sur Ryzen / Radeon, le 5 novembre 2026

| Essai | Qualité sur le corpus | Latence médiane / p95 |
|---|---|---|
| Décision Qwen3-4B, consigne actuelle | 82/93 (88,2 %) | 192 / 250 ms |
| Qwen3-4B, consigne avec exemples | 71/93 (76,3 %) | 187 / 248 ms |
| LFM2.5-1.2B Q4, même consigne | 28/93 | 250 / 287 ms |
| LFM2.5-1.2B Q4, consigne avec exemples | 33/93 | 247 / 284 ms |
| Plan d'interruption avant optimisation | 23/24 | 355 / 577 ms |
| Plan d'interruption optimisé | 23/24 | 270 / 446 ms |

Qwen reste le modèle par défaut avec la consigne actuelle. Les résultats de LFM concernent ces consignes et ce
corpus français, pas toutes ses utilisations. Gemma, Clef et Aplomb n'avaient pas été mesurés
lors de cette campagne du 5 novembre sur Ryzen / Radeon. L'essai Gemma sur RTX 4090 figure ci-dessous.
Modèles de référence : [Qwen](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507),
[LFM](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct-GGUF),
[Smart Turn et son prétraitement](https://reference-server.pipecat.ai/en/latest/_modules/pipecat/audio/turn/smart_turn/local_smart_turn_v3.html).

Smart Turn reste en **FP32**, avec `[turn] threads = 2` : son modèle CPU quantifié gagne
du temps mais passe de **62/75 à 54/75** bonnes détections au calme, et de **54/75 à 49/75**
avec bruit. Ces 75 clips sont synthétiques : ils ne remplacent pas une évaluation avec
ta voix et ton micro. L'AEC3 et la réduction de bruit WebRTC restent disponibles dans
les réglages ; ajouter un filtre IA supplémentaire sans mesure de la transcription et
de la double parole n'établirait pas une amélioration de qualité.

Les mesures détaillées sont dans `.runtime/bench/` : `audio-local.json`,
`smart-turn-local.json`, `decision.json`, `decision-lfm-fewshot.json`,
`decision-qwen-fewshot.json`, `plan-optimization.json`, `pipeline-smoke.json`.
Les métriques Live distinguent le
silence réellement observé, l'obtention de la transcription et l'audio prêt côté serveur
(`audio_ready_latency_ms`). Le silence n'est plus compté deux fois dans le graphique.

## Gemma décideur et réglages Qwen3.5 — RTX 4090, 6 novembre 2026

Comparaison **isolée, un modèle chargé à la fois**, avec llama.cpp **b11960 Vulkan** :
deux passages des mêmes 93 situations et 24 plans, consignes de production, contexte
8192, deux slots, budget de 600 ms et réflexion désactivée. Les outils sont simulés ;
ces mesures excluent transcription, synthèse vocale et concurrence avec la conversation.

| Décideur Q4_K_M | Décisions brutes justes | Médiane des deux passages | Plans justes sans repli |
|---|---|---|---|
| Qwen3-4B-Instruct-2507 | 80/93 | 27–30 ms | 23/24 à chaque passage |
| Qwen3.5-4B | 82/93 | 80–82 ms | 21/24 puis 23/24 ; deux dépassements au premier passage |
| Gemma 4 E4B, `--swa-full` | 79/93 | 37–38 ms | 22/24 à chaque passage, aucun dépassement |

**Qwen3 reste le décideur par défaut.** Gemma fonctionne, mais n'améliore pas ce corpus.
Les demandes potentielles de confirmation sont comptées séparément : le seuil de confiance
0,75 n'est pas calibré entre modèles. Deux passages ne constituent pas deux corpus indépendants.

Pour Qwen3.5 en conversation, les essais ont comparé Vulkan et CUDA 12 b11960, quatre threads,
les tailles de lots, le cache KV Q8, les checkpoints et le
[profil d'échantillonnage officiel sans réflexion](https://huggingface.co/Qwen/Qwen3.5-4B#best-practices).
Aucun profil ne gagne partout. Après chauffe des kernels, un préremplissage sans cache d'environ
4000 tokens prend environ **320 ms avec CUDA contre 384 ms avec Vulkan** ; le gain sur les
réponses courtes n'est pas constant. Le profil officiel à température 0,7 associé au cache Q8
a notamment transformé une heure d'outil **17:42 en 22:42**. Cela justifie de conserver comme
référence la température **0,2**, le cache **f16**, `--reasoning off`, le contexte **24576** et
la limite de **400 tokens**, sans annoncer un réglage universellement optimal.

Le profil d'exécution reste **Vulkan, threads automatiques et Flash Attention automatique**.
Les petits gains initiaux avec quatre threads ne se répètent pas au contrôle final :
réponse complète médiane de 374 ms pour la référence, 383 ms avec quatre threads Vulkan
et 405 ms avec quatre threads CUDA, sous la charge GPU de fond observée. CUDA reste une
option disponible via `alt_brain.llama_dir`, sans être activé. Le profil conservé réussit
les six cas courts à chacun des trois passages à graine fixe, soit 18/18 vérifications,
et non 18 questions distinctes. Le score de format du contexte long reste strict : un
code correct inclus dans une phrase est rejeté lorsque seul le code était demandé.

Un premier essai ajoutait un cinquième modèle aux quatre serveurs déjà actifs, sous forte
pression mémoire vidéo. Son rapport de vitesse apparent « ×4 » est écarté : il ne démontre
pas que Qwen3.5 est généralement quatre fois plus rapide que Qwen3.

Pour reproduire, arrêter Localive et les autres inférences avant de lancer les commandes
suivantes. CUDA nécessite son archive b11960 et ses DLLs dans le dossier indiqué.

```powershell
.venv\Scripts\python.exe bench\decision\compare_models.py --models qwen3 qwen3.5 gemma-e4b --repeat 2 --output .runtime/bench/arbiter-models-vulkan-isolated.json
.venv\Scripts\python.exe bench\tune_qwen.py --variants baseline,threads4,batch1024,kvq8,checkpoints1024 --output .runtime/bench/qwen-tuning-vulkan.json
.venv\Scripts\python.exe bench\tune_qwen.py --llama-dir .runtime/llama/b11960-cuda12 --sampling baseline,official --output .runtime/bench/qwen-tuning-cuda.json
```

Rapports locaux : `.runtime/bench/arbiter-models-{vulkan,cuda}-isolated.json` pour les
décideurs ; `qwen-tuning-{vulkan,cuda}.json` pour l'exploration et
`qwen-tuning-*-final.json` pour les contrôles finaux. Les rapports séparent première
utilisation, préremplissage après chauffe, cache réutilisé et réponses aux outils ; les
réponses complètes sont conservées pour vérifier les scores automatiques.

### Comparaison directe Qwen3 / Qwen3.5 en conversation et décision

Nouveau contrôle avec **Qwen3-4B-Instruct-2507 et Qwen3.5-4B en Q4_K_M**, sur la
même RTX 4090 et b11960 Vulkan. Un seul modèle est chargé à la fois pour les tests
texte. La conversation utilise 24 576 tokens de contexte, un slot, température
0,2, plafond de 400 tokens et réflexion désactivée. Les deux modèles reçoivent
successivement la même consigne de production, puis la même consigne simplifiée.
Ces paramètres comparent l'usage vocal actuel ; ils ne prétendent pas constituer
un optimum universel ou le profil général recommandé par les auteurs.

| Conversation : 18 scénarios × 2 graines | Qwen3 | Qwen3.5 |
|---|---|---|
| Réponses exploitables, consigne de production | 28/36 | 26/36 |
| Réponses exploitables, consigne adaptée | 28/36 | 29/36 |
| Médiane totale, réussites communes en production (24 paires) | 217 ms | 313 ms |
| Médiane totale, réussites communes avec adaptation (26 paires) | 187 ms | 248 ms |

La revue de chaque échange et de ses arguments tolère les petites maladresses de
style, pas les actions inventées, mauvaises unités ou réponses dans la mauvaise
langue. Les médianes appariées portent sur les mêmes cas réussis par les deux
modèles, afin de ne pas avantager les échecs courts ; elles excluent l'ASR, le TTS
et les vrais services externes. L'adaptation aide Qwen3.5 à demander la durée
manquante et à utiliser directement l'historique, mais conserve des erreurs de
minuteur et le suivi d'une instruction présente dans un résultat d'outil. Chez
Qwen3, elle corrige les volets non disponibles mais fait régresser une correction
de durée. **L'écart de qualité est faible et dépend de la consigne ; Qwen3 reste
plus rapide sur les réussites communes.** Les graines répétées ne sont pas des
scénarios indépendants et le score est une revue qualitative par agent.

Le décideur conserve la consigne de production, température 0, grammaire contrainte,
contexte 8192, deux slots et délai 600 ms. Les deux corpus sont déjà connus ; aucun
réglage n'est choisi sur leurs résultats.

| Décideur, deux passages | Qwen3 | Qwen3.5 |
|---|---|---|
| Classifications brutes sur le corpus principal | 80/93 à chaque passage | 82/93 à chaque passage |
| Classifications brutes sur le second corpus | 23/24 à chaque passage | 24/24 à chaque passage |
| Médiane sur les 93 cas | 30–35 ms | 82–92 ms |
| P95 sur les 93 cas | 45–51 ms | 95–103 ms |
| Plans de reprise, après les 93 cas | 23/24 | 22/24 |
| Dépassements de délai / replis | 0 | 0 |

Après application du filtre d'écho dans le benchmark, les deux obtiennent **82/93** sur le corpus
principal. Avec le seuil de confiance actuel, Qwen3 demanderait une confirmation
sur un cas, Qwen3.5 sur onze, dont dix correctement classés. Les probabilités ne
sont pas calibrées entre modèles. Qwen3.5 produit moins de fausses interruptions
brutes (1 contre 5), mais ignore davantage de demandes à traiter (8 contre 2).
Son léger avantage de classification brute ne garantit donc pas une meilleure
interaction. Les mêmes plans rejoués après le second corpus donnent 23/24 pour les
deux : une variabilité entre exécutions est observée sur un plan Qwen3.5 à entrée
identique, sans en établir la cause (cache, slots ou variations numériques possibles).

La chaîne Live a aussi été testée en ne changeant qu'un rôle à la fois, avec les
alternatives désactivées. À décideur Qwen3 identique, chaque LLM réussit six tours
audio (heure simulée 17:42, 7 × 6 = 42, puis 44, deux passages). La médiane
`audio_ready_latency_ms` est de **640 ms pour Qwen3 (553–881)** et **836 ms pour
Qwen3.5 (679–912)**, de la fin de parole détectée au premier audio prêt côté serveur,
hors lecture par le client. Ce sont trois questions répétées, pas une étude de latence
sur tous les usages.

À LLM Qwen3 identique, les deux décideurs passent **6/6 vérifications** sur trois
scénarios audio répétés : acquiescement sans coupure, stop avec coupure, correction
Paris → Lyon avec nouvel appel météo. Les clips sont identiques, vérifiés par empreinte.
Le « Mm-hm » est toutefois transcrit « Menem » : son score valide l'absence de coupure,
pas une bonne reconnaissance de l'acquiescement. Les sorties sur la tour Eiffel ne
sont pas évaluées comme réponses de culture générale dans ce test d'interruptions.
Un premier lancement a terminé ses six tours vocaux puis échoué sur un caractère
Unicode de la console Windows ; l'affichage UTF-8 a été corrigé, les interruptions
ont été rejouées séparément et les résultats précédents conservés.

Manifestes et transcriptions : `.runtime/bench/qwen-head-live-{3-3,35-3}-*` et
`qwen-head-interruptions-{3-3,3-35}-*` ; revue : `qwen3-vs-qwen35-live-review.json`.
Le lanceur `bench/compare_live.py` refuse les ports occupés, ne sauvegarde aucun
réglage et nettoie les processus qu'il crée. Exemple après arrêt de l'app :
`python bench/compare_live.py --brain qwen3 --arbiter qwen3.5 --repeat 2 --interruptions`.
La configuration habituelle a ensuite été restaurée, avec état `Prêt` vérifié.
Ces résultats justifient de conserver Qwen3 dans les deux rôles avec les consignes
et seuils actuels ; un passage à Qwen3.5 décideur demanderait une calibration séparée
de ses demandes de confirmation.

Rapports détaillés dans `.runtime/bench/qwen3-vs-qwen35-chat-{production,adapted,review}.json`
et `qwen3-vs-qwen35-arbiter-{93,24,review}.json`. Pour reproduire les comparaisons
texte, arrêter d'abord les autres inférences GPU :

```powershell
.venv\Scripts\python.exe bench\compare_chat.py --models qwen3,qwen3.5 --profile production --output .runtime/bench/qwen3-vs-qwen35-chat-production.json
.venv\Scripts\python.exe bench\compare_chat.py --models qwen3,qwen3.5 --profile adapted --output .runtime/bench/qwen3-vs-qwen35-chat-adapted.json
.venv\Scripts\python.exe bench\decision\compare_models.py --models qwen3 qwen3.5 --repeat 2 --output .runtime/bench/qwen3-vs-qwen35-arbiter-93.json
.venv\Scripts\python.exe bench\decision\compare_models.py --models qwen3 qwen3.5 --cases bench/decision/minicpm_holdout.json --repeat 2 --output .runtime/bench/qwen3-vs-qwen35-arbiter-24.json
```

### Qwen3.5-9B comme LLM — RTX 4090, 6 novembre 2026

Le [modèle Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) fonctionne dans Localive
avec b11960 Vulkan, contexte 24 576, un slot, température 0,2, plafond 400 tokens et
`--reasoning off`. La quantification standard
[Unsloth Q4_K_M](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/blob/3885219b6810b007914f3a7950a8d1b469d598a5/Qwen3.5-9B-Q4_K_M.gguf)
a été téléchargée à révision figée, avec taille et SHA-256 vérifiés :
5 680 522 464 octets, soit 5,68 Go / 5,29 Gio. Provenance complète dans
`.runtime/bench/qwen35-9b-download.json`.

Les mêmes 18 scénarios et deux graines ont été rejoués, avec outils simulés et revue
de chaque échange. Les références 4B viennent de la comparaison précédente sur la
même machine avec les mêmes paramètres ; elles n'ont pas été relancées avec le 9B.

| LLM | Consigne actuelle | Consigne adaptée |
|---|---|---|
| Qwen3-4B-Instruct-2507 | 28/36 | 28/36 |
| Qwen3.5-4B | 26/36 | 29/36 |
| Qwen3.5-9B | **34/36** | **36/36** |

Le 9B corrige les unités des minuteurs, demande les paramètres manquants et ignore
l'instruction parasite du résultat météo. Avec la consigne actuelle, deux réponses
restent problématiques : « Je lance un minuteur » avant de demander sa durée,
et une offre de fermer les volets manuellement malgré l'absence de cette capacité.
La première formulation peut aussi exprimer une intention imminente : en la tolérant,
le score serait 35/36. La consigne adaptée corrige ces deux cas ; le 36/36 reste un
résultat sur ce petit corpus, pas une garantie générale.

Sur les réussites communes aux trois modèles, les médianes complètes sont
**217 / 313 / 436 ms** avec la consigne actuelle, **187 / 248 / 382 ms** avec
l'adaptation (Qwen3 / Qwen3.5-4B / 9B), hors ASR, TTS et services externes.
Les appels natifs et arguments sont conservés dans
`.runtime/bench/qwen35-9b-chat-{production,adapted,review}.json`.

Le 9B a aussi réellement remplacé le LLM principal pendant six tours Live, avec
Qwen3 conservé comme décideur et les autres alternatives désactivées. Les six
réponses sont correctes : heure simulée via outil, calcul et suivi du résultat.
La médiane `audio_ready_latency_ms` atteint **914 ms** (679–1116), contre **640 ms**
pour Qwen3 et **836 ms** pour Qwen3.5-4B dans les mêmes trois questions répétées.
Une mesure pendant la session relève **14 483 Mio utilisés sur 24 564 Mio** : usage
GPU total incluant décideur, ASR et applications en arrière-plan, pas la seule
allocation du 9B ni un pic garanti. Rapports : `qwen35-9b-live-*` dans `.runtime/bench/`.

Le 9B est donc un candidat plus fiable pour la conversation sur ces essais, au prix
d'une réponse plus lente. Le modèle reste installé et disponible pour les benchmarks ;
la configuration habituelle a été restaurée et son état `Prêt` vérifié. Aucun choix
de modèle ni aucune consigne de production n'a été modifié définitivement.

```powershell
# Arrêter l'app et les autres inférences GPU avant ces tests isolés.
.venv\Scripts\python.exe -X utf8 bench\compare_chat.py --models qwen3.5-9b --profile production --output .runtime/bench/qwen35-9b-chat-production.json
.venv\Scripts\python.exe -X utf8 bench\compare_chat.py --models qwen3.5-9b --profile adapted --output .runtime/bench/qwen35-9b-chat-adapted.json
.venv\Scripts\python.exe -X utf8 bench\compare_live.py --brain qwen3.5-9b --arbiter qwen3 --repeat 2 --prefix qwen35-9b-live
```

### Alléger Qwen3.5-9B — mesures du 6 novembre 2026

La vision est **déjà absente du chargement actuel** : le GGUF Q4_K_M contient 427
tenseurs de langage, aucun encodeur vision/projecteur ; aucun fichier `mmproj`
n'est installé à côté. Les propriétés du serveur indiquent vision, vidéo et audio
désactivés. Dans llama.cpp, le projecteur multimodal est un
[fichier séparé](https://github.com/ggml-org/llama.cpp/blob/b11960/tools/mtmd/README.md).
Ajouter `--no-mmproj` interdit son chargement automatique, mais n'économise donc
aucune mémoire ici. Audit : `.runtime/bench/qwen35-9b-multimodal-audit.json`.

Sept profils ont été comparés sans autres inférences GPU : Vulkan/CUDA, contexte
8k/16k/24k, cache d'attention F16/Q8, Flash Attention et quatre threads CPU. Même
GGUF, consigne adaptée, température 0,2, 400 tokens maximum, réflexion désactivée,
un slot, 18 scénarios × deux graines. Chaque profil passe **36/36** après revue des
appels et arguments, ainsi que **5/5** récupérations dans un contexte d'environ
6 000 tokens. Ce petit corpus ne mesure pas la qualité sur toute la fenêtre 24k.
Le cache est réellement utilisé : 5 959 tokens réutilisés et 42 réévalués, contre
zéro token réutilisé à froid.

| Profil | Médiane complète, cas courts | Surcroît GPU maximal observé |
|---|---:|---:|
| Vulkan, F16, 24k | 434 ms | 5 956 Mio |
| CUDA, F16, 24k | 378 ms | 6 252 Mio |
| **CUDA, Q8, 24k** | **360 ms** | **5 980 Mio** |
| CUDA, Q8, 16k | 361 ms | 5 726 Mio |
| CUDA, Q8, 8k | 379 ms | 5 506 Mio |

Un contrôle dans l'ordre inverse confirme **349 contre 408 ms** pour CUDA Q8 24k
et Vulkan 24k : gain de temps observé de **14 à 17 %**.
La latence du premier texte parlé reste presque identique dans le contrôle
(173 contre 176 ms) : le gain de durée complète ne garantit pas le même gain vocal.
Les mesures GPU sont des
instantanés après chauffe et requêtes, moins l'usage avant chargement, pas des pics
continus ni des allocations exclusives. La dérive après arrêt atteint −94 à +58 Mio
dans le premier passage. **Aucune économie GPU n'est établie face à Vulkan 24k** ;
Q8 réduit surtout le surcoût de CUDA F16. Les états récurrents du modèle restent
en F32 : quantifier le cache d'attention ne divise pas toute sa mémoire par deux.
Passer à 16k retire un tiers du contexte pour seulement environ 254 Mio de gain
supplémentaire, sans accélération mesurée ; le profil conserve donc 24k.

Le fichier [localive.qwen35-9b.toml](../localive.qwen35-9b.toml) rend ce réglage
utilisable dans l'application : 9B CUDA avec Flash Attention et cache Q8, décideur
**Qwen3-4B sur Vulkan**, alternatives comparatives désactivées. Les nouveaux champs
`decision.model` et `decision.llama_dir` évitent de charger automatiquement un
deuxième 9B pour les décisions ; `brain.extra_args` ne concerne que le principal.
Les poids Q4_K_M et les consignes de production restent identiques. Le 36/36
ci-dessus concerne la consigne adaptée du benchmark ; avec la consigne de
production, la référence précédente reste 34/36.

Ce fichier a ensuite été chargé dans le vrai `App`, sans substitution du modèle
ou de la configuration de l'arbitre : **6/6 tours vocaux corrects**, aucun échec
client, et un ordre d'arrêt classé `STOP` par l'arbitre en **24 ms**. Les propriétés
des serveurs confirment 9B/24k et Qwen3/8k, sans modalité visuelle. La médiane
`audio_ready_latency_ms` est **726 ms** (553–898), contre 914 ms lors du précédent
essai Vulkan : seulement trois questions répétées deux fois, pas un gain général
garanti. Les consignes de production ont bien été utilisées pour ces tours Live.

Usage GPU total observé après les tours : **14 737 Mio / 24 564 Mio** (14,4 Gio),
contre **22 689 Mio** (22,2 Gio) pour la configuration comparative avant arrêt.
Cette différence porte sur l'ensemble de la pile : deux serveurs au lieu de quatre,
modèle principal différent, ASR et applications en arrière-plan inclus. Elle ne
mesure pas une réduction des poids du 9B ni un gain dû à la suppression de la vision.
Manifestes, transcriptions et revue : `.runtime/bench/qwen35-9b-profile-validation-*`.
Les 180 tests passent. La configuration habituelle a été rétablie et son état
`Prêt` vérifié ; les deux fichiers TOML sont restés inchangés pendant la validation.

Après arrêt de l'instance habituelle, depuis la racine du dépôt :

```powershell
.venv\Scripts\python.exe -m localive --config localive.qwen35-9b.toml
```

Ce profil est chargé sur les valeurs par défaut, sans fusion avec `localive.toml` ;
les modifications de la page Réglages seront enregistrées dans le fichier choisi.
Le lancement sans `--config` conserve la configuration habituelle.

Rapports : `.runtime/bench/qwen35-9b-optimization{,-control,-review}.json`.
Pour reproduire le balayage après arrêt de l'app et des autres inférences GPU :

```powershell
.venv\Scripts\python.exe -X utf8 bench\optimize_qwen35_9b.py --output .runtime/bench/qwen35-9b-optimization.json
```

### Un seul Qwen3.5-9B pour la conversation et le décideur

Localive permet déjà de partager un modèle avec `[decision] dedicated = false`.
Le principal démarre alors avec `--parallel 2 --kv-unified` et l'arbitre utilise
le même serveur. Les poids ne sont chargés qu'une fois ; les deux conversations
gardent leurs historiques et états distincts. Deux processus séparés chargent au
contraire chacun leurs poids et buffers GPU. Le petit arbitre séparé réserve ses
propres slots et caches, mais partage encore le GPU physique avec le principal.

Le cache unifié de 24 576 tokens constitue **un pool commun**, pas deux réserves
de 24k simultanées. Chaque séquence peut atteindre 24k si la place est disponible ;
la conversation doit laisser de la marge aux requêtes du décideur. Le planificateur
peut en outre poser deux questions simultanées : avec la génération principale,
cela fait trois requêtes pour deux slots. L'attente consomme le budget de 600 ms.
Voir les [calculs de contexte b11960](https://github.com/ggml-org/llama.cpp/blob/b11960/src/llama-context.cpp)
et le [fonctionnement des batches](https://github.com/ggml-org/llama.cpp/blob/b11960/tools/server/README-dev.md#batching).

Les décisions à vide ont été remesurées sur les mêmes 93 situations et 24 plans,
deux passages après préchauffage, via le vrai `Arbiter`, consigne et seuil 0,75
inchangés. Qwen3 utilise Vulkan/8k ; le 9B utilise CUDA, cache Q8, contexte 24k et
deux slots, comme le profil partagé envisagé. Ce sont des configurations complètes
différentes, pas une isolation de l'effet du nombre de paramètres.

| Décideur à vide | Qwen3-4B | Qwen3.5-9B |
|---|---:|---:|
| Décisions correctes brutes, chaque passage | 80/93 | 78/93 |
| Après filtre d'écho, chaque passage | 82/93 | 78/93 |
| Médiane de décision | 31–35 ms | 79 ms |
| P95 de décision | 49–53 ms | 94 ms |
| Expirations du délai de 600 ms | 0 | 0 |
| Confirmations qui seraient demandées | 1/93 | 18/93 |
| Plans corrects | 23/24, 23/24 | 21/24, 20/24 |

Le 9B n'est donc **pas intrinsèquement trop lent** pour décider. Avec les consignes
actuelles, il n'améliore toutefois pas la qualité de ce corpus et son score de
confiance déclenche davantage de confirmations. Ces probabilités ne sont pas
calibrées entre modèles. Les deux points d'écart brut ne suffisent pas à établir
un classement général ; certains cas ADD/NEW et annulations dépendent fortement
de l'interprétation du contexte. Les répétitions ne sont pas de nouvelles tâches.
Rapports : `.runtime/bench/qwen35-shared-reference-{qwen3,9b}.json` et
`qwen35-shared-idle-review.json`.

Le partage a ensuite été comparé au duo 9B + Qwen3 sur six situations simples
répétées deux fois, à vide, avant le premier token d'un nouveau contexte et pendant
la génération. La charge principale est identique : **5 988 tokens**, cache
effectivement nul, génération de 256 tokens maximum. Le budget de décision reste
600 ms ; aucune décision n'interrompt réellement la charge synthétique.

| Phase de l'appel au décideur | Un 9B partagé, 2 slots | 9B + Qwen3 séparé |
|---|---:|---:|
| À vide, médiane sur 12 appels | 83 ms | 41 ms |
| Avant le premier token du long contexte | ~600 ms, **11/12 replis** | 60 ms, **0/12 replis** |
| Pendant la génération, médiane | 126 ms, 0 repli | 64 ms, 0 repli |
| Planification pendant génération, 4 appels | 276 ms, 0 repli | 117 ms, 0 repli |
| Mémoire GPU supplémentaire après chauffe | 5 966 Mio | 9 579 Mio |

Le partage économise ici **3 613 Mio, soit environ 3,5 Gio**, en instantanés GPU
moins l'usage avant chargement, hors ASR/TTS. La dérive après arrêt est de −56/0 Mio
(partagé/séparé) ; ce ne sont pas des pics continus. Le mode partagé fonctionne
vite quand le modèle est libre ou génère déjà ; c'est l'attente lors du traitement
du long contexte à froid qui provoque les replis. Augmenter le délai masquerait
ces expirations en faisant attendre plus longtemps l'interruption.

Les timings prouvent la fenêtre HTTP avant premier contenu, pas un chevauchement
exact des kernels GPU. Le seul succès partagé de cette phase part environ 0,8 ms
après la requête principale, contre environ 22 ms pour les suivants. Les 16 appels
de classification/planification pendant la génération sont entièrement chevauchés
dans chaque mode. À vide et pendant la génération, les classifications passent
12/12 dans les deux modes. Les plans passent seulement **2/4 chacun**, avec des
erreurs sémantiques différentes : lumière immédiate classée FIX par le 9B, conversion
Fahrenheit classée AFTER par Qwen3. Cela ne constitue pas un test audio d'interruption
ni une validation générale des décisions.

Le premier lancement partagé, interrompu avant les cas par une connexion HTTP
réutilisée fermée, est exclu. Le client de charge utilise désormais une connexion
neuve, comme le produit. Rapports : `.runtime/bench/shared-qwen35-9b.json`
(séparé terminé), `shared-qwen35-9b-retry.json` (partagé terminé) et
`shared-qwen35-9b-review.json`. L'ordre n'est pas randomisé. La configuration
habituelle a été restaurée et vérifiée prête après les mesures.

Le choix est donc explicite : `decision.dedicated=false` économise les poids du
petit modèle, tandis que le décideur séparé protège mieux le délai d'interruption
avec les réglages et la charge testés. Les fichiers de configuration actifs et
le profil 9B précédent n'ont pas été modifiés par cette comparaison.

### MiniCPM5 comme décideur — même machine et même corpus

Les versions finales officielles [MiniCPM5-1B](https://huggingface.co/openbmb/MiniCPM5-1B-GGUF)
et [MiniCPM5-2B](https://huggingface.co/openbmb/MiniCPM5-2B-GGUF), en Q4_K_M, ont été
téléchargées avec révision figée et SHA-256 vérifié. Deux passages isolés avec b11960
Vulkan, la consigne française actuelle, température 0, `--min-p 0`, réflexion désactivée
et budget de 600 ms donnent :

| Décideur | Décisions brutes justes | Latence médiane | Plans justes |
|---|---|---|---|
| Qwen3-4B, référence remesurée | 80/93 aux deux passages | 26–35 ms | 23/24 |
| MiniCPM5-1B | 29/93 puis 32/93 | 19–20 ms | 6/24 |
| MiniCPM5-2B | 40/93 puis 38/93 | 27 ms | 18/24 |

Aucun dépassement du délai ni repli dans ces essais. Le rendu du template ferme bien
le bloc de réflexion avant la génération ; les réponses de contrôle n'émettent pas de
raisonnement. Les étiquettes occupent un ou deux tokens : le budget de quatre tokens
du classificateur suffit. Les faibles scores portent donc sur les décisions produites,
pas sur une indisponibilité du serveur. **Qwen3 reste le décideur** ; ces mesures ne
jugent pas MiniCPM5 en mode raisonnement ou avec une consigne spécifiquement adaptée.

```powershell
.venv\Scripts\python.exe bench\decision\compare_models.py --models qwen3 minicpm5-1b minicpm5-2b --repeat 2 --output .runtime/bench/arbiter-minicpm5-vulkan-isolated.json
```

Comme ci-dessus, arrêter les autres serveurs avant cette comparaison. Rapports locaux :
`arbiter-minicpm5-vulkan-isolated.json`, `minicpm5-probe.json` (template, tokenisation et
réponses de contrôle), `minicpm5-downloads.json` (provenance et empreintes), dans
`.runtime/bench/`. Les modèles téléchargés ne sont pas activés au démarrage de Localive.

### MiniCPM5-2B : consignes adaptées et remplacement du LLM

Le décideur a ensuite été testé avec trois consignes figées : production, français
concis et anglais concis, ces deux dernières accompagnées d'une situation en JSON.
Sur les 93 cas de développement, elles obtiennent respectivement **40/93, 30/93 et
39/93**, à chacun des deux passages. Le choix de la meilleure adaptation (anglais)
a été enregistré avant de lire 24 nouveaux cas équilibrés, écrits séparément.
Sur ce contrôle, MiniCPM obtient **13/24 avec la consigne initiale, 11/24 avec
l'adaptation**, contre **23/24 pour Qwen3**, identiques aux deux passages. Aucun
délai dépassé ; la consigne adaptée introduit six fausses interruptions par passage.
Elle ne constitue donc pas une amélioration du décideur. Ces répétitions ne sont
pas des tâches indépendantes et les probabilités de confiance ne sont pas calibrées
entre modèles. Le benchmark et les consignes sont dans
`bench/decision/minicpm_prompts.py`, les résultats dans
`.runtime/bench/minicpm-prompts.json`.

Le remplacement du **LLM conversationnel** a également été essayé : Qwen3-4B,
Qwen3.5-4B et MiniCPM5-2B tournent successivement seuls, avec b11960 Vulkan,
24 576 tokens de contexte, un slot et les mêmes 18 scénarios (deux graines).
Les outils renvoient des données simulées ; aucune action domestique n'est exécutée.
Une consigne anglaise plus courte adapte MiniCPM au français oral, aux paramètres
manquants, à l'historique et aux appels d'outils. MiniCPM utilise `min_p=0`,
`top_p=0.95`, température 0.2 sans réflexion. Un essai supplémentaire sur 18 cas
active la réflexion avec température 1, selon le
[profil de raisonnement officiel](https://huggingface.co/openbmb/MiniCPM5-2B),
et un plafond de 2 048 tokens choisi pour cet essai.

Après revue des échanges complets, en tolérant les petites maladresses de style
mais pas les mauvaises durées, les actions inventées ou les inversions de personne :

| LLM et consigne | Réponses exploitables après revue | Latence médiane totale LLM |
|---|---|---|
| Qwen3-4B, production | 28/36 | 229 ms |
| Qwen3.5-4B, production | 26/36 | 328 ms |
| MiniCPM5-2B, production sans réflexion | 16/36 | 119 ms |
| MiniCPM5-2B, adaptée sans réflexion | 16/36 | 129 ms |
| MiniCPM5-2B, adaptée avec réflexion | 13/18 | 383 ms |

Les deux premières variantes MiniCPM portent sur les mêmes 18 situations répétées
avec deux graines ; le mode réflexion n'a qu'un passage. Cette revue qualitative
par agent et ce petit corpus ne sont pas un classement général des modèles.
Les latences incluent la génération et les boucles d'outils simulés, mais pas
l'ASR, le TTS ni un service externe. La consigne adaptée corrige la réponse sur
Canberra mais n'améliore pas le score global. Le profil avec réflexion corrige les demandes de
précision, les actions non disponibles et le résultat d'outil contenant une
instruction parasite ; elle laisse notamment **12 secondes au lieu de 12 minutes**
et **30 secondes au lieu de 3 minutes**, ainsi que des erreurs de français et
d'explication. Les appels d'outils natifs sont bien parsés : les mauvais nombres
sont réellement générés, pas convertis par erreur depuis des chaînes. **Qwen3
reste donc le LLM principal.** Il n'est pas sans défauts : les essais relèvent aussi
chez lui des paramètres devinés et le suivi d'une instruction dans un résultat d'outil.
L'essai avec réflexion modifie aussi la température et le budget de tokens ; son
gain ne peut pas être attribué à la seule activation du raisonnement.

Enfin, MiniCPM a réellement remplacé le modèle principal sur le port 8081 pendant
une session Live, avec la consigne adaptée ; le décideur restait Qwen afin d'isoler
le changement de LLM. Les trois tours audio réussissent : heure via `clock__read`,
7 × 6 = 42, puis ajout de 2 = 44. L'audio était prêt respectivement en **474, 507
et 444 ms** (métrique serveur `audio_ready_latency_ms`). Ce petit test confirme
le fonctionnement ASR → LLM → outil → TTS et le suivi du dialogue, pas la qualité
générale. La configuration habituelle Qwen a ensuite été restaurée et son état
`Prêt` vérifié. Aucun réglage MiniCPM n'a été enregistré dans `localive.toml`.

```powershell
# Arrêter les autres serveurs GPU avant les comparaisons isolées.
.venv\Scripts\python.exe bench\compare_chat.py --models qwen3,qwen3.5,minicpm5-2b --profile production --output .runtime/bench/minicpm5-chat-production.json
.venv\Scripts\python.exe bench\compare_chat.py --models minicpm5-2b --profile adapted --output .runtime/bench/minicpm5-chat-adapted.json
.venv\Scripts\python.exe bench\compare_chat.py --models minicpm5-2b --profile adapted --reasoning on --temperature 1 --max-tokens 2048 --seeds 42 --output .runtime/bench/minicpm5-chat-thinking.json
```

Les rapports conservent les appels natifs, leurs arguments, les réponses complètes
et les latences. Leurs scores automatiques doivent être complétés par la revue
qualitative : ils peuvent rejeter une clarification sans point d'interrogation,
ou accepter une réponse en anglais. Les verdicts ligne par ligne et leur grille
sont dans `.runtime/bench/minicpm5-chat-review-{qwen,mini}.json`. Rapports Live :
`.runtime/bench/minicpm5-adapted-live.json` et `minicpm5-live-manifest.json`.


## Prompts et compatibilité Cornelius, 9 novembre 2026

Banc `bench/prompts/run.py` : la vraie configuration de session de Cornelius (générée par son
propre code : consignes Live, délégation, carte Luna, outils horloge, météo, bus, mémoire,
`web_search`) passe par la vraie chaîne de Localive (session, décideur, plan, cerveau). Seules
la voix et les résultats d'outils sont simulés. Radeon 890M, Vulkan, cache de prompt chaud.

| Version | Vérifications | Premier son médian |
|---|---|---|
| Consignes d'avant (Qwen3-4B) | 36/40 | 500 ms |
| Consignes retravaillées (Qwen3-4B, deux passages) | 108/110 | 461 ms |
| Gemma E4B, mêmes consignes | 54/55 | 469 ms |
| Qwen3.5-4B, mêmes consignes | 50/55 | 508 ms |

Corrigé par la nouvelle version : souvenirs inventés au lieu d'appeler `memory__recall`,
recherche web lancée pour raconter une histoire, correction « en anglais » qui ne traduisait
que les deux phrases déjà dites, émojis lus à voix haute, relances systématiques.

Le prompt de Cornelius fait environ 3 600 jetons : sans cache, la Radeon 890M met 12 s à le
lire. L'heure à la minute et la caméra ne sont donc plus dans le prompt système (qui changeait
chaque minute et faisait relire tout l'historique) : elles accompagnent chaque réponse. La date
reste en cache : chaque jeton relu à chaque réponse coûte environ 7 ms sur cette carte.

Décideur sur le même corpus de 93 situations et 24 plans (`bench/decision/compare_models.py`,
arbitre dédié) : 80/93 décisions brutes, aucun repli, p95 45 ms ; plans 23/24.
