# Réglages

Tous les réglages de la page d'administration de Localive (<http://127.0.0.1:8787>), avec leur valeur par défaut. Ils sont enregistrés dans `localive.toml` (section et nom indiqués), que l'on peut aussi modifier à la main. « Redémarrage » : la valeur s'applique au prochain démarrage de Localive.

*Page générée à partir du code : `python -m localive.settings > docs/reglages.md`.*

## Voix

La voix de Cornelius : Phonon intégré (CPU), Kokoro / serveur compatible OpenAI (/v1/audio/speech) ou Piper via Wyoming. Cornelius choisit une voix OpenAI (Cedar, Marin…) : Localive la traduit en voix masculine ou féminine.

| Réglage | Clé | Par défaut | Description |
|---|---|---|---|
| Moteur vocal | `voice.backend` | `phonon` | Phonon (par défaut) : intégré sur CPU. Compatible OpenAI : Kokoro-FastAPI ou Speaches. Wyoming : serveur Piper sur le réseau local. Choix : `phonon`, `openai`, `wyoming`. Redémarrage. |
| Voix quand le client n'en choisit pas | `voice.default` | `female` | Les clients comme Cornelius imposent généralement la leur. Choix : `female`, `male`. |
| Voix féminine | `voice.female` | `Ninon` | Ninon (par défaut), Tiphaine, Hélène, Noémie, Coralie, Maëlys, Orianne, Solène, Lilou… ou un fichier .safetensors. Redémarrage. |
| Voix masculine | `voice.male` | `Lucien` | Lucien (par défaut), Arnaud, Augustin, Jules, Baptiste, Marius… Redémarrage. |
| Adresse du serveur vocal | `voice.external_url` | *(vide)* | URL HTTP(S) pour Kokoro / OpenAI (ex. http://127.0.0.1:8880) ou hôte:port pour Wyoming Piper (ex. 127.0.0.1:10200). Redémarrage. |
| Modèle vocal externe | `voice.external_model` | `kokoro` | Nom du modèle envoyé à /v1/audio/speech (ex. kokoro). Redémarrage. |
| Voix féminine externe | `voice.external_female` | *(vide)* | Identifiant de voix féminine (ex. ff_siwis pour Kokoro ; laisser vide pour la voix par défaut de Piper). Redémarrage. |
| Voix masculine externe | `voice.external_male` | *(vide)* | Identifiant de voix masculine (ex. am_adam pour Kokoro, ou ff_siwis qui est une voix féminine française ; laisser vide pour la voix par défaut de Piper). Redémarrage. |
| Parler pendant une attente | `voice.wait_notice` | non | Après deux secondes sans réponse, la voix prononce une courte phrase. Si un outil attend encore son résultat, elle le signale. Une fois par demande. |
| Cœurs de calcul de la voix | `voice.threads` | `4` | 4 est le plus rapide sur la plupart des processeurs ; davantage ralentit. De 1 à 32. Redémarrage. Avancé. |
| Modèle Phonon | `voice.model` | `hf://gradium/phonon-40m-fr` | hf://gradium/phonon-40m-fr ou un dossier local. Redémarrage. Avancé. |
| Code langue TTS (Kokoro) | `voice.lang_code` | *(vide)* | Optionnel : « f » pour le phonémiseur français de Kokoro-FastAPI. Laisser vide pour un serveur OpenAI standard. Redémarrage. Avancé. |
| Vitesse de parole | `voice.speed` | `1.0` | 1,0 : vitesse normale. De 0.25 à 4. Avancé. |
| Fréquence PCM du serveur vocal | `voice.sample_rate` | `24000` Hz | 24 000 Hz par défaut pour Kokoro et les serveurs OpenAI. De 8000 à 48000 Hz. Redémarrage. Avancé. |
| Clé du serveur vocal externe | `voice.api_key` | *(secret)* | Optionnelle, si le serveur vocal demande un jeton Bearer. Redémarrage. Avancé. |
| Délai maximal de synthèse vocale | `voice.timeout_s` | `20.0` s | De 2 à 120 s. Avancé. |

## Écoute

Comment Localive entend : détection de la voix (Silero), transcription (Parakeet, Speaches/Whisper ou Wyoming) et réaction quand on parle pendant que Cornelius répond.

| Réglage | Clé | Par défaut | Description |
|---|---|---|---|
| Moteur de transcription | `asr.backend` | `photon` | Parakeet Ultra (Photon, par défaut), serveur compatible OpenAI (Speaches / Faster-Whisper) ou Wyoming Whisper. Choix : `photon`, `openai`, `wyoming`. Redémarrage. |
| Adresse du serveur de transcription | `asr.external_url` | *(vide)* | URL HTTP(S) pour Speaches / OpenAI (ex. http://127.0.0.1:8000) ou hôte:port pour Wyoming Whisper (ex. 127.0.0.1:10300). Redémarrage. |
| Modèle de transcription externe | `asr.external_model` | `deepdml/faster-whisper-large-v3-turbo-ct2` | Ex. deepdml/faster-whisper-large-v3-turbo-ct2 ou Systran/faster-whisper-large-v3. Redémarrage. |
| Langue de transcription | `asr.language` | `fr` | Code langue ISO (fr, en…). Redémarrage. |
| Sensibilité du micro | `vad.threshold` | `0.5` | Plus bas : capte les voix lointaines ou faibles, mais aussi plus de bruit. 0,5 convient à la plupart des pièces. De 0.2 à 0.9. |
| Silence de fin de phrase (sans fin de tour intelligente) | `vad.silence_ms` | `600` ms | Utilisé seulement si la fin de tour intelligente est désactivée. De 250 à 2000 ms. Avancé. |
| Parole minimale | `vad.min_speech_ms` | `160` ms | En dessous, un son est ignoré (clic, toux). Une phrase déjà transcrite est toujours gardée. De 60 à 600 ms. Avancé. |
| Parole avant de baisser la voix | `vad.barge_in_ms` | `350` ms | Quand quelqu'un parle pendant que Cornelius répond : durée avant la première vérification et la baisse de volume. De 150 à 1500 ms. |
| Volume dès qu'une voix est entendue | `vad.duck_early_level` | `0.4` | Pendant que Cornelius parle (1 = aucune baisse). De 0 à 1. |
| Volume quand une personne lui parle | `vad.duck_level` | `0.12` | Une fois la voix confirmée (et l'écho écarté), le temps de décider. De 0 à 1. |
| Vocabulaire des appareils après transcription | `asr.device_vocabulary` | non | Dans les nouvelles conversations, aide à reconnaître les noms et alias des appareils Home Assistant déclarés par le client. Un nom partagé demande une précision. Les sous-titres conservent les mots entendus. |
| Processeur de la transcription | `asr.device` | `auto` | auto : la carte NVIDIA si elle est disponible, sinon le processeur. Choix : `auto`, `cuda`, `cpu`. Redémarrage. Avancé. |
| Modèle de transcription | `asr.model` | `moondream/parakeet-ultra` | moondream/parakeet-ultra (par défaut). Redémarrage. Avancé. |
| Contexte initial (prompt Whisper) | `asr.prompt` | *(vide)* | Optionnel : guide la ponctuation et les noms propres du foyer. Avancé. |
| Mots-clés (hotwords Faster-Whisper) | `asr.hotwords` | *(vide)* | Optionnel : noms d'appareils ou prénoms séparés par des virgules. Avancé. |
| Clé du serveur de transcription | `asr.api_key` | *(secret)* | Optionnelle, si le serveur STT demande un jeton Bearer. Redémarrage. Avancé. |
| Délai maximal de transcription | `asr.timeout_s` | `20.0` s | De 2 à 120 s. Avancé. |

## Fin de tour

Savoir quand vous avez fini de parler. Smart Turn écoute l'intonation à chaque pause : Cornelius répond dès 200 ms si la phrase est finie, et attend si elle ne l'est pas (« mets un minuteur de… »).

| Réglage | Clé | Par défaut | Description |
|---|---|---|---|
| Fin de tour intelligente | `turn.enabled` | oui | Désactivée : Localive attend un silence fixe, plus lent. Redémarrage. |
| Raccrocher sur une formule de fin | `turn.end_on_goodbye` | non | Dans les nouvelles conversations, « au revoir » ou « c’est bon, merci » ferme la connexion. Un simple « merci » permet de continuer. |
| Détection du raccrochage | `turn.goodbye_mode` | `phrases` | Les formules prédéfinies sont instantanées. Le décideur local d3-mini peut aussi reconnaître une fin de conversation exprimée autrement. En cas de doute ou de modèle indisponible, la conversation reste ouverte. Choix : `phrases`, `d3`. |
| Attendre la réponse avant de réécouter | `turn.wait_for_reply` | non | Dans les nouvelles conversations, le micro est ignoré entre la fin de votre demande et le début de la réponse. Une voix dans la pièce ne peut alors ni l’interrompre ni la remplacer. Une phrase d’attente (« Je cherche ça… ») garde le micro fermé ; la réponse le rouvre, et vous pouvez l’interrompre. Réglage par défaut : Cornelius envoie son propre choix (Voix de Cornelius), qui le remplace. |
| Pause avant de vérifier | `turn.check_ms` | `200` ms | Plus court : répond plus vite, mais vérifie plus souvent. De 100 à 800 ms. |
| Seuil « phrase finie » | `turn.threshold` | `0.5` | Plus haut : attend davantage d'être sûr que vous avez fini. De 0.1 à 0.95. |
| Attente maximale si la phrase semble inachevée | `turn.max_silence_ms` | `3000` ms | Ensuite, Cornelius vous invite à continuer. De 600 à 6000 ms. |
| Pause de fin quand vous lui coupez la parole | `turn.over_silence_ms` | `700` ms | Pendant qu'il parle, la fin de votre phrase se décide sur les mots après cette pause. Montez-la si vos phrases sont coupées en deux. De 300 à 2000 ms. |
| Cœurs de calcul de Smart Turn | `turn.threads` | `2` | De 1 à 16. Redémarrage. Avancé. |

## Interruptions

Quand quelqu'un parle pendant que Cornelius réfléchit ou répond, un décideur choisit : ignorer (« mm-hm », télé, quelqu'un d'autre), compléter la demande, s'arrêter ou passer à une nouvelle demande. Puis il choisit de finir d'abord, de corriger ou de répondre tout de suite.

| Réglage | Clé | Par défaut | Description |
|---|---|---|---|
| Mode naturel (décideur d'interruption) | `decision.enabled` | oui | Activé : un petit modèle juge ce que vous dites pendant que Cornelius réfléchit ou parle : ignorer un « mm-hm » ou la télé, compléter, s'arrêter, changer de sujet. Désactivé, mode classique : aucun modèle de décision n'est chargé (environ 3 Go de mémoire vidéo et du calcul en moins) ; « stop » ou « chut » coupent la réponse, et toute autre phrase dite pendant qu'il parle devient une nouvelle demande, à la fin de sa phrase en cours. Redémarrage. |
| Nom de l'assistant | `decision.assistant_name` | `Cornelius` | « Julie, viens voir » s'adresse à quelqu'un d'autre, « Cornelius, … » non. |
| Délai maximal d'une décision | `decision.timeout_ms` | `600` ms | Au-delà, les règles simples s'appliquent. De 100 à 2000 ms. |
| Serveur dédié au décideur | `decision.dedicated` | oui | Oui (conseillé) : décisions en environ 50 ms, environ 3 Go de mémoire vidéo. Non : il partage le cerveau principal et attend parfois qu'il ait fini de lire (0,3 à 5 s). Redémarrage. |
| Modèle du décideur | `decision.model` | `models/Qwen3-4B-Instruct-2507/Qwen3-4B-Instruct-2507-Q4_K_M.gguf` | Qwen3-4B par défaut, quel que soit le cerveau : c'est le décideur mesuré, et un gros cerveau ne charge pas un deuxième gros modèle. Redémarrage. Avancé. |
| Dossier llama.cpp du décideur | `decision.llama_dir` | *(vide)* | Vide : le même moteur que le cerveau principal. Redémarrage. Avancé. |
| Port local du décideur | `decision.port` | `8084` | De 1024 à 65535. Redémarrage. Avancé. |
| Contexte du décideur | `decision.context` | `8192` jetons | De 2048 à 32768 jetons. Redémarrage. Avancé. |

## Cerveau

Le modèle de langage qui répond et appelle les outils de Cornelius : Qwen3-4B par défaut, sur la carte graphique via le llama.cpp de Localive, ou un modèle servi par Ollama ou LM Studio.

| Réglage | Clé | Par défaut | Description |
|---|---|---|---|
| Moteur du cerveau | `brain.backend` | `localive` | Localive (conseillé) : llama.cpp intégré, réglé et mesuré pour la voix. Ollama ou LM Studio : le modèle de votre choix, servi par ce logiciel (qui doit être lancé, avec son serveur activé). Le décideur reste sur le llama.cpp de Localive. Choix : `localive`, `ollama`, `lmstudio`, `openai`. Redémarrage. |
| Modèle du serveur externe | `brain.external_model` | *(vide)* | La liste vient du serveur choisi ci-dessus. Ignoré avec le moteur Localive. Redémarrage. |
| Adresse du serveur externe | `brain.external_url` | *(vide)* | Vide : l'adresse habituelle (Ollama http://127.0.0.1:11434/v1, LM Studio http://127.0.0.1:1234/v1). Obligatoire pour « Autre serveur ». Redémarrage. |
| Contexte demandé à Ollama | `brain.ollama_context` | `16384` jetons | Ollama garde sinon son contexte par défaut (souvent 4 096 jetons) et coupe sans le dire le début des consignes de Cornelius (environ 8 000 jetons). 16 384 suffit ; plus demande de la mémoire vidéo et peut beaucoup ralentir un gros modèle. 0 : celui du serveur. Changer cette valeur recharge le modèle dans Ollama (jusqu'à deux minutes). De 0 à 131072 jetons. Redémarrage. |
| Modèle de raisonnement | `brain.model` | `models/Qwen3-4B-Instruct-2507/Qwen3-4B-Instruct-2507-Q4_K_M.gguf` | Les modèles GGUF trouvés dans le dossier models. Qwen3-4B (par défaut) est le plus juste et le plus rapide dans nos mesures ; un modèle plus gros demande plus de mémoire vidéo. Le décideur garde Qwen3-4B (page Interruptions). Redémarrage. |
| Niveau de réflexion | `brain.reasoning_level` | `off` | Avec Qwen3.5, réfléchir davantage peut retarder le premier son. Le modèle par défaut Qwen3-4B-Instruct-2507 répond directement. Choix : `off`, `low`, `medium`, `high`. |
| Niveau de réflexion de LM Studio | `llm.lmstudio_reasoning` | `auto` | Le modèle indique les niveaux disponibles. Désactivé privilégie le premier son. Le choix s’applique dès la prochaine demande. Choix : `auto`, `off`, `on`, `low`, `medium`, `high`. |
| Créativité | `llm.temperature` | `0.2` | Bas (0,2) : réponses précises et stables. Plus haut : plus variées. De 0 à 1.5. |
| Longueur maximale d'une réponse | `llm.max_tokens` | `400` jetons | Environ 400 jetons : une histoire courte. Plus : réponses plus longues possibles. De 64 à 4096 jetons. |
| Échanges gardés en mémoire | `llm.history_turns` | `12` | Le contexte est de toute façon réduit pour tenir dans le modèle. De 1 à 50. |
| Choisir les outils après la transcription | `routing.enabled` | oui | Qwen reçoit directement les petits catalogues. d3-mini trie à partir de 6 outils, ou dès 2 si leurs schémas totalisent au moins 3 072 octets. Redémarrage. |
| Fichier GGUF du décideur d’outils | `routing.model` | `models/decision/vllm-d3-mini-Q8_0.gguf` | Modèle SystemOne partagé avec le raccrochage vocal en mode d3. d3-mini Q8_0 par défaut. Redémarrage. Avancé. |
| Port local du décideur d’outils | `routing.port` | `8085` | De 1024 à 65535. Redémarrage. Avancé. |
| Dossier llama.cpp du décideur d’outils | `routing.llama_dir` | `.runtime/llama/b11960-vulkan` | Le moteur Vulkan est utilisé par défaut, même avec un cerveau sur CPU. Laisser vide pour utiliser le moteur du cerveau. Redémarrage. Avancé. |
| Couches du décideur d’outils sur la carte graphique | `routing.gpu_layers` | `99` | La valeur par défaut de 99 place tout d3-mini sur le GPU. À -1, reprend le nombre de couches du cerveau si le moteur est partagé, ou utilise 99 couches avec un moteur distinct. À 0, le décideur tourne sur CPU. De -1 à 999. Redémarrage. Avancé. |
| Temps maximal pour choisir les outils | `routing.timeout_ms` | `0` ms | À 0, Localive adapte le délai aux décisions mesurées (400 à 8 000 ms). Passé ce délai, le cerveau reçoit tous les outils. De 0 à 10000 ms. Avancé. |
| Délai maximal d'un outil de Cornelius | `llm.tool_timeout_s` | `60.0` s | Au-delà, Cornelius répond que l'outil n'a pas répondu (agent, Home Assistant lent…). De 5 à 300 s. |
| Contexte | `brain.context` | `24576` jetons | Cornelius envoie de longues consignes et des outils : 24 576 jetons conseillés. De 4096 à 131072 jetons. Redémarrage. |
| Nom publié | `llm.model` | `qwen` | Le nom sous lequel Cornelius voit ce cerveau (/v1/models). Redémarrage. Avancé. |
| Dossier llama.cpp | `brain.llama_dir` | `.runtime/llama/b11960-vulkan` | Le moteur : Vulkan (toutes cartes), CUDA (NVIDIA) ou CPU. Redémarrage. Avancé. |
| Couches sur la carte graphique | `brain.gpu_layers` | `99` | 99 : tout le modèle. Moins : une partie reste en mémoire vive (plus lent). De 0 à 999. Redémarrage. Avancé. |
| Options llama-server | `brain.extra_args` | *(vide)* | Options supplémentaires du cerveau principal (pas du décideur). Redémarrage. Avancé. |
| Clé du serveur externe | `llm.api_key` | *(secret)* | Seulement pour un serveur externe qui en demande une. Redémarrage. Avancé. |

## Recherche web

L'outil web_search de l'API Live (module Web de Cornelius), assuré par SearXNG. Localive installe et lance SearXNG dans Docker, sur cette machine uniquement.

| Réglage | Clé | Par défaut | Description |
|---|---|---|---|
| Recherche web | `search.engine` | `searxng` | off : Cornelius dit simplement qu'il ne peut pas vérifier. Choix : `searxng`, `off`. |
| Gérer SearXNG dans Docker | `search.managed` | oui | Non : Localive utilise l'adresse ci-dessous telle quelle (votre propre SearXNG). Redémarrage. |
| Adresse de SearXNG | `search.url` | `http://127.0.0.1:8888` | Le format JSON doit être activé. Redémarrage. |
| Langue des résultats | `search.language` | `fr-FR` | fr-FR, en-US… |
| Résultats lus par le cerveau | `search.max_results` | `5` | Plus : réponses mieux informées, mais lecture plus longue. De 1 à 10. |
| Démarrer Docker Desktop si besoin | `search.start_docker` | oui | Windows : lance Docker Desktop s'il est installé mais arrêté. Redémarrage. Avancé. |
| Image Docker | `search.image` | `searxng/searxng:latest` | Redémarrage. Avancé. |
| Nom du conteneur Docker | `search.container` | `localive-searxng` | Choisissez un nouveau nom et un port disponible pour créer une autre instance. Redémarrage. Avancé. |
| Délai maximal d'une recherche | `search.timeout_s` | `6.0` s | De 1 à 30 s. Avancé. |

## Écho

Annulation d'écho côté serveur (WebRTC AEC3) : retire du micro la voix que Localive envoie au haut-parleur. Désactivée par défaut : la baisse de volume et le filtre d'écho textuel ont donné de meilleures décisions dans nos essais. À essayer dans votre pièce.

| Réglage | Clé | Par défaut | Description |
|---|---|---|---|
| Annulation d'écho côté serveur | `echo.enabled` | non | Prise en compte à la prochaine conversation. |
| Gain automatique du micro | `echo.auto_gain_control` | non |  |
| Filtre des basses (souffle, vibrations) | `echo.high_pass_filter` | oui | Avancé. |
| Réduction de bruit | `echo.noise_suppression` | non | Peut dégrader la transcription : à mesurer chez vous. Avancé. |
| Transcrire le signal nettoyé | `echo.clean_transcription` | non | Non (conseillé) : le signal nettoyé sert à détecter la voix, la transcription entend le micro intact. Avancé. |
| Délai estimé haut-parleur → micro | `echo.delay_hint_ms` | `100` ms | Point de départ ; l'annulation mesure ensuite le vrai délai. De 0 à 500 ms. Avancé. |

## Serveur et sécurité

Où Localive écoute et qui peut l'utiliser. Pour un Cornelius sur une autre machine, écoutez sur 0.0.0.0 et définissez un jeton.

| Réglage | Clé | Par défaut | Description |
|---|---|---|---|
| Adresse d'écoute | `server.host` | `127.0.0.1` | 127.0.0.1 : cette machine seulement. 0.0.0.0 : tout le réseau local. Redémarrage. |
| Port | `server.port` | `8787` | 8787 par défaut ; il fait partie de l'adresse donnée à Cornelius. De 1 à 65535. Redémarrage. |
| Jeton d'accès | `server.api_key` | *(secret)* | Vide : aucun jeton (déconseillé sur le réseau). Sinon, Cornelius doit l'envoyer comme clé API. Redémarrage. |
| Jeton Hugging Face | `huggingface.token` | *(secret)* | Pour télécharger la voix Phonon (modèle à conditions). Vide : variable HF_TOKEN, puis `hf auth login`. Redémarrage. |
| Hors ligne | `huggingface.offline` | non | Ne plus contacter Hugging Face : tous les modèles doivent être téléchargés. Redémarrage. Avancé. |
