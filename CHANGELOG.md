# Journal des modifications

## 0.3.1 — 25 novembre 2026

- Le tri conjoint des outils et des adieux raccourcit les libellés des modules. Sur
  le HX 470 avec 18 outils et 200 caractères d’historique, sa médiane passe de 669 à
  572 ms. [Conditions de mesure](docs/mesures.md)
- Les demandes explicites de fin de session passent directement au décideur.
  Un « OK » peut confirmer une proposition en attente. Après une réponse terminée,
  il évite de renvoyer tout le catalogue.

- Un client peut ouvrir `/live` en WebSocket ou WebRTC, envoyer ses consignes et modifier ses
  outils en session. Le client WAV du dépôt permet de l’essayer sans configurer Cornelius.
- Qwen reçoit directement les petits catalogues. d3-mini se charge à partir de six outils,
  ou dès deux si le catalogue atteint 3 072 octets. Un changement de catalogue en cours
  de session applique les mêmes seuils. La recherche web prépare
  SearXNG à la demande. Les sessions partagent les poids de Silero et Parakeet évite de
  transcrire une seconde fois le même extrait en fin de pause.
- Un seul modèle répond aux demandes et exécute la boucle d’outils. Les installations
  existantes peuvent conserver leur ancienne section `[expert]` jusqu’à la prochaine sauvegarde.
- L’installation standard inclut d3-mini Q8_0, qui sélectionne les outils après la transcription
  sur le GPU via Vulkan. Le catalogue
  complet reste disponible en cas de doute. Les consignes vocales communes ont été raccourcies.
- L’administration affiche l’adresse Live en haut de l’accueil et regroupe la navigation
  dans un menu sur mobile. L’adresse peut être copiée sur une connexion HTTP locale.
- L’accueil préserve la sélection de texte pendant son actualisation. Les réglages sont validés avant écriture et la clé
  du fournisseur est liée à son origine. L’accès réseau au journal exige un jeton.
- Un arrêt immédiat conserve les phrases intégralement diffusées. Les annonces en attente
  sont dites à la fin de la lecture. Une sortie Ollama tronquée ne lance plus d’action.
- **Arrêt sans erreur de Parakeet** : Localive attend la fin de son arrêt avant de quitter.
  L'erreur « Failed to stop Photon reporter … cannot schedule new futures after shutdown »
  n'apparaît plus au redémarrage.
- **Contexte demandé à Ollama** (page Cerveau, 16 384 jetons par défaut) : Localive passe par
  l'API propre d'Ollama pour choisir le contexte. Par son API compatible OpenAI, Ollama gardait
  son contexte par défaut (4 096 jetons) et coupait sans le dire le début des consignes de
  Cornelius (environ 8 000 jetons). Le journal signale aussi un prompt tronqué par un autre
  serveur.
- **Ollama : réponses des modèles qui réfléchissent** : Localive lui demande de
  répondre sans réflexion préalable (`reasoning_effort: none`). Un modèle comme Qwen3.5 ou
  bonsai passait sinon toute sa longueur de réponse à réfléchir et ne disait rien ; si cela
  arrive encore, le journal le signale.
- **Catalogue de modèles par famille** (page Cerveau) : Qwen, Gemma, Mistral, OpenAI
  (gpt-oss), Liquid (LFM) et MiniCPM, depuis les dépôts GGUF officiels quand ils existent ; et
  n'importe quel GGUF d'un dépôt Hugging Face public, choisi par son nom puis son fichier.
- **Cerveau servi par Ollama ou LM Studio** (page Cerveau, « Moteur du cerveau ») : la liste
  des modèles vient du serveur choisi ; le décideur reste sur le llama.cpp de Localive.
- **Mode classique** : l'interrupteur « Mode naturel » de la page Interruptions, éteint, ne
  charge aucun modèle de décision ; « stop » coupe, toute autre phrase devient une nouvelle
  demande.
- **Cerveau préchauffé au démarrage** : Localive garde les consignes et les outils de la
  dernière conversation (`.runtime/brain-prefix.json`) et les fait lire au cerveau dès qu'il
  est prêt. La première conversation après un démarrage n'attend plus 11 à 19 s.
- **Début de conversation sans relecture complète du prompt** : Cornelius écrit l'heure à la
  minute dans la description de `alarms__set`, ce qui changeait le prompt à chaque minute.
  Localive retire cette phrase et donne la date et l'heure avec chaque réponse. Qwen3.5-9B
  relisait 8 000 jetons (20 s sur Radeon 890M) à chaque nouvelle conversation.
- **Téléchargement des modèles depuis l'administration** (page Cerveau) : Qwen3.5, Gemma 4,
  MiniCPM5… avec barre de progression, vérification SHA-256 et reprise ; le modèle apparaît
  aussitôt dans la liste « Modèle de raisonnement ».
- **Génération 70 % plus rapide sur carte intégrée** : flash-attention et cache KV 8 bits
  par défaut (Radeon 890M, Qwen3-4B, contexte de Cornelius : 13 → 22,6 jetons/s). Une option
  `-fa` ou `-ctk` dans « Options llama-server » garde la main.
- Les appels d'outil n'écrivent plus chaque paramètre vide (`"time": null…`) : Localive les
  remet avant de les transmettre à Cornelius (20 à 25 jetons de moins par appel).
- Les llama-server s'arrêtent avec Localive, même quand on ferme sa fenêtre ; avant, ils
  restaient lancés et le démarrage suivant les réutilisait avec leurs anciens réglages.
- L'installeur télécharge sur demande d'autres modèles de raisonnement (`-Models gemma-e4b`,
  `--models all`…), proposés ensuite dans la liste déroulante de la page Cerveau.
- Si le modèle de raisonnement choisi manque, Localive démarre avec Qwen3-4B au lieu de
  s'arrêter.
- Le journal dit pourquoi une décision n'a pas abouti (délai dépassé : carte graphique occupée).

## 0.3.0 — 9 novembre 2026

Localive devient un serveur Live complet pour Cornelius : on le branche avec une adresse, sans
module ni correctif.

### Ajouts

- **WebRTC compatible OpenAI** : le miroir de Cornelius se connecte à Localive comme à OpenAI
  (offre SDP, WebSocket de contrôle `attach`, canal `oai-events`).
- **SearXNG intégré** : Localive installe, lance et surveille SearXNG dans Docker pour l'outil
  `web_search` ; Docker Desktop est démarré s'il est arrêté (Windows).
- **Administration** : état des moteurs, conversations en cours, marche à suivre pour Cornelius,
  72 réglages expliqués avec leurs valeurs par défaut, journal en direct, redémarrage.
- **Installation Linux** (`scripts/install-linux.sh`, service systemd) et installeur commun
  (`python -m localive.install`).
- Documentation (`docs/`), licence PolyForm Noncommercial, banc de prompts sur la vraie
  configuration de Cornelius.

### Améliorations

- Consignes du cerveau réécrites pour un seul modèle local : mémoire consultée au lieu d'être
  inventée, pas de recherche web pour une histoire, corrections qui refont toute la réponse,
  annonces de Cornelius appliquées, émojis et markdown retirés de la voix.
- Le prompt reste en cache : l'heure et la caméra ne sont plus dans les consignes système.
- Une correction par-dessus l'assistant est décidée sur les mots après 700 ms, au lieu d'attendre
  3 s ; une phrase déjà transcrite n'est plus jetée parce que Silero l'a jugée trop courte.
- Les consignes ponctuelles ne sont plus des messages système en cours de conversation (refusés
  par Qwen3.5 et Gemma).
- Contexte borné à la fenêtre du modèle, résultats d'outils limités, réponse forcée après 5 tours
  d'outils.

### Changements

- Les cerveaux de comparaison sont retirés : le modèle de raisonnement se choisit dans une
  liste déroulante (page Cerveau), parmi les GGUF du dossier `models`. Le décideur garde
  Qwen3-4B, le modèle mesuré, quel que soit le cerveau. Les anciennes sections `[alt_brain]`
  sont ignorées.
- Accès au réseau local : l'administration propose d'abord l'adresse du réseau domestique
  (avant Tailscale et les cartes virtuelles WSL / Hyper-V).
- Une erreur non bloquante devient `voice.provider_warning` : Cornelius ferme la conversation sur
  tout événement `error`.
- La page de test (playground) est retirée, ainsi que `integrations/cornelius-localive.patch` :
  Cornelius n'a plus besoin de correctif.
- Sans jeton d'accès, les réglages ne sont modifiables que depuis la machine de Localive ; une page
  web d'un autre site ne peut plus joindre l'API.

## 0.2.0 — 5 novembre 2026

Passerelle Live locale : Parakeet, Qwen, Phonon, fin de tour Smart Turn, décideur d'interruption.
