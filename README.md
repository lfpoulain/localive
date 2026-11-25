<div align="center">

<img src="docs/assets/banniere.svg" alt="Localive, serveur vocal local" width="100%" />

<h3>Une passerelle Live pour Cornelius et vos applications, avec un moteur vocal 100 % local.</h3>

<p>
  <a href="https://github.com/lfpoulain/Cornelius"><img alt="Pour le miroir Cornelius" src="https://img.shields.io/badge/Cornelius-serveur%20vocal-107f71?style=for-the-badge" /></a>
</p>

<p>
  <img alt="Windows et Linux" src="https://img.shields.io/badge/Windows%20%7C%20Linux-0b1312?style=flat-square&logo=linux&logoColor=white" />
  <img alt="Python 3.12" src="https://img.shields.io/badge/Python-3.12-3fb7a3?style=flat-square&logo=python&logoColor=white" />
  <img alt="Compatible API Live d'OpenAI" src="https://img.shields.io/badge/API%20Live-compatible-3fb7a3?style=flat-square" />
  <img alt="En français" src="https://img.shields.io/badge/langue-fran%C3%A7ais-3fb7a3?style=flat-square" />
  <a href="LICENSE"><img alt="Licence PolyForm Noncommercial 1.0.0" src="https://img.shields.io/badge/licence-PolyForm%20NC%201.0.0-b39dff?style=flat-square" /></a>
  <img alt="Stade expérimental" src="https://img.shields.io/badge/stade-exp%C3%A9rimental-ffb35c?style=flat-square" />
</p>

<p>
  <a href="#-ce-que-fait-localive">Fonctionnalités</a> ·
  <a href="#-ce-quil-faut">Prérequis</a> ·
  <a href="#-installer">Installer</a> ·
  <a href="#essayer-le-point-dentrée-live">Essayer /live</a> ·
  <a href="#-brancher-cornelius">Brancher Cornelius</a> ·
  <a href="docs/README.md">Documentation</a> ·
  <a href="#-confidentialité">Confidentialité</a> ·
  <a href="#-licence">Licence</a>
</p>

</div>

---

**Localive héberge sur votre ordinateur le point d’entrée vocal Live utilisé par Cornelius.** Vous parlez, Parakeet transcrit votre demande, Qwen prépare la réponse et Phonon la lit en français. Avec les modèles fournis par défaut, ces trois étapes se déroulent sur votre machine.

| Pour connecter… | Adresse sur la même machine | Premier essai |
|---|---|---|
| Le miroir Cornelius et ses modules | `http://127.0.0.1:8787/v1` dans **Serveur Live** | [Brancher Cornelius](docs/cornelius.md) |
| Une application qui envoie son micro et ses outils | `ws://127.0.0.1:8787/live` | [Envoyer un WAV et recevoir une réponse](docs/protocole.md#essayer-avec-un-fichier-wav) |

Votre client choisit les consignes, déclare ses fonctions à la connexion et peut les modifier pendant la conversation. Le transport WebRTC est aussi disponible. Les événements et formats audio pris en charge sont décrits dans le [protocole Live de Localive](docs/protocole.md).

## ✨ Ce que fait Localive

<table>
  <tr>
    <td width="50%" valign="top">
      <h4>👂 Il écoute</h4>
      Détection de la voix (Silero), transcription (Parakeet) et fin de tour intelligente (Smart Turn) : il répond dès que vous avez fini, et attend quand votre phrase est inachevée (« mets un minuteur de… »).
    </td>
    <td width="50%" valign="top">
      <h4>🧠 Il réfléchit</h4>
      Qwen3-4B reçoit directement les petits catalogues d’outils. Pour un catalogue plus fourni, d3-mini sélectionne sur GPU les fonctions utiles à la demande.
    </td>
  </tr>
  <tr>
    <td valign="top">
      <h4>🗣️ Il parle</h4>
      Phonon synthétise la réponse sur le processeur avec la voix française choisie par le client. Les nombres et les heures sont prononcés à la française.
    </td>
    <td valign="top">
      <h4>✋ Il se laisse interrompre</h4>
      Parlez pendant qu'il répond : sa voix baisse, et un décideur choisit d'ignorer (« mm-hm », la télé, quelqu'un d'autre), de s'arrêter, de corriger sa réponse ou de traiter votre nouvelle demande, tout de suite ou juste après.
    </td>
  </tr>
  <tr>
    <td valign="top">
      <h4>🔎 Il cherche sur le web</h4>
      Quand un client déclare l’outil hébergé <code>web_search</code>, Localive prépare SearXNG. Par défaut, il utilise un conteneur Docker local.
    </td>
    <td valign="top">
      <h4>⚙️ Il se règle simplement</h4>
      Une page d'administration explique chaque réglage, avec de bonnes valeurs par défaut : on n'y touche que si on le souhaite.
    </td>
  </tr>
</table>

## 🧰 Ce qu'il faut

| | |
|---|---|
| 💻 **Ordinateur** | Windows 10/11 ou Linux x86-64 (testé sous Windows ; Linux en cours de validation) |
| 🎮 **Carte graphique** | GPU Vulkan (AMD, Intel ou NVIDIA) pour Qwen et son arbitre d’interruptions. d3-mini se charge dès 6 outils, dès 2 si leurs schémas totalisent au moins 3 072 octets, ou si vous activez le raccrochage par d3. Un profil CPU est aussi disponible. |
| 🧠 **Mémoire vive** | Sur un GPU intégré comme la Radeon 890M, les trois moteurs et le système se partagent la RAM. Voir le [profil matériel](docs/installation.md#radeon-890m-du-ryzen-ai-9-hx-470). |
| 💾 **Disque** | Environ 9 Go de modèles pour le profil par défaut, auxquels s’ajoutent les moteurs et l’environnement Python |
| 🐳 **Docker** | Pour la recherche web (Docker Desktop sous Windows). Sans Docker, tout le reste fonctionne |
| 🤗 **Compte Hugging Face** | Gratuit, pour télécharger la voix Phonon (modèle à conditions) |

> [!NOTE]
> Une carte NVIDIA accélère aussi la transcription (CUDA). Sans elle, Parakeet tourne sur le processeur, un peu plus lentement.

## 🚀 Installer

**1. Récupérer Localive et l’outil `uv` sous Windows**. Pour Linux, suivez les [instructions d’installation Linux](docs/installation.md#linux).

```powershell
git clone https://github.com/lfpoulain/localive.git
cd localive
winget install --id=astral-sh.uv -e
```

**2. Autoriser la voix Phonon** : acceptez ses conditions sur [huggingface.co/gradium/phonon-40m-fr](https://huggingface.co/gradium/phonon-40m-fr), puis connectez votre compte au moment indiqué par l'installeur (`hf auth login`).

**3. Lancer l'installeur**

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup.ps1      # Windows
```

```bash
scripts/install-linux.sh --systemd                               # Linux (service au démarrage)
```

L’installeur télécharge llama.cpp, Qwen, le routeur d3-mini Q8_0 (5,16 Go), la transcription, la fin de tour et la voix. Il vérifie les poids GGUF par SHA-256 et reprend les téléchargements interrompus. Après une mise à jour, Localive récupère aussi d3-mini en arrière-plan si ses poids manquent et que le tri des outils ou le raccrochage par d3 est activé. [Installation](docs/installation.md)

## ▶️ Lancer

Sous Windows, double-cliquez sur **`Lancer Localive.cmd`**. La page d’administration affiche l’état des moteurs. Sous Linux, lancez `scripts/start.sh` ou utilisez le service `localive` installé avec `--systemd`.

L’administration sur **<http://127.0.0.1:8787>** affiche l’état des moteurs et les conversations en cours. La page **Connecter un client** donne l’adresse WebSocket de votre application et l’adresse HTTP à renseigner dans Cornelius.

## Essayer le point d’entrée Live

Une fois le serveur prêt, ouvrez un deuxième terminal à la racine du dépôt. Le client
d’exemple envoie une question enregistrée et écrit la réponse dans `reponse.wav`.

```powershell
.venv\Scripts\python.exe scripts/live_client.py --url ws://127.0.0.1:8787/live --save reponse.wav tests/fixtures/wake_calculation.wav
```

Sous Linux, utilisez `.venv/bin/python`. Pour votre propre application, la première
trame WebSocket peut être `{"type":"session.start","session":{}}`. Après
`session.started`, envoyez le micro en PCM16 mono à 16 kHz. Le même serveur accepte
le chemin `/v1/live/sessions` utilisé par Cornelius. Les formats audio, les événements
et l’exemple de fonction personnalisée figurent dans le [guide du protocole](docs/protocole.md).

## 🔌 Brancher Cornelius

1. Dans Cornelius : **Administration → Mise en route → Voix de Cornelius**, fournisseur **OpenAI**.
2. Dans **Serveur Live**, collez l'adresse affichée par Localive, par exemple `http://127.0.0.1:8787/v1`.
3. Cornelius exige une clé `OPENAI_API_KEY` : gardez la vôtre si vous en avez une, sinon n'importe quelle valeur. Si vous définissez un jeton Localive, il doit être identique à cette clé.

Cornelius sur une autre machine ? Dans l'administration, **Serveur et sécurité** : adresse d'écoute `0.0.0.0` et un jeton d'accès. Tout est expliqué dans [Brancher Cornelius](docs/cornelius.md).

## 📚 Documentation

| Page | Contenu |
|---|---|
| [Installation](docs/installation.md) | Windows, Linux, carte graphique, Hugging Face, Docker, mise à jour |
| [Brancher Cornelius](docs/cornelius.md) | Serveur Live, clé, réseau local, satellites, dépannage |
| [Administration](docs/administration.md) | La page d'administration, le redémarrage, le journal |
| [Réglages](docs/reglages.md) | Tous les réglages, leurs valeurs par défaut et leur effet |
| [Fonctionnement](docs/fonctionnement.md) | Écoute, fin de tour, interruptions, consignes, recherche web |
| [Protocole](docs/protocole.md) | Brancher son client WebSocket ou WebRTC, envoyer l’audio, déclarer et exécuter ses outils |
| [Mesures](docs/mesures.md) | Les campagnes qui ont fixé les choix par défaut |
| [Développement](docs/developpement.md) | Tests, bancs de mesure, structure du code |

## 🔒 Confidentialité

Avec la configuration par défaut, l'audio, la transcription et les réponses sont traités sur votre ordinateur. Si vous configurez un serveur de modèles distant, celui-ci reçoit le texte du dialogue, les consignes, les schémas d'outils sélectionnés et les résultats nécessaires à la réponse. Les requêtes de recherche web partent via SearXNG vers les moteurs de recherche publics.

Le journal local contient des transcriptions, des arguments et des extraits de résultats d'outils. Il est conservé dans `.runtime/logs/localive.log` avec rotation (2 Mo par fichier, trois sauvegardes). Relisez les extraits avant de les joindre à un signalement.

Pour préchauffer le modèle au démarrage, Localive conserve aussi les consignes initiales et les schémas d’outils de la dernière session dans `.runtime/brain-prefix.json`.

> [!IMPORTANT]
> Photon, le moteur de transcription de Parakeet, envoie toutes les 60 secondes un rapport d'usage à `api.moondream.ai` (nom de la machine, carte graphique, nombre de requêtes ; ni audio ni texte), sans option pour le couper. Pour le bloquer, ajoutez `0.0.0.0 api.moondream.ai` et `:: api.moondream.ai` au fichier `hosts` (Windows : `C:\Windows\System32\drivers\etc\hosts`, en administrateur ; Linux : `/etc/hosts`). Localive fonctionne normalement.

Par défaut, Localive écoute sur `127.0.0.1`. L'accès aux réglages, au journal et à l'API d'administration depuis le réseau exige un jeton Localive.

## 📜 Licence

Le code de Localive est sous [PolyForm Noncommercial 1.0.0](LICENSE) : libre pour un usage personnel, familial, éducatif ou associatif, pas pour un usage commercial.

Les modèles téléchargés gardent leur propre licence, et certains sont **non commerciaux** :

| Modèle | Rôle | Licence |
|---|---|---|
| [Phonon 40m-fr](https://huggingface.co/gradium/phonon-40m-fr) | Voix | CC BY-NC 4.0, à conditions |
| [Parakeet Ultra](https://huggingface.co/moondream/parakeet-ultra) via Photon | Transcription | Poids CC BY 4.0 ; moteur Photon et noyaux CUDA propriétaires (M87 Labs) |
| [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) | Conversation et arbitrage des interruptions | Apache 2.0 |
| [d3-mini Q8_0](https://huggingface.co/ggml-org/vllm-d3-mini-GGUF) | Sélection des outils et raccrochage vocal optionnel | Apache 2.0 |
| [Smart Turn v3.2](https://huggingface.co/pipecat-ai/smart-turn-v3) | Fin de tour | BSD 2-Clause |
| [Silero VAD v5](https://github.com/snakers4/silero-vad) | Détection de voix | MIT (fourni dans `src/localive/assets`) |
| [SearXNG](https://github.com/searxng/searxng) | Recherche web | AGPL-3.0 (image Docker officielle, non modifiée) |

Polices de l'administration : DM Sans, DM Mono et Epilogue, sous SIL Open Font License.

<div align="center">
<sub>Un projet des Frères Poulain, pour l'écosystème <a href="https://github.com/lfpoulain/Cornelius">Cornelius</a>.</sub>
</div>
