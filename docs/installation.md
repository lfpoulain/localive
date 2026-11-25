# Installation

Localive s'installe dans son dossier : un environnement Python (`.venv`), le moteur llama.cpp et
les modèles (`models/`, `.runtime/`). Rien n'est installé ailleurs, sauf l'outil `uv` et, pour la
recherche web, Docker.

## Ce que l'installeur télécharge

| Élément | Taille | Vérification |
|---|---|---|
| llama.cpp b11960 (CPU ou Vulkan sous Windows et Linux, CUDA aussi sous Linux) | 23 à 183 Mo | SHA-256 de la publication GitHub |
| Qwen3-4B-Instruct-2507 Q4_K_M (conversation et interruptions) | 2,50 Go | SHA-256 publié par Hugging Face |
| d3-mini Q8_0 (sélection des outils) | 5,16 Go | SHA-256 du GGUF publié par ggml-org |
| Smart Turn v3.2 (fin de tour) | 33 Mo | SHA-256 publié par Hugging Face |
| Parakeet Ultra (transcription) et Phonon 40m-fr (voix) | 1,4 Go | Cache Hugging Face |
| PyTorch pour Parakeet | Taille variable selon CPU ou CUDA | Index officiel PyTorch pour le paquet CPU |

Une installation interrompue reprend où elle s'était arrêtée ; relancer l'installeur ne
retélécharge rien de ce qui est déjà vérifié.

Sur Windows et Linux x86-64, une nouvelle installation avec transcription sur CPU prend
PyTorch sur son index CPU. Cela évite de télécharger les bibliothèques CUDA de PyTorch
sur le HX 470. Les bibliothèques propres à Parakeet restent nécessaires. L’installeur
conserve une version compatible de PyTorch déjà présente. [Index PyTorch CPU](https://download.pytorch.org/whl/cpu/torch/)

## Avant de commencer : la voix Phonon

Phonon est un modèle « à conditions » : il faut les accepter une fois.

1. Créez un compte gratuit sur [huggingface.co](https://huggingface.co).
2. Ouvrez [huggingface.co/gradium/phonon-40m-fr](https://huggingface.co/gradium/phonon-40m-fr) et acceptez les conditions.
3. Pendant l'installation, connectez votre compte : `.venv\Scripts\hf.exe auth login` (Windows) ou
   `.venv/bin/hf auth login` (Linux). Vous pouvez aussi définir la variable `HF_TOKEN`, ou coller un
   jeton dans l'administration (Serveur et sécurité → Jeton Hugging Face).

Pas encore prêt ? Installez avec `-SkipVoice` (Windows) ou `--skip-voice` (Linux), puis relancez
l'installeur plus tard.

## Windows

```powershell
winget install --id=astral-sh.uv -e        # puis rouvrir PowerShell
git clone https://github.com/lfpoulain/localive.git
cd localive
powershell -ExecutionPolicy Bypass -File scripts/setup.ps1
```

L’option `-Device cpu|cuda` choisit le processeur de transcription. Le réglage automatique
utilise une carte NVIDIA disponible. L’option `-Llama cpu|vulkan` choisit le moteur du modèle
conversationnel. Pour installer Localive sur une machine sans carte graphique, utilisez
`scripts/setup.ps1 -Device cpu -Llama cpu`. L’option `-SkipVoice` reporte Phonon et
`-Models gemma-e4b,qwen3.5-4b` télécharge d’autres modèles de raisonnement.

À la première installation, le cerveau utilise par défaut llama.cpp **Vulkan**, compatible avec les cartes graphiques
récentes (NVIDIA, AMD, Intel). Pour la recherche web, installez
[Docker Desktop](https://www.docker.com/products/docker-desktop/). La première session qui
annonce l’outil hébergé `web_search` déclenche la préparation de SearXNG en arrière-plan.
Le bouton **Préparer maintenant** de l’administration permet de l’anticiper.

Lancer : double-clic sur **`Lancer Localive.cmd`**, ou
`powershell -ExecutionPolicy Bypass -File scripts/start.ps1`.

> [!NOTE]
> Docker Desktop ne donne pas accès aux cartes AMD et Intel depuis un conteneur sous Windows :
> c'est pourquoi Localive tourne directement sous Windows, et seul SearXNG est dans Docker.

## Linux

Prérequis : `uv`, et pour Vulkan le chargeur et le pilote de la carte
(Debian/Ubuntu : `sudo apt install libvulkan1 mesa-vulkan-drivers`, ou le pilote NVIDIA).

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/lfpoulain/localive.git
cd localive
scripts/install-linux.sh --systemd
```

Options :

| Option | Effet |
|---|---|
| `--llama vulkan\|cuda\|cpu` | Moteur du cerveau. Par défaut Vulkan ; CUDA si une carte NVIDIA est là sans Vulkan |
| `--device cpu\|cuda` | Processeur de la transcription |
| `--systemd` | Installe et démarre le service utilisateur `localive` |
| `--skip-voice` | Reporte Phonon |
| `--models gemma-e4b,qwen3.5-4b` | Autres modèles de raisonnement (voir ci-dessous) |
| `--no-tool-router` | Omet le routeur d’outils à l’installation |

Avec `--systemd` : `systemctl --user status localive`, `journalctl --user -u localive -f`.
Pour qu'il démarre sans session ouverte : `sudo loginctl enable-linger $USER`.
Sans service : `scripts/start.sh`.

## Installer le décideur d’outils

L’installation standard télécharge et vérifie `models/decision/vllm-d3-mini-Q8_0.gguf`.
Après une mise à jour, un redémarrage de Localive suffit pour récupérer ce fichier
s’il manque et que le tri des outils ou le raccrochage par d3 est activé. Le transfert
de 5,16 Go se fait en arrière-plan. Sa progression est visible sur la page d’accueil de
l’administration et dans `/health` (`decision_model_download_pct`). Un transfert
interrompu reprend au prochain essai. Localive vérifie le SHA-256 avant d’utiliser le modèle.
Localive charge d3-mini en arrière-plan avec llama.cpp Vulkan quand une session
déclare au moins six outils, ou deux à cinq outils dont les schémas totalisent au
moins 3 072 octets. Le raccrochage vocal en mode `d3` charge aussi le modèle, même
si le tri des outils est désactivé. Si ce mode est activé pour toutes les nouvelles
sessions, le chargement commence dès que les moteurs vocaux sont prêts.
Le modèle reçoit la transcription et choisit les outils à présenter à Qwen. Si le
routeur est indisponible ou dépasse le délai du tour, Qwen reçoit le catalogue complet.
[Poids texte d3-mini](https://huggingface.co/ggml-org/vllm-d3-mini-GGUF/tree/8b29fd3b87373ea85fa59cc2f26251c1591d771a)

En mode hors ligne ou avec un modèle personnalisé, placez les poids au chemin
configuré dans **Cerveau → Fichier GGUF du décideur d’outils**.
Pour activer le tri sur une installation où il avait été désactivé, lancez
`powershell -ExecutionPolicy Bypass -File scripts/setup.ps1 -ToolRouter` sous Windows
ou `scripts/install-linux.sh --tool-router` sous Linux, puis redémarrez Localive.
L’option prépare le modèle configuré et enregistre son activation.
Pour omettre ce téléchargement pendant l’installation, utilisez `-NoToolRouter` sous
Windows ou `--no-tool-router` sous Linux. Au démarrage, le serveur suit les réglages
enregistrés. Pour éviter le téléchargement au démarrage, désactivez le tri dans
**Cerveau** et gardez le raccrochage désactivé ou en mode `phrases`. Un client qui
active ensuite le raccrochage en mode `d3` demande aussi la préparation du modèle.

Une fois le tri activé, Localive repère aussi les poids ajoutés pendant son exécution.
Au prochain grand catalogue ou tour qui doit être trié, il charge le modèle en
arrière-plan. L’état `ready` de `/health` indique qu’une décision de contrôle a abouti
sur `/v1/systemone`. Après un échec de téléchargement ou de démarrage, un tour
éligible peut relancer l’opération après 30 secondes. Les demandes vocales continuent
avec le catalogue complet entre-temps.

Le routeur a son propre réglage de moteur. Son profil GPU utilise
`.runtime/llama/b11960-vulkan` et `routing.gpu_layers = 99`, ce qui demande le
chargement de toutes les couches sur le GPU. Une première installation explicitement
CPU (`-Llama cpu` ou `--llama cpu`) configure aussi le routeur sur CPU.

### Radeon 890M du Ryzen AI 9 HX 470

Le HX 470 intègre une Radeon 890M à 16 unités de calcul. Elle utilise la RAM du
serveur pour ses allocations graphiques. Il faut y faire tenir les poids et caches
de trois moteurs (Qwen pour la conversation, Qwen pour les interruptions et d3-mini),
ainsi que la transcription, la voix et le système. [Fiche AMD du HX 470](https://www.amd.com/en/products/processors/laptop/ryzen/ai-400-series/amd-ryzen-ai-9-hx-470.html)

Avec les contextes par défaut, llama.cpp a déclaré **11,88 Gio de buffers Vulkan**
sur la RTX 4090 de test (4,21 pour Qwen, 3,00 pour son arbitre et 4,66 pour d3-mini).
Un serveur qui reçoit uniquement de petits catalogues utilise les deux moteurs Qwen
(environ 7,21 Gio). Ces volumes excluent la mémoire du système et des
services vocaux. Pour le profil complet
sur une 890M, nous conseillons **32 Go de RAM**. Avec 16 Go partagés, la marge est
faible. Les mesures sur Radeon du dépôt portent sur Qwen avec un HX 370. La latence
de d3-mini sur HX 470 reste à mesurer. [Mesures matérielles](mesures.md#empreinte-du-profil-gpu-par-défaut)

Sur une machine AMD, utilisez `-Device cpu -Llama vulkan` sous Windows ou
`--device cpu --llama vulkan` sous Linux. La transcription utilise alors le CPU et
llama.cpp utilise la Radeon. Si plusieurs GPU sont présents, `llama-server --list-devices`
dans le dossier du moteur Vulkan affiche leurs identifiants. Le routeur demande
`Vulkan0` par défaut. La variable d’environnement llama.cpp `LLAMA_ARG_DEVICE`
permet de choisir une autre carte pour les moteurs qui la reçoivent.

Pour vérifier le chargement, démarrez Localive avec `LLAMA_ARG_LOG_VERBOSITY=4`
dans son environnement. Dans **Journal**, la ligne « couches sur GPU » reprend le
placement annoncé par chaque moteur llama.cpp que Localive démarre. Ouvrez une
session avec un catalogue qui déclenche le tri,
par exemple six fonctions de votre application ou la configuration Cornelius de
`bench/prompts/cornelius_session.json` (cinq fonctions, 3 197 octets avant la recherche web).
Une fois le routeur prêt sur l’accueil,
cherchez `Radeon 890M` et `offloaded 33/33 layers to GPU` dans
`.runtime/logs/brain-tool-router.log`. Pour mesurer le tri seul, lancez
`python bench/prompts/route_tools.py --port 8085` pendant que le serveur est au repos.
Pour mesurer l’effet du tri sur un tel catalogue, comparez également le premier audio
avec et sans **Cerveau → Choisir les outils après la transcription**, en redémarrant
après le changement. [Banc vocal et mesures](mesures.md#refaire-la-mesure-sur-un-hx-470)

## Autres modèles de raisonnement

L'installation télécharge Qwen3-4B, le modèle mesuré. D'autres modèles s'ajoutent à tout moment
depuis la page **Cerveau** de l'administration : le catalogue ci-dessous, rangé par famille, ou
n'importe quel GGUF d'un dépôt Hugging Face public (« auteur/nom », puis choix du fichier). Ils
apparaissent ensuite dans la liste « Modèle de raisonnement ». L'installation peut aussi les
télécharger (la relancer conserve ce qui est déjà vérifié).

| Nom | Famille | Modèle | Taille |
|---|---|---|---|
| `qwen3.5-2b` | Qwen | Qwen3.5-2B | 1,3 Go |
| `qwen3.5-4b` | Qwen | Qwen3.5-4B | 2,7 Go |
| `qwen3.5-4b-mtp` | Qwen | Qwen3.5-4B MTP (UD-Q4_K_XL) | 3,0 Go |
| `qwen3.5-9b` | Qwen | Qwen3.5-9B | 5,7 Go |
| `qwen3.5-27b` | Qwen | Qwen3.5-27B | 16,7 Go |
| `qwen3.5-35b-a3b` | Qwen | Qwen3.5-35B-A3B | 22,0 Go |
| `gemma-e2b` | Gemma | Gemma 4 E2B | 3,1 Go |
| `gemma-e4b` | Gemma | Gemma 4 E4B | 5,0 Go |
| `gemma-12b` | Gemma | Gemma 4 12B | 7,1 Go |
| `gemma-26b` | Gemma | Gemma 4 26B-A4B | 13,6 Go |
| `gemma-31b` | Gemma | Gemma 4 31B | 18,3 Go |
| `ministral-3b` | Mistral | Ministral 3 3B | 2,2 Go |
| `ministral-8b` | Mistral | Ministral 3 8B | 5,2 Go |
| `ministral-14b` | Mistral | Ministral 3 14B | 8,2 Go |
| `mistral-small-24b` | Mistral | Mistral Small 3.2 24B | 14,3 Go |
| `gpt-oss-20b` | OpenAI | gpt-oss 20B | 12,1 Go |
| `lfm2.5-1.2b` | Liquid (LFM) | LFM2.5 1.2B Instruct | 0,7 Go |
| `lfm3-1.7b` | Liquid (LFM) | LFM3 1.7B | 1,1 Go |
| `lfm2.5-2.6b` | Liquid (LFM) | LFM2.5 2.6B | 1,7 Go |
| `lfm2.5-8b-a1b` | Liquid (LFM) | LFM2.5 8B-A1B | 5,2 Go |
| `lfm2.5-24b-a2b` | Liquid (LFM) | LFM2.5 24B-A2B | 14,5 Go |
| `minicpm5-1b` | MiniCPM | MiniCPM5-1B | 0,7 Go |
| `minicpm5-2b` | MiniCPM | MiniCPM5-2B | 1,6 Go |

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup.ps1 -Models gemma-e4b,qwen3.5-4b
```

`all` les télécharge tous (environ 170 Go). Sous Linux : `scripts/install-linux.sh --models gemma-e4b`.

Pour la recherche web, installez Docker (`docker` utilisable sans `sudo` par votre utilisateur).

### Changer le port ou l’image de SearXNG

Chaque conteneur géré par Localive utilise le port publié et l’image choisis lors de sa
création. À la préparation de la recherche, Localive vérifie son dossier de configuration, son image et son
port. Il reconnaît également les conteneurs des anciennes versions grâce au montage de
`.runtime/searxng`.

Si l’administration indique un conflit, ouvrez **Recherche web**. Choisissez un nouveau nom
de conteneur et une adresse locale avec un port disponible, par exemple
`http://127.0.0.1:8889`, puis redémarrez Localive. Il crée le nouveau conteneur avec
l’image demandée. L’ancien reste disponible dans Docker. Pour confier la gestion du conteneur
à un autre service, désactivez **Gérer SearXNG dans Docker** et indiquez son adresse.

Si le serveur répond mais refuse le format JSON, ajoutez `json` aux formats de recherche
dans le fichier `settings.yml` de cette instance SearXNG, redémarrez son conteneur puis
cliquez sur **Réessayer**.

> [!WARNING]
> L'installation Linux est écrite et vérifiée syntaxiquement, mais n'a pas encore été essayée sur
> une vraie machine Linux. Les retours sont bienvenus.

## Mettre à jour

```bash
git pull
```

Pour une mise à jour complète des dépendances et des moteurs, relancez l’installeur puis
redémarrez Localive avec le bouton « Redémarrer » de l’administration. Pour récupérer
les poids manquants de d3-mini, le redémarrage suffit. Vos réglages sont conservés dans
`localive.toml`, qui n’est pas suivi par Git.

## Désinstaller

Arrêtez Localive, supprimez son dossier, et si vous le souhaitez le conteneur SearXNG :
`docker rm -f localive-searxng`. Sous Linux avec le service :
`systemctl --user disable --now localive` puis supprimez `~/.config/systemd/user/localive.service`.
