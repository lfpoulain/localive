# Protocole

Localive expose un serveur vocal pour Cornelius et pour les applications qui parlent
le protocole décrit ici. Le point d’entrée canonique est `/v1/live/sessions`.
L’alias `/live` utilise les mêmes événements, transports et règles d’accès.
Si votre client utilise l’API Realtime avec `input_audio_buffer.append`, adaptez ses
événements et son format audio au protocole Live décrit ci-dessous.

## Essayer avec un fichier WAV

Démarrez Localive et vérifiez que `GET http://127.0.0.1:8787/health` renvoie
`"ready": true`. Depuis la racine du dépôt, avec l’environnement Python activé,
envoyez un enregistrement et écoutez le fichier de réponse.

```sh
python scripts/live_client.py --url ws://127.0.0.1:8787/live --save reponse.wav tests/fixtures/wake_calculation.wav
```

Vous pouvez fournir vos propres WAV PCM 16 bits, avec une demande par fichier. Le client
convertit l’entrée en mono 16 kHz et affiche les transcriptions. Sur une autre
machine, le script a uniquement besoin de Python, `aiohttp` et `numpy`. Utilisez
l’adresse du serveur dans `--url` et, si un jeton est configuré, fournissez-le
via la variable d’environnement `LOCALIVE_API_KEY`. L’option
`--report mesures.json` enregistre les délais par tour. Le profil facultatif
`--profile cornelius` ajoute les outils de démonstration du banc de mesure,
dont des réponses simulées pour la météo et la maison.

## Connecter son propre client WebSocket

1. Ouvrez `ws://127.0.0.1:8787/v1/live/sessions` et envoyez
   `{"type":"session.start","session":{}}`. Attendez `session.started`.
2. Envoyez des trames `session.input_audio.append`. Le champ `audio` contient
   les échantillons PCM16 **little-endian**, mono, 16 kHz, encodés en base64.
   Une trame de 40 ms représente 640 échantillons, soit 1 280 octets avant encodage.
   Gardez la cadence du microphone pendant les silences pour que la détection de fin de tour fonctionne.
3. Recevez les événements pendant l’envoi. Décodez le champ `delta` des
   `session.output_audio.delta` dans le même format pour jouer la voix, et
   affichez les champs `delta` de `session.input_transcript.delta` et
   `session.output_transcript.delta` si vous souhaitez des sous-titres.
   `voice.turn_completed` signale la fin de lecture d’une réponse.
4. Envoyez `{"type":"session.close"}` et attendez `session.closed`.

Dans `session.start`, vous pouvez fournir vos consignes dans `session.instructions`
et vos fonctions dans `session.tools`. Par exemple, un client qui sait lire son
capteur de température peut annoncer cette session.

```json
{"type":"session.start","session":{"tools":[{"type":"function","name":"serre__temperature","description":"Lire la température mesurée dans la serre.","parameters":{"type":"object","properties":{}}}]}}
```

À la réception d’un `response.event` dont `event.type` vaut
`response.output_item.done` et `event.item.type` vaut `function_call`, exécutez
la fonction de votre application. `event.item.arguments` contient les arguments
sous forme de chaîne JSON. Renvoyez le résultat avec le `call_id` reçu. Localive
reprend la réponse dès l’arrivée des résultats attendus.
Si la personne interrompt la réponse ou si l’attente expire, le client peut encore
renvoyer le résultat. Localive met à jour en silence le tour correspondant pour la
prochaine question, tant que ce tour figure dans l’historique.

```json
{"type":"response.item.create","item":{"type":"function_call_output","call_id":"call_exemple","output":"{\"temperature_c\":21.5}"}}
```

Pour changer les outils en cours de session, envoyez `session.update` avec la liste
complète des fonctions à conserver. Une mise à jour limitée aux consignes conserve
le catalogue en place. Localive transmet les petits catalogues directement à Qwen
et recalcule le besoin de tri chaque fois que vous modifiez la liste. [Seuils de sélection des outils](fonctionnement.md#sélection-des-outils)

### Vocabulaire des appareils après transcription

Dans **Réglages → Écoute**, activez **Vocabulaire des appareils après transcription**,
puis ouvrez une nouvelle conversation. Cornelius transmet déjà les noms et les alias
des appareils sélectionnés dans Home Assistant. Si Parakeet écrit « allume le plafonnier U »,
Localive peut ainsi proposer « Plafonnier Hue » au modèle pour choisir l’appareil.
Les sous-titres affichent toujours la transcription entendue.

Localive utilise les identifiants autorisés par `parameters.properties.entityId.enum`
de chaque outil `home_assistant__…`. La liste JSON en fin de description fournit les
champs `id`, `name` et `aliases`. Une mise à jour des outils remplace ce vocabulaire.
Les correspondances portent sur des mots entiers et privilégient le nom le plus long.
Si deux appareils d’un même outil correspondent au même nom, Localive retire cet outil
pour la demande et invite le modèle à demander une précision.

## Points d'entrée

| Méthode | Chemin | Rôle |
|---|---|---|
| `GET` (WebSocket) | `/v1/live/sessions` | Session complète : `session.start`, audio et événements |
| `POST` | `/v1/live/sessions` | Session WebRTC : `{"session": {...}, "transport": {"type": "webrtc", "sdp": "<offre>"}}` → `{"session": {"id"}, "transport": {"type": "webrtc", "sdp": "<réponse>"}}` |
| `GET` (WebSocket) | `/v1/live/sessions/{id}/attach` | Contrôle d'une session WebRTC : événements seulement |
| `POST` | `/v1/live/sessions/{id}/hangup` | Raccrocher |
| `GET` | `/v1/models`, `/v1/models/{id}` | `gpt-live-1` et les cerveaux (modèles de délégation) |
| `GET` | `/health` | État, moteurs, recherche web, transports |

En utilisant l’alias, les routes WebRTC de contrôle deviennent `/live/{id}/attach`
et `/live/{id}/hangup`.

Authentification : `Authorization: Bearer <jeton>` si `server.api_key` est défini. Une requête de
navigateur venant d'un autre site (en-tête `Origin` différent) est refusée.
Pour une application web hébergée sur un autre domaine, ouvrez la connexion Live
depuis votre backend et conservez le jeton sur celui-ci. Sur le réseau local,
remplacez `127.0.0.1` par l’adresse de la machine qui héberge Localive.

En WebRTC, la session démarre à la création ; le canal de données `oai-events` reçoit seulement
`session.started`, `session.closed` et `error` ; tout le reste passe par la WebSocket `attach`.
Une offre sans candidats ICE (envoyée avant leur collecte, comme le fait un navigateur) est
acceptée : Localive apprend l'adresse du navigateur par ses vérifications de connectivité.

## Événements du client

| Événement | Prise en charge |
|---|---|
| `session.start` | Consignes, voix, `delegation.responses` (consignes, outils, modèle), `welcome_message` / `welcome_instructions`, `localive.metrics`, `localive.end_on_goodbye`, `localive.goodbye_mode`, `localive.wait_for_reply` |
| `session.update` | Outils, consignes, voix, modèle de délégation et options `localive`. Confirmation `session.updated` |
| `session.input_audio.append` | PCM16 16 kHz, blocs d'une seconde au plus (WebSocket seulement) |
| `session.input_audio.mute` / `unmute` | Ignorer le micro ; répond `session.input_audio.muted` / `unmuted` |
| `session.instructions.append` | Consigne ajoutée pour la suite de la session |
| `session.commentary.append` | Information ou consigne à dire maintenant (dite plus tard si l'assistant parle) |
| `session.thinking.append` | Contexte silencieux (caméra) ; répond `session.thinking.appended` avec `client_event_id` |
| `response.item.create` | `function_call_output` d'un outil |
| `response.create` | Accepté (la réponse reprend d'elle-même quand les sorties d'outils sont arrivées) |
| `session.close` | Ferme la session ; répond `session.closed` |

### Raccrocher à la voix

Le serveur applique `turn.end_on_goodbye` (désactivé par défaut) et
`turn.goodbye_mode` aux nouvelles connexions. Le mode `phrases` utilise les formules
prédéfinies. Le mode `d3` utilise le décideur local, même avec un catalogue vide.
Un client peut régler ces deux options dans `session.start` ou en cours d’appel.

```json
{"type":"session.update","session":{"localive":{"end_on_goodbye":true,"goodbye_mode":"d3"}}}
```

Après une formule de fin explicite, le client reçoit la transcription puis
`session.closed` avec `reason: "user_goodbye"`. Localive ferme la WebSocket et,
en WebRTC, la connexion média. Le client peut alors repasser en veille. En mode `d3`, la
session reste ouverte pendant le chargement du modèle et si la décision est incertaine
ou hors délai. `GET /health` expose `decision_model_state`, qui vaut `ready` après
une décision de contrôle réussie.

En mode `d3`, « C’est bon » juste après un remerciement tient compte de la dernière
réponse de l’assistant. Une simple formule de politesse permet de conclure en une
décision. Une proposition d’action à confirmer laisse la conversation ouverte.

Dans **Voix**, l’option **Parler pendant une attente** permet à Phonon de jouer
une courte phrase au plus une fois par demande lente. Si un appel au client est
déjà parti et attend son résultat, la voix dit « J’attends le résultat ». Pendant la
réflexion, elle choisit parmi les 12 phrases mélangées pour la session. Le journal
indique quand le premier son de l’interjection est prêt. Les mesures de premier audio
continuent à chronométrer la réponse elle-même.

### Attendre la réponse avant de réécouter

Avec `localive.wait_for_reply` (désactivé par défaut, réglage serveur
`turn.wait_for_reply`), Localive ignore le micro dès qu’il valide la fin d’une demande.
Une voix dans la pièce ne peut alors ni interrompre ni remplacer la demande pendant
la réflexion. Une phrase d’attente (« Je cherche ça… ») garde le micro fermé. Le premier
son transmis de la réponse le rouvre et permet de l’interrompre. Si le modèle échoue
ou termine sans réponse, Localive le signale à voix haute avant de réécouter. Si la
synthèse vocale échoue aussi, le micro se rouvre à la fin du tour. L’accueil et les
annonces du client laissent le micro ouvert.

```json
{"type":"session.update","session":{"localive":{"wait_for_reply":true}}}
```

Le client n’a rien à suspendre lui-même : il peut continuer d’envoyer son audio.

### Réponse vocale directe d’un outil

Un module peut utiliser le résultat Cornelius habituel. Cet exemple montre le contenu
de `output` pour un appel à un module de vélos.

```json
{"schemaVersion":1,"status":"succeeded","summary":"Il reste quatre vélos à la gare.","result":{"available":4},"error":null}
```

Pour un outil dont les paramètres forment un objet fermé (`additionalProperties: false`),
Localive propose à Qwen un argument facultatif, `speak_summary`. Qwen peut le choisir si
le résumé d’un seul appel suffit pour répondre à toute la demande. Localive retire cet
argument avant l’envoi au module. Le module conserve son schéma, son nom et ses arguments.
Un outil qui déclare déjà une propriété de ce nom la conserve.

Après un succès, Phonon lit le résumé court (300 caractères maximum). Si
`result.accepted` vaut `true` sans confirmation, la voix dit « La demande a été transmise ».
Les erreurs, les données périmées, les appels multiples, les résultats sans résumé exploitable
et les demandes qui nécessitent une interprétation suivent le circuit habituel avec Qwen.
Dans `localive.turn`, `direct_tool_reply: true` indique que le second passage a été économisé.

## Événements du serveur

| Événement | Contenu |
|---|---|
| `session.started`, `session.closed` | `session.id` ; `reason` |
| `session.output_audio.delta` | Audio PCM16 16 kHz en temps réel, silence compris (WebSocket seulement) |
| `session.input_transcript.delta` | Ce qui a été entendu, avec `start_ms` / `end_ms` |
| `session.output_transcript.delta` | Ce qui est dit, phrase par phrase, avec `start_ms` / `end_ms` |
| `session.delegation.created` | La réponse est confiée au moteur de raisonnement. `offset_ms` situe la délégation, `delegation.id` l’identifie et `delegation.response_id` identifie sa première réponse |
| `response.event` | Enveloppe : `response.created`, `response.output_item.done` (`function_call`, `arguments` en chaîne JSON), `response.completed`, et pour la recherche web `response.output_item.added` / `done` (`web_search_call`), `response.web_search_call.in_progress` / `searching` / `completed`, message avec annotations `url_citation` |
| `voice.state` | Extension Localive pour l’orbe. États `listening`, `thinking`, `speaking`. L’état `thinking` est renouvelé pendant une attente prolongée. Avec `speaking`, `interjection: true` signale une phrase d’attente : la demande est toujours en cours. La réponse renvoie `speaking` avec `interjection: false` |
| `voice.interrupted`, `voice.turn_completed` | Vider la lecture ; fin du tour |
| `voice.provider_warning` | Problème non bloquant (outil, recherche, événement mal formé) |
| `localive.turn` | Avec `session.localive.metrics = true`, délais en millisecondes et nombre d’outils retenus pour le tour |
| `error` | Code et message dans `error`, notamment `engine_not_ready` si les moteurs chargent encore ou `invalid_event` pour une configuration de session incorrecte |

Les voix GPT-Live (`cedar`, `marin`…) sont traduites en voix masculine ou féminine
(`voice.male_names` dans `localive.toml`).

Pour afficher « Je réfléchis », utilisez `session.delegation.created`. Localive l’émet
avant le tri des outils et la génération de Qwen, y compris pour une réponse parlée
sans outil. Une délégation vers `responses` regroupe les appels successifs au modèle.
Chacun commence par `response.created` dans une enveloppe `response.event` et se termine par
`response.completed`, `response.incomplete` en cas d’interruption ou `response.failed` en cas
d’échec. Après chaque prise de parole, `session.input_transcript.delta` contient le
texte reconnu par Parakeet. Les sous-titres de l’assistant arrivent dans
`session.output_transcript.delta`. [Transcriptions Live](https://developers.openai.com/api/docs/guides/live-conversations)
[Délégations Live](https://developers.openai.com/api/docs/guides/live-delegation)

Pour mesurer un essai vocal avec votre propre client, ouvrez **Journal** dans
l’administration. Pour chaque demande avec des outils, la ligne du tour indique
si le routeur a été consulté, combien de schémas ont été transmis au modèle et le
temps consacré au tri. La ligne « premier audio prêt » mesure le délai côté serveur
entre la fin de parole estimée et le premier échantillon de réponse disponible.
Elle détaille aussi le délai du test de raccrochage d3, le tri et l’attente du premier
texte ou appel d’outil produit par le cerveau, en millisecondes, lorsque ces étapes ont eu lieu.
Le transport et la lecture sur le client s’ajoutent à ce délai.
