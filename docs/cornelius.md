# Brancher Cornelius

Localive implémente l'API Live d'OpenAI (`/v1/live/sessions`) : pour Cornelius, c'est un
serveur Live comme un autre. Aucun module ni correctif n'est nécessaire.

## En trois étapes

1. Dans Cornelius : **Administration → Mise en route → Voix de Cornelius**, fournisseur de la
   conversation **OpenAI**.
2. Dans **Serveur Live**, collez l'adresse de Localive. L'administration de Localive l'affiche
   dans **Connecter un client → Cornelius**. Utilisez `http://127.0.0.1:8787/v1` si les
   deux tournent sur la même machine.
3. **Clé OpenAI** : Cornelius exige `OPENAI_API_KEY` pour utiliser un serveur Live. Gardez la
   vôtre si vous en avez une (d'autres fonctions de Cornelius s'en servent) ; sinon, n'importe
   quelle valeur. Cornelius l'envoie à Localive, jamais à OpenAI pour la conversation.

Le réglage s'applique à la conversation suivante. Elle apparaît sur l'accueil de
l'administration de Localive.

Équivalent dans le `.env` de Cornelius : `CORNELIUS_LIVE_BASE_URL=http://127.0.0.1:8787/v1`
(la valeur saisie dans l'administration de Cornelius est prioritaire).

## Cornelius sur une autre machine

Dans l'administration de Localive, **Serveur et sécurité** :

1. **Adresse d'écoute** : `0.0.0.0` (tout le réseau local).
2. **Jeton d'accès** : choisissez un jeton, et mettez **la même valeur** dans `OPENAI_API_KEY`
   chez Cornelius (Cornelius envoie sa clé comme jeton).
3. Redémarrez Localive (bouton en haut de l'administration).

Dans Cornelius, utilisez alors l'adresse réseau affichée par Localive, par exemple
`http://192.168.1.50:8787/v1`. Le pare-feu de la machine Localive doit autoriser le port 8787
(TCP) et, pour le miroir en WebRTC, les ports UDP éphémères.

## Qui se connecte comment

| Client | Transport | Audio |
|---|---|---|
| Miroir (navigateur) | WebRTC : offre SDP via le serveur Cornelius, puis WebSocket de contrôle `…/attach` | Opus, directement entre le navigateur et Localive |
| Satellites | WebSocket `/v1/live/sessions` | PCM 16 kHz relayé par Cornelius |
| Autres clients Live | WebSocket ou WebRTC | Idem |

Le miroir échange son audio **directement** avec Localive : sur le réseau local, le navigateur
doit pouvoir joindre la machine Localive. Aucun serveur STUN extérieur n'est contacté.

## Ce qui change pour Cornelius

- La conversation, la transcription et la voix sont locales.
- Les outils de Cornelius (heure, météo, bus, maison, mémoire, cartes, agents) fonctionnent
  comme avec OpenAI : Localive les appelle et attend leur résultat.
- La recherche web (module Web de Cornelius) passe par SearXNG, sur la machine Localive.
- Le modèle de délégation choisi dans Cornelius est ignoré : Localive répond avec le
  modèle de raisonnement choisi dans son administration (page Cerveau).
- Les voix GPT-Live (Cedar, Marin…) deviennent la voix masculine ou féminine de Localive.

## Dépannage

| Symptôme | Piste |
|---|---|
| « Connexion trop longue » dans Cornelius | Localive démarre encore (accueil de l'administration), ou l'adresse est fausse |
| « Erreur Live » dès l'ouverture | Journal de Localive : jeton refusé, format audio, cerveau indisponible |
| Le miroir ne s'entend pas, les satellites oui | WebRTC bloqué : pare-feu UDP de la machine Localive, ou navigateur sur un autre réseau |
| Il ne cherche pas sur le web | Accueil de l'administration → Recherche web : Docker absent ou arrêté |
| Il n'entend pas une phrase | Journal : « Speech discarded », « Echo ignored » ; réglages Écoute |
| La première réponse est lente | Le cerveau lit les longues consignes de Cornelius (plusieurs secondes la première fois, puis en cache) |
