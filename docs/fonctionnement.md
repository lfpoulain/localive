# Fonctionnement

```
Client vocal (Cornelius ou votre application) ──WebRTC / WebSocket──▶ Localive
   micro ─────────────────────────────▶ Silero (voix ?) ─▶ Smart Turn (fini ?) ─▶ Parakeet (texte)
                                                                       │
   outils du client ◀───── appels d'outils ──── Qwen (cerveau) ◀───────┘
   web_search ──────────────────────────────▶ SearXNG (Docker, local)
   haut-parleur ◀──────────── Phonon (voix) ◀── phrases
                       décideur (Qwen dédié) : que faire quand on parle par-dessus ?
```

## Tours de parole

- **Fin de phrase** : Silero entend la pause ; à chaque pause de 200 ms, **Smart Turn v3.2**
  (environ 32 ms sur le processeur) et la transcription disent si la phrase est finie. Une reprise
  de parole pendant leur calcul invalide le verdict : la phrase reste ouverte. Inachevée
  (« mets un minuteur de… ») : Localive attend jusqu'à 3 s ; sans suite, le cerveau sait que la
  phrase était inachevée et invite à continuer au lieu de deviner.
- **Quand la transcription a des mots**, la phrase est gardée même si Silero a compté peu de
  voix (une voix atténuée par l'annulation d'écho, sous celle de l'assistant).
- **Par-dessus l'assistant**, le micro mélange les deux voix et Smart Turn se trompe : la fin de
  phrase se décide alors sur les mots, après 700 ms de pause (réglage « Pause de fin quand vous lui
  coupez la parole »).

## Interruptions

Quand on parle pendant que Cornelius parle, sa voix baisse aussitôt (40 %, puis 12 % une fois
une vraie voix confirmée et l'écho écarté) et le décideur juge la phrase entière :

- **ignorer** (mm-hm, télé, « Julie, … ») : la voix remonte, rien n'est coupé ; la phrase est
  gardée 4 s pour être recollée si la suite s'adresse finalement à lui ;
- **après** (« et Nantes ? » pendant la météo de Bordeaux) : il glisse « Je te donne Nantes
  juste après. », termine, puis enchaîne ;
- **correction** (« non, à Lyon », « en anglais ») : il finit la phrase en cours, abandonne la
  suite prévue et refait la réponse demandée, à partir de ce qui a vraiment été dit ;
- **tout de suite** (« allume la lumière », « quelqu'un à la porte ») : il finit sa phrase et
  traite la demande ;
- **clôture** (« ok merci, c'est bon ») : il se tait et se remet à écouter par défaut.

Dans **Fin de tour**, activez **Raccrocher sur une formule de fin**, puis choisissez
la **Détection du raccrochage** pour les nouvelles conversations.

- **Formules prédéfinies** reconnaît sur CPU des phrases entières comme « au revoir »
  et « OK, c’est bon, merci ».
- **Décideur local (d3-mini)** interprète la transcription finale après le filtrage
  de l’écho. Il peut reconnaître « je file, on se reparle demain ». Une demande comme
  « donne-moi la météo avant de raccrocher » doit recevoir sa réponse. Le modèle
  reçoit la phrase et le nom de l’assistant. Une formule prédéfinie peut renforcer
  une décision positive du modèle. Si le détecteur acoustique juge le tour inachevé,
  Localive attend la fin du délai de silence. Il consulte alors d3 si la transcription
  entière est une formule de fin complète, comme « On peut arrêter la session, c’est bon ».
  Une décision incertaine, un modèle indisponible ou un délai dépassé laissent la session ouverte.

Le mode d3 partage le processus utilisé pour le tri des outils. Il fonctionne aussi
sans outils et quand le tri est désactivé. La vérification dédiée du raccrochage
dispose de 800 ms au maximum. Si vous reprenez la parole pendant le calcul, la session reste ouverte.
Après la transcription d’un nouveau tour, si le catalogue mérite un tri, d3-mini
choisit en une question entre répondre directement, proposer des outils et raccrocher.
S’il propose de raccrocher ou attribue à cette possibilité un score d’au moins 0,05,
Localive lui demande de confirmer avec sa question de fin de conversation. Seule cette
confirmation ferme la connexion. Un doute sur les outils permet de transmettre le
catalogue complet à Qwen. Pour un nouveau tour sans tri, d3 répond d’abord à une question
courte sur la possibilité d’un adieu. Un score inférieur à 0,20 laisse la conversation
continuer immédiatement et Qwen reçoit les outils disponibles. Sinon, d3 vérifie la
phrase complète avant tout raccrochage. Une formule d’adieu reconnue, une interruption
ou un budget inférieur à 800 ms passe directement à cette vérification. Les deux
questions éventuelles se partagent le même délai maximal.
Le mode **Formules prédéfinies** reconnaît les expressions localement.

Quand le catalogue est assez grand pour être trié, « merci » seul reçoit une réponse
sans outil. Pour « OK », « Okay » ou « OK, merci », d3 examine la dernière réponse de l’assistant et la
demande qui l’a précédée. S’il identifie une action proposée qui attend encore un
accord, Qwen reçoit les outils correspondants. Après une action terminée, ou dans
le doute, la réponse reste sans outil. Une nouvelle demande comme « merci, allume la
lampe » et un adieu comme « c’est bon, merci » suivent le routage habituel.

Pendant qu'il réfléchit encore (rien n'a été dit), un ajout est fusionné à la demande, et une
nouvelle demande remplace l'ancienne. Si le décideur doute qu'une demande lui soit adressée
(confiance < 0,75), Cornelius pose une question fermée ; le oui ou le non qui suit y répond.

Le décideur a son propre serveur llama.cpp (même modèle, environ 3 Go de mémoire vidéo) : sur
le serveur du cerveau, il attendait chaque fois qu'une réponse lisait ses consignes (0,3 à 5 s).
Décision et plan ont chacun un budget de 600 ms ; au-delà, des règles simples s'appliquent. Le
décideur voit les deux derniers échanges, ce qui vient d'être ignoré, la demande en cours, ce qui
a déjà été dit et l'outil en cours.

## Consignes du cerveau

Qwen reçoit les consignes de la session et quelques phrases sur la réponse orale.
Le client peut définir sa propre personnalité et sa langue. Le serveur ajoute les
consignes concernant les fonctions, la mémoire personnelle ou la recherche web quand
le client déclare les capacités correspondantes. Quand il reconnaît le modèle de
consignes de délégation de Cornelius, il retire les indications de remplissage du
silence propres à l’ancien service distant.

## Sélection des outils

Localive choisit selon le catalogue déclaré à l’ouverture de la session et à chaque
`session.update` qui remplace les outils.

| Outils disponibles | Chemin suivi |
|---|---|
| 0 ou 1 | Direct vers Qwen, quelle que soit la taille du schéma |
| 2 à 5 | Direct sous 3 072 octets au total, tri par d3-mini à partir de 3 072 octets |
| 6 ou plus | Tri par d3-mini |

La taille est celle de la liste des outils au format de Qwen, sérialisée en JSON compact
UTF-8. Elle inclut noms, descriptions et schémas de paramètres, ainsi que `web_search`
quand il est disponible. Le premier catalogue qui atteint un seuil de tri déclenche
le chargement de d3-mini Q8_0 sur GPU via Vulkan. Le modèle reste ensuite en mémoire
jusqu’à l’arrêt du serveur. Les petits catalogues passent toujours directement à Qwen,
même si une autre session a déjà chargé d3-mini.

Le routeur charge uniquement le GGUF texte. Le projecteur d’image et de vidéo
est distribué dans un fichier `mmproj` séparé. [Fichiers d3-mini GGUF](https://huggingface.co/ggml-org/vllm-d3-mini-GGUF/tree/main)

Pour trier, Localive transmet la demande transcrite, un court extrait du tour précédent,
le nom de chaque outil et jusqu’aux cinq premiers mots de sa description. Au-delà de 16 outils,
les noms sont regroupés par module. Quand d3 trie aussi les adieux dans le même passage,
le nom du module apparaît une seule fois, suivi des noms de ses actions et de trois mots
au plus par description, pris parmi les huit premiers. Un seul appel à `/v1/systemone` suffit pour ce tri. Si d3-mini
signale plusieurs actions ou hésite, Qwen reçoit immédiatement le catalogue complet.
Au-delà de 24 familles, Localive passe directement le catalogue à Qwen. Les schémas
transmis à Qwen conservent leurs descriptions complètes, paramètres et ordre.

Si les poids officiels manquent, Localive les télécharge en arrière-plan lorsque le tri est
activé et que le serveur est en ligne. Pendant le transfert ou le chargement, Qwen reçoit le
catalogue complet. Une session déjà ouverte profite du tri dès que d3-mini est prêt. Le
catalogue complet sert aussi de repli si d3-mini répond trop lentement ou si le classement
reste incertain. Les options d’installation figurent
dans le [guide d’installation](installation.md). Dans **Cerveau**, le délai à zéro s’adapte
au temps de calcul des décisions. Il laisse jusqu’à huit secondes au premier essai, puis
s’ajuste entre 400 ms et huit secondes. Un délai fixe peut aussi être indiqué.

## Cache et latence

Les consignes de Cornelius et ses outils peuvent occuper plusieurs milliers de jetons. llama.cpp les garde en cache
d'une réponse à l'autre, à condition qu'elles ne changent pas : l'heure à la minute et les
observations de la caméra accompagnent donc chaque réponse au lieu d'être dans les consignes, et
les consignes ajoutées en cours de conversation viennent après la partie stable. Les tours les
plus anciens sont retirés pour tenir dans le contexte du modèle, et chaque résultat d'outil est
limité à 6 000 caractères.

## Voix et audio

- Phonon synthétise phrase par phrase, au plus une phrase d'avance quand il reste moins de 1,2 s
  d'audio à jouer ; le moteur est libéré entre les phrases pour les autres sessions.
- Les blancs ajoutés par Phonon (environ 800 ms entre deux phrases) sont ramenés à environ 120 ms ;
  les pauses aux virgules sont gardées.
- La sortie est cadencée en temps réel (blocs de 40 ms, 120 ms d'avance) ; un retard ne déclenche
  jamais de rafale de vieux audio.
- **Annulation d'écho côté serveur (option)** : WebRTC AEC3 retire du micro la voix que Localive
  envoie, quel que soit le client. Désactivée par défaut : dans nos essais, la baisse de volume et
  le filtre d'écho textuel ont donné de meilleures décisions.

## Recherche web

La première session qui déclare l’outil hébergé `{"type":"web_search"}` dans
`session.start` ou `session.update` déclenche la préparation de SearXNG en arrière-plan.
Toutes les sessions partagent cette préparation. Le bouton **Préparer maintenant**
de l’administration permet de l’anticiper.

Par défaut, Localive vérifie Docker, prépare `.runtime/searxng/settings.yml`, crée ou
relance le conteneur local et attend une réponse JSON. Sous Windows, le réglage
`search.start_docker` autorise le lancement de Docker Desktop. Pour utiliser votre
propre instance SearXNG, définissez `search.managed = false` et renseignez `search.url`.
Avec `search.engine = "off"`, la recherche hébergée est désactivée.

Chaque appel dispose de `search.timeout_s` (6 s par défaut) pour attendre la disponibilité
du service et obtenir les résultats. Si ce délai expire, la préparation continue et la
session reçoit un avertissement `web_search_failed`. Lors d’une recherche réussie, le
client reçoit les événements `web_search_call` et les sources en `url_citation`.
