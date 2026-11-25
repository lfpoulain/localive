# Administration

L'administration de Localive est sur **<http://127.0.0.1:8787>** (ou l'adresse réseau de la
machine si Localive écoute sur `0.0.0.0`). `Lancer Localive.cmd` l'ouvre automatiquement.
Sur téléphone, **Menu** ouvre la navigation. L’adresse WebSocket à copier se trouve
en haut de l’accueil. Si le navigateur bloque la copie automatique, le bouton sélectionne
l’adresse pour la commande **Copier** de votre appareil.

## Les pages

| Page | Contenu |
|---|---|
| **Accueil** | État de Localive, moteurs chargés (écoute, cerveau, voix), recherche web (« Préparer maintenant » à froid, « Réessayer » après une erreur), conversations en cours et adresse WebSocket Live |
| **Connecter un client** | Adresses WebSocket pour votre application et adresses HTTP à copier dans le réglage « Serveur Live » de Cornelius |
| **Réglages** (une page par thème) | Voix, Écoute, Fin de tour, Interruptions, Cerveau (liste déroulante du modèle), Recherche web, Écho, Serveur et sécurité |
| **Journal** | Ce que Localive entend, décide et répond, en direct ; filtre « avertissements seulement » |

## Les réglages

Chaque réglage est expliqué : à quoi il sert, quand y toucher, sa valeur par défaut. Les réglages
courants sont affichés d'office ; les autres sous **Réglages avancés**.

- **Modifier** : la barre « Enregistrer » apparaît ; rien n'est appliqué avant.
- **Valeur par défaut** : remet le réglage d'origine (il est retiré de `localive.toml`).
- **Au redémarrage** : certains réglages (modèles, ports, moteurs) ne prennent effet qu'au
  prochain démarrage. Le formulaire affiche la valeur enregistrée et permet de la corriger
  avant d'utiliser **Redémarrer Localive**.
- **Jetons** (accès Localive, Hugging Face, serveur externe) : jamais réaffichés, seulement
  « défini » ; on peut les remplacer ou les effacer.

Si vous changez de fournisseur, d’hôte, de protocole ou de port pour le cerveau externe,
saisissez la clé du nouveau serveur si nécessaire. La clé de l’ancienne origine est retirée.

Tout est enregistré dans `localive.toml`, à la racine de Localive, que l'on peut aussi modifier à
la main (voir [Réglages](reglages.md) et `localive.toml.example`).

### Essayer Qwen3.5-4B MTP

Dans **Cerveau**, ouvrez la famille **Qwen** et téléchargez **Qwen3.5-4B MTP (UD-Q4_K_XL)**.
Sélectionnez ensuite ce profil dans **Modèle de raisonnement**, enregistrez et redémarrez
Localive. Le moteur configure automatiquement le décodage spéculatif avec jusqu’à trois
jetons d’avance. Pour revenir au profil précédent, sélectionnez à nouveau son modèle,
enregistrez et redémarrez.

## Redémarrer

Le bouton **Redémarrer Localive** se trouve en haut de chaque page, à côté de l’état du
serveur, sur ordinateur comme sur téléphone. Il est disponible même si aucun réglage n’a
changé. Après confirmation, le lanceur (`Lancer Localive.cmd`, `scripts/start.ps1`,
`scripts/start.sh` ou le service systemd) relance Localive et la page se reconnecte.
Si une conversation est en cours, le serveur vous invite à réessayer lorsqu’elle est terminée.

## Accès

- Par défaut, Localive n'écoute que sur cette machine (`127.0.0.1`).
- Avec un **jeton d'accès** (Serveur et sécurité), l'administration le demande une fois par onglet,
  et Cornelius doit l'envoyer comme clé.
- Sans jeton, seule cette machine peut accéder à l’API d’administration, y compris au journal.
- Une page web d'un autre site ne peut ni lire ni modifier les réglages, ni ouvrir une
  conversation.
