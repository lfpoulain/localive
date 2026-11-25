# Contribuer à Localive

Merci de votre intérêt ! Localive fait partie de l'écosystème
[Cornelius](https://github.com/lfpoulain/Cornelius).

## Signaler un problème

Ouvrez une *issue* avec :

- votre système (Windows ou Linux), votre carte graphique et votre processeur ;
- ce que vous avez dit, ce qui s'est passé, ce que vous attendiez ;
- les lignes du **Journal** de l'administration au moment du problème.

## Proposer une modification

1. Installez Localive, puis ajoutez les dépendances de développement en suivant [Développement](docs/developpement.md).
2. Lancez les tests et le lint : `python -m pytest` et `python -m ruff check src tests`.
3. Si vous touchez aux consignes du cerveau ou au décideur, mesurez avant et après avec
   `bench/prompts/run.py` et `bench/decision/compare_models.py`, et donnez les chiffres.
4. Si vous ajoutez un réglage, expliquez-le dans `settings.py` et régénérez
   `docs/reglages.md` (`python -m localive.settings > docs/reglages.md`).

Le code et les commentaires sont en anglais ; l'interface, la documentation et les messages sont en
français.

## Licence

En contribuant, vous acceptez que votre contribution soit publiée sous la licence du projet,
[PolyForm Noncommercial 1.0.0](LICENSE).
