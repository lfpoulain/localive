import pytest

from localive.text import SentenceChunker, clean_for_speech, speakable_french


def test_chunker_releases_first_clause_early_then_sentences():
    chunker = SentenceChunker()
    out = []
    pieces = ["Il fait ", "7 degrés à Lyon. ", "Le ciel est couvert ", "toute la journée"]
    pieces += [", avec ", "un peu de pluie ce soir. Demain"]
    for piece in pieces:
        out += chunker.push(piece)
    assert out == [
        "Il fait 7 degrés à Lyon.",
        "Le ciel est couvert toute la journée, avec un peu de pluie ce soir.",
    ]
    assert chunker.flush() == "Demain"


def test_chunker_cuts_long_unpunctuated_text():
    chunker = SentenceChunker(max_chars=60)
    out = chunker.push("mot " * 40)
    assert out and all(len(piece) <= 60 for piece in out)


def test_chunker_keeps_abbreviations():
    chunker = SentenceChunker(first_min=1)
    assert chunker.push("Voici M. ") == []


@pytest.mark.parametrize(
    "source, expected",
    [
        ("Il est 14h30.", "Il est quatorze heures trente."),
        ("Il est 13:05", "Il est treize heures cinq"),
        ("Il est 1 h 01.", "Il est une heure une."),
        ("37 fois 12 font 444.", "trente-sept fois douze font quatre cent quarante-quatre."),
        ("7 × 6 = 42", "sept fois six égale quarante-deux"),
        ("Il fait 7 °C, risque 78 %.", "Il fait sept degrés, risque soixante-dix-huit pour cent."),
        ("Pi vaut 3,14.", "Pi vaut trois virgule quatorze."),
        ("Population : 2 100 000 habitants", "Population : deux millions cent mille habitants"),
        ("Tu as dit « 42 ».", "Tu as dit quarante-deux."),
        ("Voir https://example.com/2024 page 2", "Voir https://example.com/2024 page deux"),
    ],
)
def test_speakable_french(source, expected):
    assert speakable_french(source) == expected


def test_clean_for_speech_removes_markdown_and_emoji():
    assert clean_for_speech("**Bonjour** ! 😊\n- un\n- deux") == "Bonjour ! un deux"
