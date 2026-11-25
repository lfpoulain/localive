import threading

import numpy as np
import pytest

from localive.asr import Transcriber, clean
from localive.config import Config


@pytest.mark.parametrize(
    "text",
    [
        "Active les sous-titres.",
        "Que veut dire abonnez-vous en anglais ?",
        "Explique ce qu'est amara.org.",
        "Traduis thanks for watching en français.",
        "Merci d'avoir regardé mon problème, maintenant explique-moi.",
    ],
)
def test_boilerplate_words_do_not_erase_real_requests(text):
    assert clean(text) == text


@pytest.mark.parametrize(
    "text",
    ["Merci d'avoir regardé cette vidéo !", "Abonnez-vous.", "Thanks for watching!", "..."],
)
def test_standalone_boilerplate_and_empty_speech_are_filtered(text):
    assert clean(text) == ""


def test_close_waits_for_native_inference_without_unloading_a_running_model():
    entered = threading.Event()
    release = threading.Event()
    unloaded = threading.Event()

    class SlowSpeech:
        def transcribe(self, **kwargs):
            entered.set()
            assert release.wait(timeout=2)
            assert not unloaded.is_set()
            return {"text": "Bonjour."}

        def __exit__(self, *args):
            unloaded.set()

    transcriber = Transcriber(Config())
    transcriber._speech = SlowSpeech()
    future = transcriber._executor.submit(transcriber._run, np.zeros(512, np.float32))
    try:
        assert entered.wait(timeout=2)
        transcriber.close()
        transcriber.close()  # teardown can be requested twice
        assert not unloaded.is_set()
    finally:
        release.set()
        transcriber.close()
    assert future.result(timeout=2).text == "Bonjour."
    assert unloaded.wait(timeout=2)
