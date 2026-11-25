"""Exercise the real native-thread bridge shared by two voice sessions."""

import asyncio
from contextlib import contextmanager

import numpy as np

from localive.config import Config
from localive.live import RATE
from localive.tts import Synthesizer
from tests.test_live import SESSION, Client
from tests.test_tts import FakeAudio, wait_for


async def test_playback_backpressure_does_not_hold_shared_native_engine():
    class Native:
        def __init__(self):
            self.started = []
            self.finished = []

        @contextmanager
        def stream(self, text, voice):
            self.started.append(text)
            try:
                # Synthesize four seconds quickly, as the native model does. Session
                # one must pace playback without holding up session two's synthesis.
                yield FakeAudio([np.full(960, 0.2, np.float32)] * 100)
            finally:
                self.finished.append(text)

    synth = Synthesizer(Config())
    native = synth._tts = Native()
    first = Client([
        "Première phrase suffisamment longue pour occuper le lecteur. "
        "Deuxième phrase en attente derrière la première."
    ])
    second = Client(["Réponse indépendante pour la seconde session."])
    try:
        for client in (first, second):
            client.session.s.tts = synth
            await client.session.start(SESSION)
        await first.session.on_user_text("Une longue réponse")
        await wait_for(lambda: len(native.finished) == 1)
        assert first.session.output.queued > RATE * 2
        await second.session.on_user_text("Une autre demande")
        await wait_for(lambda: second.audio_seconds() > 0)
        assert first.session.output.busy  # first client is still playing
        assert len(native.started) == 2
        assert native.started[1].startswith("Réponse indépendante")
    finally:
        await asyncio.gather(first.session.close("test"), second.session.close("test"))
