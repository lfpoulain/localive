from pathlib import Path

import numpy as np
import pytest

from localive.vad import CHUNK, Segmenter, SileroVad
from scripts.live_client import load_16k

FIXTURES = Path(__file__).parent / "fixtures"


def run(segmenter: Segmenter, audio: np.ndarray, frame: int = 320):
    events = []
    for start in range(0, len(audio), frame):
        events += list(segmenter.feed(audio[start : start + frame]))
    return events


class ScriptedVad:
    def __init__(self, probabilities):
        self.probabilities = iter(probabilities)

    def reset(self):
        pass

    def __call__(self, chunk):
        return next(self.probabilities)


def test_sessions_share_silero_graphs_but_not_speech_history():
    first, second, reference = SileroVad(), SileroVad(), SileroVad()
    assert first._encoder is second._encoder and first._decoder is second._decoder
    speech = load_16k(FIXTURES / "wake_only.wav")[:32 * CHUNK].reshape(-1, CHUNK)
    expected = [reference(chunk) for chunk in speech]
    for chunk, probability in zip(speech, expected, strict=True):
        second(chunk[::-1])
        assert first(chunk) == probability
    first.reset()
    assert first(speech[0]) == expected[0]
    reference.reset()
    for chunk in speech:
        reference(chunk[::-1])
    assert second(speech[0]) == reference(speech[0])


def test_silence_produces_nothing():
    assert run(Segmenter(), np.zeros(16000 * 2, dtype=np.float32)) == []


def test_noise_produces_nothing():
    noise = np.random.default_rng(0).normal(0, 0.01, 16000 * 2).astype(np.float32)
    assert [e for e in run(Segmenter(), noise) if e.audio is not None] == []


@pytest.mark.parametrize("name", ["wake_clock.wav", "other_person.wav"])
def test_one_utterance_with_preroll(name):
    speech = load_16k(FIXTURES / name)
    silence = np.zeros(16000, dtype=np.float32)
    events = run(Segmenter(), np.concatenate([silence, speech, silence]))
    assert [e.kind for e in events] == ["start", "end"]
    utterance = events[1].audio
    assert utterance is not None
    assert 0.6 * len(speech) < len(utterance) < len(speech) + 16000


def test_two_utterances_separated_by_silence():
    speech = load_16k(FIXTURES / "wake_only.wav")
    gap = np.zeros(16000, dtype=np.float32)
    events = run(Segmenter(), np.concatenate([gap, speech, gap, speech, gap]))
    assert [e.kind for e in events] == ["start", "end", "start", "end"]


@pytest.mark.parametrize("probability", [0.35, 0.4])
@pytest.mark.parametrize("continuation_chunks, kept", [(2, False), (3, True)])
def test_hysteresis_continuation_counts_toward_minimum_speech(
    probability, continuation_chunks, kept
):
    probabilities = [0.9] * 2 + [probability] * continuation_chunks + [0.0] * 3
    segmenter = Segmenter(ScriptedVad(probabilities), silence_ms=96)
    events = list(segmenter.feed(np.zeros(len(probabilities) * CHUNK, np.float32)))
    assert [event.kind for event in events] == ["start", "end"]
    # Two strong frames plus continuation: 128 ms is too short, 160 ms is enough.
    assert (events[-1].audio is not None) == kept


@pytest.mark.parametrize("probabilities", [[0.4] * 12, [0.9, 0.4] * 6])
def test_continuation_threshold_cannot_start_speech(probabilities):
    segmenter = Segmenter(ScriptedVad(probabilities))
    assert list(segmenter.feed(np.zeros(len(probabilities) * CHUNK, np.float32))) == []
    assert not segmenter.speaking
    assert segmenter.voiced_ms == 0


def test_silence_does_not_count_toward_minimum_speech():
    segmenter = Segmenter(ScriptedVad([0.9] * 2 + [0.4] * 2 + [0.34] * 2))
    list(segmenter.feed(np.zeros(4 * CHUNK, np.float32)))
    assert segmenter.voiced_ms == 128
    list(segmenter.feed(np.zeros(2 * CHUNK, np.float32)))
    assert segmenter.voiced_ms == 128
    assert segmenter.silence_ms == 64
    assert segmenter.end_now().audio is None


def test_hysteresis_continuation_advances_barge_in_duration():
    segmenter = Segmenter(ScriptedVad([0.9] * 2 + [0.4] * 9))
    list(segmenter.feed(np.zeros(2 * CHUNK, np.float32)))
    assert segmenter.voiced_ms == 64
    list(segmenter.feed(np.zeros(8 * CHUNK, np.float32)))
    assert segmenter.voiced_ms == 320
    list(segmenter.feed(np.zeros(CHUNK, np.float32)))
    assert segmenter.voiced_ms == 352  # crosses the default 350 ms barge-in threshold


@pytest.mark.parametrize("silence_chunks, force_end", [(3, False), (1, True)])
def test_end_reports_actual_silence_before_reset(silence_chunks, force_end):
    segmenter = Segmenter(ScriptedVad([1.0] * 5 + [0.0] * silence_chunks), silence_ms=96)
    events = list(segmenter.feed(np.zeros((5 + silence_chunks) * CHUNK, np.float32)))
    event = segmenter.end_now() if force_end else events[-1]
    assert event.kind == "end"
    assert event.silence_ms == silence_chunks * 32
    assert segmenter.silence_ms == 0


def test_semantic_end_can_keep_speech_shorter_than_the_minimum():
    """Words already transcribed must not be dropped because Silero counted little speech."""
    segmenter = Segmenter(ScriptedVad([0.9] * 2 + [0.0] * 3), min_speech_ms=500)
    list(segmenter.feed(np.zeros(5 * CHUNK, np.float32)))
    assert segmenter.end_now().audio is None  # without keep_short: below the minimum
    segmenter = Segmenter(ScriptedVad([0.9] * 2 + [0.0] * 3), min_speech_ms=500)
    list(segmenter.feed(np.zeros(5 * CHUNK, np.float32)))
    assert segmenter.speaking
    event = segmenter.end_now(keep_short=True)
    assert event is not None and event.audio is not None and len(event.audio)
