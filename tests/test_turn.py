import numpy as np
import pytest

from localive.turn import SmartTurn, features


@pytest.mark.parametrize("threads", [0, -1])
def test_smart_turn_rejects_unbounded_thread_count(threads):
    with pytest.raises(ValueError, match="threads must be at least 1"):
        SmartTurn("unused.onnx", threads=threads)


def test_features_keep_only_last_eight_seconds_and_handle_silence():
    signal = np.random.default_rng(42).normal(0, 0.1, 10 * 16000).astype(np.float32)
    np.testing.assert_array_equal(features(signal), features(signal[-8 * 16000 :]))
    silence = features(np.zeros(0, np.float32))
    assert silence.shape == (1, 80, 800)
    assert silence.dtype == np.float32
    assert np.all(np.isfinite(silence))
