import asyncio
import math

import pytest

from localive.config import Config
from localive.decision import NOW_SYSTEM, STILL_SYSTEM, Arbiter, Situation, plan


class StubArbiter(Arbiter):
    def __init__(self, data):
        super().__init__(None, Config())
        self.data = data

    async def _post(self, body, timeout_ms):
        return self.data


def answer(text, **extra):
    return {"choices": [{"message": {"content": text}, **extra}]}


@pytest.mark.parametrize("metadata", [None, {}, {"content": None}, {"content": []}])
async def test_optional_logprobs_do_not_discard_a_valid_decision(metadata):
    arbiter = StubArbiter(answer("IGNORE", logprobs=metadata))
    decision = await arbiter.decide(Situation("Une histoire", "Mm-hm"))
    assert decision.label == "IGNORE"


@pytest.mark.parametrize("data", [None, {}, {"choices": None}, answer(None), answer([])])
async def test_invalid_model_response_falls_back_without_crashing_audio(data):
    decision = await StubArbiter(data).decide(Situation("Une histoire", "Stop"))
    assert decision.label is None


async def test_malformed_logprobs_are_ignored_and_absent_chosen_label_is_not_certain():
    top = [None, {"token": []}, {"token": "NEW", "logprob": float("nan")},
           {"token": "NEW", "logprob": 10000},
           {"token": "IGN", "logprob": math.log(0.9)}]
    arbiter = StubArbiter(answer("NEW", logprobs={"content": [{"top_logprobs": top}]}))
    decision = await arbiter.decide(Situation("Une histoire", "Allume"))
    assert decision.label == "NEW"
    assert decision.unsure
    assert decision.confidence == 0


@pytest.mark.parametrize("heard,expected", [("Et Nantes ?", "AFTER"), ("Non, à Lyon", "FIX")])
async def test_linking_cue_skips_discarded_continuation_inference(monkeypatch, heard, expected):
    prompts = []

    async def request(arbiter, system, user, **extra):
        prompts.append(system)
        return "NON"

    monkeypatch.setattr("localive.decision._plain", request)
    how, _ = await plan(StubArbiter({}), Situation("Météo", heard), "La suite", True)
    assert how == expected
    assert prompts == [NOW_SYSTEM]


async def test_urgent_action_does_not_wait_for_unused_continuation(monkeypatch):
    started, cancelled = asyncio.Event(), asyncio.Event()

    async def request(arbiter, system, user, **extra):
        if system == NOW_SYSTEM:
            await started.wait()
            return "OUI"
        assert system == STILL_SYSTEM
        started.set()
        try:
            await asyncio.Future()
        finally:
            cancelled.set()

    monkeypatch.setattr("localive.decision._plain", request)
    how, _ = await asyncio.wait_for(
        plan(StubArbiter({}), Situation("Météo", "Allume la lumière"), "La suite", True), 1
    )
    assert how == "NOW"
    assert cancelled.is_set()


async def test_cancel_plan_closes_both_inference_requests(monkeypatch):
    started, closed = set(), set()
    both_started = asyncio.Event()

    async def request(arbiter, system, user, **extra):
        started.add(system)
        if len(started) == 2:
            both_started.set()
        try:
            await asyncio.Future()
        finally:
            closed.add(system)

    monkeypatch.setattr("localive.decision._plain", request)
    task = asyncio.create_task(
        plan(StubArbiter({}), Situation("Météo", "Allume la lumière"), "La suite", True)
    )
    await asyncio.wait_for(both_started.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert closed == {NOW_SYSTEM, STILL_SYSTEM}


async def test_plan_obeys_configured_latency_budget(monkeypatch):
    closed = set()

    async def request(arbiter, system, user, **extra):
        try:
            await asyncio.Future()
        finally:
            closed.add(system)

    monkeypatch.setattr("localive.decision._plain", request)
    arbiter = StubArbiter({})
    arbiter.config.decision.timeout_ms = 20
    how, elapsed = await asyncio.wait_for(
        plan(arbiter, Situation("Météo", "Allume la lumière"), "La suite", True), 1
    )
    assert how == "FIX"
    assert elapsed < 500
    assert closed == {NOW_SYSTEM, STILL_SYSTEM}
