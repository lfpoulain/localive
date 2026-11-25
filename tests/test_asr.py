import asyncio
import threading

import numpy as np
import pytest
from aiohttp import web

from localive import wyoming
from localive.asr import Transcriber, clean, encode_wav_pcm16, extract_confident_transcript
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


def test_extract_confident_transcript_filters_whisper_hallucinations():
    payload = {
        "text": "Hallucination Quelle heure est-il ? Répétition",
        "segments": [
            {"text": "Merci d'avoir regardé", "no_speech_prob": 0.85, "avg_logprob": -0.2},
            {"text": "bruit", "no_speech_prob": 0.6, "avg_logprob": -1.1},
            {"text": "inintelligible", "no_speech_prob": 0.1, "avg_logprob": -1.5},
            {
                "text": "la la la la",
                "no_speech_prob": 0.05,
                "avg_logprob": -0.2,
                "compression_ratio": 2.9,
            },
            {
                "text": " Quelle heure est-il ? ",
                "no_speech_prob": 0.05,
                "avg_logprob": -0.15,
                "compression_ratio": 1.1,
            },
        ],
    }
    assert extract_confident_transcript(payload) == "Quelle heure est-il ?"
    assert extract_confident_transcript({"text": "  Bonjour.  "}) == "Bonjour."


def test_encode_wav_pcm16_produces_valid_mono_16k_wav():
    wav_bytes = encode_wav_pcm16(np.array([0.0, 0.5, -0.5, 1.5], dtype=np.float32), 16000)
    assert wav_bytes[:4] == b"RIFF"
    assert wav_bytes[8:12] == b"WAVE"
    assert len(wav_bytes) == 44 + 8


async def test_openai_stt_posts_verbose_json_and_filters_segments():
    received: dict[str, object] = {}

    async def handle(request: web.Request) -> web.Response:
        received["auth"] = request.headers.get("Authorization")
        reader = await request.multipart()
        fields: dict[str, object] = {}
        while part := await reader.next():
            if part.name == "file":
                fields["file_len"] = len(await part.read())
                fields["filename"] = part.filename
            else:
                fields[part.name] = await part.text()
        received["fields"] = fields
        return web.json_response(
            {
                "text": "Merci d'avoir regardé Allume le salon",
                "segments": [
                    {"text": "Merci d'avoir regardé", "no_speech_prob": 0.9, "avg_logprob": -0.3},
                    {"text": "Allume le salon.", "no_speech_prob": 0.02, "avg_logprob": -0.1},
                ],
            }
        )

    async def handle_models(_req: web.Request) -> web.Response:
        return web.json_response({"data": []})

    app = web.Application()
    app.router.add_get("/v1/models", handle_models)
    app.router.add_post("/v1/audio/transcriptions", handle)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]

    config = Config()
    config.asr.backend = "openai"
    config.asr.external_url = f"http://127.0.0.1:{port}"
    config.asr.external_model = "deepdml/faster-whisper-large-v3-turbo-ct2"
    config.asr.language = "fr"
    config.asr.prompt = "Salon, Chambre"
    config.asr.hotwords = "Cornelius, WLED"
    config.asr.api_key = "secret-stt"
    transcriber = Transcriber(config)
    await transcriber.start()
    try:
        result = await transcriber.transcribe(np.zeros(1600, dtype=np.float32))
    finally:
        await transcriber.aclose()
        await runner.cleanup()

    assert result.text == "Allume le salon."
    assert received["auth"] == "Bearer secret-stt"
    fields = received["fields"]
    assert fields["model"] == "deepdml/faster-whisper-large-v3-turbo-ct2"
    assert fields["language"] == "fr"
    assert fields["prompt"] == "Salon, Chambre"
    assert fields["hotwords"] == "Cornelius, WLED"
    assert fields["response_format"] == "verbose_json"
    assert fields["vad_filter"] == "false"
    assert fields["filename"] == "speech.wav"
    assert fields["file_len"] == 44 + 3200


async def test_wyoming_stt_streams_pcm_and_returns_transcript():
    events_seen: list[str] = []
    audio_bytes = bytearray()

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            while event := await wyoming.read_event(reader):
                events_seen.append(event.type)
                if event.type == "audio-chunk":
                    audio_bytes.extend(event.payload)
                elif event.type == "audio-stop":
                    writer.write(wyoming.encode_event("transcript", {"text": "  Il fait beau. "}))
                    await writer.drain()
                    break
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    config = Config()
    config.asr.backend = "wyoming"
    config.asr.external_url = f"127.0.0.1:{port}"
    transcriber = Transcriber(config)
    await transcriber.start()
    try:
        result = await transcriber.transcribe(np.zeros(800, dtype=np.float32))
    finally:
        await transcriber.aclose()
        server.close()
        await server.wait_closed()

    assert result.text == "Il fait beau."
    assert events_seen[0] == "transcribe"
    assert "audio-start" in events_seen and events_seen[-1] == "audio-stop"
    assert len(audio_bytes) == 1600


async def test_external_stt_probe_reports_unreachable_server_without_blocking_startup():
    config = Config()
    config.asr.backend = "openai"
    config.asr.external_url = "http://127.0.0.1:1"
    transcriber = Transcriber(config)
    await transcriber.start()
    try:
        assert transcriber.probe_error is not None
        assert "injoignable" in transcriber.probe_error
    finally:
        await transcriber.aclose()


async def test_wyoming_read_event_rejects_oversized_payload():
    reader = asyncio.StreamReader()
    reader.feed_data(b'{"type":"audio-chunk","payload_length":99999999}\n')
    reader.feed_eof()
    with pytest.raises(ValueError, match="trop volumineuse"):
        await wyoming.read_event(reader)
