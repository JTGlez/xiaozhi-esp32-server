"""MiniMax needs its own host for TTS and its own request options for the LLM."""

from unittest.mock import MagicMock

from core.providers.llm.openai.openai import LLMProvider
from core.providers.tts.minimax_httpstream import TTSProvider


def tts(tmp_path, **config):
    return TTSProvider(
        {"api_key": "k", "model": "m", "output_dir": str(tmp_path), **config}, True
    )


def test_tts_uses_configured_host_without_group_id(tmp_path):
    assert (
        tts(tmp_path, host="api.minimax.io").api_url
        == "https://api.minimax.io/v1/t2a_v2"
    )


def test_tts_keeps_default_host_and_group_id(tmp_path):
    assert (
        tts(tmp_path, group_id="g").api_url
        == "https://api.minimaxi.com/v1/t2a_v2?GroupId=g"
    )


def test_llm_sends_configured_extra_body():
    thinking = {"thinking": {"type": "disabled"}}
    llm = LLMProvider(
        {
            "api_key": "k",
            "base_url": "https://api.minimax.io/v1",
            "model_name": "m",
            "extra_body": thinking,
        }
    )
    llm.client = MagicMock()
    llm.client.chat.completions.create.return_value = MagicMock(
        __iter__=lambda self: iter(())
    )

    list(llm.response("s", [{"role": "user", "content": "hola"}]))

    assert llm.client.chat.completions.create.call_args.kwargs["extra_body"] == thinking
