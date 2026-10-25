import os
import re
from pathlib import Path

from .settings import required_env


def language_for(text, default="en-IN"):
    if re.search(r"[\u0900-\u097f]", text):
        return "hi-IN"
    if re.search(r"[\u0b80-\u0bff]", text):
        return "ta-IN"
    if re.search(r"[\u0c00-\u0c7f]", text):
        return "te-IN"
    if re.search(r"[\u0c80-\u0cff]", text):
        return "kn-IN"
    if re.search(r"[\u0d00-\u0d7f]", text):
        return "ml-IN"
    if re.search(r"[\u0980-\u09ff]", text):
        return "bn-IN"
    if re.search(r"[\u0a00-\u0a7f]", text):
        return "pa-IN"
    if re.search(r"[\u0a80-\u0aff]", text):
        return "gu-IN"
    return default


def transcribe(audio_path, language_code=None):
    required_env("GNANI_API_KEY")
    from gnani.stt import GnaniSTTClient

    client = GnaniSTTClient()
    language = language_code or os.environ.get("GNANI_LANGUAGE_CODE", "en-IN")
    result = client.transcribe(str(audio_path), language_code=language)
    return result.get("transcript", "").strip()


def synthesize(text, output_path, language_code=None):
    required_env("GNANI_API_KEY")
    from gnani.tts import GnaniTTSClient

    client = GnaniTTSClient()
    language = language_code or language_for(text, os.environ.get("GNANI_LANGUAGE_CODE", "en-IN"))
    client.synthesize(
        text,
        voice=os.environ.get("GNANI_TTS_VOICE", "Kaveri"),
        model=os.environ.get("GNANI_TTS_MODEL", "timbre-v2.5"),
        language=language,
        output_file=str(Path(output_path)),
    )
