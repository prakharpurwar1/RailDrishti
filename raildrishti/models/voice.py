"""Voice I/O: speech-to-text (pluggable, offline-first) and text-to-speech (browser-native).

No paid API and no API key. Text-to-speech runs in the browser via the Web Speech
API (``SpeechSynthesis``). Speech-to-text is pluggable and degrades gracefully to
text-only if no backend is installed:

    faster-whisper  (offline, auto-downloads a small model)  ->  vosk  ->  Google (online)
"""
from __future__ import annotations

import json
import os
import tempfile
import wave

# Maps the assistant's language codes to BCP-47 tags for the browser voice.
LANG_CODES = {"en": "en-IN", "hi": "hi-IN", "ta": "ta-IN", "te": "te-IN",
              "bn": "bn-IN", "mr": "mr-IN", "kn": "kn-IN", "gu": "gu-IN"}

_STT = None  # cached (backend_name, handle)


def _load_stt():
    """Return (backend_name, handle). Tries faster-whisper, then vosk, then Google."""
    global _STT
    if _STT is not None:
        return _STT
    try:
        from faster_whisper import WhisperModel
        size = os.environ.get("RAILDRISHTI_WHISPER", "base")
        _STT = ("faster_whisper", WhisperModel(size, device="cpu", compute_type="int8"))
        return _STT
    except Exception:
        pass
    try:
        from vosk import Model, SetLogLevel  # noqa: F401
        path = os.environ.get("RAILDRISHTI_VOSK_MODEL", "")
        if path and os.path.isdir(path):
            SetLogLevel(-1)
            _STT = ("vosk", Model(path))
            return _STT
    except Exception:
        pass
    try:
        import speech_recognition as sr
        _STT = ("google", sr.Recognizer())
        return _STT
    except Exception:
        pass
    _STT = ("none", None)
    return _STT


def stt_available() -> bool:
    return _load_stt()[0] != "none"


def stt_backend() -> str:
    return _load_stt()[0]


def transcribe(audio_bytes: bytes, language: str = "en") -> str | None:
    """Transcribe recorded WAV bytes to text. Returns None if no backend or on failure."""
    backend, handle = _load_stt()
    if backend == "none" or not audio_bytes:
        return None
    fd, path = tempfile.mkstemp(suffix=".wav")
    with os.fdopen(fd, "wb") as f:
        f.write(audio_bytes)
    try:
        if backend == "faster_whisper":
            segments, _ = handle.transcribe(path, language=language or None, beam_size=1)
            return (" ".join(s.text for s in segments).strip()) or None
        if backend == "vosk":
            from vosk import KaldiRecognizer
            wf = wave.open(path, "rb")
            rec = KaldiRecognizer(handle, wf.getframerate())
            out = []
            while True:
                data = wf.readframes(4000)
                if not data:
                    break
                if rec.AcceptWaveform(data):
                    out.append(json.loads(rec.Result()).get("text", ""))
            out.append(json.loads(rec.FinalResult()).get("text", ""))
            return (" ".join(out).strip()) or None
        if backend == "google":
            import speech_recognition as sr
            with sr.AudioFile(path) as src:
                audio = handle.record(src)
            return handle.recognize_google(audio, language=LANG_CODES.get(language, "en-IN"))
    except Exception:
        return None
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
    return None


def speak_html(text: str, language: str = "en") -> str:
    """Self-contained browser TTS widget. Free, no API key.

    Voices depend on the operating system / browser; if the requested language
    voice is unavailable the browser falls back to its default voice.
    """
    lang = LANG_CODES.get(language, "en-IN")
    payload = json.dumps(text)
    return f"""
    <div style="font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;margin:2px 0">
      <button onclick="rdSpeak()" style="background:#0B2A66;color:#fff;border:none;
        border-radius:8px;padding:6px 14px;font-weight:600;font-size:0.85rem;cursor:pointer">
        &#128266; Play
      </button>
      <script>
        function rdSpeak() {{
          const u = new SpeechSynthesisUtterance({payload});
          u.lang = "{lang}"; u.rate = 0.98;
          window.speechSynthesis.cancel();
          window.speechSynthesis.speak(u);
        }}
      </script>
    </div>
    """
