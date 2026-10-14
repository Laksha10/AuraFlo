# audio_checkin.py
# The event-triggered audio loop: ask -> record -> Whisper -> VADER.
# It listens for the "pose_held" event and runs in a background thread,
# so the camera window keeps updating while the mic is open.

import threading
import queue
import numpy as np
import sounddevice as sd
import whisper

import config
from sentiment import analyze_sentiment

print("Loading Whisper model (first run downloads it)...")
_model = whisper.load_model(config.WHISPER_MODEL)


def ask_question():
    print(f"🗣️  {config.CHECKIN_QUESTION}")
    if not config.USE_SPOKEN_QUESTION:
        return
    try:
        import pyttsx3                      # imported here so the app still runs without it
        engine = pyttsx3.init()
        engine.say(config.CHECKIN_QUESTION)
        engine.runAndWait()
        engine.stop()
    except Exception as error:
        print("Spoken question failed, showing it on screen only:", error)


def record_audio():
    print("🎙️  Recording...")
    num_samples = int(config.SAMPLE_RATE * config.RECORD_SECONDS)
    audio = sd.rec(num_samples, samplerate=config.SAMPLE_RATE, channels=1, dtype="float32")
    sd.wait()                               # block this (background) thread until recording ends
    return np.squeeze(audio)                # shape (N, 1) -> (N,)


def transcribe(audio):
    # Guard 1: if the recording is basically silent, don't even ask Whisper
    # (on silence Whisper can "hallucinate" words nobody said).
    loudness = np.sqrt(np.mean(audio ** 2))
    if loudness < config.SILENCE_RMS:
        print("🔇 Silence detected - skipping Whisper")
        return ""

    result = _model.transcribe(audio, fp16=False, language="en", condition_on_previous_text=False)

    # Guard 2: drop any piece Whisper itself thinks probably wasn't speech.
    kept_parts = []
    for segment in result["segments"]:
        if segment["no_speech_prob"] < config.NO_SPEECH_PROB_LIMIT:
            kept_parts.append(segment["text"].strip())
    text = " ".join(kept_parts)
    print("📝 Heard:", text)
    return text


class AudioCheckin:
    def __init__(self):
        self.busy = False                   # the "do not disturb" flag
        self.results = queue.Queue()        # finished check-ins wait here for the main loop

    def on_pose_held(self, event_data):
        """Listener for the 'pose_held' event."""
        if self.busy:
            print("⏳ Check-in already running - ignoring duplicate trigger")
            return
        self.busy = True
        worker = threading.Thread(target=self._run_checkin, args=(event_data,), daemon=True)
        worker.start()

    def _run_checkin(self, event_data):
        text = ""
        try:
            ask_question()
            audio = record_audio()
            text = transcribe(audio)
        except Exception as error:
            print("❌ Check-in failed:", error)

        label, compound = analyze_sentiment(text)
        print(f"💬 Sentiment: {label} ({compound:.2f})")
        self.results.put({
            "pose": event_data["pose"],
            "text": text,
            "label": label,
            "compound": compound,
        })
        self.busy = False