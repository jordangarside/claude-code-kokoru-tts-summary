"""Sound effect generation for chimes and drop tones."""

import os
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf


def generate_chime(sample_rate: int = 24000) -> np.ndarray:
    """Generate two-note chime (G5 -> C6) for interrupts.

    Returns:
        Audio as float32 numpy array.
    """
    def make_note(freq: float, duration: float, amplitude: float = 0.25) -> np.ndarray:
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        # Fundamental + harmonics
        note = amplitude * np.sin(2 * np.pi * freq * t)
        note += amplitude * 0.3 * np.sin(2 * np.pi * freq * 2 * t)
        note += amplitude * 0.1 * np.sin(2 * np.pi * freq * 3 * t)
        # Envelope with attack and decay
        envelope = np.exp(-t * 8)
        attack = int(len(t) * 0.05)
        envelope[:attack] *= np.linspace(0, 1, attack)
        return note * envelope

    note1 = make_note(784, 0.08)  # G5
    note2 = make_note(1047, 0.08)  # C6
    gap = np.zeros(int(sample_rate * 0.03))
    chime = np.concatenate([note1, gap, note2])

    # Fade out
    fade = int(sample_rate * 0.02)
    if fade > 0:
        chime[-fade:] *= np.linspace(1, 0, fade)

    return chime.astype(np.float32)


def generate_drop_tone(sample_rate: int = 24000) -> np.ndarray:
    """Generate soft kalimba-like pluck for dropped messages.

    Returns:
        Audio as float32 numpy array.
    """
    duration = 0.15
    t = np.linspace(0, duration, int(sample_rate * duration), False)

    # Base frequency - E5, gentle and musical
    freq = 659

    # Fundamental with decaying harmonics (kalimba/music box character)
    tone = np.sin(2 * np.pi * freq * t)
    tone += 0.5 * np.sin(2 * np.pi * freq * 2 * t) * np.exp(-t * 20)
    tone += 0.25 * np.sin(2 * np.pi * freq * 3 * t) * np.exp(-t * 30)
    tone += 0.1 * np.sin(2 * np.pi * freq * 4 * t) * np.exp(-t * 40)

    # Pluck envelope - quick attack, smooth decay
    attack_time = 0.005
    attack_samples = int(sample_rate * attack_time)
    envelope = np.exp(-t * 10)
    envelope[:attack_samples] = np.linspace(0, 1, attack_samples)

    pluck = tone * envelope * 0.18

    # Soft fade out
    fade = int(sample_rate * 0.03)
    pluck[-fade:] *= np.linspace(1, 0, fade)

    return pluck.astype(np.float32)


def save_audio(audio: np.ndarray, sample_rate: int = 24000) -> Path:
    """Save audio to a temporary WAV file.

    Args:
        audio: Audio data as numpy array.
        sample_rate: Sample rate in Hz.

    Returns:
        Path to the temporary file.
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        path = Path(f.name)
    sf.write(path, audio, sample_rate)
    return path


class SoundManager:
    """Manages sound effect files."""

    def __init__(self, sample_rate: int = 24000):
        self.sample_rate = sample_rate
        self.chime_file: Path | None = None
        self.drop_file: Path | None = None

    def init_sounds(self) -> None:
        """Generate and save sound effect files."""
        self.chime_file = save_audio(generate_chime(self.sample_rate), self.sample_rate)
        self.drop_file = save_audio(generate_drop_tone(self.sample_rate), self.sample_rate)

    def cleanup(self) -> None:
        """Delete sound effect files."""
        for f in [self.chime_file, self.drop_file]:
            if f and f.exists():
                try:
                    os.unlink(f)
                except OSError:
                    pass
        self.chime_file = None
        self.drop_file = None
