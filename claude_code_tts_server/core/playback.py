"""Audio playback utilities."""

import asyncio
import logging
import subprocess
import sys
from pathlib import Path

log = logging.getLogger("tts-server")


def get_player() -> list[str] | None:
    """Get audio player command for this platform.

    Returns:
        List of command arguments for the audio player, or None if not found.
    """
    if sys.platform == "darwin":
        return ["afplay"]

    for player in [["mpv", "--no-terminal"], ["paplay"], ["aplay"]]:
        try:
            result = subprocess.run(
                ["which", player[0]], capture_output=True, check=False
            )
            if result.returncode == 0:
                return player
        except OSError:
            pass
    return None


def play_sound_async(audio_file: Path | str) -> subprocess.Popen | None:
    """Play sound without blocking (fire-and-forget).

    Args:
        audio_file: Path to the audio file to play.

    Returns:
        The subprocess, or None if playback failed.
    """
    player = get_player()
    if player and audio_file:
        return subprocess.Popen(
            player + [str(audio_file)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    return None


class AudioPlayer:
    """Manages audio playback with interrupt support."""

    def __init__(self):
        self.current_process: subprocess.Popen | None = None
        self.play_start_time: float | None = None
        self._current_audio_file: Path | None = None

    def is_playing(self) -> bool:
        """Check if audio is currently playing."""
        return self.current_process is not None and self.current_process.poll() is None

    def get_elapsed_time(self) -> float | None:
        """Get time elapsed since playback started."""
        if self.play_start_time is None:
            return None
        import time
        return time.monotonic() - self.play_start_time

    def play(self, audio_file: Path) -> bool:
        """Start playing an audio file.

        Args:
            audio_file: Path to the audio file.

        Returns:
            True if playback started, False otherwise.
        """
        import time

        player = get_player()
        if not player:
            return False

        self.current_process = subprocess.Popen(
            player + [str(audio_file)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self.play_start_time = time.monotonic()
        self._current_audio_file = audio_file
        return True

    def stop(self) -> Path | None:
        """Stop currently playing audio.

        Returns:
            Path to the audio file that was playing, for cleanup.
        """
        audio_file = self._current_audio_file

        if self.current_process and self.current_process.poll() is None:
            self.current_process.terminate()
            try:
                self.current_process.wait(timeout=0.1)
            except subprocess.TimeoutExpired:
                self.current_process.kill()

        self.current_process = None
        self.play_start_time = None
        self._current_audio_file = None

        return audio_file

    def check_finished(self) -> Path | None:
        """Check if playback finished and return the file for cleanup.

        Returns:
            Path to the finished audio file if playback ended, None otherwise.
        """
        if self.current_process and self.current_process.poll() is not None:
            audio_file = self._current_audio_file
            self.current_process = None
            self.play_start_time = None
            self._current_audio_file = None
            return audio_file
        return None

    async def play_chime(self, chime_file: Path | None, max_wait: float = 0.5) -> None:
        """Play chime sound with brief wait.

        Args:
            chime_file: Path to the chime sound file.
            max_wait: Maximum time to wait for chime to finish.
        """
        if not chime_file:
            return

        player = get_player()
        if not player:
            return

        log.debug("Playing chime")
        proc = subprocess.Popen(
            player + [str(chime_file)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # Wait briefly for chime (but not forever)
        elapsed = 0.0
        while elapsed < max_wait:
            if proc.poll() is not None:
                break
            await asyncio.sleep(0.05)
            elapsed += 0.05

        if proc.poll() is None:
            proc.terminate()

    def play_drop_sound(self, drop_file: Path | None) -> None:
        """Play drop tone (fire-and-forget).

        Args:
            drop_file: Path to the drop sound file.
        """
        if drop_file:
            log.debug("Playing drop tone")
            play_sound_async(drop_file)
