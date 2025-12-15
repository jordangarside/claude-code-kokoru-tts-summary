#!/usr/bin/env python3
"""
Kokoro TTS Server for Claude Code audio hooks.
Async server that loads the model once and handles requests with cancellation support.

Usage:
    uv run kokoro-server.py [--port 20202] [--voice af_heart]

Features:
    - Instant ping/pong health checks (even while generating/playing)
    - Background audio generation - new audio prepares while current plays
    - Interrupts only when new audio is ready (no silence gaps)
    - Plays transition chime on interruption
    - Handles rapid requests by playing snippets of each
"""

import asyncio
import subprocess
import sys
import signal
import argparse
import tempfile
import os
import time
import warnings
from collections import deque

# Suppress torch warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Global state
pipeline = None
voice = None
current_process = None
request_queue = asyncio.Queue()  # Incoming text requests
ready_queue = asyncio.Queue()     # Generated audio ready to play
shutdown_event = asyncio.Event()

# Audio settings
SNIPPET_DURATION = 0.8  # seconds of audio to play when skipping
PLAYER_STARTUP_DELAY = 0.15  # seconds to wait for audio player to start
RAPID_REQUEST_WINDOW = 0.3  # seconds to wait for rapid requests


def generate_transition_sound(sample_rate=24000):
    """Generate a pleasant two-note chime for transitions."""
    import numpy as np

    # Two-note chime: G5 -> C6 (perfect fourth, pleasant interval)
    note1_freq = 784  # G5
    note2_freq = 1047  # C6
    note_duration = 0.08
    gap = 0.03

    t1 = np.linspace(0, note_duration, int(sample_rate * note_duration), False)
    t2 = np.linspace(0, note_duration, int(sample_rate * note_duration), False)

    # Generate notes with harmonics for warmth
    def make_note(t, freq, amplitude=0.25):
        note = amplitude * np.sin(2 * np.pi * freq * t)
        note += amplitude * 0.3 * np.sin(2 * np.pi * freq * 2 * t)
        note += amplitude * 0.1 * np.sin(2 * np.pi * freq * 3 * t)
        envelope = np.exp(-t * 8)
        attack_samples = int(len(t) * 0.05)
        envelope[:attack_samples] *= np.linspace(0, 1, attack_samples)
        return note * envelope

    note1 = make_note(t1, note1_freq)
    note2 = make_note(t2, note2_freq)

    gap_samples = int(sample_rate * gap)
    chime = np.concatenate([note1, np.zeros(gap_samples), note2])

    fade_samples = int(sample_rate * 0.02)
    if fade_samples > 0:
        chime[-fade_samples:] *= np.linspace(1, 0, fade_samples)

    return chime.astype(np.float32)


def generate_audio(text):
    """Generate audio from text using Kokoro pipeline."""
    import numpy as np
    all_audio = []
    for _, _, audio in pipeline(text, voice=voice):
        all_audio.append(audio)

    if not all_audio:
        return None

    return np.concatenate(all_audio)


def save_audio(audio, sample_rate=24000):
    """Save audio to a temporary WAV file."""
    import soundfile as sf
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        audio_file = f.name
    sf.write(audio_file, audio, sample_rate)
    return audio_file


def get_audio_player():
    """Get the appropriate audio player command for this platform."""
    if sys.platform == 'darwin':
        return ['afplay']

    for player in [['mpv', '--no-terminal'], ['paplay'], ['aplay']]:
        try:
            if subprocess.run(['which', player[0]], capture_output=True).returncode == 0:
                return player
        except:
            pass

    return None


async def play_audio_file(audio_file, max_duration=None, check_ready_queue=False):
    """Play an audio file, optionally limiting duration.

    If check_ready_queue=True, monitors ready_queue and returns True if interrupted.
    """
    global current_process

    player = get_audio_player()
    if not player:
        print("[kokoro-tts-server] No audio player found", file=sys.stderr)
        return False

    cmd = player + [audio_file]
    current_process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    interrupted = False

    try:
        if max_duration:
            await asyncio.sleep(PLAYER_STARTUP_DELAY + max_duration)
            if current_process and current_process.poll() is None:
                current_process.terminate()
                try:
                    current_process.wait(timeout=0.1)
                except:
                    current_process.kill()
        else:
            while current_process and current_process.poll() is None:
                # Check if new audio is ready to play
                if check_ready_queue and not ready_queue.empty():
                    current_process.terminate()
                    try:
                        current_process.wait(timeout=0.1)
                    except:
                        current_process.kill()
                    interrupted = True
                    break
                await asyncio.sleep(0.05)
    except asyncio.CancelledError:
        if current_process and current_process.poll() is None:
            current_process.terminate()
        raise
    finally:
        current_process = None

    return interrupted


def cancel_current_audio():
    """Cancel any currently playing audio."""
    global current_process
    if current_process and current_process.poll() is None:
        current_process.terminate()
        try:
            current_process.wait(timeout=0.1)
        except:
            current_process.kill()
        current_process = None
        return True
    return False


async def play_transition_sound():
    """Play a pleasant chime to indicate transition."""
    chime = generate_transition_sound()
    audio_file = save_audio(chime)
    try:
        await play_audio_file(audio_file, max_duration=0.2)
    finally:
        try:
            os.unlink(audio_file)
        except:
            pass


async def handle_client(reader, writer):
    """Handle incoming client connection."""
    data = b''
    try:
        while True:
            chunk = await asyncio.wait_for(reader.read(4096), timeout=0.3)
            if not chunk:
                break
            data += chunk
            if data.strip() == b"ping":
                try:
                    writer.write(b"pong")
                    await writer.drain()
                except:
                    pass
                writer.close()
                await writer.wait_closed()
                return
    except asyncio.TimeoutError:
        pass
    except Exception as e:
        print(f"[kokoro-tts-server] Read error: {e}", file=sys.stderr)

    try:
        writer.close()
        await writer.wait_closed()
    except:
        pass

    text = data.decode('utf-8', errors='ignore').strip()
    if text and text != "ping":
        await request_queue.put((text, time.time()))


async def generation_worker():
    """Generate audio in background, put ready audio in ready_queue."""
    print("[kokoro-tts-server] Generation worker started", flush=True)

    while not shutdown_event.is_set():
        try:
            # Wait for a request
            try:
                text, timestamp = await asyncio.wait_for(
                    request_queue.get(),
                    timeout=0.5
                )
            except asyncio.TimeoutError:
                continue

            # Brief wait to collect rapid follow-up requests
            await asyncio.sleep(RAPID_REQUEST_WINDOW)

            # Collect all queued requests
            requests = [(text, timestamp)]
            while not request_queue.empty():
                try:
                    next_req = request_queue.get_nowait()
                    requests.append(next_req)
                except asyncio.QueueEmpty:
                    break

            print(f"[kokoro-tts-server] Generating {len(requests)} request(s)", flush=True)

            # Generate audio for all requests
            generated = []
            loop = asyncio.get_event_loop()

            for req_text, req_ts in requests:
                audio = await loop.run_in_executor(None, generate_audio, req_text)
                if audio is not None and len(audio) > 0:
                    generated.append((req_text, audio))

            if generated:
                # Put all generated audio in ready queue
                await ready_queue.put(generated)

        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"[kokoro-tts-server] Generation worker error: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()


async def playback_worker():
    """Play audio from ready_queue, interrupt if new audio becomes ready."""
    print("[kokoro-tts-server] Playback worker started", flush=True)

    while not shutdown_event.is_set():
        try:
            # Wait for ready audio
            try:
                generated_batch = await asyncio.wait_for(
                    ready_queue.get(),
                    timeout=0.5
                )
            except asyncio.TimeoutError:
                continue

            # Play the batch
            if len(generated_batch) > 1:
                # Multiple requests - play snippets of all but last
                for i, (text, audio) in enumerate(generated_batch[:-1]):
                    print(f"[kokoro-tts-server] Snippet {i+1}/{len(generated_batch)-1}: {text[:50]}...", flush=True)

                    audio_file = save_audio(audio)
                    try:
                        await play_audio_file(audio_file, max_duration=SNIPPET_DURATION)
                    finally:
                        try:
                            os.unlink(audio_file)
                        except:
                            pass

                    await play_transition_sound()
                    await asyncio.sleep(0.05)

            # Play the last (or only) one in full
            text, audio = generated_batch[-1]
            print(f"[kokoro-tts-server] Playing: {text[:80]}{'...' if len(text) > 80 else ''}", flush=True)

            audio_file = save_audio(audio)
            try:
                interrupted = await play_audio_file(audio_file, check_ready_queue=True)
                if interrupted:
                    print("[kokoro-tts-server] Interrupted - new audio ready", flush=True)
                    await play_transition_sound()
            finally:
                try:
                    os.unlink(audio_file)
                except:
                    pass

        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"[kokoro-tts-server] Playback worker error: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()


async def run_server(port):
    """Run the async TCP server."""
    server = await asyncio.start_server(handle_client, 'localhost', port)

    addr = server.sockets[0].getsockname()
    print(f"[kokoro-tts-server] Listening on {addr[0]}:{addr[1]}", flush=True)
    print(f"[kokoro-tts-server] SSH to remote with: ssh -R {port}:localhost:{port} user@server", flush=True)
    print(f"[kokoro-tts-server] Press Ctrl+C to stop", flush=True)
    print("", flush=True)

    # Start workers
    gen_task = asyncio.create_task(generation_worker())
    play_task = asyncio.create_task(playback_worker())

    try:
        await server.serve_forever()
    except asyncio.CancelledError:
        pass
    finally:
        shutdown_event.set()
        gen_task.cancel()
        play_task.cancel()
        try:
            await gen_task
        except asyncio.CancelledError:
            pass
        try:
            await play_task
        except asyncio.CancelledError:
            pass
        server.close()
        await server.wait_closed()


def main():
    global pipeline, voice

    parser = argparse.ArgumentParser(description='Kokoro TTS server (async)')
    parser.add_argument('--port', type=int, default=20202, help='Port to listen on (default: 20202)')
    parser.add_argument('--voice', default='af_heart', help='Kokoro voice (default: af_heart)')
    parser.add_argument('--lang', default='a', help='Language code (default: a for American English)')
    args = parser.parse_args()

    voice = args.voice

    print(f"[kokoro-tts-server] Loading Kokoro model...", flush=True)
    try:
        from kokoro import KPipeline
        import soundfile as sf
        import numpy as np
    except ImportError as e:
        print(f"[kokoro-tts-server] Error: {e}", file=sys.stderr)
        print("[kokoro-tts-server] Install with: uv pip install kokoro soundfile numpy", file=sys.stderr)
        sys.exit(1)

    pipeline = KPipeline(lang_code=args.lang, repo_id='hexgrad/Kokoro-82M')
    print(f"[kokoro-tts-server] Model loaded, voice: {args.voice}", flush=True)

    def signal_handler(sig, frame):
        print("\n[kokoro-tts-server] Shutting down...", flush=True)
        cancel_current_audio()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        asyncio.run(run_server(args.port))
    except KeyboardInterrupt:
        print("\n[kokoro-tts-server] Shutting down...", flush=True)
        cancel_current_audio()


if __name__ == '__main__':
    main()
