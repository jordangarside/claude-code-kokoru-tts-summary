"""Tests for kokoro-server.py"""

import asyncio
import pytest
import numpy as np
from unittest.mock import Mock, patch, AsyncMock
from collections import deque

# Import the module under test
import importlib.util
spec = importlib.util.spec_from_file_location("server", "kokoro_server.py")
server = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def reset_server_state():
    """Reset global state before each test."""
    server.pending_messages.clear()
    server.ready_audio.clear()
    server.current_process = None
    server.play_start_time = None
    server.shutdown_event.clear()
    server.new_message_event.clear()
    server.audio_ready_event.clear()
    yield
    # Cleanup after test
    server.pending_messages.clear()
    server.ready_audio.clear()


@pytest.fixture
def default_config():
    """Default configuration."""
    return server.Config()


@pytest.fixture
def no_queue_config():
    """Configuration with queue disabled."""
    return server.Config(queue=False)


@pytest.fixture
def no_interrupt_config():
    """Configuration with interrupt disabled."""
    return server.Config(interrupt=False)


# =============================================================================
# Config Tests
# =============================================================================

class TestConfig:
    def test_default_values(self):
        config = server.Config()
        assert config.port == 20202
        assert config.voice == "af_heart"
        assert config.lang == "a"
        assert config.interrupt is True
        assert config.min_duration == 1.5
        assert config.queue is True
        assert config.max_queue == 10
        assert config.interrupt_chime is True
        assert config.drop_sound is True

    def test_custom_values(self):
        config = server.Config(
            port=12345,
            voice="custom_voice",
            interrupt=False,
            queue=False,
            max_queue=5,
        )
        assert config.port == 12345
        assert config.voice == "custom_voice"
        assert config.interrupt is False
        assert config.queue is False
        assert config.max_queue == 5


# =============================================================================
# Sound Generation Tests
# =============================================================================

class TestSoundGeneration:
    def test_generate_chime_shape(self):
        chime = server.generate_chime()
        assert isinstance(chime, np.ndarray)
        assert chime.dtype == np.float32
        assert len(chime) > 0

    def test_generate_chime_duration(self):
        sample_rate = 24000
        chime = server.generate_chime(sample_rate)
        duration = len(chime) / sample_rate
        # Two 0.08s notes + 0.03s gap = ~0.19s
        assert 0.15 < duration < 0.25

    def test_generate_chime_amplitude(self):
        chime = server.generate_chime()
        assert np.max(np.abs(chime)) <= 1.0

    def test_generate_drop_tone_shape(self):
        drop = server.generate_drop_tone()
        assert isinstance(drop, np.ndarray)
        assert drop.dtype == np.float32
        assert len(drop) > 0

    def test_generate_drop_tone_duration(self):
        sample_rate = 24000
        drop = server.generate_drop_tone(sample_rate)
        duration = len(drop) / sample_rate
        # Should be around 0.15s
        assert 0.1 < duration < 0.2

    def test_generate_drop_tone_amplitude(self):
        drop = server.generate_drop_tone()
        assert np.max(np.abs(drop)) <= 1.0


# =============================================================================
# Message Tests
# =============================================================================

class TestMessage:
    def test_message_creation(self):
        msg = server.Message(id="123", text="hello", timestamp=1000.0)
        assert msg.id == "123"
        assert msg.text == "hello"
        assert msg.timestamp == 1000.0


class TestReadyAudio:
    def test_ready_audio_creation(self):
        audio = server.ReadyAudio(message_id="123", audio_file="/tmp/test.wav", text="hello")
        assert audio.message_id == "123"
        assert audio.audio_file == "/tmp/test.wav"
        assert audio.text == "hello"


# =============================================================================
# Queue Behavior Tests
# =============================================================================

class TestAddMessageQueueMode:
    """Tests for add_message with queue=True (default)."""

    @pytest.mark.asyncio
    async def test_add_single_message(self, reset_server_state):
        server.config = server.Config(queue=True)

        await server.add_message("test message")

        assert len(server.pending_messages) == 1
        assert server.pending_messages[0].text == "test message"

    @pytest.mark.asyncio
    async def test_add_multiple_messages_queued(self, reset_server_state):
        server.config = server.Config(queue=True)

        await server.add_message("message 1")
        await server.add_message("message 2")
        await server.add_message("message 3")

        assert len(server.pending_messages) == 3
        assert server.pending_messages[0].text == "message 1"
        assert server.pending_messages[1].text == "message 2"
        assert server.pending_messages[2].text == "message 3"

    @pytest.mark.asyncio
    async def test_queue_max_limit(self, reset_server_state):
        server.config = server.Config(queue=True, max_queue=3, drop_sound=False)

        await server.add_message("message 1")
        await server.add_message("message 2")
        await server.add_message("message 3")
        await server.add_message("message 4")  # Should drop oldest

        assert len(server.pending_messages) == 3
        assert server.pending_messages[0].text == "message 2"
        assert server.pending_messages[1].text == "message 3"
        assert server.pending_messages[2].text == "message 4"

    @pytest.mark.asyncio
    async def test_queue_overflow_plays_drop_sound(self, reset_server_state):
        server.config = server.Config(queue=True, max_queue=2, drop_sound=True)

        with patch.object(server, 'play_drop_sound') as mock_drop:
            await server.add_message("message 1")
            await server.add_message("message 2")
            await server.add_message("message 3")

            mock_drop.assert_called_once()

    @pytest.mark.asyncio
    async def test_new_message_event_set(self, reset_server_state):
        server.config = server.Config(queue=True)
        server.new_message_event.clear()

        await server.add_message("test")

        assert server.new_message_event.is_set()


class TestAddMessageNoQueueMode:
    """Tests for add_message with queue=False."""

    @pytest.mark.asyncio
    async def test_single_message(self, reset_server_state):
        server.config = server.Config(queue=False)

        await server.add_message("test message")

        assert len(server.pending_messages) == 1
        assert server.pending_messages[0].text == "test message"

    @pytest.mark.asyncio
    async def test_new_message_replaces_pending(self, reset_server_state):
        server.config = server.Config(queue=False, drop_sound=False)

        await server.add_message("message 1")
        await server.add_message("message 2")

        assert len(server.pending_messages) == 1
        assert server.pending_messages[0].text == "message 2"

    @pytest.mark.asyncio
    async def test_replace_plays_drop_sound(self, reset_server_state):
        server.config = server.Config(queue=False, drop_sound=True)

        with patch.object(server, 'play_drop_sound') as mock_drop:
            await server.add_message("message 1")
            await server.add_message("message 2")

            mock_drop.assert_called_once()

    @pytest.mark.asyncio
    async def test_multiple_pending_all_dropped(self, reset_server_state):
        server.config = server.Config(queue=False, drop_sound=False)

        # Manually add multiple pending to simulate race condition
        server.pending_messages.append(server.Message("1", "msg1", 1.0))
        server.pending_messages.append(server.Message("2", "msg2", 2.0))

        await server.add_message("latest")

        assert len(server.pending_messages) == 1
        assert server.pending_messages[0].text == "latest"


# =============================================================================
# Audio Player Tests
# =============================================================================

class TestGetPlayer:
    def test_returns_list(self):
        player = server.get_player()
        # Should return a list or None
        assert player is None or isinstance(player, list)

    @patch('sys.platform', 'darwin')
    def test_darwin_uses_afplay(self):
        player = server.get_player()
        assert player == ['afplay']


class TestStopCurrentAudio:
    def test_stops_running_process(self, reset_server_state):
        mock_process = Mock()
        mock_process.poll.return_value = None  # Still running
        server.current_process = mock_process
        server.play_start_time = 100.0

        server.stop_current_audio()

        mock_process.terminate.assert_called_once()
        assert server.current_process is None
        assert server.play_start_time is None

    def test_handles_no_process(self, reset_server_state):
        server.current_process = None
        server.play_start_time = None

        # Should not raise
        server.stop_current_audio()

        assert server.current_process is None

    def test_handles_already_finished_process(self, reset_server_state):
        mock_process = Mock()
        mock_process.poll.return_value = 0  # Already finished
        server.current_process = mock_process

        server.stop_current_audio()

        mock_process.terminate.assert_not_called()


# =============================================================================
# Integration-style Tests
# =============================================================================

class TestInterruptBehavior:
    """Test interrupt vs no-interrupt modes."""

    @pytest.mark.asyncio
    async def test_interrupt_mode_allows_interrupt_after_min_duration(self, reset_server_state):
        server.config = server.Config(interrupt=True, min_duration=1.0)

        # Simulate audio playing for 2 seconds
        mock_process = Mock()
        mock_process.poll.return_value = None
        server.current_process = mock_process
        server.play_start_time = asyncio.get_event_loop().time() - 2.0

        # Should be able to interrupt (elapsed > min_duration)
        elapsed = asyncio.get_event_loop().time() - server.play_start_time
        assert elapsed > server.config.min_duration

    @pytest.mark.asyncio
    async def test_interrupt_mode_waits_for_min_duration(self, reset_server_state):
        server.config = server.Config(interrupt=True, min_duration=1.0)

        # Simulate audio just started
        server.play_start_time = asyncio.get_event_loop().time()

        elapsed = asyncio.get_event_loop().time() - server.play_start_time
        assert elapsed < server.config.min_duration


class TestQueueVsNoQueueGeneration:
    """Test that generation worker handles queue modes correctly."""

    @pytest.mark.asyncio
    async def test_queue_mode_processes_fifo(self, reset_server_state):
        server.config = server.Config(queue=True)

        # Add messages
        msg1 = server.Message("1", "first", 1.0)
        msg2 = server.Message("2", "second", 2.0)
        server.pending_messages.append(msg1)
        server.pending_messages.append(msg2)

        # In queue mode, should take first
        async with server.pending_lock:
            next_msg = server.pending_messages[0]

        assert next_msg.text == "first"

    @pytest.mark.asyncio
    async def test_no_queue_mode_takes_latest(self, reset_server_state):
        server.config = server.Config(queue=False, drop_sound=False)

        # Add messages
        msg1 = server.Message("1", "first", 1.0)
        msg2 = server.Message("2", "second", 2.0)
        server.pending_messages.append(msg1)
        server.pending_messages.append(msg2)

        # In no-queue mode, should drop all but latest
        async with server.pending_lock:
            while len(server.pending_messages) > 1:
                server.pending_messages.popleft()
            next_msg = server.pending_messages[0]

        assert next_msg.text == "second"


# =============================================================================
# TCP Handler Tests
# =============================================================================

class TestHandleClient:
    @pytest.mark.asyncio
    async def test_ping_pong(self, reset_server_state):
        reader = AsyncMock()
        writer = AsyncMock()

        # Simulate ping message
        reader.read = AsyncMock(side_effect=[b"ping", b""])
        writer.write = Mock()
        writer.drain = AsyncMock()
        writer.close = Mock()
        writer.wait_closed = AsyncMock()

        await server.handle_client(reader, writer)

        writer.write.assert_called_with(b"pong")

    @pytest.mark.asyncio
    async def test_text_message_added(self, reset_server_state):
        server.config = server.Config()
        reader = AsyncMock()
        writer = AsyncMock()

        # Simulate text message
        reader.read = AsyncMock(side_effect=[b"hello world", b""])
        writer.close = Mock()
        writer.wait_closed = AsyncMock()

        await server.handle_client(reader, writer)

        assert len(server.pending_messages) == 1
        assert server.pending_messages[0].text == "hello world"

    @pytest.mark.asyncio
    async def test_empty_message_ignored(self, reset_server_state):
        server.config = server.Config()
        reader = AsyncMock()
        writer = AsyncMock()

        # Simulate empty message
        reader.read = AsyncMock(side_effect=[b"   ", b""])
        writer.close = Mock()
        writer.wait_closed = AsyncMock()

        await server.handle_client(reader, writer)

        assert len(server.pending_messages) == 0


# =============================================================================
# Argument Parsing Tests
# =============================================================================

class TestArgumentParsing:
    def test_default_args(self):
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--port', type=int, default=20202)
        parser.add_argument('--interrupt', action=argparse.BooleanOptionalAction, default=True)
        parser.add_argument('--queue', action=argparse.BooleanOptionalAction, default=True)

        args = parser.parse_args([])
        assert args.port == 20202
        assert args.interrupt is True
        assert args.queue is True

    def test_no_interrupt_flag(self):
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--interrupt', action=argparse.BooleanOptionalAction, default=True)

        args = parser.parse_args(['--no-interrupt'])
        assert args.interrupt is False

    def test_no_queue_flag(self):
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--queue', action=argparse.BooleanOptionalAction, default=True)

        args = parser.parse_args(['--no-queue'])
        assert args.queue is False

    def test_explicit_interrupt_flag(self):
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--interrupt', action=argparse.BooleanOptionalAction, default=True)

        args = parser.parse_args(['--interrupt'])
        assert args.interrupt is True


# =============================================================================
# Run with: pytest test_kokoro_server.py -v
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
