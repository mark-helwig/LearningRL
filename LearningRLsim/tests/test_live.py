"""The live sink: it serves, it never blocks the rollout, and it says what is
wrong when a port is taken.

These bind a real socket on a free port -- no MuJoCo, no run directory, no
container, so `static` still holds.
"""

from __future__ import annotations

import json
import socket
import threading

import numpy as np
import pytest

from LearningRLsim.data import channels
from LearningRLsim.data.live_sink import LiveError, LiveSink

pytestmark = pytest.mark.static


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@pytest.fixture
def sink():
    live = LiveSink(port=free_port())
    yield live
    live.close()


def test_it_serves_the_bag_s_channels(sink):
    sink.open(channels.ALL)
    with socket.create_connection((sink.host, sink.port), timeout=5) as client:
        assert client.getpeername()[1] == sink.port


def test_a_taken_port_names_the_likely_cause(sink):
    """Host networking means a shared port namespace, so this is the failure
    people will actually hit."""
    sink.open(channels.ALL)
    second = LiveSink(port=sink.port)
    with pytest.raises(LiveError, match="already in use"):
        second.open(channels.ALL)


def test_writing_with_nobody_watching_is_a_no_op(sink):
    sink.open(channels.ALL)
    for step in range(200):
        sink.write(channels.SIM_STATE, step, {"t": step * 0.005})
    sink.close()  # a clean shutdown is the assertion: nothing blocked, nothing raised


def test_a_full_queue_drops_the_oldest_and_never_blocks():
    """The rollout must not wait on a viewer. With no server thread draining
    it, a hundred messages into a ten-deep queue must still return."""
    live = LiveSink(port=free_port(), queue_size=10)
    live._thread = threading.current_thread()  # pretend it is open, drain nothing

    for step in range(100):
        live.write(channels.SIM_STATE, step, {"t": step})

    assert live._queue.qsize() == 10
    assert live._dropped == 90
    newest = json.loads(live._queue.queue[-1][2])
    assert newest["t"] == 99


def test_write_before_open_is_dropped_not_queued():
    live = LiveSink(port=free_port())
    live.write(channels.SIM_STATE, 0, {"t": 0.0})
    assert live._queue.empty()


def test_the_payload_is_encoded_before_write_returns():
    """Payloads carry arrays the loop rewrites in place -- `command.stiffness`
    is rewritten every policy step -- so a message must be bytes by the time
    `write` returns, not a reference the server thread reads a step later."""
    live = LiveSink(port=free_port())
    live._thread = threading.current_thread()  # queued, never drained

    payload = {"t": 0.0, "stiffness": np.array([1.0, 2.0])}
    live.write(channels.ACTUATOR_CMD, 0, payload)
    payload["stiffness"][0] = 999.0

    assert json.loads(live._queue.get_nowait()[2])["stiffness"] == [1.0, 2.0]
