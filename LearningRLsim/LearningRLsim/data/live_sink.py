"""The live sink: a Foxglove WebSocket server, so a rollout can be watched as
it runs rather than after.

Same channels as the bag, same JSON encoding, different rate -- `channels.py`
is the single definition and this sink just advertises it. A viewer connected
here sees the 3D robot off `/tf` exactly as it would from a written file, which
is the point: nothing about the live view is a second, differently-derived
picture of the run.

**Three rules, in the order they matter.**

1. **The rollout never blocks on a viewer.** `write()` drops a serialized
   message into a bounded queue and returns; a background thread owns the
   socket. If the queue is full the OLDEST message is discarded, not the
   newest -- a viewer that has fallen behind wants to see now, not a minute
   ago -- and the count is reported once at close.
2. **No client attached is a no-op, not an error.** `send_message` only walks
   subscribed clients, so a run with nobody watching costs a dict lookup per
   message. Live streaming is a convenience; the bag is the deliverable.
3. **A sink that cannot open must not end the rollout.** `Recorder` catches
   that and degrades to an unrecorded-to-live run with a warning. The most
   likely cause by far is a port collision: two concurrent runs on one
   machine land in the same port namespace.

**Why a thread and not the rollout's own event loop.** The rollout is
synchronous, tight, and must stay that way -- interleaving `await` into a
physics loop would put socket latency inside the control path. The thread runs
one asyncio loop, polls the queue, and does nothing else.
"""

from __future__ import annotations

import asyncio
import json
import logging
import queue
import socket
import threading
from typing import Any

from LearningRLsim.data.channels import Channel, Rate
from LearningRLsim.data.encoding import jsonable
from LearningRLsim.data.recorder import Sink

DEFAULT_PORT = 8765
DEFAULT_HOST = "127.0.0.1"
"""Loopback, not 0.0.0.0: Foxglove desktop on the same machine connects to
ws://localhost:<port>. Binding the wildcard would additionally offer the stream
to the network, which is a different decision and belongs behind
`--live-host`."""

QUEUE_SIZE = 2048
"""~40 s of control-rate traffic across all channels. Deep enough that a
viewer's hiccup loses nothing, shallow enough that it cannot hide a sink that
has stopped draining."""

POLL_INTERVAL = 0.002
FLUSH_TIMEOUT = 1.0
OPEN_TIMEOUT = 5.0


class LiveError(RuntimeError):
    """The server could not be started."""


def _silent_logger() -> logging.Logger:
    """`FoxgloveServer`'s default logger prints every connection at INFO to
    stderr, interleaved with the rollout's own output. Ours stays at WARNING
    so a failure is still visible and a working stream is silent."""
    logger = logging.getLogger("LearningRLsim.live")
    logger.setLevel(logging.WARNING)
    return logger


class LiveSink(Sink):
    def __init__(
        self,
        port: int = DEFAULT_PORT,
        host: str = DEFAULT_HOST,
        physics_decimation: int = 1,
        queue_size: int = QUEUE_SIZE,
    ) -> None:
        self.port = port
        self.host = host
        self.physics_decimation = physics_decimation
        self._queue: queue.Queue[tuple[str, int, bytes]] = queue.Queue(maxsize=queue_size)
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._error: BaseException | None = None
        self._dropped = 0
        self._latched: dict[str, tuple[int, bytes]] = {}

    @property
    def name(self) -> str:
        return f"LiveSink(ws://{self.host}:{self.port})"

    def open(self, channels: tuple[Channel, ...]) -> None:
        self._probe_port()
        self._thread = threading.Thread(target=self._serve, args=(channels,), name="LearningRLsim-live", daemon=True)
        self._thread.start()
        if not self._ready.wait(OPEN_TIMEOUT):
            raise LiveError(f"the Foxglove server did not come up within {OPEN_TIMEOUT:g}s on port {self.port}.")
        if self._error is not None:
            raise LiveError(f"the Foxglove server failed to start: {self._error}")
        print(f"[LearningRLsim] streaming to ws://{self.host}:{self.port} -- in Foxglove, 'Open connection'.")

    def _probe_port(self) -> None:
        """Bind the port ourselves first, purely for the error message.

        asyncio's own failure surfaces inside the server task, where the only
        symptom this side would see is a server that never reports itself
        open. SO_REUSEADDR matches what `create_server` does, so the probe
        succeeds exactly when the real bind would.
        """
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((self.host, self.port))
        except OSError as exc:
            raise LiveError(
                f"port {self.port} is already in use ({exc.strerror}) -- another sim2sim run, "
                f"or another user's, is streaming on it. Pass --live-port <n> or set LEARNINGRL_FOXGLOVE_PORT."
            ) from exc
        finally:
            probe.close()

    def write(self, channel: Channel, t_ns: int, payload: dict[str, Any]) -> None:
        if self._thread is None:
            return
        data = json.dumps(jsonable(payload)).encode("utf-8")
        if channel.rate is Rate.ONCE:
            # `/meta/run` goes out at t=0, before anyone has had a chance to
            # connect. Latched and replayed on subscribe, so a viewer that
            # joins mid-rollout still learns which policy it is watching --
            # the same reason the bag carries it at all.
            self._latched[channel.topic] = (t_ns, data)
        try:
            self._queue.put_nowait((channel.topic, t_ns, data))
        except queue.Full:
            # Discard the oldest, then take the slot. A viewer behind by a
            # full queue wants the present, and the bag has the rest.
            self._dropped += 1
            try:
                self._queue.get_nowait()
                self._queue.put_nowait((channel.topic, t_ns, data))
            except (queue.Empty, queue.Full):
                pass

    def close(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=OPEN_TIMEOUT)
        self._thread = None
        if self._dropped:
            print(f"[LearningRLsim] live stream dropped {self._dropped} message(s); the bag is unaffected.")

    # -- the server thread ---------------------------------------------------
    def _serve(self, channels: tuple[Channel, ...]) -> None:
        try:
            asyncio.run(self._main(channels))
        except BaseException as exc:  # noqa: BLE001 -- reported to the opening thread, never raised here
            self._error = exc
        finally:
            self._ready.set()

    async def _main(self, channels: tuple[Channel, ...]) -> None:
        from foxglove_websocket.server import FoxgloveServer

        server = FoxgloveServer(self.host, self.port, "LearningRLsim", logger=_silent_logger())
        server.start()
        topics: dict[int, str] = {}
        try:
            await asyncio.wait_for(server.wait_opened(), OPEN_TIMEOUT)
            ids = {
                channel.topic: await server.add_channel(
                    {
                        "topic": channel.topic,
                        "encoding": "json",
                        # The same name the bag registers, so a Foxglove panel
                        # written against one works against the other -- and
                        # so `/tf` renders, which it only does when the schema
                        # is called `foxglove.FrameTransforms`.
                        "schemaName": channel.schema_name,
                        "schema": json.dumps(channel.schema),
                        "schemaEncoding": "jsonschema",
                    }
                )
                for channel in channels
            }
            topics.update({chan_id: topic for topic, chan_id in ids.items()})
            server.set_listener(_Latch(self, topics))
            self._ready.set()
            await self._pump(server, ids)
        finally:
            server.close()
            await server.wait_closed()

    async def _pump(self, server: Any, ids: dict[str, int]) -> None:
        while not self._stop.is_set():
            if not await self._drain(server, ids):
                await asyncio.sleep(POLL_INTERVAL)
        # A rollout that just ended still has its last few messages in the
        # queue; a viewer watching the fall wants them.
        deadline = asyncio.get_running_loop().time() + FLUSH_TIMEOUT
        while asyncio.get_running_loop().time() < deadline:
            if not await self._drain(server, ids):
                break

    async def replay_latched(self, server: Any, chan_id: int, topic: str) -> None:
        """Send the latched message for `topic`, if there is one."""
        latched = self._latched.get(topic)
        if latched is not None:
            await server.send_message(chan_id, *latched)

    async def _drain(self, server: Any, ids: dict[str, int]) -> bool:
        sent = False
        while True:
            try:
                topic, t_ns, data = self._queue.get_nowait()
            except queue.Empty:
                return sent
            await server.send_message(ids[topic], t_ns, data)
            sent = True


class _Latch:
    """Replays the write-once channels to a client that subscribes late.

    `FoxgloveServerListener`'s hooks are all optional at runtime -- the server
    calls what it finds -- so this does not subclass it and drag the import
    into module scope.
    """

    def __init__(self, sink: LiveSink, topics: dict[int, str]) -> None:
        self._sink = sink
        self._topics = topics

    async def on_subscribe(self, server: Any, channel_id: int) -> None:
        await self._sink.replay_latched(server, channel_id, self._topics.get(channel_id, ""))

    async def on_unsubscribe(self, server: Any, channel_id: int) -> None:
        pass
