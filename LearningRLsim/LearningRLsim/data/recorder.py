"""Fan-out from the rollout to any number of sinks, each with its own decimation.

The loop calls `record(topic, t, payload)` and does not know or care what is
attached. Sinks decimate independently because they want different things: the
file wants physics-rate torque when you are chasing a sim2sim gap, while a
live viewer stutters on it -- twelve channels of 200 Hz float arrays saturate
a WebSocket long before they tell anyone anything. Decimation is therefore the
SINK's property, not the recorder's: the same rollout writes a 200 Hz bag and
streams a 50 Hz view of itself.

**The rollout must never block on a sink.** A recording failure degrades the
run to an unrecorded one with a warning; it does not end it. The file is the
deliverable, but a lost bag is still better than a lost rollout.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from LearningRLsim.data.channels import ALL, BY_TOPIC, Channel, Rate


class Sink(ABC):
    """One destination for recorded messages."""

    physics_decimation: int = 1
    """Keep one in N physics-rate messages. Control-rate channels are never
    thinned: they are already at the rate a policy acts on, and dropping one
    would make this destination unable to answer "what did the policy see"."""

    @abstractmethod
    def open(self, channels: tuple[Channel, ...]) -> None: ...

    @abstractmethod
    def write(self, channel: Channel, t_ns: int, payload: dict[str, Any]) -> None: ...

    @abstractmethod
    def close(self) -> None: ...

    @property
    def name(self) -> str:
        return type(self).__name__


class Recorder:
    """Owns the sinks and the per-sink decimation.

    Timestamps are **sim time in nanoseconds**, for both `log_time` and
    `publish_time`. Viewers scrub on `log_time`, and sim time makes two bags
    from different runs directly comparable -- wall time does not, and would
    make a slow machine look like a slow robot.
    """

    def __init__(self, sinks: list[Sink]) -> None:
        self.sinks = sinks
        self._physics_counter = 0
        self._gates: dict[int, bool] = {}
        self._failed: set[str] = set()

    def __enter__(self) -> Recorder:
        for sink in list(self.sinks):
            try:
                sink.open(ALL)
            except Exception as exc:  # noqa: BLE001 -- a sink must not end the rollout
                print(f"[LearningRLsim] {sink.name} failed to open ({exc}); not recording to it.")
                self.sinks.remove(sink)
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def tick_physics(self) -> bool:
        """Open or close each sink's gate for this physics step. Once per step.

        Latched flags rather than arithmetic at each `record()` call: several
        physics-rate channels are written per step, and a sink must not take
        the torque of a step whose contact it dropped.
        """
        self._gates = {id(sink): self._physics_counter % max(1, sink.physics_decimation) == 0 for sink in self.sinks}
        self._physics_counter += 1
        return any(self._gates.values())

    def record(self, topic: str, t: float, payload: dict[str, Any]) -> None:
        channel = BY_TOPIC[topic]
        physics = channel.rate is Rate.PHYSICS
        t_ns = int(round(t * 1e9))
        for sink in self.sinks:
            if physics and not self._gates.get(id(sink), True):
                continue
            try:
                sink.write(channel, t_ns, payload)
            except Exception as exc:  # noqa: BLE001
                if sink.name not in self._failed:
                    self._failed.add(sink.name)
                    print(f"[LearningRLsim] {sink.name} write failed ({exc}); continuing unrecorded.")

    def close(self) -> None:
        for sink in self.sinks:
            try:
                sink.close()
            except Exception as exc:  # noqa: BLE001
                print(f"[LearningRLsim] {sink.name} failed to close cleanly: {exc}")
        self.sinks = []
