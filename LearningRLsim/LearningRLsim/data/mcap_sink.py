"""The MCAP file sink -- Phase 1's deliverable.

Writer settings that matter, and why:

- **zstd chunk compression, ~4 MB chunks.** Physics-rate float arrays
  compress well, which is what makes plain JSON affordable.
- **The writer builds the summary index.** That index is what makes
  Foxglove's seek instant; a bag without one plays but scrubs badly.
- **No per-message flush.** Flushing per message would close a chunk per
  message, destroying both the compression ratio and the index.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mcap.writer import CompressionType, Writer

from LearningRLsim.data.channels import Channel
from LearningRLsim.data.encoding import jsonable
from LearningRLsim.data.recorder import Sink

CHUNK_SIZE = 4 * 1024 * 1024


class McapSink(Sink):
    def __init__(self, path: str | Path, physics_decimation: int = 1) -> None:
        self.path = Path(path)
        self.physics_decimation = physics_decimation
        self._stream = None
        self._writer: Writer | None = None
        self._channel_ids: dict[str, int] = {}
        self._sequence: dict[str, int] = {}

    def open(self, channels: tuple[Channel, ...]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open("wb")
        self._writer = Writer(self._stream, compression=CompressionType.ZSTD, chunk_size=CHUNK_SIZE)
        self._writer.start(profile="", library="LearningRLsim")
        for channel in channels:
            schema_id = self._writer.register_schema(
                name=channel.schema_name,
                encoding="jsonschema",
                data=json.dumps(channel.schema).encode("utf-8"),
            )
            self._channel_ids[channel.topic] = self._writer.register_channel(
                topic=channel.topic,
                message_encoding="json",
                schema_id=schema_id,
                metadata={"rate": channel.rate.value, "description": channel.description},
            )
            self._sequence[channel.topic] = 0

    def write(self, channel: Channel, t_ns: int, payload: dict[str, Any]) -> None:
        if self._writer is None:
            return
        self._writer.add_message(
            channel_id=self._channel_ids[channel.topic],
            log_time=t_ns,
            publish_time=t_ns,
            sequence=self._sequence[channel.topic],
            data=json.dumps(jsonable(payload)).encode("utf-8"),
        )
        self._sequence[channel.topic] += 1

    def close(self) -> None:
        if self._writer is not None:
            self._writer.finish()
            self._writer = None
        if self._stream is not None:
            self._stream.close()
            self._stream = None
