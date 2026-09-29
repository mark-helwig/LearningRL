"""The recording layer: channel definitions once, sinks attached to them.

`channels.py` is the single schema definition. `recorder.py` fans messages
out. `mcap_sink.py` writes the file. A live Foxglove WebSocket sink attaches
to the same channels in Phase 2 without the schemas moving.
"""
