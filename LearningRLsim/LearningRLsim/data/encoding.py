"""numpy -> plain JSON, shared by every sink.

Lives here rather than in `mcap_sink.py` because the file and the live socket
must encode a payload identically -- a bag and the stream someone watched it
being written on cannot disagree about what a message said.

Encoding happens on the ROLLOUT's thread, not a sink's. Payloads carry numpy
arrays the loop mutates in place (`command.stiffness` is rewritten every
policy step), so handing the array to a background thread would race the next
step. `json.dumps` of a few dozen floats costs microseconds; a stale gain plot
costs a day.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def jsonable(value: Any) -> Any:
    """numpy -> plain JSON. Non-finite floats become null.

    `json.dumps` emits bare `NaN`/`Infinity` for those, which is valid Python
    but NOT valid JSON -- Foxglove and `mcap cat` both reject the message. A
    diverging rollout produces exactly those values, so this is the path that
    matters most.
    """
    if isinstance(value, np.ndarray):
        return [jsonable(v) for v in value.tolist()]
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value
