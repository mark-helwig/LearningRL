"""The exported ONNX policy, wrapped to one `numpy in, numpy out` call.

Deterministic inference: the exported graph is a bare MLP whose output is the
Gaussian MEAN, so there is no sampling to reproduce. LearningRLlab's runner sets
`obs_normalization = False`, so there is no normalizer to apply either --
but nothing in `config.yaml` records that, so it is an assumption, recorded
as one in `/meta/run` rather than left implicit.

Tensor names are read off the graph, never hardcoded. The exporter has
already changed them once (`x`/`linear_3` in an earlier revision,
`obs`/`actions` in the current one), and a future recurrent export would add
hidden-state inputs -- which must fail loudly here rather than silently
running a stateful policy statelessly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime

from LearningRLsim.runconfig import RunConfig


class PolicyError(RuntimeError):
    """The ONNX policy is missing, unloadable, or does not fit the config."""


class Policy:
    def __init__(self, path: str | Path, cfg: RunConfig) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise PolicyError(f"{self.path}: no such file")
        # ONNX external data is resolved RELATIVE TO THE MODEL FILE, so a
        # policy.onnx copied without its .data sibling loads and then fails
        # with a weight-shaped error much later.
        external = self.path.with_suffix(self.path.suffix + ".data")
        if not external.is_file():
            raise PolicyError(
                f"{external} is missing. The exporter writes the weights as ONNX external "
                f"data, which must sit beside policy.onnx -- copy the run directory, not "
                f"just the .onnx."
            )

        self.session = onnxruntime.InferenceSession(str(self.path), providers=["CPUExecutionProvider"])

        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise PolicyError(
                f"{self.path}: expected one input and one output, got "
                f"{[i.name for i in inputs]} -> {[o.name for o in outputs]}. A multi-input "
                f"graph is a recurrent or otherwise stateful export, which LearningRLsim's "
                f"feed-forward rollout would run incorrectly."
            )
        self.input_name = inputs[0].name
        self.output_name = outputs[0].name

        _check_dim(inputs[0].shape, cfg.observations.total_dim, "input", self.path, "observation")
        _check_dim(outputs[0].shape, cfg.actions.total_dim, "output", self.path, "action")

        self.obs_dim = cfg.observations.total_dim
        self.action_dim = cfg.actions.total_dim

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        """One deterministic forward pass. Returns the RAW action vector."""
        batched = np.asarray(obs, dtype=np.float32).reshape(1, self.obs_dim)
        out = self.session.run([self.output_name], {self.input_name: batched})[0]
        return np.asarray(out, dtype=np.float64).reshape(self.action_dim)


def _check_dim(shape: list, expected: int, which: str, path: Path, what: str) -> None:
    """The last axis must match; leading axes are the batch and may be symbolic."""
    if not shape:
        return
    last = shape[-1]
    if isinstance(last, int) and last != expected:
        raise PolicyError(
            f"{path}: the graph's {which} is {shape}, but config.yaml declares a "
            f"{expected}-dim {what} vector. The policy and the config describe different "
            f"runs."
        )
