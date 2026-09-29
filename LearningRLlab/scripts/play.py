"""Playback / evaluation entry point for LearningRLlab tasks.

Thin wrapper over ``scripts/reinforcement_learning/rsl_rl/play_rsl_rl.py`` in the
pinned isaaclab checkout, for the same reason ``train.py`` exists (see its
docstring). Unlike ``train_rsl_rl.py``, this module has no ``run(argv)`` -- its logic
is module-scope code under ``if __name__ == "__main__":`` -- so it's executed via
``runpy.run_path(..., run_name="__main__")`` with ``sys.argv``/``sys.path`` swapped
in and restored after, matching upstream's own dispatcher. ``sys.path`` needs the
script's own directory added since it does a bare ``import cli_args``.

Sim2sim export: on by default, ``--no-export`` turns it off. Upstream's script has no
hook between ``runner.load()`` and its rollout loop, so the export rides on the ONNX
export it already performs -- see ``_install_export_hook``.

Video: with Kit running (the training wrapper defaults every train/play invocation to
``--viz kit``, headless), bare ``--video`` works like a normal Isaac Sim install;
``--viz newton_rtx``/``newton_gl`` still work too, avoiding Kit's renderer.

MUST be launched with ``cwd=/workspace/isaaclab`` (F3), same as ``train.py``.
"""

import os
import runpy
import sys
from pathlib import Path

import warp as wp

# See train.py: must be set before the first Isaac Lab import.
wp.config.enable_backward = False

from torch.distributed.elastic.multiprocessing.errors import record

import LearningRLlab.tasks  # noqa: F401  (registers the LearningRL tasks with gymnasium)

# See train.py's comment on the same line.
_PLAY_RSL_RL = (
    Path(os.environ.get("ISAACLAB_PATH", "/workspace/isaaclab"))
    / "scripts"
    / "reinforcement_learning"
    / "rsl_rl"
    / "play_rsl_rl.py"
)


def _split_export_args(argv: list[str]) -> tuple[list[str], bool, str | None, str | None]:
    """Strip LearningRL-only flags before upstream's argparse sees them, and sniff ``--task``."""
    remaining: list[str] = []
    export, robot_name, task = True, None, None
    index = 0
    while index < len(argv):
        arg = argv[index]
        if arg == "--no-export":
            export = False
        elif arg == "--robot-name":
            robot_name = argv[index + 1] if index + 1 < len(argv) else None
            index += 1
        elif arg.startswith("--robot-name="):
            robot_name = arg.split("=", 1)[1]
        else:
            if arg == "--task" and index + 1 < len(argv):
                task = argv[index + 1]
            elif arg.startswith("--task="):
                task = arg.split("=", 1)[1]
            remaining.append(arg)
        index += 1
    return remaining, export, robot_name, task


def _install_export_hook(task: str | None, robot_name: str | None) -> None:
    """Redirect upstream's ONNX export into ``<run_dir>/<run_name>/`` and drop a
    ``config.yaml`` beside it.

    Upstream calls ``runner.load()`` then ``runner.export_policy_to_onnx()`` before it
    starts stepping, so patching the pair gives both the checkpoint path and a live env
    -- the only point where the policy is loaded but the env has not yet stepped (see
    ``LearningRLlab.export``).
    """
    from rsl_rl.runners import OnPolicyRunner

    from LearningRLlab.export import resolve_export_dir, write_export

    original_load = OnPolicyRunner.load
    original_export = OnPolicyRunner.export_policy_to_onnx

    def load(self, path, *args, **kwargs):
        self._learningrl_resume_path = path
        return original_load(self, path, *args, **kwargs)

    def export_policy_to_onnx(self, path, filename="policy.onnx", **kwargs):
        resume_path = getattr(self, "_learningrl_resume_path", None)
        if resume_path is None:
            return original_export(self, path, filename=filename, **kwargs)
        export_dir = resolve_export_dir(resume_path)
        export_dir.mkdir(parents=True, exist_ok=True)
        result = original_export(self, str(export_dir), filename=filename, **kwargs)
        written = write_export(
            self.env.unwrapped, task=task or "unknown", resume_path=resume_path, robot_name=robot_name
        )
        print(f"[INFO] Exported sim2sim policy + config to: {written}")
        return result

    OnPolicyRunner.load = load
    OnPolicyRunner.export_policy_to_onnx = export_policy_to_onnx


@record
def main():
    """Run LearningRLlab playback via play_rsl_rl.py, executed the same way upstream's own dispatcher executes it."""
    original_argv = sys.argv
    original_path = list(sys.path)
    forwarded, export, robot_name, task = _split_export_args(sys.argv[1:])
    try:
        if export:
            _install_export_hook(task, robot_name)
        sys.argv = [str(_PLAY_RSL_RL)] + forwarded
        sys.path.insert(0, str(_PLAY_RSL_RL.parent))
        runpy.run_path(str(_PLAY_RSL_RL), run_name="__main__")
    finally:
        sys.argv = original_argv
        sys.path[:] = original_path


if __name__ == "__main__":
    main()
