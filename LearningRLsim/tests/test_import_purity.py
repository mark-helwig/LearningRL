"""Invariant 1: LearningRLsim imports no Isaac Lab, Isaac Sim, Newton, warp or torch
-- ever, including transitively.

A sim2sim check that shared the training stack's dynamics code would not be
checking anything, and the standalone CPU-only mode is a hard requirement
rather than a nicety. Both are one `import` away from being silently lost, in a
way nothing else here would notice: in-container every forbidden package is
importable and everything would keep working.

Runs in a SUBPROCESS so `sys.modules` is clean -- by the time the rest of the
suite has run, the test process has plenty of these loaded for its own reasons.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytestmark = pytest.mark.static

# Not the monorepo root, whose `LearningRLsim/` directory would shadow the package.
HERE = Path(__file__).resolve().parent

FORBIDDEN = (
    "isaaclab",
    "isaacsim",
    "omni",
    "carb",
    "pxr",
    "newton",
    "warp",
    "torch",
    "rsl_rl",
)

# Every module in the package. Listed explicitly rather than walked, so that a
# new module has to be added here consciously.
MODULES = (
    "LearningRLsim",
    "LearningRLsim.registry",
    "LearningRLsim.runconfig",
    "LearningRLsim.library",
    "LearningRLsim.observations",
    "LearningRLsim.actions",
    "LearningRLsim.context",
    "LearningRLsim.model",
    "LearningRLsim.objects",
    "LearningRLsim.policy",
    "LearningRLsim.loop",
    "LearningRLsim.eval",
    "LearningRLsim.commands.terms",
    "LearningRLsim.commands.sources",
    "LearningRLsim.commands.scripted",
    "LearningRLsim.commands.keyboard",
    "LearningRLsim.data.channels",
    "LearningRLsim.data.encoding",
    "LearningRLsim.data.recorder",
    "LearningRLsim.data.mcap_sink",
    "LearningRLsim.data.live_sink",
    "LearningRLsim.data.scene",
    "LearningRLsim.data.transforms",
)


def _imported_roots(module: str) -> set[str]:
    script = textwrap.dedent(f"""
        import sys, json
        import {module}
        print(json.dumps(sorted({{m.split(".")[0] for m in sys.modules}})))
    """)
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False, cwd=HERE)
    if result.returncode != 0:
        pytest.fail(f"importing {module} failed:\n{result.stderr}")
    import json

    return set(json.loads(result.stdout.strip().splitlines()[-1]))


@pytest.mark.parametrize("module", MODULES)
def test_module_pulls_in_nothing_gpu_bound(module):
    offenders = sorted(_imported_roots(module) & set(FORBIDDEN))
    assert not offenders, (
        f"importing {module} pulled in {offenders}. LearningRLsim must import no Isaac Lab, "
        f"Isaac Sim, Newton, warp or torch, including transitively -- it has to run on a "
        f"CPU-only host with no simulator installed."
    )


def test_the_whole_package_is_importable_without_robot_description():
    """`robot_description` is imported lazily, inside `resolve_mjcf`, so a
    missing one is a clear error at model-resolution time rather than an
    ImportError at startup."""
    script = textwrap.dedent("""
        import sys
        class Hide:
            def find_spec(self, name, path=None, target=None):
                if name.split('.')[0] == 'robot_description':
                    raise ModuleNotFoundError(name, name=name)
                return None
        sys.meta_path.insert(0, Hide())
        import LearningRLsim.eval
        print("ok")
    """)
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False, cwd=HERE)
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
