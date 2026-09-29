"""Training entry point for LearningRLlab tasks.

Thin wrapper over ``scripts/reinforcement_learning/rsl_rl/train_rsl_rl.py`` in the
pinned isaaclab checkout (not a pip package). That module imports only
``isaaclab_tasks`` at module scope, so it can't see LearningRLlab's tasks -- and a
module-qualified task id doesn't help either, since the gym lookup strips any prefix
without importing it. So this script does ``import LearningRLlab.tasks`` first (registering
the task with gymnasium), then loads ``train_rsl_rl.py`` by file path and calls its
``run(argv)``, the same way upstream's own dispatcher does. It also needs
``scripts/reinforcement_learning/`` on ``sys.path`` first, since ``train_rsl_rl.py``
does ``from common import (...)`` and we aren't the script that gets that for free.

MUST be launched with ``cwd=/workspace/isaaclab`` (F3): ``train_rsl_rl.py``'s log
directory is CWD-relative. REPO 1's ``docker exec -w`` pins this.
"""

import os
import sys
from pathlib import Path

import warp as wp

# Warp captures this flag at module creation, so it must be set before the first
# Isaac Lab import (which imports warp transitively). Isaac Lab does not use Warp
# autodiff; skipping adjoint codegen roughly halves cold kernel-cache build time.
wp.config.enable_backward = False

from torch.distributed.elastic.multiprocessing.errors import record

import LearningRLlab.tasks  # noqa: F401  (registers the LearningRL tasks with gymnasium)

# ISAACLAB_PATH is exported by isaaclab.sh before it execs the interpreter; falls
# back to the fixed container workspace path in case this is invoked some other way.
_RL_SCRIPTS_ROOT = Path(os.environ.get("ISAACLAB_PATH", "/workspace/isaaclab")) / "scripts" / "reinforcement_learning"
if str(_RL_SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_RL_SCRIPTS_ROOT))

from common import import_local_module  # noqa: E402  (needs the sys.path insert above)


@record
def main():
    """Run LearningRLlab training via train_rsl_rl.py's run(), loaded the way upstream's own dispatcher loads it."""
    module = import_local_module("learningrl_train_rsl_rl", _RL_SCRIPTS_ROOT / "rsl_rl" / "train_rsl_rl.py")
    module.run(sys.argv[1:])


if __name__ == "__main__":
    main()
