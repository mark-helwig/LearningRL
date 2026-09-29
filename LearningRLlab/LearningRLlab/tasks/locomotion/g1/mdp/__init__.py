"""Task-local MDP terms for LearningRL-G1-Walk-v0.

Upstream's `mdp` packages are re-exported wholesale so `g1_env_cfg.py` can reach
every stock term through this one namespace; the task-local terms below are
imported after them, so a local name of the same spelling wins.

Re-export upstream's *leaf* modules, never `isaaclab.envs.mdp` itself. That
package binds its names lazily, so `from isaaclab.envs.mdp import *` forces every
submodule to resolve -- including `actions.binary_joint_actions`, which chains
`Articulation` -> `isaaclab.sim` -> `import pxr`. Hydra imports this cfg *before*
`AppLauncher` starts Kit, so that `pxr` pulls the pip `usd-core` USD into the
process; Kit then dlopens its own `omni.usd.libs` USD and the two copies of the
same symbols corrupt the heap (`free(): invalid pointer`, exit 134). The leaf
modules below are pxr-free and between them export every stock name the cfg
uses. The action/command *implementation* classes that the old wildcard also
dragged in are reached at runtime via each cfg's `class_type`, so re-exporting
them here is unnecessary -- and is exactly what made the import unsafe.

The `velocity.mdp` wildcard stays: its own `__all__` holds only the eight
locomotion-specific terms, and it reaches `isaaclab.envs.mdp` through a runtime
fallback that `import *` does not follow.
"""

from isaaclab.envs.mdp.actions.actions_cfg import *  # noqa: F403
from isaaclab.envs.mdp.commands.commands_cfg import *  # noqa: F403
from isaaclab.envs.mdp.curriculums import *  # noqa: F403
from isaaclab.envs.mdp.events import *  # noqa: F403
from isaaclab.envs.mdp.observations import *  # noqa: F403
from isaaclab.envs.mdp.rewards import *  # noqa: F403
from isaaclab.envs.mdp.terminations import *  # noqa: F403

from isaaclab_tasks.manager_based.locomotion.velocity.config.spot.mdp import foot_clearance_reward
from isaaclab_tasks.manager_based.locomotion.velocity.mdp import *  # noqa: F403

from .events import view_middle_env
from .rewards import energy, feet_gait
