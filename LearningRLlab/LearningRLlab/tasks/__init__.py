"""Task registration entry point for LearningRLlab.

Importing this package registers every Gymnasium environment LearningRLlab defines, via
``isaaclab_tasks.utils.import_packages``, which recurses into every submodule whose
qualified name does not contain one of the given substrings.

The ``".mdp"`` substring is deliberate: a per-task ``mdp`` package registers no
environment, and each task's env cfg imports its own ``mdp`` anyway, so walking into
them here would only import the same modules by a second path.
"""

from isaaclab_tasks.utils import import_packages

import_packages(__name__, ["utils", ".mdp"])
