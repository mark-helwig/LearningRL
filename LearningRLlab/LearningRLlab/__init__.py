"""LearningRL's Isaac Lab extension: the locomotion tasks.

This package stays free of any Omniverse (``omni.*``) imports at module scope, so it
can be imported -- for task discovery, AST tooling, or a plain
``python -c "import LearningRLlab"`` -- without a running simulation app. Anything that
needs ``omni.*`` belongs behind a function call, never at import time.
"""

from . import tasks  # noqa: F401
