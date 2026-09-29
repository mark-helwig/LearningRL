"""Shared MDP terms for LearningRLlab tasks.

``symmetry`` -- the left/right mirror for rsl_rl's symmetry augmentation, shared by every walking task.

Placement rule
--------------
Reward, observation, event, termination, and command terms start life in the
task-local ``mdp/`` package of whichever task first needs them (for example
``LearningRLlab/tasks/locomotion/g1/mdp/``). A term only moves here once a *second* task needs the
same term -- and it moves by relocating the implementation, never by copying it, so
there is exactly one definition to maintain at any time.

Shared terms follow the convention the task-local ``mdp``
packages use:

- implementation in a plain module (``rewards.py``, ``terminations.py``,
  ``events.py``, ...), never inline in ``__init__.py``;
- ``__init__.py`` imports the public symbols from those modules by name, after any
  wildcard re-export of an upstream ``mdp`` package, so a local term of the same
  spelling wins.
"""
