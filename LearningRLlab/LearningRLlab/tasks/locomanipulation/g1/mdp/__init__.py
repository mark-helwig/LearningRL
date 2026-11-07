"""Task-local MDP terms for LearningRL-G1-LocoManipulation-v0.

The walk task's ``mdp`` namespace first -- which is itself upstream's pxr-free leaf modules plus the
walk's own terms (F6) -- then this task's, so a local name of the same spelling wins.

``HandPosCommand`` itself is not exported: it imports ``CommandTerm``, which reaches ``pxr`` (F9).
The env cfg names it through ``HandPosCommandCfg.class_type``.
"""

from LearningRLlab.tasks.locomotion.g1.mdp import *  # noqa: F403

from .commands_cfg import HandPosCommandCfg
from .curriculums import linear_ramp
from .rewards import hand_pos_tracking_reward
