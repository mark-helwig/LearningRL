"""Validation and mesh-inspection tools.

`robot_schema.py` and `mesh_ops.py` have no Isaac Lab dependency (they run
anywhere Python 3.12 + numpy + pyyaml are available, including a machine
without Isaac Lab). `validate.py` imports Isaac Lab (for Level D/E) and
MuJoCo (opportunistically, for Level B) lazily inside functions, so Level A
also runs anywhere; Levels D and E only run inside the real Isaac Lab
environment on Linux.
"""
