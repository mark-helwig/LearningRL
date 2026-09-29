"""Per-robot Isaac Lab configs. Each subpackage owns its `robot.py` (an
`ArticulationCfg` + actuator groups, spawned directly from URDF via
`urdf_common.RobotUrdfFileCfg`) and `robot.yaml` (names the source URDF
and what `robot_description.tools.validate` should expect from it).
"""
