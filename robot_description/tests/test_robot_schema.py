"""Every `robots/*/robot.yaml` parses under the strict schema, and the
schema itself rejects the mistakes it exists to catch.

No Isaac Lab, no vendor/, no build/ needed -- `robot_schema.py` is pure
Python + pyyaml, so this runs on Windows with nothing else installed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from robot_description.tools.robot_schema import (
    RobotConfig,
    RobotYamlError,
    discover_robot_yamls,
    load_robot_yaml,
)

REPO_ROOT = Path(__file__).parent.parent
ROBOTS_DIR = REPO_ROOT / "robot_description" / "robots"


# robot -> the gitman source name its robot.yaml must point at.
ROBOT_VENDORS = {
    "roboparty_rpo": "rpo_description",
    "unitree_g1": "unitree_ros",
}


@pytest.mark.static
def test_discovers_every_robot() -> None:
    found = discover_robot_yamls(ROBOTS_DIR)
    names = sorted(p.parent.name for p in found)
    assert names == sorted(ROBOT_VENDORS)


@pytest.mark.static
@pytest.mark.parametrize("robot", sorted(ROBOT_VENDORS))
def test_robot_yaml_parses(robot: str) -> None:
    path = ROBOTS_DIR / robot / "robot.yaml"
    cfg = load_robot_yaml(path)
    assert cfg.schema_version == 1
    assert cfg.robot == robot
    assert cfg.source.vendor == ROBOT_VENDORS[robot]
    assert not Path(cfg.source.path).is_absolute()
    assert cfg.validate.expected_dof > 0
    assert cfg.validate.total_mass.value > 0.0


@pytest.mark.static
def test_g1_robot_yaml_values() -> None:
    """Pin the numbers the spec calls out explicitly, so a future edit to
    robot.yaml that silently changes them fails a test instead of only
    showing up much later as a wrong mass/DoF at spawn time."""
    cfg = load_robot_yaml(ROBOTS_DIR / "unitree_g1" / "robot.yaml")
    assert cfg.enabled is True
    assert cfg.validate.expected_dof == 29
    assert cfg.validate.total_mass.value == pytest.approx(33.341)
    assert cfg.validate.total_mass.rel_tol == pytest.approx(0.01)
    assert cfg.validate.mesh_budget_mb == pytest.approx(200.0)
    assert set(cfg.validate.nominal_joint_pos) == {
        ".*_hip_pitch_joint",
        ".*_knee_joint",
        ".*_ankle_pitch_joint",
    }


@pytest.mark.static
def test_unknown_top_level_key_is_rejected() -> None:
    data = {
        "schema_version": 1,
        "robot": "unitree_g1",
        "source": {"vendor": "unitree_ros", "path": "robots/g1_description/g1_29dof_rev_1_0.urdf"},
        "validate": {"expected_dof": 29, "total_mass": {"value": 33.341}},
        "not_a_real_key": True,
    }
    with pytest.raises(RobotYamlError, match="unknown key"):
        RobotConfig.from_dict(data, "test")


@pytest.mark.static
def test_unknown_nested_key_is_rejected() -> None:
    data = {
        "schema_version": 1,
        "robot": "unitree_g1",
        "source": {
            "vendor": "unitree_ros",
            "path": "robots/g1_description/g1_29dof_rev_1_0.urdf",
            "typo_key": 1,
        },
        "validate": {"expected_dof": 29, "total_mass": {"value": 33.341}},
    }
    with pytest.raises(RobotYamlError, match="unknown key"):
        RobotConfig.from_dict(data, "test")


@pytest.mark.static
def test_wrong_schema_version_is_rejected() -> None:
    data = {
        "schema_version": 2,
        "robot": "unitree_g1",
        "source": {"vendor": "unitree_ros", "path": "x.urdf"},
        "validate": {"expected_dof": 29, "total_mass": {"value": 33.341}},
    }
    with pytest.raises(RobotYamlError, match="schema_version"):
        RobotConfig.from_dict(data, "test")


@pytest.mark.static
def test_missing_required_key_is_rejected() -> None:
    data = {
        "schema_version": 1,
        "robot": "unitree_g1",
        "source": {"vendor": "unitree_ros"},  # missing 'path'
        "validate": {"expected_dof": 29, "total_mass": {"value": 33.341}},
    }
    with pytest.raises(RobotYamlError, match="path"):
        RobotConfig.from_dict(data, "test")


@pytest.mark.static
@pytest.mark.parametrize(
    "bad_path",
    [
        "/abs/path.urdf",  # POSIX absolute
        "C:/vendor/path.urdf",  # Windows absolute, forward slashes
        r"C:\vendor\path.urdf",  # Windows absolute, backslashes
        r"vendor\path.urdf",  # relative but backslash-separated
    ],
)
def test_absolute_source_path_is_rejected(bad_path: str) -> None:
    """source.path must be relative AND forward-slash.

    The Windows shapes matter because authoring happens on Windows:
    `PurePosixPath("C:/x").is_absolute()` is False, so a POSIX-only check lets a
    Windows absolute path through, and on the Linux target it silently degrades to a
    nonsensical relative name (file-not-found) rather than a clear schema error.
    """
    data = {
        "schema_version": 1,
        "robot": "unitree_g1",
        "source": {"vendor": "unitree_ros", "path": bad_path},
        "validate": {"expected_dof": 29, "total_mass": {"value": 33.341}},
    }
    with pytest.raises(RobotYamlError, match="absolute|Windows-style"):
        RobotConfig.from_dict(data, "test")


@pytest.mark.static
def test_robot_name_must_match_parent_directory(tmp_path: Path) -> None:
    robot_dir = tmp_path / "unitree_g1"
    robot_dir.mkdir()
    bad_yaml = robot_dir / "robot.yaml"
    bad_yaml.write_text(
        "schema_version: 1\n"
        "robot: some_other_name\n"
        "source:\n  vendor: unitree_ros\n  path: x.urdf\n"
        "validate:\n  expected_dof: 29\n  total_mass:\n    value: 33.341\n",
        encoding="utf-8",
    )
    with pytest.raises(RobotYamlError, match="does not match parent directory"):
        load_robot_yaml(bad_yaml)
