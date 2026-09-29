"""Validate one or more robots, Levels A, B, D, E.

    python -m robot_description.tools.validate [ROBOT ...] [--all] [--sim] \\
        [--strict] [--json]

One line per check (`PASS`/`FAIL`/`SKIP` + detail), then a per-robot summary.
Exit 1 if ANY check FAILs across ANY selected robot -- no warn-and-continue.
`--strict` promotes every SKIP to FAIL (use this in CI once Level B's `mujoco`
dependency exists). `--json` emits machine-readable output for
CI scripts.

Every check here exists to protect the model's mass, inertia and joint
numbers. Every failure message names the offending
link/joint/value directly; "something is wrong, check the logs" is not
acceptable output from this file.

Levels, and what each needs:

  A. Source XML + numpy. No simulator. Seconds. Needs `vendor/<vendor>/` only.
  B. Self-collision at nominal pose, via MuJoCo, on the SOURCE URDF (both
     Unitree URDFs carry a `<mujoco>` compiler block, so this loads directly).
     SKIP (-> FAIL under --strict) if `mujoco` is not importable -- this repo
     does NOT depend on mujoco (Isaac Lab + Newton only); it is already
     present as a MuJoCo-Warp dependency, so this is opportunistic.
  D. `robot.py` wiring, pure Python + Isaac Lab (for `ArticulationCfg` objects,
     not simulation). Cheapest per-check value in the whole file.
  E. `--sim`, opt-in. Newton, 1 env, headless. Never in the default path.

No Level C: robots spawn directly from URDF via `UrdfFileCfg`, no separate
offline USD build to check against. Level E is the first point anything
actually spawns a `UsdPhysics` articulation from the converted asset.
"""

from __future__ import annotations

import argparse
import importlib
import json as json_module
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from robot_description import paths
from robot_description.tools import mesh_ops
from robot_description.tools.robot_schema import (
    RobotConfig,
    RobotYamlError,
    discover_robot_yamls,
    load_robot_yaml,
)

Status = str  # "PASS" | "FAIL" | "SKIP"


@dataclass
class CheckResult:
    level: str  # "A".."E"
    check_id: str  # e.g. "A4"
    name: str
    status: Status
    detail: str = ""

    def as_dict(self) -> dict:
        return {
            "level": self.level,
            "id": self.check_id,
            "name": self.name,
            "status": self.status,
            "detail": self.detail,
        }

    def line(self) -> str:
        return f"[{self.check_id}] {self.status:4s} {self.name} -- {self.detail}".rstrip(" -")


# ---------------------------------------------------------------------------
# URDF parsing (shared by Level A and Level B/C cross-checks). Pure
# xml.etree + numpy -- same dependency footprint as mesh_ops.py.
# ---------------------------------------------------------------------------
@dataclass
class UrdfJoint:
    name: str
    type: str  # "revolute" | "continuous" | "prismatic" | "fixed" | ...
    lower: float | None
    upper: float | None
    effort: float | None
    velocity: float | None


@dataclass
class UrdfLink:
    name: str
    mass: float | None
    inertia: np.ndarray | None  # 3x3 symmetric, or None if no <inertial>


@dataclass
class ParsedUrdf:
    joints: list[UrdfJoint]
    links: list[UrdfLink]

    @property
    def actuated_joint_names(self) -> list[str]:
        return [j.name for j in self.joints if j.type in ("revolute", "continuous", "prismatic")]

    @property
    def total_mass(self) -> float:
        return sum(link.mass for link in self.links if link.mass is not None)


def parse_urdf(path: Path) -> ParsedUrdf:
    import xml.etree.ElementTree as ET

    tree = ET.parse(path)
    root = tree.getroot()

    joints: list[UrdfJoint] = []
    for joint_el in root.findall("joint"):
        name = joint_el.get("name", "<unnamed>")
        jtype = joint_el.get("type", "fixed")
        limit_el = joint_el.find("limit")
        lower = upper = effort = velocity = None
        if limit_el is not None:
            lower = _float_or_none(limit_el.get("lower"))
            upper = _float_or_none(limit_el.get("upper"))
            effort = _float_or_none(limit_el.get("effort"))
            velocity = _float_or_none(limit_el.get("velocity"))
        joints.append(UrdfJoint(name, jtype, lower, upper, effort, velocity))

    links: list[UrdfLink] = []
    for link_el in root.findall("link"):
        name = link_el.get("name", "<unnamed>")
        inertial_el = link_el.find("inertial")
        mass = None
        inertia = None
        if inertial_el is not None:
            mass_el = inertial_el.find("mass")
            if mass_el is not None:
                mass = _float_or_none(mass_el.get("value"))
            inertia_el = inertial_el.find("inertia")
            if inertia_el is not None:
                ixx = float(inertia_el.get("ixx", "0"))
                ixy = float(inertia_el.get("ixy", "0"))
                ixz = float(inertia_el.get("ixz", "0"))
                iyy = float(inertia_el.get("iyy", "0"))
                iyz = float(inertia_el.get("iyz", "0"))
                izz = float(inertia_el.get("izz", "0"))
                # Eigenvalues of a real symmetric tensor are invariant to which
                # orthonormal frame it's expressed in (rotating frames rotates
                # the eigenVECTORS, not the eigenVALUES) -- so we do not need
                # to account for <inertial><origin rpy="..."/> here at all.
                inertia = np.array(
                    [[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]], dtype=np.float64
                )
        links.append(UrdfLink(name, mass, inertia))

    return ParsedUrdf(joints=joints, links=links)


def _float_or_none(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Shared eigenvalue / triangle-inequality check (A4 and C6)
# ---------------------------------------------------------------------------
def _eigen_and_triangle(label: str, inertia_diag_or_matrix: np.ndarray) -> tuple[bool, bool, str]:
    """Returns `(positive_definite, triangle_ok, detail)`.

    `triangle_ok` is the physical-realizability check: for principal moments
    sorted ascending (l1 <= l2 <= l3), a valid rigid-body inertia tensor
    satisfies l1 + l2 >= l3 (the largest moment cannot exceed the sum of the
    other two). A tensor can pass the positive-eigenvalue check and still be
    physically unrealizable -- these are reported separately, per the spec.
    """
    if inertia_diag_or_matrix.ndim == 1:
        eigvals = np.sort(inertia_diag_or_matrix)
    else:
        eigvals = np.linalg.eigvalsh(inertia_diag_or_matrix)
        eigvals.sort()

    l1, l2, l3 = eigvals
    positive_definite = bool(l1 > 0.0)
    triangle_ok = bool(l1 + l2 >= l3)

    detail = f"{label}: eigenvalues=[{l1:.6g}, {l2:.6g}, {l3:.6g}]"
    if not positive_definite:
        detail += f" -- NOT POSITIVE DEFINITE (min eigenvalue {l1:.6g} <= 0)"
    if not triangle_ok:
        detail += f" -- TRIANGLE INEQUALITY VIOLATED (l1+l2={l1 + l2:.6g} < l3={l3:.6g})"
    return positive_definite, triangle_ok, detail


# ---------------------------------------------------------------------------
# Level A -- source XML + numpy
# ---------------------------------------------------------------------------
def run_level_a(source_path: Path, cfg: RobotConfig) -> list[CheckResult]:
    results: list[CheckResult] = []
    parsed = parse_urdf(source_path)

    # A1 -- mesh refs
    refs = mesh_ops.iter_mesh_refs(source_path)
    audit = mesh_ops.check_mesh_refs(refs)
    if audit.ok:
        results.append(
            CheckResult("A", "A1", "mesh references", "PASS", f"{len(refs)} refs, all resolve")
        )
    else:
        lines = []
        for ref in audit.missing:
            lines.append(f"MISSING {ref.raw_filename} (link {ref.link_name}, {ref.kind})")
        for ref, actual in audit.case_mismatches:
            lines.append(
                f"CASE MISMATCH {ref.raw_filename} (link {ref.link_name}, {ref.kind}) "
                f"-- actual file is {actual.name}"
            )
        results.append(CheckResult("A", "A1", "mesh references", "FAIL", "; ".join(lines)))

    # A2 -- total mass
    total_mass = parsed.total_mass
    expected = cfg.validate.total_mass.value
    rel_tol = cfg.validate.total_mass.rel_tol
    if expected > 0 and abs(total_mass - expected) <= rel_tol * expected:
        results.append(
            CheckResult(
                "A", "A2", "total mass", "PASS", f"{total_mass:.4g} kg (expected {expected:.4g} kg)"
            )
        )
    else:
        results.append(
            CheckResult(
                "A",
                "A2",
                "total mass",
                "FAIL",
                f"URDF total mass {total_mass:.4g} kg vs expected {expected:.4g} kg "
                f"(rel_tol {rel_tol}) -- a vendor URDF swap or variant change?",
            )
        )

    # A3 -- massless links
    allowlist = set(cfg.validate.massless_links)
    min_mass = cfg.validate.min_link_mass
    offenders = [
        link.name
        for link in parsed.links
        if (link.mass is None or link.mass < min_mass) and link.name not in allowlist
    ]
    if offenders:
        results.append(
            CheckResult(
                "A",
                "A3",
                "no un-allowlisted near-zero-mass links",
                "FAIL",
                f"link(s) with mass < {min_mass} not in validate.massless_links: {offenders} "
                f"-- an ill-conditioned mass matrix, unless deliberately allowlisted",
            )
        )
    else:
        results.append(CheckResult("A", "A3", "no un-allowlisted near-zero-mass links", "PASS", ""))

    # A4 -- inertia eigenvalues + triangle inequality
    bad_pd = []
    bad_triangle = []
    for link in parsed.links:
        if link.inertia is None:
            continue
        pd, tri, detail = _eigen_and_triangle(link.name, link.inertia)
        if not pd:
            bad_pd.append(detail)
        if not tri:
            bad_triangle.append(detail)
    if bad_pd:
        results.append(
            CheckResult("A", "A4", "inertia positive-definite (URDF)", "FAIL", "; ".join(bad_pd))
        )
    else:
        results.append(CheckResult("A", "A4", "inertia positive-definite (URDF)", "PASS", ""))
    if bad_triangle:
        results.append(
            CheckResult(
                "A",
                "A4",
                "inertia triangle inequality (URDF)",
                "FAIL",
                "; ".join(bad_triangle),
            )
        )
    else:
        results.append(CheckResult("A", "A4", "inertia triangle inequality (URDF)", "PASS", ""))

    # A5 -- joint limits
    bad_limits = []
    for joint in parsed.joints:
        if joint.type not in ("revolute", "prismatic"):
            continue  # "continuous" joints have no <limit> lower/upper by definition
        problems = []
        if joint.lower is None or joint.upper is None:
            problems.append("missing lower/upper")
        elif not (math.isfinite(joint.lower) and math.isfinite(joint.upper)):
            problems.append("non-finite lower/upper")
        elif not (joint.lower < joint.upper):
            problems.append(f"lower({joint.lower}) >= upper({joint.upper})")
        if joint.effort is None or not (joint.effort > 0):
            problems.append(f"effort={joint.effort} (must be > 0)")
        if joint.velocity is None or not (joint.velocity > 0):
            problems.append(f"velocity={joint.velocity} (must be > 0)")
        if problems:
            bad_limits.append(f"{joint.name}: {', '.join(problems)}")
    if bad_limits:
        results.append(
            CheckResult("A", "A5", "joint limits well-formed", "FAIL", "; ".join(bad_limits))
        )
    else:
        results.append(CheckResult("A", "A5", "joint limits well-formed", "PASS", ""))

    # A6 -- nominal-pose margin
    margin = cfg.validate.joint_limit_margin
    joints_by_name = {j.name: j for j in parsed.joints}
    zero_match_patterns = []
    margin_violations = []
    for pattern, nominal_value in cfg.validate.nominal_joint_pos.items():
        matched = [name for name in joints_by_name if re.fullmatch(pattern, name)]
        if not matched:
            zero_match_patterns.append(pattern)
            continue
        for name in matched:
            joint = joints_by_name[name]
            if joint.lower is None or joint.upper is None:
                margin_violations.append(f"{name}: no limits to check margin against")
                continue
            if not (joint.lower + margin <= nominal_value <= joint.upper - margin):
                lo, hi = joint.lower + margin, joint.upper - margin
                margin_violations.append(
                    f"{name}: nominal {nominal_value} not within [{lo:.4g}, {hi:.4g}] "
                    f"(limits [{joint.lower}, {joint.upper}], margin {margin})"
                )
    if zero_match_patterns or margin_violations:
        detail_parts = []
        if zero_match_patterns:
            detail_parts.append(f"pattern(s) matched zero joints: {zero_match_patterns}")
        if margin_violations:
            detail_parts.append("; ".join(margin_violations))
        results.append(
            CheckResult(
                "A", "A6", "nominal pose within limit margin", "FAIL", "; ".join(detail_parts)
            )
        )
    else:
        results.append(CheckResult("A", "A6", "nominal pose within limit margin", "PASS", ""))

    # A7 -- DoF count
    n_actuated = len(parsed.actuated_joint_names)
    if n_actuated == cfg.validate.expected_dof:
        results.append(CheckResult("A", "A7", "DoF count", "PASS", f"{n_actuated} actuated joints"))
    else:
        results.append(
            CheckResult(
                "A",
                "A7",
                "DoF count",
                "FAIL",
                f"URDF has {n_actuated} actuated (revolute/continuous/prismatic) joints, "
                f"expected {cfg.validate.expected_dof} -- silent variant swap?",
            )
        )

    return results


# ---------------------------------------------------------------------------
# Level B -- self-collision at nominal pose, via MuJoCo, on the source URDF
# ---------------------------------------------------------------------------
def run_level_b(source_path: Path, cfg: RobotConfig) -> list[CheckResult]:
    try:
        import mujoco
    except ImportError:
        return [
            CheckResult(
                "B",
                "B1",
                "self-collision at nominal pose (MuJoCo)",
                "SKIP",
                "mujoco is not importable in this environment -- NOT a dependency of "
                "this repo (Isaac Lab + Newton only); it is expected to already be "
                "present as a MuJoCo-Warp dependency. Do not add "
                "it to pyproject.toml to force this check.",
            )
        ]

    try:
        model = mujoco.MjModel.from_xml_path(str(source_path))
    except Exception as exc:  # noqa: BLE001 - report any loader failure as a check failure
        return [
            CheckResult(
                "B",
                "B1",
                "self-collision at nominal pose (MuJoCo)",
                "FAIL",
                f"MjModel.from_xml_path({source_path}) raised: {exc}",
            )
        ]

    data = mujoco.MjData(model)

    unmatched_patterns = []
    for pattern, value in cfg.validate.nominal_joint_pos.items():
        matched_any = False
        for joint_id in range(model.njnt):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            if name is not None and re.fullmatch(pattern, name):
                qpos_adr = model.jnt_qposadr[joint_id]
                data.qpos[qpos_adr] = value
                matched_any = True
        if not matched_any:
            unmatched_patterns.append(pattern)

    mujoco.mj_forward(model, data)

    contacts = []
    for i in range(data.ncon):
        contact = data.contact[i]
        geom1_body = model.body(model.geom_bodyid[contact.geom1]).name
        geom2_body = model.body(model.geom_bodyid[contact.geom2]).name
        # MuJoCo's compiler excludes parent-child body pairs from contact
        # generation by default (the "filterparent" behavior) -- so anything
        # that shows up in data.contact here is a genuine, non-adjacent
        # self-collision at this pose, not a kinematic-tree artifact.
        depth = -float(contact.dist)  # negative dist = penetration
        contacts.append((depth, geom1_body, geom2_body))

    results = []
    if unmatched_patterns:
        results.append(
            CheckResult(
                "B",
                "B0",
                "nominal_joint_pos patterns resolve (MuJoCo model)",
                "FAIL",
                f"pattern(s) matched zero MuJoCo joints: {unmatched_patterns}",
            )
        )

    if contacts:
        contacts.sort(key=lambda c: c[0], reverse=True)  # deepest first
        allow = {frozenset(pair) for pair in cfg.validate.allow_self_collision_pairs}
        genuine = [c for c in contacts if frozenset((c[1], c[2])) not in allow]
        if genuine:
            detail = "; ".join(f"{a} <-> {b} (penetration {d:.4g} m)" for d, a, b in genuine)
            results.append(
                CheckResult("B", "B1", "self-collision at nominal pose (MuJoCo)", "FAIL", detail)
            )
        else:
            results.append(
                CheckResult(
                    "B",
                    "B1",
                    "self-collision at nominal pose (MuJoCo)",
                    "PASS",
                    f"{len(contacts)} contact(s), all in allow_self_collision_pairs",
                )
            )
    else:
        results.append(
            CheckResult("B", "B1", "self-collision at nominal pose (MuJoCo)", "PASS", "0 contacts")
        )

    return results


# ---------------------------------------------------------------------------
# Level D -- robot.py wiring, pure Python + Isaac Lab (no simulation)
# ---------------------------------------------------------------------------
def run_level_d(robot: str, cfg: RobotConfig, real_joint_names: list[str]) -> list[CheckResult]:
    try:
        robot_module = importlib.import_module(f"robot_description.robots.{robot}.robot")
    except ImportError as exc:
        return [
            CheckResult(
                "D",
                "D0",
                f"import robot_description.robots.{robot}.robot",
                "FAIL",
                f"{exc} -- Level D requires Isaac Lab to be importable",
            )
        ]

    articulation_cfg = getattr(robot_module, f"{_upper_prefix(robot)}_CFG", None)
    if articulation_cfg is None:
        return [
            CheckResult(
                "D",
                "D0",
                "locate <ROBOT>_CFG in robot.py",
                "FAIL",
                f"no *_CFG export found in {robot_module}",
            )
        ]

    results: list[CheckResult] = []
    real_set = set(real_joint_names)

    # D1 -- actuator partition: zero-claimed or double-claimed joints
    claims: dict[str, list[str]] = {name: [] for name in real_joint_names}
    for group_name, actuator in articulation_cfg.actuators.items():
        patterns = actuator.joint_names_expr
        for name in real_joint_names:
            if any(re.fullmatch(pattern, name) for pattern in patterns):
                claims[name].append(group_name)
    unclaimed = [name for name, groups in claims.items() if not groups]
    double_claimed = {name: groups for name, groups in claims.items() if len(groups) > 1}
    if unclaimed or double_claimed:
        results.append(
            CheckResult(
                "D",
                "D1",
                "actuator groups partition all real joints",
                "FAIL",
                f"unclaimed (silent limp limb): {unclaimed}; double-claimed "
                f"(Isaac Lab requires disjoint groups): {double_claimed}",
            )
        )
    else:
        results.append(
            CheckResult("D", "D1", "actuator groups partition all real joints", "PASS", "")
        )

    # D2 -- every regex key in stiffness/damping/armature/effort/velocity limit
    # dicts matches >=1 joint WITHIN THAT ACTUATOR's own resolved joint set
    # (a key matching only joints in a DIFFERENT actuator is exactly as silent
    # a typo as matching nothing at all).
    dead_keys = []
    for group_name, actuator in articulation_cfg.actuators.items():
        own_joints = [
            name
            for name in real_joint_names
            if any(re.fullmatch(p, name) for p in actuator.joint_names_expr)
        ]
        for field_name in ("stiffness", "damping", "armature", "effort_limit", "velocity_limit"):
            value = getattr(actuator, field_name, None)
            if not isinstance(value, dict):
                continue
            for pattern in value:
                if not any(re.fullmatch(pattern, name) for name in own_joints):
                    dead_keys.append(f"{group_name}.{field_name}['{pattern}']")
    if dead_keys:
        results.append(
            CheckResult(
                "D",
                "D2",
                "no dead regex keys in actuator dicts",
                "FAIL",
                f"key(s) matching zero joints in their own actuator group: {dead_keys}",
            )
        )
    else:
        results.append(CheckResult("D", "D2", "no dead regex keys in actuator dicts", "PASS", ""))

    # D3 -- no dead keys in init_state.joint_pos. Does NOT require every real
    # joint to be covered: robot.py's init pose may address only some joints
    # by regex, relying on Isaac Lab's zero-default for the rest (matching
    # upstream unitree_rl_lab's own convention) -- a dead pattern (typo'd
    # joint name) is still exactly as silent a bug as ever, so that half of
    # the old check stays.
    joint_pos = articulation_cfg.init_state.joint_pos
    dead_pos_keys = [
        pattern
        for pattern in joint_pos
        if not any(re.fullmatch(pattern, name) for name in real_joint_names)
    ]
    if dead_pos_keys:
        results.append(
            CheckResult(
                "D",
                "D3",
                "no dead keys in init_state.joint_pos",
                "FAIL",
                f"dead pattern(s), matching zero real joints: {dead_pos_keys}",
            )
        )
    else:
        results.append(CheckResult("D", "D3", "no dead keys in init_state.joint_pos", "PASS", ""))

    # D4 -- robot.py's init_state.joint_pos agrees with robot.yaml's nominal_joint_pos
    mismatches = []
    for pattern, expected_value in cfg.validate.nominal_joint_pos.items():
        actual_value = joint_pos.get(pattern)
        if actual_value is None:
            mismatches.append(
                f"'{pattern}' present in robot.yaml but not in robot.py init_state.joint_pos"
            )
        elif abs(actual_value - expected_value) > 1e-9:
            mismatches.append(
                f"'{pattern}': robot.py={actual_value} vs robot.yaml={expected_value}"
            )
    if mismatches:
        results.append(
            CheckResult(
                "D",
                "D4",
                "robot.py nominal pose agrees with robot.yaml",
                "FAIL",
                "; ".join(mismatches) + " -- Level B tests the pose robot.yaml describes, "
                "not the one the robot actually spawns at, unless these agree",
            )
        )
    else:
        results.append(
            CheckResult("D", "D4", "robot.py nominal pose agrees with robot.yaml", "PASS", "")
        )

    # D5 -- <ROBOT>_JOINT_NAMES is a permutation of the real joint set
    pinned_names = getattr(robot_module, f"{_upper_prefix(robot)}_JOINT_NAMES", None)
    if pinned_names is None:
        results.append(
            CheckResult(
                "D",
                "D5",
                "pinned JOINT_NAMES is a permutation of real joints",
                "SKIP",
                "no *_JOINT_NAMES export found",
            )
        )
    else:
        pinned_set = set(pinned_names)
        missing = real_set - pinned_set
        extra = pinned_set - real_set
        duplicated = len(pinned_names) != len(pinned_set)
        if missing or extra or duplicated:
            results.append(
                CheckResult(
                    "D",
                    "D5",
                    "pinned JOINT_NAMES is a permutation of real joints",
                    "FAIL",
                    f"missing: {sorted(missing)}; extra: {sorted(extra)}; duplicated: {duplicated}",
                )
            )
        else:
            results.append(
                CheckResult(
                    "D", "D5", "pinned JOINT_NAMES is a permutation of real joints", "PASS", ""
                )
            )

    return results


def _upper_prefix(robot: str) -> str:
    """`unitree_g1` -> `G1`, matching this repo's `G1_CFG`/`G1_JOINT_NAMES`
    naming convention (the part after the manufacturer prefix, upper-cased)."""
    parts = robot.split("_")
    return parts[-1].upper() if len(parts) > 1 else robot.upper()


# ---------------------------------------------------------------------------
# Level E -- --sim, opt-in. Newton, 1 env, headless.
# ---------------------------------------------------------------------------
def run_level_e(robot: str, cfg: RobotConfig, real_joint_names: list[str]) -> list[CheckResult]:
    """NOT exercised on any machine this repo was authored on (Windows, no
    GPU) -- best-effort against the documented Isaac Lab
    `Articulation`/`SimulationContext` API, to be smoke-tested on the Linux
    box. Deliberately simple: 1 env, no scene manager, no task -- just enough
    to prove the robot spawns and Isaac Lab's joint bookkeeping agrees with
    this repo's.
    """
    try:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import Articulation

        robot_module = importlib.import_module(f"robot_description.robots.{robot}.robot")
        articulation_cfg = getattr(robot_module, f"{_upper_prefix(robot)}_CFG")
    except ImportError as exc:
        return [CheckResult("E", "E0", "Level E imports", "FAIL", str(exc))]

    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 200.0)
    sim = sim_utils.SimulationContext(sim_cfg)
    try:
        robot_asset = Articulation(articulation_cfg.replace(prim_path="/World/Robot"))
        sim.reset()

        results: list[CheckResult] = []

        n_joints = robot_asset.num_joints
        if n_joints == cfg.validate.expected_dof:
            results.append(
                CheckResult("E", "E1", "num_joints == expected_dof", "PASS", str(n_joints))
            )
        else:
            results.append(
                CheckResult(
                    "E",
                    "E1",
                    "num_joints == expected_dof",
                    "FAIL",
                    f"{n_joints} != {cfg.validate.expected_dof}",
                )
            )

        # Never trust a hardcoded index into joint-ordered arrays.
        # Compare by NAME SET, not position, so
        # this check stays valid even if joint_ordering behaves unexpectedly
        # under Newton/MJWarp.
        sim_joint_names = set(robot_asset.joint_names)
        if sim_joint_names == set(real_joint_names):
            results.append(CheckResult("E", "E2", "joint_names matches real joint set", "PASS", ""))
        else:
            results.append(
                CheckResult(
                    "E",
                    "E2",
                    "joint_names matches real joint set",
                    "FAIL",
                    f"symmetric difference: {sorted(sim_joint_names ^ set(real_joint_names))}",
                )
            )

        default_pos = robot_asset.data.default_joint_pos
        soft_limits = robot_asset.data.soft_joint_pos_limits
        # F5: under Newton/MJWarp mid-refactor, these may be warp arrays, not
        # torch tensors -- convert defensively rather than assuming .cpu()/.numpy().
        default_pos_np = _to_numpy(default_pos)
        soft_limits_np = _to_numpy(soft_limits)
        within = bool(
            np.all(default_pos_np >= soft_limits_np[..., 0])
            and np.all(default_pos_np <= soft_limits_np[..., 1])
        )
        results.append(
            CheckResult(
                "E",
                "E3",
                "default_joint_pos within soft_joint_pos_limits",
                "PASS" if within else "FAIL",
                ""
                if within
                else "at least one joint's default position is outside its soft limits",
            )
        )
        return results
    finally:
        sim.clear_instance()


def _to_numpy(array_like) -> np.ndarray:
    """Best-effort conversion for F5 (Newton may return warp arrays, not
    torch tensors, mid-refactor). Tries numpy() directly, then a warp/torch
    round-trip; never assumes one specific array library."""
    if hasattr(array_like, "numpy"):
        return np.asarray(array_like.numpy())
    if hasattr(array_like, "cpu"):
        return np.asarray(array_like.cpu())
    return np.asarray(array_like)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m robot_description.tools.validate",
        description="Validate robots/*/robot.yaml-described robots, Levels A, B, D, E.",
    )
    parser.add_argument("robots", nargs="*", metavar="ROBOT")
    parser.add_argument(
        "--all", action="store_true", help="validate every robot with enabled: true"
    )
    parser.add_argument(
        "--sim", action="store_true", help="also run Level E (requires a running Newton sim)"
    )
    parser.add_argument("--strict", action="store_true", help="promote every SKIP to FAIL")
    parser.add_argument(
        "--json", action="store_true", help="emit machine-readable JSON instead of text"
    )
    return parser


def _validate_one(robot: str, cfg: RobotConfig, args: argparse.Namespace) -> list[CheckResult]:
    all_results: list[CheckResult] = []

    vendor_dir = paths.VENDOR_DIR / cfg.source.vendor
    source_path = vendor_dir / cfg.source.path
    if not source_path.is_file():
        all_results.append(
            CheckResult(
                "A",
                "A0",
                "source file present",
                "FAIL",
                f"{source_path} not found -- run `gitman install --skip-changes`",
            )
        )
        return all_results  # nothing else can run without the source URDF

    parsed_urdf = parse_urdf(source_path)

    all_results.extend(run_level_a(source_path, cfg))
    all_results.extend(run_level_b(source_path, cfg))
    all_results.extend(run_level_d(robot, cfg, parsed_urdf.actuated_joint_names))
    if args.sim:
        all_results.extend(run_level_e(robot, cfg, parsed_urdf.actuated_joint_names))
    else:
        all_results.append(CheckResult("E", "E*", "simulation checks", "SKIP", "pass --sim to run"))

    return all_results


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    robots_dir = paths.PACKAGE_ROOT / "robots"
    robot_yaml_paths = discover_robot_yamls(robots_dir)
    configs: dict[str, RobotConfig] = {}
    for path in robot_yaml_paths:
        try:
            configs[path.parent.name] = load_robot_yaml(path)
        except RobotYamlError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1

    if args.robots:
        selected = list(args.robots)
        unknown = [name for name in selected if name not in configs]
        if unknown:
            print(f"ERROR: unknown robot(s) {unknown}. Known: {sorted(configs)}", file=sys.stderr)
            return 1
    elif args.all:
        selected = [name for name, cfg in configs.items() if cfg.enabled]
    else:
        parser.error("specify ROBOT [ROBOT ...] or --all")
        return 2

    overall_ok = True
    json_report: dict[str, list[dict]] = {}

    for robot in selected:
        results = _validate_one(robot, configs[robot], args)
        json_report[robot] = [r.as_dict() for r in results]

        effective_statuses = [
            ("FAIL" if (args.strict and r.status == "SKIP") else r.status) for r in results
        ]
        robot_ok = all(status != "FAIL" for status in effective_statuses)
        overall_ok = overall_ok and robot_ok

        if not args.json:
            print(f"=== {robot} ===")
            for result, status in zip(results, effective_statuses, strict=True):
                if status == result.status:
                    print(result.line())
                else:
                    print(
                        f"[{result.check_id}] {status:4s} {result.name} -- "
                        f"{result.detail} (was SKIP, promoted by --strict)"
                    )
            print(f"--- {robot}: {'PASS' if robot_ok else 'FAIL'} ---\n")

    if args.json:
        print(json_module.dumps(json_report, indent=2))

    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
