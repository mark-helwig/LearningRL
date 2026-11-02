"""MuJoCo model loading, the joint remap, and base-state accessors.

Everything that knows about MuJoCo indexing lives here. The rest of the
package asks for `sim.joint_pos` and gets an array in `config.yaml`'s
`joint_order`, so no observation or action term ever indexes `qpos` itself.

Two things this module exists to get right:

**The remap is by name, always.** `joint_order` is Isaac Lab's articulation
ordering. For the G1 it happens to equal the MJCF's hinge-joint order
byte-for-byte -- verified -- but that is luck, not contract, and a wrong
remap produces a robot that stands up and falls over plausibly rather than
raising.

**Free-joint velocity frames are mixed.** `qvel[0:3]` is linear velocity in
WORLD coordinates while `qvel[3:6]` is angular velocity in the BODY frame.
So `base_ang_vel` is `qvel[3:6]` verbatim and `base_lin_vel` needs `R.T @
qvel[0:3]`. Getting this backwards is the classic silent sim2sim bug; there
is a test pinning `qvel[3:6]` against the MJCF's own pelvis gyro.

**The model is built through `MjSpec` unconditionally**, whether or not the
run asks for scene objects. A `from_xml_path`-when-empty split would mean
every baseline bag came from a code path the object runs never exercise, so
any compile difference between the two would read as a policy difference.
`test_mjspec_load_is_the_same_plant` pins the equivalence instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from LearningRLsim.objects import ObjectSpec
from LearningRLsim.runconfig import RunConfig


class ModelError(RuntimeError):
    """The MJCF could not be resolved, loaded, or reconciled with the config."""


# Gravity DIRECTION, not the acceleration: Isaac Lab's `projected_gravity`
# observation is the unit gravity vector rotated into the root frame.
_GRAVITY_DIR_W = np.array([0.0, 0.0, -1.0])

# MuJoCo's enums compared as PLAIN INTS, because the model's type arrays are
# numpy scalars and the two do not compare reliably across the `mujoco>=3.8,<4`
# range this package declares. On 3.14 `np.int32(3) == mjtJoint.mjJNT_HINGE` is
# True but `np.int32(3) in (mjtJoint.mjJNT_HINGE,)` is False -- containment
# compares with the enum on the left, whose reflected `__eq__` declines numpy
# scalars, so it silently falls through to identity. On 3.8.1 both are True.
# That difference turned every hinge joint into "not a hinge joint" on a
# standalone install and nothing in-container could have caught it.
_JNT_HINGE = int(mujoco.mjtJoint.mjJNT_HINGE)
_JNT_SLIDE = int(mujoco.mjtJoint.mjJNT_SLIDE)
_JNT_FREE = int(mujoco.mjtJoint.mjJNT_FREE)
_TRN_JOINT = int(mujoco.mjtTrn.mjTRN_JOINT)

_GEOM_TYPES = {
    "box": mujoco.mjtGeom.mjGEOM_BOX,
    "sphere": mujoco.mjtGeom.mjGEOM_SPHERE,
    "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
}
"""`ObjectSpec.shape` -> the MuJoCo primitive. Handed to `MjSpec` rather than
compared against a model array, so the enum is used as an enum here."""


def resolve_mjcf(cfg: RunConfig) -> Path:
    """Locate the MuJoCo XML for `cfg.robot` via `robot_description`.

    `robot_description.paths` is import-safe on a CPU-only host -- the
    package `__init__` is bare and `paths` pulls in only `robot_schema`. It is
    `robot_description.robots.*` that eagerly imports Isaac Lab; nothing here
    touches that subpackage.
    """
    if cfg.robot is None:
        raise ModelError(
            f"{cfg.path}: `robot` is null, so there is nothing to look up in "
            f"robot_description. The exporter derives it from the spawn config's "
            f"usd_dir basename; a null means that lookup failed at export time."
        )

    try:
        from robot_description.paths import mjcf_path
    except ImportError as exc:
        raise ModelError(
            f"robot_description is not importable ({exc}). LearningRLsim does not vendor robot "
            f"models; install it first -- `pip install -e ./robot_description`."
        ) from exc

    path = Path(mjcf_path(cfg.robot))
    # `mjcf_path` deliberately does not stat the result (the vendor tree may
    # not be checked out yet), so the existence check is ours to make and ours
    # to explain.
    if not path.is_file():
        raise ModelError(
            f"{cfg.robot}'s MuJoCo model is declared as {path}, which does not exist. "
            f"Run `gitman install --skip-changes` in robot_description to fetch the "
            f"vendor tree."
        )
    return path


def resolve_urdf(cfg: RunConfig) -> Path | None:
    """The URDF for `cfg.robot`, or None -- viewer geometry only, never physics.

    Deliberately non-raising, unlike `resolve_mjcf`. Nothing in the rollout
    reads this file: it exists so the bag can name the meshes a 3D viewer
    should hang off the link frames, and so `/tf` can carry the links MuJoCo
    merged away. A run that cannot find it should still produce its bag.

    `robot_description.paths.urdf_path` is itself TOTAL -- it returns a guess
    rather than raising on a missing vendor checkout -- so the `is_file` check
    below is what actually decides.
    """
    if cfg.robot is None:
        return None
    try:
        from robot_description.paths import urdf_path
    except ImportError:
        return None
    path = Path(urdf_path(cfg.robot))
    return path if path.is_file() else None


@dataclass
class SimOverrides:
    """Physics parameters the exporter does NOT record, so LearningRLsim must assume.

    Each one is a real sim2sim confound, and every value here is written into
    the bag's `/meta/run` marked as assumed rather than read, so a divergence
    number is never ambiguous about which of these it might be measuring. The
    defaults reproduce the training-side values documented in LearningRLlab's
    `scenes.py` / `G1_CFG`.
    """

    spawn_height: float = 0.8  # G1_CFG.init_state.pos[2]; the fallback for an export without init_state
    friction: float = 1.0  # FLAT_TERRAIN_CFG static == dynamic == 1.0, on the TERRAIN
    gravity: float = -9.81
    spawn_height_source: str = "assumed"
    """"read" when `spawn_height` came from config.yaml's `init_state`."""

    def as_meta(self) -> dict[str, dict[str, object]]:
        return {
            "spawn_height": {"value": self.spawn_height, "source": self.spawn_height_source},
            "friction": {
                "value": self.friction,
                "source": "assumed",
                "note": "applied to every geom EXCEPT an injected scene object's, which keeps its own.",
            },
            "gravity": {"value": self.gravity, "source": "assumed"},
        }


def _add_object(spec, obj: ObjectSpec) -> None:
    """Inject one free-floating primitive into a spec's worldbody.

    The body, its freejoint and its geom are all named after the object, so a
    later `mj_name2id` miss can say which one it was looking for.
    """
    body = spec.worldbody.add_body(name=obj.name, pos=list(obj.initial_pos))
    body.add_freejoint(name=f"{obj.name}_free")
    # MuJoCo's geom size is always three numbers; a sphere reads one of them
    # and a cylinder two, so the unused slots are padded rather than omitted.
    size = list(obj.size) + [0.0] * (3 - len(obj.size))
    body.add_geom(
        name=f"{obj.name}_geom",
        type=_GEOM_TYPES[obj.shape],
        size=size,
        mass=obj.mass,
        rgba=list(obj.rgba),
    )


class RobotModel:
    """An `MjModel`/`MjData` pair plus the `joint_order` -> MuJoCo remap."""

    def __init__(
        self,
        mjcf: Path,
        cfg: RunConfig,
        overrides: SimOverrides,
        *,
        objects: tuple[ObjectSpec, ...] = (),
    ) -> None:
        self.cfg = cfg
        self.mjcf = mjcf
        self.overrides = overrides
        self.objects = tuple(objects)

        try:
            spec = mujoco.MjSpec.from_file(str(mjcf))
            for obj in self.objects:
                _add_object(spec, obj)
            self.model = spec.compile()
        except Exception as exc:  # noqa: BLE001 -- any loader failure, reported with the path
            raise ModelError(f"loading {mjcf} through MjSpec failed: {exc}") from exc
        self.data = mujoco.MjData(self.model)

        # Before the body map and the base lookup, both of which have to know
        # which bodies this class itself injected.
        self._resolve_objects()
        self._build_joint_map()
        self._build_actuator_map()
        self._build_body_map()
        self._find_base()
        self._apply_config_to_model()
        self._apply_overrides()

        self.reset()

    # -- construction -------------------------------------------------------
    def _resolve_objects(self) -> None:
        """Ids for every injected object, and the body set the rest must exclude."""
        self.object_body_ids: list[int] = []
        self.object_geom_ids: list[int] = []
        self.object_qpos_adr: list[int] = []
        self.object_dof_adr: list[int] = []
        self.object_body_names: list[str] = [obj.name for obj in self.objects]

        for obj in self.objects:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, obj.name)
            geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"{obj.name}_geom")
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{obj.name}_free")
            if body_id < 0 or geom_id < 0 or joint_id < 0:
                raise ModelError(
                    f"scene object {obj.name!r} was injected into the spec but the compiled model has "
                    f"no body/geom/freejoint by that name (ids {body_id}/{geom_id}/{joint_id}). The "
                    f"likeliest cause is a name the MJCF already uses."
                )
            self.object_body_ids.append(int(body_id))
            self.object_geom_ids.append(int(geom_id))
            self.object_qpos_adr.append(int(self.model.jnt_qposadr[joint_id]))
            self.object_dof_adr.append(int(self.model.jnt_dofadr[joint_id]))

        self._object_bodies = set(self.object_body_ids)

    def _build_joint_map(self) -> None:
        """`joint_order` -> MuJoCo joint id, qpos address, and dof address."""
        by_name = {}
        for jid in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, jid)
            if name is not None:
                by_name[name] = jid

        missing = [j for j in self.cfg.joint_order if j not in by_name]
        if missing:
            raise ModelError(
                f"{self.mjcf}: the model has no joint(s) {missing}, which config.yaml's "
                f"joint_order names. Model joints: {sorted(by_name)}"
            )

        self.joint_ids = np.array([by_name[j] for j in self.cfg.joint_order], dtype=np.int32)

        wrong_type = [
            j
            for j, jid in zip(self.cfg.joint_order, self.joint_ids)
            if int(self.model.jnt_type[jid]) not in (_JNT_HINGE, _JNT_SLIDE)
        ]
        if wrong_type:
            raise ModelError(
                f"{self.mjcf}: joint(s) {wrong_type} are not hinge/slide joints, so they do "
                f"not have the single scalar position an actuated joint needs."
            )

        self.qpos_adr = self.model.jnt_qposadr[self.joint_ids].copy()
        self.dof_adr = self.model.jnt_dofadr[self.joint_ids].copy()

    def _build_actuator_map(self) -> None:
        """Actuator id per joint, matched on the actuator's TARGET, not its name.

        The G1's `<motor>` elements happen to be named after their joints, but
        nothing requires that, and an actuator silently matched by a
        coincidental name would apply torque to the wrong joint.
        """
        by_joint: dict[int, int] = {}
        for aid in range(self.model.nu):
            if int(self.model.actuator_trntype[aid]) != _TRN_JOINT:
                continue
            by_joint.setdefault(int(self.model.actuator_trnid[aid, 0]), aid)

        missing = [j for j, jid in zip(self.cfg.joint_order, self.joint_ids) if int(jid) not in by_joint]
        if missing:
            raise ModelError(
                f"{self.mjcf}: no joint-transmission actuator drives {missing}. LearningRLsim "
                f"applies computed torques through `data.ctrl`, so every actuated joint "
                f"needs a <motor> (or equivalent) targeting it."
            )
        self.actuator_ids = np.array([by_joint[int(j)] for j in self.joint_ids], dtype=np.int32)

    def _build_body_map(self) -> None:
        """Every named body, in model order -- the link frames a 3D viewer needs.

        Body 0 is MuJoCo's `worldbody`, which is the parent of these frames
        rather than one of them. An unnamed body has no URDF link to attach a
        mesh to, so there is nothing a viewer could do with its pose.

        Injected scene objects are skipped. These names are handed to
        `urdf_static_frames` and `LinkTransforms` as the ROBOT's link frames,
        and offering a scene object to the URDF as a link would both break that
        walk and draw a crate as part of the robot. Object frames are
        published deliberately instead -- see `data/transforms.py`.
        """
        ids, names = [], []
        for bid in range(1, self.model.nbody):
            if bid in self._object_bodies:
                continue
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, bid)
            if name is not None:
                ids.append(bid)
                names.append(name)
        self.body_ids = np.array(ids, dtype=np.int32)
        self.body_names: list[str] = names
        # Each body's parent BODY name, or None for a child of the worldbody.
        # This is the robot's kinematic structure, which is the shape a viewer
        # wants its transform tree in -- see `data/transforms.py`.
        self.body_parents: list[str | None] = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, int(self.model.body_parentid[bid])) or None
            for bid in ids
        ]

    def _find_base(self) -> None:
        """The free joint and the body it floats -- the robot's root.

        Every scene object carries a freejoint of its own, so the raw count is
        `1 + len(objects)`. The ones this class injected are subtracted by
        body id -- the exact set `_add_object` created, not a heuristic -- and
        the hard error is kept for whatever is left.
        """
        free = [
            jid
            for jid in range(self.model.njnt)
            if int(self.model.jnt_type[jid]) == _JNT_FREE and int(self.model.jnt_bodyid[jid]) not in self._object_bodies
        ]
        if len(free) != 1:
            excluded = f", excluding the injected object(s) {self.object_body_names}" if self.objects else ""
            raise ModelError(
                f"{self.mjcf}: expected exactly one free joint (the floating base), found "
                f"{len(free)}{excluded}. A fixed-base or multi-body model is not a locomotion scene."
            )
        self.free_joint_id = free[0]
        self.base_qpos_adr = int(self.model.jnt_qposadr[self.free_joint_id])
        self.base_dof_adr = int(self.model.jnt_dofadr[self.free_joint_id])
        self.base_body_id = int(self.model.jnt_bodyid[self.free_joint_id])

    def _apply_config_to_model(self) -> None:
        """Push the exported per-joint actuator parameters into the model.

        Armature and joint friction are training-side dynamics, not policy
        interface: Isaac Lab applies them from the actuator cfg, and this
        vendor MJCF declares neither (no `<default>`, no per-joint
        `armature=`). Leaving MuJoCo's zeros in place would compare a policy
        against a measurably different plant -- the G1's hip armature is
        0.0102 kg m^2, which is not a rounding difference at 200 Hz -- and the
        gap would read as a sim2sim finding.

        These come from `config.yaml`, so they are recorded as read, not
        assumed.
        """
        self.model.opt.timestep = self.cfg.sim_dt
        self.model.dof_armature[self.dof_adr] = self.cfg.actuators.armature
        self.model.dof_frictionloss[self.dof_adr] = self.cfg.actuators.friction

    def _apply_overrides(self) -> None:
        self.model.opt.gravity[:] = (0.0, 0.0, self.overrides.gravity)
        # Sliding friction only (column 0); torsional/rolling are left at the
        # model's own values, since Isaac Lab's material has no analogue for
        # them and inventing one would be a silent divergence of its own.
        #
        # `--friction` is the TERRAIN's, reproducing FLAT_TERRAIN_CFG. An
        # injected object keeps its own declared value, or a sweep of one
        # would silently move the other.
        terrain = np.ones(self.model.ngeom, dtype=bool)
        terrain[self.object_geom_ids] = False
        self.model.geom_friction[terrain, 0] = self.overrides.friction
        for obj, gid in zip(self.objects, self.object_geom_ids):
            self.model.geom_friction[gid, 0] = obj.friction

    # -- lifecycle ----------------------------------------------------------
    def reset(self) -> None:
        """Nominal pose: default joint positions, upright, at `spawn_height`."""
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.qpos_adr] = self.cfg.default_joint_pos
        self.data.qpos[self.base_qpos_adr + 0 : self.base_qpos_adr + 3] = (
            0.0,
            0.0,
            self.overrides.spawn_height,
        )
        # MuJoCo free-joint quaternion order is (w, x, y, z).
        self.data.qpos[self.base_qpos_adr + 3 : self.base_qpos_adr + 7] = (1.0, 0.0, 0.0, 0.0)
        # `mj_resetData` already restored these from `qpos0`; written out so
        # the whole scene's reset invariant reads in one place, as the robot's
        # pose does. A deferred object goes back to staging, which is what
        # re-arms placement after a mid-rollout reset.
        for obj, adr in zip(self.objects, self.object_qpos_adr):
            self.data.qpos[adr : adr + 3] = obj.initial_pos
            self.data.qpos[adr + 3 : adr + 7] = (1.0, 0.0, 0.0, 0.0)
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def step(self, torque: np.ndarray) -> None:
        self.data.ctrl[self.actuator_ids] = torque
        mujoco.mj_step(self.model, self.data)

    # -- state, all in `joint_order` ---------------------------------------
    @property
    def time(self) -> float:
        return float(self.data.time)

    @property
    def joint_pos(self) -> np.ndarray:
        return self.data.qpos[self.qpos_adr]

    @property
    def joint_vel(self) -> np.ndarray:
        return self.data.qvel[self.dof_adr]

    @property
    def base_pos_w(self) -> np.ndarray:
        return self.data.qpos[self.base_qpos_adr : self.base_qpos_adr + 3]

    @property
    def base_quat_w(self) -> np.ndarray:
        """(w, x, y, z), MuJoCo's order."""
        return self.data.qpos[self.base_qpos_adr + 3 : self.base_qpos_adr + 7]

    @property
    def base_rot_w(self) -> np.ndarray:
        """World<-body rotation matrix of the root body."""
        return self.data.xmat[self.base_body_id].reshape(3, 3)

    @property
    def base_lin_vel_w(self) -> np.ndarray:
        return self.data.qvel[self.base_dof_adr : self.base_dof_adr + 3]

    @property
    def base_lin_vel_b(self) -> np.ndarray:
        """`qvel[0:3]` is WORLD-frame -- rotate it in, unlike the angular half."""
        return self.base_rot_w.T @ self.base_lin_vel_w

    @property
    def base_ang_vel_b(self) -> np.ndarray:
        """`qvel[3:6]` is already BODY-frame -- no rotation. See the module docstring."""
        return self.data.qvel[self.base_dof_adr + 3 : self.base_dof_adr + 6]

    @property
    def projected_gravity_b(self) -> np.ndarray:
        return self.base_rot_w.T @ _GRAVITY_DIR_W

    @property
    def heading_w(self) -> float:
        """Yaw of the base frame, as Isaac Lab's `heading_w` defines it: the
        world-frame direction the body x-axis points in."""
        forward_w = self.base_rot_w @ np.array([1.0, 0.0, 0.0])
        return float(np.arctan2(forward_w[1], forward_w[0]))

    # -- link poses, for a 3D viewer ----------------------------------------
    def refresh_link_poses(self) -> None:
        """Recompute every body's world pose from the CURRENT `qpos`.

        `mj_step` runs `mj_forward` and only then integrates, so on return
        `xpos`/`xquat` describe the state *before* the integration while
        `qpos` describes the state after it. The gap is exactly `qvel * dt` --
        measured, not assumed -- and `mj_kinematics` closes it exactly.

        Without this the 3D view is drawn one physics step (5 ms at 200 Hz)
        behind every number plotted next to it, which is small enough to look
        fine and wrong enough to misread a foot-strike against `/sim/contact`.

        Cheap and side-effect-free in the rollout: the next `mj_step`'s own
        `mj_forward` recomputes all of this from scratch.

        NOTE: `base_rot_w` reads `xmat` and so carries this same one-step lag
        into `base_lin_vel_b`, `projected_gravity_b` and `heading_w` -- i.e.
        into the observation vector, where Isaac Lab has no equivalent lag.
        That is a live sim2sim confound and deliberately NOT fixed here:
        closing it changes every rollout, so it is a decision, not a cleanup.
        """
        mujoco.mj_kinematics(self.model, self.data)

    @property
    def body_pos_w(self) -> np.ndarray:
        """(nbody, 3) world position of every named body, in `body_names` order."""
        return self.data.xpos[self.body_ids]

    @property
    def body_quat_w(self) -> np.ndarray:
        """(nbody, 4) world orientation, MuJoCo's (w, x, y, z) order."""
        return self.data.xquat[self.body_ids]

    # -- contacts -----------------------------------------------------------
    def body_ids_matching(self, patterns: tuple[str, ...]) -> tuple[list[int], list[str]]:
        """Bodies whose names fully match any of `patterns` (regexes)."""
        import re

        ids: list[int] = []
        names: list[str] = []
        for bid in range(self.model.nbody):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, bid)
            if name is not None and any(re.fullmatch(p, name) for p in patterns):
                ids.append(bid)
                names.append(name)
        return ids, names

    def body_contact_force(self, body_id: int) -> np.ndarray:
        """Net external force on a body, world frame.

        `cfrc_ext` is (torque, force) -- force is the SECOND triple. The
        MJCF's foot contact geoms are unnamed (four spheres per ankle roll
        link), so a per-geom lookup is unavailable and this body-level total
        is the available signal.

        On a FOOT it sums the ground reaction and any object reaction; there
        is no per-contact split here. `/sim/object` carries the object's side
        of a collision independently.
        """
        return self.data.cfrc_ext[body_id][3:6]

    # -- scene objects ------------------------------------------------------
    def object_pose_w(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """`(position, quaternion)` from `qpos`, NOT from the one-step-stale `xpos`."""
        adr = self.object_qpos_adr[index]
        return self.data.qpos[adr : adr + 3], self.data.qpos[adr + 3 : adr + 7]

    def object_vel_w(self, index: int) -> np.ndarray:
        """(6,) free-joint velocity: linear WORLD, angular BODY -- as the base's is."""
        adr = self.object_dof_adr[index]
        return self.data.qvel[adr : adr + 6]

    def object_contact_force(self, index: int) -> np.ndarray:
        """Net external force on the object, world frame. Includes floor support."""
        return self.data.cfrc_ext[self.object_body_ids[index]][3:6]

    def place_object(self, index: int, pos_w, quat_w=(1.0, 0.0, 0.0, 0.0)) -> None:
        """Teleport an object to `pos_w`, at rest.

        Deliberately does NOT call `mj_forward`. Doing so would refresh `xmat`
        mid-policy-step and close, for that one step only, the deliberate
        one-step orientation lag in the observation path -- at exactly the
        instant under study. The next `mj_step` picks the pose up.
        """
        adr, dof = self.object_qpos_adr[index], self.object_dof_adr[index]
        self.data.qpos[adr : adr + 3] = pos_w
        self.data.qpos[adr + 3 : adr + 7] = quat_w
        self.data.qvel[dof : dof + 6] = 0.0
