"""The bag's channels and their JSON schemas -- defined once, shared by every sink.

Encoding is deliberately `jsonschema`/`json` for every channel: no `.proto`,
no codegen, inspectable with `mcap cat`, and zstd chunk compression absorbs
most of the size penalty. Protobuf gets added later and only for what Foxglove
must render in 3D (`foxglove.FrameTransform`, `foxglove.SceneUpdate`), which
is the one thing JSON cannot buy.

Channels are split by RATE, because that is what a sink decimates on:

- ``ONCE``    -- written at t=0 only.
- ``CONTROL`` -- once per policy step (50 Hz).
- ``PHYSICS`` -- once per physics step (200 Hz).

`/meta/run` is the highest-value channel here and the easiest to skip. It
carries `config.yaml` verbatim, the hashes of the policy and the MJCF, and
every assumed-vs-read physics parameter. **A sim2sim bag whose policy you
cannot identify from the file is an anecdote.**

`/actuator/cmd` carries the stiffness/damping in effect, so a bag records
the PD law the policy actually ran against, not just the one it was exported
with.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Rate(Enum):
    ONCE = "once"
    CONTROL = "control"
    PHYSICS = "physics"


@dataclass(frozen=True)
class Channel:
    topic: str
    rate: Rate
    schema: dict
    description: str
    well_known_schema: str | None = None
    """A Foxglove schema name, for the channels it renders natively.

    Foxglove picks a renderer off the SCHEMA NAME, never the topic -- `/tf`
    carrying a schema named `tf` is an unrecognised blob it will happily plot
    and never draw. Only `/tf` needs this today; everything else is ours to
    name, and gets the topic with the slashes turned into dots.
    """

    @property
    def schema_name(self) -> str:
        """The name this channel's schema is registered under in the bag."""
        return self.well_known_schema or self.topic.lstrip("/").replace("/", ".")


def _array(description: str, items: str = "number") -> dict:
    return {"type": "array", "items": {"type": items}, "description": description}


def _number(description: str) -> dict:
    return {"type": "number", "description": description}


def _object(properties: dict) -> dict:
    return {"type": "object", "properties": properties}


_TIME = _object({"sec": {"type": "integer", "minimum": 0}, "nsec": {"type": "integer", "minimum": 0}})
_VECTOR3 = _object({"x": _number("m"), "y": _number("m"), "z": _number("m")})
_QUATERNION = _object({"x": _number(""), "y": _number(""), "z": _number(""), "w": _number("")})

META_RUN = Channel(
    topic="/meta/run",
    rate=Rate.ONCE,
    description="Everything needed to identify what produced this bag.",
    # `config` is the exported YAML parsed into JSON, kept whole rather than
    # summarised: the exporter already describes the entire policy interface,
    # so embedding it costs one message and makes the bag self-contained.
    schema=_object(
        {
            "learningrlsim_version": {"type": "string"},
            "task": {"type": "string"},
            "robot": {"type": "string"},
            "run_dir": {"type": "string"},
            "mjcf": {"type": "string"},
            # Viewer geometry only -- nothing in the rollout reads it.
            "urdf": {"type": ["string", "null"], "description": "meshes for /tf's link frames"},
            "seed": {"type": "integer"},
            "config": {"type": "object", "description": "config.yaml, verbatim"},
            "sha256": _object(
                {
                    "policy.onnx": {"type": "string"},
                    "policy.onnx.data": {"type": "string"},
                    "mjcf": {"type": "string"},
                }
            ),
            "git": _object({"LearningRLsim": {"type": "string"}, "robot_description": {"type": "string"}}),
            "timeline": {
                "type": ["object", "null"],
                "description": "The scripted command timeline, verbatim; null if none.",
            },
            "physics": {
                "type": "object",
                "description": (
                    "Physics parameters with, for each, whether it was READ from config.yaml "
                    "or ASSUMED by LearningRLsim. The exporter records neither terrain friction, "
                    "spawn height nor gravity, so a divergence must never be ambiguous about "
                    "which of them it might be measuring."
                ),
            },
            "joint_order": _array("config.yaml's joint order", items="string"),
        }
    ),
)

SIM_STATE = Channel(
    topic="/sim/state",
    rate=Rate.PHYSICS,
    description="Raw MuJoCo state, plus the base pose and twist.",
    schema=_object(
        {
            "t": _number("sim time, seconds"),
            "joint_pos": _array("in joint_order"),
            "joint_vel": _array("in joint_order"),
            "base_pos_w": _array("world position of the root body"),
            "base_quat_w": _array("(w, x, y, z), MuJoCo's order"),
            "base_lin_vel_b": _array("root-frame linear velocity"),
            "base_ang_vel_b": _array("root-frame angular velocity"),
            "base_height": _number("world z of the root body"),
        }
    ),
)

POLICY_OBS = Channel(
    topic="/policy/obs",
    rate=Rate.CONTROL,
    description="The observation vector fed to the policy, split by term name.",
    schema=_object(
        {
            "t": _number("sim time, seconds"),
            "vector": _array("the whole post-clip, post-scale vector"),
            "terms": {
                "type": "object",
                "description": "term name -> that term's slice, so a plot can name what it shows",
            },
        }
    ),
)

POLICY_ACTION = Channel(
    topic="/policy/action",
    rate=Rate.CONTROL,
    description="Raw ONNX output, and each term's processed value.",
    schema=_object(
        {
            "t": _number("sim time, seconds"),
            "raw": _array("the policy output verbatim, pre-scale/offset/clip"),
            "terms": {"type": "object", "description": "term name -> its raw slice"},
        }
    ),
)

ACTUATOR_CMD = Channel(
    topic="/actuator/cmd",
    rate=Rate.CONTROL,
    description="Joint position targets and THE GAINS IN EFFECT.",
    schema=_object(
        {
            "t": _number("sim time, seconds"),
            "joint_pos_target": _array("in joint_order"),
            "stiffness": _array("live kp"),
            "damping": _array("live kd"),
        }
    ),
)

ACTUATOR_TORQUE = Channel(
    topic="/actuator/torque",
    rate=Rate.PHYSICS,
    description="Applied torque and whether the effort limit bound it.",
    schema=_object(
        {
            "t": _number("sim time, seconds"),
            "computed": _array("kp*(q_des-q) + kd*(0-qd), before clipping"),
            "applied": _array("after clipping to effort_limit"),
            "saturated": _array("1 where the limit bound the torque", items="integer"),
            "tracking_error": _array("q_des - q, in joint_order"),
        }
    ),
)

SIM_CONTACT = Channel(
    topic="/sim/contact",
    rate=Rate.PHYSICS,
    description="Per-foot ground reaction, from cfrc_ext on the foot bodies.",
    schema=_object(
        {
            "t": _number("sim time, seconds"),
            "bodies": _array("foot body names", items="string"),
            "normal_force": _array("world-z component of the net external force"),
            "in_contact": _array("1 where normal_force exceeds the threshold", items="integer"),
            # Strictly ADDITIVE. `normal_force` and `in_contact` keep their
            # exact original meaning for bags written before this field existed.
            "force": {
                "type": "array",
                "items": _array("(x, y, z) newtons, world frame"),
                "description": "net external force on each body, world frame (cfrc_ext[3:6])",
            },
        }
    ),
)

SIM_OBJECT = Channel(
    topic="/sim/object",
    rate=Rate.PHYSICS,
    description="Pose, velocity and net force of each injected scene object.",
    # Physics rate deliberately: an impact transient is over within a control
    # step. A live sink thins it per its own decimation, so a viewer pays
    # nothing for it.
    #
    # `shape`/`half_extents` are constant but ride along anyway, so this topic
    # alone is enough to redraw the object without cross-reading `/meta/run`.
    schema=_object(
        {
            "t": _number("sim time, seconds"),
            "name": {"type": "string"},
            "shape": {"type": "string", "description": "box | sphere | cylinder"},
            "half_extents": _array("MuJoCo's convention -- HALF the drawn extent"),
            "placed": {"type": "integer", "description": "1 once a deferred object has been placed"},
            "pos_w": _array("world position, from qpos"),
            "quat_w": _array("(w, x, y, z), MuJoCo's order"),
            "lin_vel_w": _array("world-frame linear velocity"),
            "speed": _number("|lin_vel_w|, m/s -- what contact is detected on"),
            "force_w": _array("net external force, world frame (cfrc_ext[3:6]); includes floor support"),
        }
    ),
)

SIM_SCENE = Channel(
    topic="/sim/scene",
    rate=Rate.CONTROL,
    well_known_schema="foxglove.SceneUpdate",
    description="The scene objects as drawable primitives, frame-locked to /tf.",
    # CONTROL rate, not ONCE, for two independent reasons: the 3D panel
    # rebuilds entity state on seek, so a single message at t=0 risks the
    # object not rendering when you scrub into the middle of a bag; and with
    # deferred spawning and a keyboard
    # bind the scene genuinely changes mid-run. `live_sink`'s latched replay
    # would cover a late subscriber for a ONCE channel and does nothing at all
    # for file playback.
    #
    # Same cadence as `/tf`, which is what the primitives are locked to, so
    # geometry and pose can never be one message apart.
    schema=_object(
        {
            "deletions": {"type": "array", "items": {"type": "object"}},
            "entities": {
                "type": "array",
                "items": _object(
                    {
                        "timestamp": _TIME,
                        "frame_id": {"type": "string", "description": "the object's own /tf frame"},
                        "id": {"type": "string"},
                        "frame_locked": {"type": "boolean"},
                        "cubes": {"type": "array", "items": {"type": "object"}},
                        "spheres": {"type": "array", "items": {"type": "object"}},
                        "cylinders": {"type": "array", "items": {"type": "object"}},
                    }
                ),
            },
        }
    ),
)

CMD_VELOCITY = Channel(
    topic="/cmd/velocity",
    rate=Rate.CONTROL,
    description="The commanded velocity, per command term.",
    schema=_object(
        {
            "t": _number("sim time, seconds"),
            "commands": {"type": "object", "description": "term name -> its value"},
        }
    ),
)

CMD_TRACKING = Channel(
    topic="/cmd/tracking",
    rate=Rate.CONTROL,
    description="How far the robot is from a motion command's current reference frame. Empty unless the run has one.",
    schema=_object(
        {
            "t": _number("sim time, seconds"),
            "name": {"type": "string", "description": "the command term"},
            "frame": _number("index of the clip frame the policy was shown"),
            "joint_rmse": _number("rms joint position error, radians"),
            "anchor_pos_error": _number("anchor body position error, metres"),
            "anchor_height_error": _number("anchor body height error, metres"),
            "anchor_ori_error": _number("anchor body orientation error, radians"),
        }
    ),
)

TF = Channel(
    topic="/tf",
    rate=Rate.CONTROL,
    well_known_schema="foxglove.FrameTransforms",
    description="World pose of every robot link, for a 3D viewer.",
    # Foxglove's own schema, transcribed -- the NAME above is what makes it
    # render, but writing the real shape keeps the bag self-describing under
    # `mcap cat` like every other channel here.
    #
    # Control rate, not physics: this is what a viewer draws, and 50 Hz is
    # already smoother than anyone scrubs. It also makes `--bag-rate control`
    # and `physics` produce an identically watchable 3D view.
    schema=_object(
        {
            "transforms": {
                "type": "array",
                "items": _object(
                    {
                        "timestamp": _TIME,
                        "parent_frame_id": {"type": "string"},
                        "child_frame_id": {"type": "string"},
                        "translation": _VECTOR3,
                        "rotation": _QUATERNION,
                    }
                ),
            }
        }
    ),
)

JOINT_STATES = Channel(
    topic="/joint/states",
    rate=Rate.CONTROL,
    well_known_schema="foxglove.JointStates",
    description="The same pose as /tf, encoded as joint angles instead of link frames.",
    # Deliberately redundant with `/tf`. Foxglove's URDF layer will either read
    # a frame per link (`controlMode: "transforms"`) or run its own forward
    # kinematics from these (`controlMode: "jointStates"`), and which of the
    # two a given build gets right is not something this repo can control. Both
    # are written so a bag can be viewed either way without re-running.
    schema=_object(
        {
            "timestamp": _TIME,
            "joints": {
                "type": "array",
                "items": _object(
                    {
                        "name": {"type": "string"},
                        "position": _number("radians for a revolute joint, metres for a prismatic one"),
                        "velocity": _number("rad/s or m/s"),
                    }
                ),
            },
        }
    ),
)


ALL: tuple[Channel, ...] = (
    META_RUN,
    SIM_STATE,
    POLICY_OBS,
    POLICY_ACTION,
    ACTUATOR_CMD,
    ACTUATOR_TORQUE,
    SIM_CONTACT,
    SIM_OBJECT,
    SIM_SCENE,
    CMD_VELOCITY,
    CMD_TRACKING,
    TF,
    JOINT_STATES,
)

BY_TOPIC = {channel.topic: channel for channel in ALL}
