"""The rollout: policy at `policy_frequency_hz`, physics at `sim_frequency_hz`.

Step order is Isaac Lab's `ManagerBasedRLEnv.step`, which matters because the
command resample and the observation both happen AFTER the physics substeps:

    process actions (once)
    decimation x { apply actions; compute torque; mj_step }
    update commands
    assemble the next observation

The PD law reproduces `IdealPDActuator`:

    tau = clip(kp*(q_des - q) + kd*(0 - qd), -effort_limit, +effort_limit)

with `q̇_des = 0` and `tau_ff = 0`. It runs **every physics substep**, not
every policy step -- recomputing it at 50 Hz instead of 200 Hz is a different
controller and a large, entirely artificial sim2sim gap.
"""

from __future__ import annotations

import math
import re
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from itertools import count

import numpy as np

from LearningRLsim import actions as action_lib
from LearningRLsim import observations as obs_lib
from LearningRLsim.commands.sources import CommandManager
from LearningRLsim.context import EvalContext
from LearningRLsim.data.recorder import Recorder
from LearningRLsim.data.scene import scene_message
from LearningRLsim.data.transforms import LinkTransforms, joint_states_message
from LearningRLsim.model import RobotModel
from LearningRLsim.policy import Policy
from LearningRLsim.runconfig import RunConfig

DEFAULT_FOOT_BODIES = (".*_ankle_roll_link",)
"""`FEET_CONTACT.bodies` for all three robots, but it lives in each robot's
`sensor.py`, which imports `isaaclab.sensors` -- unreadable on a CPU-only
host. A flag with this default until it is hoisted into `robot.yaml`."""

CONTACT_FORCE_THRESHOLD = 1.0  # newtons

MAX_PACING_DEBT = 0.5
"""Seconds behind wall clock before a pacer gives up on catching up.

A rollout that cannot hold the rate -- a laptop, a contact-heavy step -- must
not then sprint to make up the lost time, which is exactly when a hand-driven
robot runs away from its driver."""


class Pacer:
    """Holds sim time to wall time, for a rollout someone is watching.

    Off by default: a headless run recording a bag should finish as fast as
    the machine allows, and 15 s of sim in 2 s of wall clock is the normal
    case. It is only live streaming and interactive control that need the two
    clocks to agree -- a keyboard cannot drive a robot that has already
    finished walking.
    """

    def __init__(self, speed: float = 1.0) -> None:
        if speed <= 0:
            raise ValueError(f"pacing speed must be positive, got {speed}")
        self.speed = float(speed)
        self._origin = time.monotonic()

    def start(self) -> None:
        self._origin = time.monotonic()

    def wait(self, sim_time: float) -> None:
        delay = self._origin + sim_time / self.speed - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        elif delay < -MAX_PACING_DEBT:
            self._origin = time.monotonic() - sim_time / self.speed


@dataclass
class RolloutResult:
    steps: int = 0
    sim_time: float = 0.0
    fell: bool = False
    fall_time: float | None = None
    distance: float = 0.0
    max_tracking_error: float = 0.0
    saturated_steps: int = 0
    bag_path: str | None = None
    events: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"  steps            {self.steps} ({self.sim_time:.2f} s of sim time)",
            f"  distance         {self.distance:.3f} m",
            f"  max |q_des - q|  {self.max_tracking_error:.4f} rad",
            f"  effort-saturated {self.saturated_steps} physics step(s)",
        ]
        if self.bag_path:
            lines.append(f"  bag              {self.bag_path}")
        if self.events:
            lines.append(f"  events           {', '.join(self.events)}")
        lines.append(f"  FELL at t={self.fall_time:.2f} s" if self.fell else "  upright throughout")
        return "\n".join(lines)


class Rollout:
    def __init__(
        self,
        cfg: RunConfig,
        sim: RobotModel,
        policy: Policy,
        commands: CommandManager,
        recorder: Recorder,
        foot_bodies: tuple[str, ...] = DEFAULT_FOOT_BODIES,
        min_base_height: float = 0.4,
        transforms: LinkTransforms | None = None,
        pacer: Pacer | None = None,
    ) -> None:
        self.cfg = cfg
        self.sim = sim
        self.policy = policy
        self.commands = commands
        self.recorder = recorder
        self.min_base_height = min_base_height
        self.transforms = transforms
        self.pacer = pacer
        self.result = RolloutResult()

        self.action_terms = action_lib.build(cfg)
        self.command = action_lib.ActuatorCommand.nominal(cfg)

        # Read off the sim rather than taken as a parameter: the objects are
        # already in the plant by the time a Rollout exists, and a second
        # declaration of them could disagree with the one that was compiled.
        self.objects = sim.objects
        self._placed: list[bool] = []

        self.foot_ids, self.foot_names = sim.body_ids_matching(foot_bodies)
        if not self.foot_ids:
            raise ValueError(
                f"no body in {sim.mjcf.name} matches {list(foot_bodies)}, so there is no "
                f"contact to record. Pass --foot-bodies with a pattern that matches; the "
                f"model's bodies are listed by `--list-bodies`."
            )

        self.ctx = EvalContext(
            cfg=cfg,
            sim=sim,
            commands=commands,
            # The raw action at reset, per term. This is what the policy's
            # `last_action` slot reads on the very first inference.
            last_action=self._reset_action(),
        )

    def _reset_action(self) -> np.ndarray:
        raw = np.zeros(self.cfg.actions.total_dim)
        for term, cfg_term in zip(self.action_terms, self.cfg.actions.terms):
            start, stop = cfg_term.slice
            raw[start:stop] = term.reset_action
        return raw

    def run(self, duration: float) -> RolloutResult:
        cfg = self.cfg
        # Held on the instance as well as returned, so a rollout ended by
        # Ctrl-C still has a result to report -- an interactive session is
        # ended that way as often as by the `x` key.
        result = self.result = RolloutResult()

        self.sim.reset()
        self._reset_objects()
        self.commands.reset()
        self.ctx.last_action = self._reset_action()
        self.commands.update(0.0, self.ctx)

        self._record_meta_dependent_start()
        obs = obs_lib.assemble(self.ctx)

        start_xy = self.sim.base_pos_w[:2].copy()
        # An interactive rollout has no length: it ends when the driver says
        # so, the robot falls, or Ctrl-C. `inf` is how eval.py says that.
        steps = count() if math.isinf(duration) else range(int(round(duration * cfg.policy_frequency_hz)))
        if self.pacer is not None:
            self.pacer.start()

        try:
            self._steps(steps, result, obs)
        finally:
            result.sim_time = self.sim.time
            result.distance = float(np.linalg.norm(self.sim.base_pos_w[:2] - start_xy))
        return result

    def _steps(self, steps: Iterable[int], result: RolloutResult, obs: np.ndarray) -> None:
        """The policy loop itself. Split out of `run` so that an interrupt or a
        fall still lands in `run`'s finally and closes out the result."""
        cfg = self.cfg
        stop = False
        for _ in steps:
            raw = self.policy(obs)

            for term, cfg_term in zip(self.action_terms, cfg.actions.terms):
                start, end = cfg_term.slice
                term.process(raw[start:end])

            # The command the substeps below are about to start from. Without
            # this, `/actuator/cmd` would be recorded before any term had
            # written into `self.command`, so every message carried the
            # PREVIOUS policy step's targets -- a 20 ms lag against its own
            # timestamp. `apply` is a plain write, so running it here and
            # again at the top of the substep loop costs nothing.
            for term in self.action_terms:
                term.apply(self.command)

            t_control = self.sim.time
            self._record_control(t_control, obs, raw)

            for _ in range(cfg.decimation):
                for term in self.action_terms:
                    term.apply(self.command)
                torque, computed, saturated = self._pd()
                self.recorder.tick_physics()
                self._record_physics(self.sim.time, computed, torque, saturated)
                self.sim.step(torque)
                result.steps += 1
                if int(saturated.sum()) > 0:
                    result.saturated_steps += 1

            # Post-physics, exactly as the env does it: the command a policy
            # step sees is the one resampled after the previous step's physics.
            self.commands.update(self.sim.time, self.ctx)
            self.ctx.last_action = raw
            obs = obs_lib.assemble(self.ctx)

            error = float(np.max(np.abs(self.command.joint_pos_target - self.sim.joint_pos)))
            result.max_tracking_error = max(result.max_tracking_error, error)

            for event in self.commands.drain_events():
                result.events.append(f"{event.kind}@{event.t:.2f}s")
                if event.kind == "reset":
                    self.sim.reset()
                    # The WHOLE scene resets, which re-arms a deferred object:
                    # it is back at staging, so `place_object` works again.
                    self._reset_objects()
                    obs = obs_lib.assemble(self.ctx)
                elif event.kind == "place_object":
                    self._place_from_event(event)
                elif event.kind == "stop":
                    stop = True

            if not result.fell and self._has_fallen():
                result.fell = True
                result.fall_time = self.sim.time
                stop = True

            if stop:
                break

            if self.pacer is not None:
                self.pacer.wait(self.sim.time)

    # -- scene objects ------------------------------------------------------
    def _reset_objects(self) -> None:
        """Re-arm placement."""
        self._placed = [not obj.deferred for obj in self.objects]

    def _place_from_event(self, event) -> None:
        """Drop every deferred object in front of the base, along its heading.

        `ahead`/`lateral` come from the spec, overridable per event -- which is
        what `--object-ahead` and a swept timeline both go through.
        """
        deferred = [i for i, obj in enumerate(self.objects) if obj.deferred]
        if not deferred:
            raise ValueError(
                "a `place_object` event fired but the scene declares no deferred object. "
                "Either the scene's objects are all `at_start`, or no --scene was given: "
                "the operator asked for a placement and would otherwise silently not get one."
            )

        heading = self.sim.heading_w
        forward = np.array([math.cos(heading), math.sin(heading), 0.0])
        left = np.array([-math.sin(heading), math.cos(heading), 0.0])
        base = self.sim.base_pos_w

        for index in deferred:
            obj = self.objects[index]
            ahead = float(event.payload.get("ahead", obj.ahead))
            lateral = float(event.payload.get("lateral", obj.lateral))
            pos = base + ahead * forward + lateral * left
            pos[2] = obj.half_height
            self.sim.place_object(index, pos)
            self._placed[index] = True

    def _pd(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """`IdealPDActuator.compute`, with q̇_des = 0 and tau_ff = 0."""
        error_pos = self.command.joint_pos_target - self.sim.joint_pos
        error_vel = -self.sim.joint_vel
        computed = self.command.stiffness * error_pos + self.command.damping * error_vel
        limit = self.cfg.actuators.effort_limit
        applied = np.clip(computed, -limit, limit)
        return applied, computed, (np.abs(computed) > limit).astype(np.int8)

    def _has_fallen(self) -> bool:
        """Base below `min_base_height`, or tipped past horizontal.

        `projected_gravity_b[2]` is -1 upright and 0 on its side, so a value
        above -0.5 is past 60 degrees of tilt.
        """
        return bool(self.sim.base_pos_w[2] < self.min_base_height or self.sim.projected_gravity_b[2] > -0.5)

    # -- recording ----------------------------------------------------------
    def _record_meta_dependent_start(self) -> None:
        self.recorder.tick_physics()
        self._record_physics(
            self.sim.time,
            np.zeros(self.cfg.num_joints),
            np.zeros(self.cfg.num_joints),
            np.zeros(self.cfg.num_joints, dtype=np.int8),
        )

    def _record_control(self, t: float, obs: np.ndarray, raw: np.ndarray) -> None:
        self.recorder.record(
            "/policy/obs",
            t,
            {
                "t": t,
                "vector": obs,
                "terms": {term.name: obs[term.slice[0] : term.slice[1]] for term in self.cfg.observations.terms},
            },
        )
        self.recorder.record(
            "/policy/action",
            t,
            {
                "t": t,
                "raw": raw,
                "terms": {term.name: raw[term.slice[0] : term.slice[1]] for term in self.cfg.actions.terms},
            },
        )
        self.recorder.record(
            "/actuator/cmd",
            t,
            {
                "t": t,
                "joint_pos_target": self.command.joint_pos_target,
                "stiffness": self.command.stiffness,
                "damping": self.command.damping,
            },
        )
        self.recorder.record(
            "/cmd/velocity",
            t,
            {"t": t, "commands": {name: term.value() for name, term in self.commands.terms.items()}},
        )
        if self.transforms is not None:
            # Read straight off `data.xpos`/`xquat`, so this is the pose the
            # policy is about to act from -- the same instant `/policy/obs`
            # above describes, which is what makes the 3D view and the plots
            # line up under the scrubber.
            self.recorder.record("/tf", t, self.transforms.message(t))
        if self.objects:
            # Same cadence as the `/tf` frames the primitives are locked to,
            # so geometry and pose can never be one message apart.
            self.recorder.record("/sim/scene", t, scene_message(self.objects, t))
        # Not gated on `self.transforms`: joint angles need no URDF, and they
        # are the fallback for a viewer that will not place links from frames.
        self.recorder.record("/joint/states", t, joint_states_message(self.sim, t))

    def _record_physics(self, t: float, computed: np.ndarray, applied: np.ndarray, saturated: np.ndarray) -> None:
        self.recorder.record(
            "/sim/state",
            t,
            {
                "t": t,
                "joint_pos": self.sim.joint_pos,
                "joint_vel": self.sim.joint_vel,
                "base_pos_w": self.sim.base_pos_w,
                "base_quat_w": self.sim.base_quat_w,
                "base_lin_vel_b": self.sim.base_lin_vel_b,
                "base_ang_vel_b": self.sim.base_ang_vel_b,
                "base_height": float(self.sim.base_pos_w[2]),
            },
        )
        self.recorder.record(
            "/actuator/torque",
            t,
            {
                "t": t,
                "computed": computed,
                "applied": applied,
                "saturated": saturated,
                "tracking_error": self.command.joint_pos_target - self.sim.joint_pos,
            },
        )
        net = np.array([self.sim.body_contact_force(bid) for bid in self.foot_ids])
        forces = [float(f) for f in net[:, 2]]
        self.recorder.record(
            "/sim/contact",
            t,
            {
                "t": t,
                "bodies": self.foot_names,
                # Unchanged, byte for byte: older bags carry these two.
                # `force` is the addition.
                "normal_force": forces,
                "in_contact": [int(f > CONTACT_FORCE_THRESHOLD) for f in forces],
                "force": net,
            },
        )
        self._record_objects(t)

    def _record_objects(self, t: float) -> None:
        """`/sim/object` per object."""
        for index, obj in enumerate(self.objects):
            position, quaternion = self.sim.object_pose_w(index)
            velocity = self.sim.object_vel_w(index)
            force = self.sim.object_contact_force(index)
            speed = float(np.linalg.norm(velocity[:3]))
            self.recorder.record(
                "/sim/object",
                t,
                {
                    "t": t,
                    "name": obj.name,
                    "shape": obj.shape,
                    "half_extents": list(obj.half_extents),
                    "placed": int(self._placed[index]),
                    "pos_w": position,
                    "quat_w": quaternion,
                    "lin_vel_w": velocity[:3],
                    "speed": speed,
                    "force_w": force,
                },
            )


def parse_patterns(raw: str) -> tuple[str, ...]:
    """Comma-separated regexes, each compiled now so a bad one fails at parse."""
    patterns = tuple(p.strip() for p in raw.split(",") if p.strip())
    for pattern in patterns:
        re.compile(pattern)
    return patterns
