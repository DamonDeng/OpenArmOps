"""Offline (no UDP, no hardware) replay of a recorded VR log through
the same VR→IK pipeline the live motion worker uses.

What this script does
---------------------
For each LEFT/RIGHT controller packet in the log, in order:

1. Build a Pinocchio SE3 from the wire pose (same construction as the
   live worker's ``_controller_pose``).
2. Detect grip rising/falling edges using the same threshold as
   ``config.VR_GRIP_ENABLE_THRESHOLD``.
3. On rising edge, take a fresh snapshot — controller pose at that
   moment + the *pretend* arm TCP pose (FK of "all joints at zero").
4. On each subsequent engaged tick, compute
   ``delta = ctrl_snapshot⁻¹ * ctrl_now``, apply VR position/rotation
   scale, apply the per-arm axis remap, then ``arm_target =
   arm_snapshot * delta``.
5. Run the same ``CartesianIKSolver.solve()`` the live worker uses
   against this target, with the seed = previous successful q (or
   all-zero on the first solve).
6. Record the result: convergence, position/rotation error, joint
   clamping, whether position-priority fallback was used, and whether
   IK declared the target unusable.

What this script does NOT do
----------------------------
* No UDP. No socket open.
* No motor commands. Nothing touches the live app.
* No joint-space ramp / lead cap simulation. Those operate AFTER IK
  chooses a joint target; their failure modes don't affect "is the IK
  target reachable?" which is what we're testing.
* No gripper / trigger logic. Gripper isn't part of the 7-DOF IK; we
  drop it. Trigger is irrelevant offline.

Usage
-----
    python3 -m openarm_controller_ui_lerobot.tools.replay_vr_log_offline \\
        path/to/vr_log_*.jsonl

Output: prints progress as it works through the file, then a summary
table. Pass ``--csv path.csv`` to also write the per-tick numbers for
plotting in a spreadsheet.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pinocchio as pin

# Reuse the live code where it makes sense — we want this to find the
# same bugs the live path would, not a parallel re-implementation.
from openarm_controller_ui_lerobot import config
from openarm_controller_ui_lerobot.ik_solver import (
    CartesianIKSolver,
    IKResult,
    pose_from_xyzrpy,
    pose_to_xyzrpy,
)

logger = logging.getLogger(__name__)


# Real per-arm joint limits from hardware_info/joint_limits.md (what the
# live app uses with USE_FULL_LIMITS=True). Hardcoded so we don't need
# a hardware-connected RobotService.
_LEFT_JOINT_LIMITS = {
    # joint_1 upper bound mirrors the runtime override in
    # robot_service.arm_config_snapshot() (75°→120° for VR teleop reach).
    "joint_1": (-75.0, +120.0),
    "joint_2": (-90.0, +9.0),     # mirrored
    "joint_3": (-85.0, +85.0),
    "joint_4": (0.0, +135.0),
    "joint_5": (-85.0, +85.0),
    "joint_6": (-40.0, +40.0),
    "joint_7": (-80.0, +80.0),
    "gripper": (-65.0, 0.0),
}
_RIGHT_JOINT_LIMITS = {
    # joint_1 upper bound mirrors the runtime override in
    # robot_service.arm_config_snapshot() (75°→120° for VR teleop reach).
    "joint_1": (-75.0, +120.0),
    "joint_2": (-9.0, +90.0),     # asymmetric, mirror of left
    "joint_3": (-85.0, +85.0),
    "joint_4": (0.0, +135.0),
    "joint_5": (-85.0, +85.0),
    "joint_6": (-40.0, +40.0),
    "joint_7": (-80.0, +80.0),
    "gripper": (-65.0, 0.0),
}


@dataclass
class Packet:
    """One controller datagram from the .jsonl, parsed."""
    line_no: int       # 1-based source line for cross-reference
    t: float
    kind: str          # "LEFT" or "RIGHT"; HEAD/MODE are skipped
    pos: np.ndarray    # (3,) translation
    quat: np.ndarray   # (4,) qx qy qz qw
    trigger: float
    grip: float
    a: int
    b: int
    x: int
    y: int


def parse_log(path: Path) -> list[Packet]:
    """Read .jsonl, keep only LEFT/RIGHT packets with full pose payload."""
    packets: list[Packet] = []
    with path.open("r", encoding="utf-8") as f:
        for ln_no, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            tokens = str(rec.get("raw", "")).split()
            if len(tokens) < 16:
                continue
            kind = tokens[0].upper()
            if kind not in ("LEFT", "RIGHT"):
                continue
            try:
                packets.append(Packet(
                    line_no=ln_no,
                    t=float(rec.get("t", 0.0)),
                    kind=kind,
                    pos=np.array([float(tokens[1]), float(tokens[2]),
                                  float(tokens[3])], dtype=float),
                    quat=np.array([float(tokens[4]), float(tokens[5]),
                                   float(tokens[6]), float(tokens[7])],
                                  dtype=float),
                    trigger=float(tokens[8]),
                    grip=float(tokens[9]),
                    # Wire format with kind kept at index 0 puts buttons
                    # at 10..13. Live receiver uses 9..12 because it
                    # strips the kind first; here we keep it.
                    a=int(float(tokens[10])),
                    b=int(float(tokens[11])),
                    x=int(float(tokens[12])),
                    y=int(float(tokens[13])),
                ))
            except (ValueError, IndexError):
                continue
    packets.sort(key=lambda p: p.t)
    return packets


def controller_pose(pkt: Packet) -> pin.SE3:
    """Mirror of MotionWorker._controller_pose."""
    q = pkt.quat
    n = float(np.linalg.norm(q))
    if n < 1e-9:
        q = np.array([0.0, 0.0, 0.0, 1.0])
    else:
        q = q / n
    qx, qy, qz, qw = q
    R = np.array([
        [1 - 2*(qy*qy + qz*qz),   2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),       1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),       2*(qy*qz + qx*qw),     1 - 2*(qx*qx + qy*qy)],
    ])
    return pin.SE3(R, pkt.pos.copy())


# Per-process override for the right-arm translation remap. Set by
# main() when --remap-right is supplied, so we can sweep candidate
# remaps without editing config.py.
_REMAP_RIGHT_OVERRIDE: np.ndarray | None = None

# When true, the remapped delta is treated as a world-frame motion
# applied at the home pose (target.t = home.t + delta_world). This
# matches the live motion_worker post the 2026-05-28 fix, so it is
# the default. Pass --legacy-body-delta to reproduce the pre-fix
# behaviour for A/B comparison against old recordings.
_APPLY_DELTA_IN_WORLD: bool = True


def apply_vr_remap(arm: str, delta: pin.SE3) -> pin.SE3:
    """Apply the per-arm VR-to-robot axis remap to an SE3 delta.

    NOTE: this differs from MotionWorker._apply_vr_remap, which uses
    log → matrix-multiply → exp. That decomposition is incorrect when
    the delta has a non-zero rotation component, because log(SE3) of a
    pose with rotation does NOT yield translation-equal-to-the-pose's
    translation — the screw-axis interpretation mixes translation and
    rotation, and applying a permutation matrix to the screw-axis
    translation distorts the result.

    Correct math: extract translation and rotation directly from the
    SE3, remap each in its own space (translation gets M_t @ t,
    rotation gets a similarity transform M_r R M_r^T), and rebuild a
    new SE3.
    """
    if arm == "left":
        M_t = np.array(config.VR_TRANSLATION_REMAP_LEFT, dtype=float)
        M_r = np.array(config.VR_ROTATION_REMAP_LEFT, dtype=float)
    else:
        if _REMAP_RIGHT_OVERRIDE is not None:
            M_t = _REMAP_RIGHT_OVERRIDE
        else:
            M_t = np.array(config.VR_TRANSLATION_REMAP_RIGHT, dtype=float)
        M_r = np.array(config.VR_ROTATION_REMAP_RIGHT, dtype=float)
    t_new = M_t @ delta.translation
    R_new = M_r @ delta.rotation @ M_r.T
    return pin.SE3(R_new, t_new)


@dataclass
class Snapshot:
    controller: pin.SE3
    arm: pin.SE3


@dataclass
class TickResult:
    line_no: int
    t_rel: float            # seconds since first packet
    arm: str
    grip: float
    grip_engaged: bool
    snapshot_taken: bool
    target_x: float | None
    target_y: float | None
    target_z: float | None
    pos_err_mm: float | None
    rot_err_deg: float | None
    iters: int | None
    converged: bool | None
    usable: bool | None
    clamped: bool | None
    position_priority_used: bool | None
    error: str = ""


class ArmReplay:
    """Per-arm VR-replay state. One of these per arm; receives the
    arm's packets in order via ``feed``.
    """

    def __init__(self, arm: str, urdf_path: str, vr_pos_scale: float,
                 vr_rot_scale: float):
        self.arm = arm
        limits = _LEFT_JOINT_LIMITS if arm == "left" else _RIGHT_JOINT_LIMITS
        self.solver = CartesianIKSolver(urdf_path, arm, limits)
        # Pretend the arm starts at all-joints-zero. FK gives the home
        # TCP pose; that's our arm-snapshot baseline.
        self.q_zero = [0.0] * 7
        self.home_tcp: pin.SE3 = self.solver.forward_kinematics(self.q_zero)
        self.snapshot: Snapshot | None = None
        self.last_q_deg: list[float] | None = None
        self.vr_pos_scale = vr_pos_scale
        self.vr_rot_scale = vr_rot_scale
        # Counters for the summary line.
        self.n_seen = 0
        self.n_engaged = 0
        self.n_solves = 0
        self.n_unusable = 0
        self.n_clamped = 0
        self.n_pos_priority = 0
        self.n_strict_converged = 0

    def feed(self, pkt: Packet, t0: float) -> TickResult:
        self.n_seen += 1
        t_rel = pkt.t - t0
        result = TickResult(
            line_no=pkt.line_no, t_rel=t_rel, arm=self.arm,
            grip=pkt.grip,
            grip_engaged=False, snapshot_taken=False,
            target_x=None, target_y=None, target_z=None,
            pos_err_mm=None, rot_err_deg=None, iters=None,
            converged=None, usable=None, clamped=None,
            position_priority_used=None,
        )

        ctrl_pose = controller_pose(pkt)
        grip_engaged = pkt.grip >= config.VR_GRIP_ENABLE_THRESHOLD
        result.grip_engaged = grip_engaged

        if not grip_engaged:
            # Released: drop snapshot, do nothing.
            self.snapshot = None
            return result

        self.n_engaged += 1
        if self.snapshot is None:
            # Rising edge — take a fresh snapshot. Arm is "at zero" so
            # its baseline TCP is home_tcp.
            self.snapshot = Snapshot(controller=ctrl_pose, arm=self.home_tcp)
            result.snapshot_taken = True
            # On the very first engaged packet, target == arm_snapshot
            # → IK solving the home pose returns ~q_zero. We still run
            # IK so the report has a row for this tick.
            target_pose = self.snapshot.arm
        else:
            # Continuing — compute delta and apply.
            delta = self.snapshot.controller.actInv(ctrl_pose)
            # Scale translation and rotation directly on the SE3
            # components (NOT via log/exp). Translation just scales as
            # itself; rotation is scaled as an angle by extracting the
            # axis-angle and re-exponentiating just the rotation part.
            if self.vr_pos_scale != 1.0:
                t_scaled = delta.translation * self.vr_pos_scale
            else:
                t_scaled = delta.translation
            if self.vr_rot_scale != 1.0:
                # Convert rotation matrix to axis-angle, scale the
                # angle, rebuild. Pinocchio's pin.log on SO(3) gives
                # axis*angle; pin.exp inverts it.
                w = pin.log3(delta.rotation)         # axis * angle
                w_scaled = w * self.vr_rot_scale
                R_scaled = pin.exp3(w_scaled)
            else:
                R_scaled = delta.rotation
            delta = pin.SE3(R_scaled, t_scaled)
            delta = apply_vr_remap(self.arm, delta)
            if _APPLY_DELTA_IN_WORLD:
                # Treat the remapped delta as expressed in the robot's
                # WORLD frame, not the EE-local frame: target =
                # home_world ∘ delta_world. Pinocchio's `*` left-applies
                # in body frame, so re-express delta in body frame
                # via similarity through R_home.
                R_home = self.snapshot.arm.rotation
                t_world = delta.translation
                R_world = delta.rotation
                t_body = R_home.T @ t_world
                R_body = R_home.T @ R_world @ R_home
                delta = pin.SE3(R_body, t_body)
            target_pose = self.snapshot.arm * delta

        # Run IK.
        seed = self.last_q_deg if self.last_q_deg is not None else self.q_zero
        try:
            ik: IKResult = self.solver.solve(target_pose, q_seed_deg_7=seed)
        except Exception as e:
            result.error = f"solver raised: {e!r}"
            return result

        self.n_solves += 1
        if ik.usable:
            self.last_q_deg = list(ik.q_deg)
        else:
            self.n_unusable += 1
        if ik.clamped:
            self.n_clamped += 1
        if ik.position_priority_used:
            self.n_pos_priority += 1
        if ik.converged:
            self.n_strict_converged += 1

        x, y, z, _, _, _ = pose_to_xyzrpy(target_pose)
        result.target_x = x
        result.target_y = y
        result.target_z = z
        result.pos_err_mm = ik.pos_err_mm
        result.rot_err_deg = ik.rot_err_deg
        result.iters = ik.iters
        result.converged = ik.converged
        result.usable = ik.usable
        result.clamped = ik.clamped
        result.position_priority_used = ik.position_priority_used
        return result


def replay(
    packets: list[Packet],
    urdf_path: str,
    vr_pos_scale: float,
    vr_rot_scale: float,
    verbose: bool,
) -> tuple[list[TickResult], dict[str, ArmReplay]]:
    arms = {
        "left":  ArmReplay("left", urdf_path, vr_pos_scale, vr_rot_scale),
        "right": ArmReplay("right", urdf_path, vr_pos_scale, vr_rot_scale),
    }
    if not packets:
        return [], arms
    t0 = packets[0].t
    results: list[TickResult] = []
    for pkt in packets:
        # Packet kind is upper-case ("LEFT"/"RIGHT"); arms dict uses
        # lower-case keys to match the rest of the codebase.
        arm_key = pkt.kind.lower()
        if arm_key not in arms:
            continue
        r = arms[arm_key].feed(pkt, t0)
        results.append(r)
        if verbose and (r.snapshot_taken or r.error
                        or (r.usable is False and r.grip_engaged)):
            tag = ("SNAP" if r.snapshot_taken
                   else "ERR " if r.error
                   else "UNUS")
            target = (f"({r.target_x:+.3f}, {r.target_y:+.3f}, "
                      f"{r.target_z:+.3f})"
                      if r.target_x is not None else "—")
            ik_brief = (f"pos={r.pos_err_mm:6.2f}mm rot={r.rot_err_deg:5.2f}°"
                        if r.pos_err_mm is not None else "—")
            extra = (f"  [{r.error}]" if r.error
                     else f"  prio={'Y' if r.position_priority_used else 'n'}"
                          f"  clamp={'Y' if r.clamped else 'n'}")
            logger.info(
                f"{tag}  ln{r.line_no:5d}  t={r.t_rel:7.3f}s  "
                f"{r.arm:5s}  target={target}  {ik_brief}{extra}"
            )
    return results, arms


def write_csv(results: list[TickResult], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([f.name for f in dataclasses.fields(TickResult)])
        for r in results:
            row = []
            for f_field in dataclasses.fields(TickResult):
                v = getattr(r, f_field.name)
                row.append("" if v is None else v)
            w.writerow(row)


def print_summary(arms: dict[str, ArmReplay]) -> None:
    print()
    print(f"{'arm':5}  {'pkts':>6}  {'engaged':>7}  {'solves':>6}  "
          f"{'unusable':>8}  {'clamped':>7}  {'pos-prio':>8}  "
          f"{'strict-conv':>11}")
    print("-" * 70)
    for name, ar in arms.items():
        print(f"{name:5}  {ar.n_seen:>6d}  {ar.n_engaged:>7d}  "
              f"{ar.n_solves:>6d}  {ar.n_unusable:>8d}  {ar.n_clamped:>7d}  "
              f"{ar.n_pos_priority:>8d}  {ar.n_strict_converged:>11d}")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(
        description="Offline replay of a VR log through the IK pipeline. "
                    "No UDP, no hardware. Pretends both arms start at zero "
                    "joints and reports IK behaviour for each engaged "
                    "controller packet."
    )
    parser.add_argument("input", type=Path,
                        help="Path to the .jsonl recording.")
    parser.add_argument("--vr-pos-scale", type=float,
                        default=config.INITIAL_VR_POS_SCALE,
                        help="VR translation gain (matches the live System "
                             f"tab knob, default {config.INITIAL_VR_POS_SCALE}).")
    parser.add_argument("--vr-rot-scale", type=float,
                        default=config.INITIAL_VR_ROT_SCALE,
                        help="VR rotation gain (default "
                             f"{config.INITIAL_VR_ROT_SCALE}).")
    parser.add_argument("--urdf", type=Path,
                        default=Path(config.GRAVITY_URDF_PATH),
                        help="URDF file (defaults to the one the live app uses).")
    parser.add_argument("--csv", type=Path, default=None,
                        help="Optional path to dump per-tick results as CSV.")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Log notable events (snapshot, unusable, error).")
    parser.add_argument(
        "--legacy-body-delta", action="store_true",
        help="Reproduce the pre-2026-05-28 behaviour: compose the remapped "
             "delta in EE-body frame (target = snap ∘ delta). Useful only "
             "for A/B comparison against old recordings — the live worker "
             "now applies delta in world frame, which is the new default.",
    )
    parser.add_argument(
        "--remap-right", type=str, default=None,
        help="Override VR_TRANSLATION_REMAP_RIGHT for this run only. "
             "Format: 9 comma-separated floats in row-major order. "
             "Example: '0,0,1, -1,0,0, 0,1,0' for "
             "robot_x=+vr_z, robot_y=-vr_x, robot_z=+vr_y. config.py is not modified.",
    )
    args = parser.parse_args(argv)

    global _APPLY_DELTA_IN_WORLD
    _APPLY_DELTA_IN_WORLD = not bool(args.legacy_body_delta)
    if _APPLY_DELTA_IN_WORLD:
        logger.info("Composing delta in WORLD frame (matches live worker).")
    else:
        logger.info("--legacy-body-delta: composing delta in EE-body frame "
                    "(pre-2026-05-28 behaviour).")

    if args.remap_right is not None:
        try:
            vals = [float(x) for x in args.remap_right.replace(",", " ").split()]
        except ValueError:
            logger.error("--remap-right must be 9 numbers")
            return 2
        if len(vals) != 9:
            logger.error(f"--remap-right needs 9 numbers, got {len(vals)}")
            return 2
        # _REMAP_RIGHT_OVERRIDE is module-level; we already have a
        # `global` declaration above for _APPLY_DELTA_IN_WORLD, but
        # _REMAP_RIGHT_OVERRIDE needs its own.
        global _REMAP_RIGHT_OVERRIDE
        _REMAP_RIGHT_OVERRIDE = np.array(vals, dtype=float).reshape(3, 3)
        logger.info(f"Override right remap = {_REMAP_RIGHT_OVERRIDE.tolist()}")

    if not args.input.exists():
        logger.error(f"input not found: {args.input}")
        return 2
    if not args.urdf.exists():
        logger.error(f"URDF not found: {args.urdf}")
        return 2

    packets = parse_log(args.input)
    if not packets:
        logger.error("input has no LEFT/RIGHT packets")
        return 1
    by_kind: dict[str, int] = {}
    for p in packets:
        by_kind[p.kind] = by_kind.get(p.kind, 0) + 1
    duration = packets[-1].t - packets[0].t
    logger.info(
        f"Loaded {len(packets)} controller packets from {args.input.name} "
        f"({by_kind}, duration {duration:.2f}s)"
    )
    logger.info(
        f"VR_GRIP_ENABLE_THRESHOLD={config.VR_GRIP_ENABLE_THRESHOLD}, "
        f"vr_pos_scale={args.vr_pos_scale}, vr_rot_scale={args.vr_rot_scale}"
    )

    results, arms = replay(
        packets, str(args.urdf),
        args.vr_pos_scale, args.vr_rot_scale,
        verbose=args.verbose,
    )

    print_summary(arms)

    if args.csv is not None:
        write_csv(results, args.csv)
        logger.info(f"Wrote per-tick CSV: {args.csv}")

    # Exit code: non-zero if any solve was unusable in either arm —
    # makes it easy to wire into a "did this log replay cleanly?" check.
    total_unusable = sum(a.n_unusable for a in arms.values())
    return 0 if total_unusable == 0 else 3


if __name__ == "__main__":
    sys.exit(main())
