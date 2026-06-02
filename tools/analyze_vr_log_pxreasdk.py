"""Offline diagnosis of a PXREASDK VR recording.

For a given ``vr_log_*.jsonl`` captured by the new XRoboToolkit
(PXREASDK) backend, replay every frame through the same VR→IK pipeline
``motion_worker._vr_tick`` runs live, and report the numbers we'd need
to localise tremor in the arm:

* Input-side pose jitter (per-axis stddev) measured during quiescent
  sub-windows (grip held, controller "still" by dead-band threshold).
* Output-side jitter — same windows, after EMA + dead-band + remap.
* IK behaviour: how many solves were strict-6-DOF success, how many
  fell back to position-priority, how many got clamped to joint limits,
  how many came back unusable.
* Arm-freeze events: stretches where IK declared unusable, so the live
  worker would have pinned the arm.
* Effective target-update rate vs. raw frame rate (the gap is what
  the dead-band swallows).

No hardware, no UDP, no sockets. Pretends both arms start at zero
joints; that's a poor proxy for the real "VR enable" pose but the
relative-mode pipeline is shape-invariant — for the questions above
(jitter, IK failure rate, freezes) the home-pose seed is fine.

Usage:

    python3 -m openarm_controller_ui_lerobot.tools.analyze_vr_log_pxreasdk \\
        local_data/vr_recordings/vr_log_20260602_120137.jsonl \\
        [--csv out.csv] [--still-window-sec 1.0]
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import logging
import math
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pinocchio as pin

from openarm_controller_ui_lerobot import config
from openarm_controller_ui_lerobot.ik_solver import (
    CartesianIKSolver,
    IKResult,
    pose_to_xyzrpy,
)

logger = logging.getLogger(__name__)


_LEFT_JOINT_LIMITS = {
    "joint_1": (-75.0, +120.0),
    "joint_2": (-90.0, +9.0),
    "joint_3": (-85.0, +85.0),
    "joint_4": (0.0, +135.0),
    "joint_5": (-85.0, +85.0),
    "joint_6": (-40.0, +40.0),
    "joint_7": (-80.0, +80.0),
    "gripper": (-65.0, 0.0),
}
_RIGHT_JOINT_LIMITS = {
    "joint_1": (-75.0, +120.0),
    "joint_2": (-9.0, +90.0),
    "joint_3": (-85.0, +85.0),
    "joint_4": (0.0, +135.0),
    "joint_5": (-85.0, +85.0),
    "joint_6": (-40.0, +40.0),
    "joint_7": (-80.0, +80.0),
    "gripper": (-65.0, 0.0),
}


@dataclass
class Frame:
    """One Tracking frame from the recording, both controllers' fields."""
    line_no: int
    t: float                          # monotonic seconds (recorder)
    ts_ns: int                        # device timestamp, ns
    left_pose: np.ndarray             # (7,) tx ty tz qx qy qz qw
    right_pose: np.ndarray
    left_grip: float
    right_grip: float
    left_trigger: float
    right_trigger: float


def parse_log(path: Path) -> list[Frame]:
    frames: list[Frame] = []
    with path.open("r", encoding="utf-8") as f:
        for ln_no, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            raw = rec.get("raw")
            if not isinstance(raw, str):
                continue
            try:
                outer = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if outer.get("functionName") != "Tracking":
                continue
            inner_str = outer.get("value", "")
            try:
                inner = json.loads(inner_str)
            except json.JSONDecodeError:
                continue
            ctrl = inner.get("Controller") or {}
            left = ctrl.get("left") or {}
            right = ctrl.get("right") or {}

            def parse_pose(d: dict) -> np.ndarray | None:
                s = d.get("pose", "")
                try:
                    vals = [float(x) for x in s.split(",")]
                except ValueError:
                    return None
                if len(vals) < 7:
                    return None
                return np.array(vals[:7], dtype=float)

            lp = parse_pose(left)
            rp = parse_pose(right)
            if lp is None or rp is None:
                continue

            frames.append(Frame(
                line_no=ln_no,
                t=float(rec.get("t", 0.0)),
                ts_ns=int(inner.get("timeStampNs", 0)),
                left_pose=lp,
                right_pose=rp,
                left_grip=float(left.get("grip", 0.0)),
                right_grip=float(right.get("grip", 0.0)),
                left_trigger=float(left.get("trigger", 0.0)),
                right_trigger=float(right.get("trigger", 0.0)),
            ))
    frames.sort(key=lambda fr: fr.t)
    return frames


def pose_to_se3(pose: np.ndarray) -> pin.SE3:
    qx, qy, qz, qw = pose[3:7]
    n = math.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    if n < 1e-9:
        qx, qy, qz, qw = 0.0, 0.0, 0.0, 1.0
    else:
        qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    R = np.array([
        [1 - 2*(qy*qy + qz*qz),   2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),       1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),       2*(qy*qz + qx*qw),     1 - 2*(qx*qx + qy*qy)],
    ])
    t = pose[:3].astype(float)
    return pin.SE3(R, t)


@dataclass
class _Snap:
    controller: pin.SE3
    arm: pin.SE3


@dataclass
class ArmStats:
    """Accumulators for one arm across the whole replay."""
    arm: str
    n_frames: int = 0
    n_grip_engaged: int = 0
    n_snapshot_taken: int = 0
    n_dead_band_hold: int = 0          # engaged, but inside dead-band
    n_target_emitted: int = 0          # frames where motion_worker would update cart_target

    # IK outcomes among frames we ran a solve on.
    n_solves: int = 0
    n_strict_converged: int = 0
    n_position_priority: int = 0
    n_clamped: int = 0
    n_unusable: int = 0

    # Distribution of IK errors (only for solves we ran).
    pos_err_mm: list[float] = field(default_factory=list)
    rot_err_deg: list[float] = field(default_factory=list)
    iters: list[int] = field(default_factory=list)

    # Freeze events: contiguous runs of unusable frames while grip is engaged.
    freeze_events: list[tuple[float, float]] = field(default_factory=list)
    _freeze_open_t: float | None = None

    # Quiescent jitter: when the arm is engaged AND the dead-band says
    # "still", we collect raw and filtered SE3 translations to measure
    # noise floor on input vs output.
    quiet_raw_pos: list[np.ndarray] = field(default_factory=list)
    quiet_filt_pos: list[np.ndarray] = field(default_factory=list)
    quiet_raw_quat: list[np.ndarray] = field(default_factory=list)
    quiet_filt_quat: list[np.ndarray] = field(default_factory=list)


class ArmReplay:
    """Per-arm replay state — mirrors the relevant motion_worker fields.

    We model the relative ``_vr_tick`` path because that is the live
    default and the path that owns the EMA filter + hysteretic
    dead-band, which are the two primary suspects for tremor.
    """

    def __init__(self, arm: str, urdf_path: str,
                 vr_pos_scale: float, vr_rot_scale: float):
        self.arm = arm
        limits = _LEFT_JOINT_LIMITS if arm == "left" else _RIGHT_JOINT_LIMITS
        self.solver = CartesianIKSolver(urdf_path, arm, limits)
        # Pretend the arm starts at all-joints-zero — same simplification
        # as the existing replay tool.
        self.q_zero = [0.0] * 7
        self.home_tcp = self.solver.forward_kinematics(self.q_zero)

        self.snapshot: _Snap | None = None
        self.filt_pose: pin.SE3 | None = None
        self.moving: bool = False
        self.last_q_deg: list[float] | None = None
        self.last_target: pin.SE3 | None = None
        self.vr_pos_scale = vr_pos_scale
        self.vr_rot_scale = vr_rot_scale

        # remap matrices — same per-arm matrices the live worker uses.
        if arm == "left":
            t_src = config.VR_TRANSLATION_REMAP_LEFT
            r_src = config.VR_ROTATION_REMAP_LEFT
        else:
            t_src = config.VR_TRANSLATION_REMAP_RIGHT
            r_src = config.VR_ROTATION_REMAP_RIGHT
        self._M_t = np.array(t_src, dtype=float)
        self._M_r = np.array(r_src, dtype=float)

        self.stats = ArmStats(arm=arm)

    def _filter(self, raw: pin.SE3) -> pin.SE3:
        """Mirror of MotionWorker._filter_pose."""
        alpha = config.VR_POSE_FILTER_ALPHA
        if alpha >= 1.0:
            self.filt_pose = raw
            return raw
        prev = self.filt_pose
        if prev is None:
            self.filt_pose = raw
            return raw
        t_new = (1.0 - alpha) * prev.translation + alpha * raw.translation
        R_delta = raw.rotation @ prev.rotation.T
        try:
            w = pin.log3(R_delta)
        except Exception:
            self.filt_pose = raw
            return raw
        R_step = pin.exp3(alpha * w)
        R_new = R_step @ prev.rotation
        filt = pin.SE3(R_new, t_new)
        self.filt_pose = filt
        return filt

    def feed(self, frame: Frame, raw_pose: np.ndarray, grip: float) -> dict:
        """Run one frame through the pipeline; update stats; return a
        per-frame metrics dict suitable for CSV.
        """
        s = self.stats
        s.n_frames += 1
        out: dict = {
            "line_no": frame.line_no,
            "t": frame.t,
            "arm": self.arm,
            "grip": grip,
            "engaged": False,
            "snapshot": False,
            "moving": False,
            "target_x": None, "target_y": None, "target_z": None,
            "ik_pos_err_mm": None,
            "ik_rot_err_deg": None,
            "ik_iters": None,
            "ik_usable": None,
            "ik_clamped": None,
            "ik_pos_priority": None,
            "ik_strict_conv": None,
        }

        engaged = grip >= config.VR_GRIP_ENABLE_THRESHOLD
        out["engaged"] = engaged

        if not engaged:
            # released → drop snapshot + filter, freeze arm
            self.snapshot = None
            self.filt_pose = None
            self.moving = False
            # Closing freeze-event if one was open.
            if s._freeze_open_t is not None:
                s.freeze_events.append((s._freeze_open_t, frame.t))
                s._freeze_open_t = None
            return out

        s.n_grip_engaged += 1
        ctrl_raw = pose_to_se3(raw_pose)
        ctrl_filt = self._filter(ctrl_raw)

        if self.snapshot is None:
            # rising edge: snap + emit (target == arm_snapshot)
            self.snapshot = _Snap(controller=ctrl_filt, arm=self.home_tcp)
            self.moving = False
            s.n_snapshot_taken += 1
            out["snapshot"] = True
            target = self.home_tcp
            s.n_target_emitted += 1
        else:
            snap = self.snapshot
            delta = snap.controller.actInv(ctrl_filt)
            trans_norm = float(np.linalg.norm(delta.translation))
            rot_norm = float(np.linalg.norm(pin.log3(delta.rotation)))
            moving = self.moving
            if moving:
                if (trans_norm < config.VR_DEAD_BAND_POS_M_IN
                        and rot_norm < config.VR_DEAD_BAND_ROT_RAD_IN):
                    moving = False
            else:
                if (trans_norm > config.VR_DEAD_BAND_POS_M_OUT
                        or rot_norm > config.VR_DEAD_BAND_ROT_RAD_OUT):
                    moving = True
            self.moving = moving
            out["moving"] = moving

            if not moving:
                # Stay pinned to the snapshot — quiescent.
                target = snap.arm
                s.n_dead_band_hold += 1
                # quiescent input/output samples for jitter floor
                s.quiet_raw_pos.append(ctrl_raw.translation.copy())
                s.quiet_filt_pos.append(ctrl_filt.translation.copy())
                s.quiet_raw_quat.append(raw_pose[3:7].copy())
                # extract a rough "filt quat" by taking log3 of filt R
                # (we just want a stddev of rotation noise; magnitudes
                # across frames are what matters).
                try:
                    w_filt = pin.log3(ctrl_filt.rotation)
                except Exception:
                    w_filt = np.zeros(3)
                s.quiet_filt_quat.append(w_filt.copy())
                # Hold-target frames don't trigger a new IK solve in
                # the live code (cart_target stays equal). But the live
                # worker still calls _cartesian_tick → solver.solve()
                # every tick. We replicate to surface IK churn that
                # would happen in practice.
            else:
                # Build the same target the live worker would.
                if self.vr_pos_scale != 1.0:
                    t_scaled = delta.translation * self.vr_pos_scale
                else:
                    t_scaled = delta.translation
                if self.vr_rot_scale != 1.0:
                    w = pin.log3(delta.rotation)
                    R_scaled = pin.exp3(w * self.vr_rot_scale)
                else:
                    R_scaled = delta.rotation
                delta = pin.SE3(R_scaled, t_scaled)
                t_new = self._M_t @ delta.translation
                R_new = self._M_r @ delta.rotation @ self._M_r.T
                delta = pin.SE3(R_new, t_new)
                R_snap = snap.arm.rotation
                delta = pin.SE3(R_snap.T @ delta.rotation @ R_snap,
                                R_snap.T @ delta.translation)
                target = snap.arm * delta
                s.n_target_emitted += 1

        # IK solve
        seed = self.last_q_deg if self.last_q_deg is not None else self.q_zero
        try:
            ik: IKResult = self.solver.solve(target, q_seed_deg_7=seed)
        except Exception as e:
            out["error"] = repr(e)
            return out

        s.n_solves += 1
        if ik.usable:
            self.last_q_deg = list(ik.q_deg)
            if s._freeze_open_t is not None:
                s.freeze_events.append((s._freeze_open_t, frame.t))
                s._freeze_open_t = None
        else:
            s.n_unusable += 1
            if s._freeze_open_t is None:
                s._freeze_open_t = frame.t
        if ik.clamped:
            s.n_clamped += 1
        if ik.position_priority_used:
            s.n_position_priority += 1
        if ik.converged:
            s.n_strict_converged += 1
        s.pos_err_mm.append(ik.pos_err_mm)
        s.rot_err_deg.append(ik.rot_err_deg)
        s.iters.append(ik.iters)
        x, y, z, _, _, _ = pose_to_xyzrpy(target)
        out["target_x"], out["target_y"], out["target_z"] = x, y, z
        out["ik_pos_err_mm"] = ik.pos_err_mm
        out["ik_rot_err_deg"] = ik.rot_err_deg
        out["ik_iters"] = ik.iters
        out["ik_usable"] = bool(ik.usable)
        out["ik_clamped"] = bool(ik.clamped)
        out["ik_pos_priority"] = bool(ik.position_priority_used)
        out["ik_strict_conv"] = bool(ik.converged)
        self.last_target = target
        return out


def _stats_for(values: list[float]) -> tuple[float, float, float, float]:
    """Return (median, mean, p95, max). Empty → all zeros."""
    if not values:
        return 0.0, 0.0, 0.0, 0.0
    s = sorted(values)
    n = len(s)
    median = s[n // 2]
    mean = sum(s) / n
    p95 = s[max(0, int(round(0.95 * (n - 1))))]
    return median, mean, p95, s[-1]


def _per_axis_stddev(samples: list[np.ndarray]) -> tuple[float, float, float]:
    if len(samples) < 2:
        return 0.0, 0.0, 0.0
    arr = np.stack(samples, axis=0)
    return tuple(float(x) for x in arr.std(axis=0))  # type: ignore[return-value]


def print_summary(arms: dict[str, ArmReplay], duration: float, n_frames: int) -> None:
    print()
    print("=" * 78)
    print(f"VR replay summary  ({n_frames} frames, {duration:.2f}s wall, "
          f"{n_frames / max(duration, 1e-6):.1f} Hz aggregate)")
    print("=" * 78)
    for name, ar in arms.items():
        s = ar.stats
        print()
        print(f"  Arm {name.upper():5}  ({s.n_frames} frames)")
        engaged_pct = 100.0 * s.n_grip_engaged / max(s.n_frames, 1)
        print(f"    grip engaged frames     : {s.n_grip_engaged:6d} "
              f"({engaged_pct:5.1f}%)")
        if s.n_grip_engaged:
            hold_pct = 100.0 * s.n_dead_band_hold / s.n_grip_engaged
            emit_pct = 100.0 * s.n_target_emitted / s.n_grip_engaged
            print(f"    snapshots (rising edges): {s.n_snapshot_taken:6d}")
            print(f"    dead-band holds (engaged): {s.n_dead_band_hold:5d} "
                  f"({hold_pct:5.1f}% of engaged)")
            print(f"    target emitted to IK    : {s.n_target_emitted:6d} "
                  f"({emit_pct:5.1f}% of engaged)")
        if s.n_solves:
            print(f"    IK solves               : {s.n_solves:6d}")
            print(f"    IK strict-converged     : {s.n_strict_converged:6d}  "
                  f"({100*s.n_strict_converged/s.n_solves:5.1f}%)")
            print(f"    IK position-priority    : {s.n_position_priority:6d}  "
                  f"({100*s.n_position_priority/s.n_solves:5.1f}%)")
            print(f"    IK clamped              : {s.n_clamped:6d}  "
                  f"({100*s.n_clamped/s.n_solves:5.1f}%)")
            print(f"    IK unusable (would freeze): {s.n_unusable:4d}  "
                  f"({100*s.n_unusable/s.n_solves:5.1f}%)")
            med, mean, p95, mx = _stats_for(s.pos_err_mm)
            print(f"    pos_err mm  med={med:7.3f}  mean={mean:7.3f}  "
                  f"p95={p95:7.3f}  max={mx:7.3f}")
            med, mean, p95, mx = _stats_for(s.rot_err_deg)
            print(f"    rot_err deg med={med:6.3f}  mean={mean:6.3f}  "
                  f"p95={p95:6.3f}  max={mx:6.3f}")
            med, mean, p95, mx = _stats_for([float(x) for x in s.iters])
            print(f"    iters       med={med:6.1f}  mean={mean:6.2f}  "
                  f"p95={p95:6.1f}  max={mx:6.0f}")

        # Freeze events: stretches of consecutive unusable frames.
        # Filter to "noticeable" ones (≥2 frames, ≈ ≥22ms at 90 Hz).
        durations = [(b - a) for (a, b) in s.freeze_events]
        long_freezes = [d for d in durations if d > 0.05]  # ≥50 ms
        print(f"    freeze events           : {len(s.freeze_events):6d}"
              f"  (>{50}ms: {len(long_freezes)})")
        if durations:
            med, mean, p95, mx = _stats_for(durations)
            print(f"    freeze dur sec med={med:.3f}  mean={mean:.3f}  "
                  f"p95={p95:.3f}  max={mx:.3f}")

        if s.quiet_raw_pos:
            ax_raw = _per_axis_stddev(s.quiet_raw_pos)
            ax_filt = _per_axis_stddev(s.quiet_filt_pos)
            print(f"    quiescent samples (engaged + in-deadband): {len(s.quiet_raw_pos)}")
            print(f"    raw  pos stddev (mm)  x={ax_raw[0]*1000:6.2f} "
                  f"y={ax_raw[1]*1000:6.2f} z={ax_raw[2]*1000:6.2f}")
            print(f"    filt pos stddev (mm)  x={ax_filt[0]*1000:6.2f} "
                  f"y={ax_filt[1]*1000:6.2f} z={ax_filt[2]*1000:6.2f}")
            ax_raw_q = _per_axis_stddev(s.quiet_raw_quat)
            ax_filt_q = _per_axis_stddev(s.quiet_filt_quat)
            print(f"    raw  quat stddev      x={ax_raw_q[0]:.5f} "
                  f"y={ax_raw_q[1]:.5f} z={ax_raw_q[2]:.5f} "
                  f"w={ax_raw_q[3] if len(ax_raw_q)>3 else 0:.5f}")
            # filt rotation noise expressed as log3 stddev (radians):
            print(f"    filt rot stddev (deg) x={math.degrees(ax_filt_q[0]):6.3f} "
                  f"y={math.degrees(ax_filt_q[1]):6.3f} "
                  f"z={math.degrees(ax_filt_q[2]):6.3f}")


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    # union of all keys (some rows may have an "error")
    extra = {k for r in rows for k in r.keys()}
    for k in extra:
        if k not in fieldnames:
            fieldnames.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s: %(message)s")
    p = argparse.ArgumentParser(
        description="Replay a PXREASDK VR log through the live VR→IK "
                    "pipeline and report jitter / IK / freeze metrics."
    )
    p.add_argument("input", type=Path)
    p.add_argument("--vr-pos-scale", type=float,
                   default=config.INITIAL_VR_POS_SCALE)
    p.add_argument("--vr-rot-scale", type=float,
                   default=config.INITIAL_VR_ROT_SCALE)
    p.add_argument("--urdf", type=Path, default=Path(config.GRAVITY_URDF_PATH))
    p.add_argument("--csv", type=Path, default=None)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    if not args.input.exists():
        logger.error(f"input not found: {args.input}")
        return 2

    frames = parse_log(args.input)
    if not frames:
        logger.error("no Tracking frames parsed")
        return 1
    duration = frames[-1].t - frames[0].t
    logger.info(
        f"Loaded {len(frames)} Tracking frames ({duration:.2f}s wall, "
        f"{len(frames)/max(duration,1e-6):.1f} Hz aggregate). "
        f"Pipeline: filter alpha={config.VR_POSE_FILTER_ALPHA}, "
        f"db_pos OUT={config.VR_DEAD_BAND_POS_M_OUT*1000:.1f}mm "
        f"IN={config.VR_DEAD_BAND_POS_M_IN*1000:.1f}mm, "
        f"db_rot OUT={math.degrees(config.VR_DEAD_BAND_ROT_RAD_OUT):.2f}° "
        f"IN={math.degrees(config.VR_DEAD_BAND_ROT_RAD_IN):.2f}°, "
        f"grip>={config.VR_GRIP_ENABLE_THRESHOLD}, "
        f"vr_pos_scale={args.vr_pos_scale}, vr_rot_scale={args.vr_rot_scale}"
    )

    # Inter-frame interval distribution → freeze-from-input check.
    intervals = [frames[i].t - frames[i-1].t for i in range(1, len(frames))]
    med, mean, p95, mx = _stats_for(intervals)
    logger.info(
        f"frame interval (ms): med={med*1000:.2f}  mean={mean*1000:.2f}  "
        f"p95={p95*1000:.2f}  max={mx*1000:.2f}  "
        f"(>=100ms gaps: {sum(1 for x in intervals if x>=0.1)})"
    )

    arms = {
        "left":  ArmReplay("left",  str(args.urdf),
                           args.vr_pos_scale, args.vr_rot_scale),
        "right": ArmReplay("right", str(args.urdf),
                           args.vr_pos_scale, args.vr_rot_scale),
    }
    rows: list[dict] = []
    for fr in frames:
        rows.append(arms["left"].feed(fr, fr.left_pose, fr.left_grip))
        rows.append(arms["right"].feed(fr, fr.right_pose, fr.right_grip))

    print_summary(arms, duration, len(frames))

    if args.csv is not None:
        write_csv(rows, args.csv)
        logger.info(f"Wrote per-frame CSV: {args.csv}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
