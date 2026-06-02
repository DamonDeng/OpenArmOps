"""Per-arm cartesian IK using Pinocchio.

Iterative damped least-squares (Levenberg–Marquardt) solver. Given a target
SE(3) pose for the end-effector frame (``openarm_{arm}_hand_tcp``), returns
a 7-vector of joint angles in degrees that places the arm's EE at that pose
— or raises on convergence failure.

Design notes:

- **One ``Model`` / ``Data`` per arm.** Same URDF as gravity comp, but each
  solver focuses on its own arm's joint indices; the other arm's q is held
  at zero since it doesn't affect this arm's kinematics.

- **Seeded IK.** Each call accepts a ``q_seed`` (7 joint angles in degrees)
  so the solver starts close to the previous solution. At 30 Hz teleop the
  pose delta per tick is tiny and IK converges in 1–3 iterations.

- **Joint limit enforcement.** The solver itself does not know about joint
  limits. We clamp the returned q to each joint's software limits after
  convergence. If the clamp moves q significantly, we report it via the
  return-tuple's ``clamped`` flag so the caller can warn / freeze.

- **Damping.** Damped pseudo-inverse avoids blowing up near singularities.
  Damping coefficient is a constant (``_DAMPING``); rare tuning.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pinocchio as pin

logger = logging.getLogger(__name__)

_DAMPING = 1e-6
# Raised from 40 to 80 on 2026-06-02. Bench (replay_vr_log_20260602_120137,
# 2000 right-arm targets at vr_scale 1.0/1.0): 663 µs/solve at cap=40 vs
# 1236 µs at cap=80 — still well inside the 33 ms motion-worker budget.
# The extra iterations rescue ~1% of genuinely-hard cases that 40 iters
# couldn't reach; the rest converge in <10 iters either way.
_MAX_ITERS = 80
_POS_TOL = 1e-4      # 0.1 mm  — strict 6-DOF convergence
_ROT_TOL = 1e-3      # ~0.06°  — strict 6-DOF convergence
_STEP_ALPHA = 1.0    # full step per iteration

# "Usable" tolerances. Position is the primary criterion — if the
# arm got within this much of the requested position, the result
# is "usable" and the caller should drive the motors. Rotation
# tolerance is split into two bands depending on whether we used
# the position-priority pass:
#  - strict-only result: rot_err must be small (full 6-DOF success)
#  - position-priority result: rotation deliberately abandoned, so
#    we don't gate on it — just position and a generous safety
#    ceiling so a fully-broken result still gets caught.
_USABLE_POS_TOL = 10e-3                      # 10 mm position error
# Raised from 5 mm to 10 mm on 2026-06-02 after live testing showed the
# old threshold flagged a long tail of "barely missed" solves as
# unusable: pos_err in the 5-8 mm range with rot_err ≈ 0° (strict
# converged, just shy of the cutoff). 5-8 mm is below human grasping
# precision, so freezing the arm there feels like tremor rather than
# a real reachability problem. 10 mm still gates anything visibly
# off-target while clearing the noise band (offline replay p95 was
# 5.7 mm, so this admits the long tail without admitting genuine
# unreachables which sit in the 100 mm+ range).
_USABLE_ROT_TOL_STRICT = math.radians(15.0)  # 15° rot if strict was used
_USABLE_ROT_TOL_RELAXED = math.radians(120.0)  # generous ceiling when
                                                # position-priority is in
                                                # play (only catches truly
                                                # unreachable wrist poses)

# Boundary-clamp fallback (pass 3). Runs when both strict and
# position-priority passes fail "usable" — the requested pose is past
# the arm's workspace. Instead of freezing, walk the position back
# along the line from current EE toward the requested target while
# holding rotation strict, and pick the largest fraction t ∈ (0, 1]
# that's reachable. Result: arm stretches toward the target with
# correct wrist orientation, stopping at the workspace edge rather
# than freezing mid-motion.
_BOUNDARY_BISECT_ITERS = 5    # 5 IK calls → t precision ~0.03 (~3% of delta)
_BOUNDARY_MIN_DELTA_M = 1e-3  # if requested - current is <1 mm, the issue
                              # isn't position reach (probably orientation
                              # or seed); skip the fallback.


@dataclass
class IKResult:
    q_deg: list[float]      # 7 joint angles
    converged: bool         # strict 6-DOF convergence (both pos AND rot tight)
    iters: int
    pos_err_mm: float
    rot_err_deg: float
    clamped: bool           # True if joint limits modified the solution
    # Whether the result is good enough to send to motors. True if the
    # POSITION error is within _USABLE_POS_TOL even when the strict
    # convergence test failed — caller can check this rather than
    # `converged` to decide between "drive the arm" and "freeze".
    usable: bool = False
    # Was a position-priority fallback pass run? Useful for telemetry
    # and so the UI can mention "orientation relaxed to reach target".
    position_priority_used: bool = False
    # True when both strict and position-priority were unusable, and the
    # boundary bisection (pass 3) returned the largest reachable
    # fraction along the line from current EE to the requested position.
    # The arm can still be driven (orientation is strict, position is
    # the partial target) — the caller should treat this as a soft
    # "workspace edge" rather than a freeze. The remaining distance to
    # the original requested position is in pos_err_mm.
    boundary_clamped: bool = False


class CartesianIKSolver:
    """Numerical IK for a single arm in a shared bimanual URDF."""

    def __init__(
        self,
        urdf_path: str,
        arm_side: str,
        joint_limits: dict[str, tuple[float, float]],
    ) -> None:
        """
        arm_side: 'left' or 'right'. Determines which 7 joints and which TCP
                  frame the solver operates on.
        joint_limits: per-joint (lo_deg, hi_deg) for joint_1..joint_7. Same
                  shape as OpenArmFollowerConfig.joint_limits. Gripper
                  ignored (not part of the 7-DOF arm kinematics).
        """
        self.arm_side = arm_side
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()

        self._joint_urdf_names = [f"openarm_{arm_side}_joint{i+1}" for i in range(7)]
        self._q_indices = [self.model.idx_qs[self.model.getJointId(n)]
                           for n in self._joint_urdf_names]
        self._v_indices = [self.model.idx_vs[self.model.getJointId(n)]
                           for n in self._joint_urdf_names]

        tcp_name = f"openarm_{arm_side}_hand_tcp"
        self._tcp_fid = self.model.getFrameId(tcp_name)
        if self._tcp_fid >= self.model.nframes:
            raise ValueError(f"frame {tcp_name!r} not found in URDF")

        # Per-joint limits for j1..j7 in degrees.
        self._limits_deg = [
            joint_limits[f"joint_{i+1}"] for i in range(7)
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def forward_kinematics(self, q_deg_7: list[float]) -> pin.SE3:
        """Return the TCP SE(3) placement for the given 7 joint angles."""
        q = np.zeros(self.model.nq)
        for i, v in enumerate(q_deg_7):
            q[self._q_indices[i]] = np.radians(v)
        pin.framesForwardKinematics(self.model, self.data, q)
        return self.data.oMf[self._tcp_fid].copy()

    def solve(
        self,
        target_pose: pin.SE3,
        q_seed_deg_7: list[float],
        boundary_fallback: bool = True,
    ) -> IKResult:
        """Solve IK for a single target. Position is the priority metric.

        Strategy:
          1. Run strict 6-DOF damped least squares.
          2. If strict's POSITION error is already within _USABLE_POS_TOL
             AND its rotation error is within strict tolerance, that's
             the best possible outcome — full 6-DOF success. Return.
          3. Otherwise run the position-priority pass (3-row Jacobian,
             4-D null space) from the original user seed.
          4. Pick the pass with lower POSITION error. (Position is the
             user's primary input — translations come from controller
             deltas; rotation often comes along passively. We never
             pick a result with a worse position just because its
             rotation was a bit better.)
          5. If both passes left position past _USABLE_POS_TOL and
             ``boundary_fallback`` is True, run pass 3 — bisection on
             t in (0, 1] for partial-position targets — and return the
             largest reachable fraction with strict orientation. With
             ``boundary_fallback=False`` this step is skipped and the
             arm freezes on unreachable targets (pre-pass-3 behavior).
        """
        q = np.zeros(self.model.nq)
        for i, v in enumerate(q_seed_deg_7):
            q[self._q_indices[i]] = np.radians(v)

        # Pass 1: strict 6-DOF.
        q_strict, pos_err_strict, rot_err_strict, iters_strict = self._iterate(
            q.copy(), target_pose, position_only=False,
        )
        strict_pos_ok = pos_err_strict < _USABLE_POS_TOL
        strict_full_success = (strict_pos_ok
                               and rot_err_strict < _USABLE_ROT_TOL_STRICT)

        if strict_full_success:
            # Best case: 6-DOF target genuinely reachable.
            return self._finalize_result(
                q_strict, pos_err_strict, rot_err_strict, iters_strict,
                converged=(pos_err_strict < _POS_TOL
                           and rot_err_strict < _ROT_TOL),
                position_priority_used=False,
                target_pose=target_pose,
            )

        # Pass 2: position-priority. Always run when strict didn't
        # achieve a full 6-DOF success — even if strict's position
        # was tight but rotation was off, the relaxed pass may find
        # a different joint config with better OR comparable position
        # and acceptable rotation drift, which is the user-preferred
        # outcome.
        q_seed_arr = np.zeros(self.model.nq)
        for i, v in enumerate(q_seed_deg_7):
            q_seed_arr[self._q_indices[i]] = np.radians(v)
        q_relaxed, pos_err_relaxed, rot_err_relaxed, iters_relaxed = self._iterate(
            q_seed_arr, target_pose, position_only=True,
        )

        # Pick the pass with the lower position error. Strict is only
        # preferred when its position is meaningfully better than the
        # relaxed pass's; otherwise we use relaxed even with rotation
        # drift, since position is the primary criterion the user
        # gives us.
        strict_meaningfully_better = pos_err_strict + 1e-4 < pos_err_relaxed

        # Pick which of the first two passes is the better candidate.
        # Used both for the early-return path and as the fallback if
        # the boundary bisection (pass 3) finds nothing.
        if strict_meaningfully_better:
            best_q = q_strict
            best_pos_err = pos_err_strict
            best_rot_err = rot_err_strict
            best_iters = iters_strict
            best_pos_priority = False
        else:
            best_q = q_relaxed
            best_pos_err = pos_err_relaxed
            best_rot_err = rot_err_relaxed
            best_iters = iters_strict + iters_relaxed
            best_pos_priority = True

        # If either of the first two passes was good enough on
        # POSITION, we don't need the boundary bisection — usability
        # already passes (see _finalize_result). Return now.
        if best_pos_err < _USABLE_POS_TOL:
            return self._finalize_result(
                best_q, best_pos_err, best_rot_err, best_iters,
                converged=False, position_priority_used=best_pos_priority,
                target_pose=target_pose,
            )

        # Pass 3: boundary bisection. Both strict and relaxed left
        # position too far from target — the request is past the
        # workspace. Walk the position back along (current_ee → target)
        # while holding orientation strict, and find the largest t in
        # (0, 1] that gives a usable strict-6DOF solve. Result is the
        # arm extending toward the user's hand, stopping at the edge.
        # Skipped when boundary_fallback=False (caller wants the
        # pre-pass-3 freeze-on-unreachable behavior).
        boundary_result = (
            self._boundary_bisect(q.copy(), q_seed_arr.copy(), target_pose)
            if boundary_fallback else None
        )
        if boundary_result is not None:
            q_b, pos_err_b, rot_err_b, iters_b = boundary_result
            return self._finalize_result(
                q_b, pos_err_b, rot_err_b,
                iters_strict + iters_relaxed + iters_b,
                converged=False, position_priority_used=False,
                target_pose=target_pose,
                boundary_clamped=True,
            )

        # Boundary bisection found nothing reachable along the line
        # — fall back to whichever of the first two passes had lower
        # position error. Caller will see usable=False and freeze.
        return self._finalize_result(
            best_q, best_pos_err, best_rot_err, best_iters,
            converged=False, position_priority_used=best_pos_priority,
            target_pose=target_pose,
        )

    # ------------------------------------------------------------------
    # Internal: one iterative DLS pass.
    # ------------------------------------------------------------------
    def _iterate(
        self,
        q: np.ndarray,
        target_pose: pin.SE3,
        position_only: bool = False,
    ) -> tuple[np.ndarray, float, float, int]:
        """Run the damped-least-squares iteration on a copy of q. Returns
        (final_q, pos_err_m, rot_err_rad, iters_used).

        Two modes:
          - position_only=False (default, strict): solve full 6-DOF.
            All 6 rows of the Jacobian are used; rotation tracking is
            enforced.
          - position_only=True: solve only the 3 translation rows of
            the system. Rotation is completely unconstrained, leaving
            4 dimensions of null-space for the 7-DOF arm to use to
            satisfy joint limits and reach the target position. This
            is what handles "user pushed X+5cm from zero pose, but
            the elbow joint limit prevents pure-X motion at that
            orientation" — the wrist is allowed to rotate freely so
            the elbow can lift to reach forward.

        Joint-limit avoidance: at every iteration we project the joint
        update step away from joints already at their limits, by
        zeroing dq components that would push deeper into the limit.
        Without this, the iteration parks against the limit and gets
        stuck even though there's still a reachable solution nearby.
        """
        pos_err_m = rot_err_rad = 0.0
        # Snapshot limits in radians for fast per-iter checks.
        limits_rad = np.array(
            [(np.radians(lo), np.radians(hi)) for (lo, hi) in self._limits_deg],
            dtype=float,
        )
        for it in range(_MAX_ITERS):
            pin.framesForwardKinematics(self.model, self.data, q)
            current = self.data.oMf[self._tcp_fid]
            err_local = pin.log(current.actInv(target_pose)).vector
            pos_err_m = float(np.linalg.norm(err_local[:3]))
            rot_err_rad = float(np.linalg.norm(err_local[3:]))
            if pos_err_m < _POS_TOL and rot_err_rad < _ROT_TOL:
                break

            J = pin.computeFrameJacobian(
                self.model, self.data, q, self._tcp_fid,
                pin.ReferenceFrame.LOCAL,
            )
            J_arm = J[:, self._v_indices]

            if position_only:
                # Use only the 3 translation rows. With 7 DOF this is
                # an under-determined system → there's null-space to
                # spare for joint-limit avoidance.
                J_used = J_arm[:3, :]
                err_used = err_local[:3]
                rows = 3
            else:
                J_used = J_arm
                err_used = err_local
                rows = 6

            JJt = J_used @ J_used.T
            damped = JJt + _DAMPING * np.eye(rows)
            dq_arm = J_used.T @ np.linalg.solve(damped, err_used)

            # Joint-limit avoidance: zero out dq components that would
            # push a joint deeper into its limit. If joint_4 is already
            # at its lower limit (0 rad) and dq_arm[3] is negative, the
            # boundary projection masks that step away. Without this,
            # the solver pins joints against limits and stops moving.
            for k, idx in enumerate(self._v_indices):
                lo, hi = limits_rad[k]
                cur = q[idx]
                if cur <= lo and dq_arm[k] < 0:
                    dq_arm[k] = 0.0
                elif cur >= hi and dq_arm[k] > 0:
                    dq_arm[k] = 0.0

            for k, idx in enumerate(self._v_indices):
                q[idx] += _STEP_ALPHA * dq_arm[k]
                # Hard clamp inside the iteration loop too — avoids
                # ever leaving the legal region (the gradient logic
                # above only zeros steps, but rounding and the fixed
                # alpha can still nudge a joint slightly past). This
                # also makes the post-hoc clamp in _finalize_result
                # mostly a no-op for in-loop solutions.
                lo, hi = limits_rad[k]
                if q[idx] < lo:
                    q[idx] = lo
                elif q[idx] > hi:
                    q[idx] = hi

        return q, pos_err_m, rot_err_rad, it + 1

    def _boundary_bisect(
        self,
        q_seed: np.ndarray,
        q_seed_orig: np.ndarray,
        target_pose: pin.SE3,
    ) -> Optional[tuple[np.ndarray, float, float, int]]:
        """Pass 3: line-search the largest reachable fraction toward the
        target while holding orientation strict.

        Returns (q, pos_err_to_partial_target, rot_err_to_partial_target,
        total_iters) for the best found t, or None if no t > 0 produces
        a usable strict-6DOF solve.

        ``q_seed`` is the joint config to use both for FK (current EE)
        and as the warm-start for the inner IK calls. We use the seed
        rather than the strict-pass result because the strict pass
        likely converged to a contorted joint config trying to reach
        an unreachable pose.
        """
        # Current EE pose (FK on seed q).
        pin.framesForwardKinematics(self.model, self.data, q_seed)
        current_pose = self.data.oMf[self._tcp_fid].copy()
        delta = target_pose.translation - current_pose.translation
        if float(np.linalg.norm(delta)) < _BOUNDARY_MIN_DELTA_M:
            # Position barely moved — bisection has nothing to do.
            return None

        target_rot = target_pose.rotation
        total_iters = 0
        # Bisection on t ∈ (0, 1]. lo is the largest known-reachable
        # fraction, hi is the smallest known-unreachable. Start by
        # probing t=1 (already known unreachable from passes 1+2) and
        # bisect downward. Result of each probe = (q, pos_err, rot_err).
        lo = 0.0
        hi = 1.0
        best: Optional[tuple[np.ndarray, float, float]] = None
        # First probe at t=0.5, then bisect for _BOUNDARY_BISECT_ITERS-1
        # rounds. With 5 total probes we narrow t to ~1/32 of the delta.
        t = 0.5
        for _ in range(_BOUNDARY_BISECT_ITERS):
            partial_pos = current_pose.translation + t * delta
            partial_pose = pin.SE3(target_rot, partial_pos)
            q_try, pos_err, rot_err, iters = self._iterate(
                q_seed_orig.copy(), partial_pose, position_only=False,
            )
            total_iters += iters
            usable = (pos_err < _USABLE_POS_TOL
                      and rot_err < _USABLE_ROT_TOL_STRICT)
            if usable:
                best = (q_try, pos_err, rot_err)
                lo = t
                t = (t + hi) / 2.0
            else:
                hi = t
                t = (lo + t) / 2.0

        if best is None:
            return None
        q_b, pos_err_b, rot_err_b = best
        return q_b, pos_err_b, rot_err_b, total_iters

    def _finalize_result(
        self,
        q: np.ndarray,
        pos_err_m: float,
        rot_err_rad: float,
        iters: int,
        converged: bool,
        position_priority_used: bool,
        target_pose: pin.SE3,
        boundary_clamped: bool = False,
    ) -> IKResult:
        """Common tail: extract our 7 joint angles, clamp to limits,
        re-evaluate position error after clamp, mark ``usable``, build
        the IKResult.
        """
        q_deg = [float(np.degrees(q[self._q_indices[i]])) for i in range(7)]

        clamped = False
        for i, (lo, hi) in enumerate(self._limits_deg):
            if q_deg[i] < lo:
                q_deg[i] = lo
                clamped = True
            elif q_deg[i] > hi:
                q_deg[i] = hi
                clamped = True

        # If we clamped, the position error from inside the iteration
        # no longer reflects what the motors will actually produce.
        # Re-run FK on the clamped angles and recompute the residual
        # against the *target* — only that error matters for "usable".
        # Boundary-clamped solutions also recompute against the
        # original target (passed in `target_pose`) so pos_err_mm
        # reports remaining distance to the user's requested point,
        # not zero (the partial point that the bisection used).
        if clamped or boundary_clamped:
            q_full = np.zeros(self.model.nq)
            for i, deg in enumerate(q_deg):
                q_full[self._q_indices[i]] = np.radians(deg)
            pin.framesForwardKinematics(self.model, self.data, q_full)
            current = self.data.oMf[self._tcp_fid]
            err_local = pin.log(current.actInv(target_pose)).vector
            pos_err_m = float(np.linalg.norm(err_local[:3]))
            rot_err_rad = float(np.linalg.norm(err_local[3:]))
            # Strict 6-DOF "converged" no longer applies after a clamp
            # or after a boundary truncation. Recompute against the
            # strict thresholds.
            converged = (pos_err_m < _POS_TOL and rot_err_rad < _ROT_TOL)

        # Usability rule:
        #   - Position is the primary criterion (must be within tolerance).
        #   - If we used position-priority, the relaxed rot tolerance
        #     applies — rotation residual is mostly informational
        #     because that pass deliberately ignored it.
        #   - If strict was used, the tighter rot tolerance applies
        #     because strict was supposed to satisfy rotation too.
        #   - Boundary-clamped: solution is usable by definition
        #     (the bisection only kept passes that converged with
        #     strict tolerances on the partial target). pos_err_mm
        #     here measures distance to the *original* target, which
        #     is expected to exceed _USABLE_POS_TOL — that's the whole
        #     point of this fallback. Force-mark usable so the caller
        #     drives the arm to the boundary.
        if boundary_clamped:
            usable = True
        else:
            rot_tol = (_USABLE_ROT_TOL_RELAXED if position_priority_used
                       else _USABLE_ROT_TOL_STRICT)
            usable = (pos_err_m < _USABLE_POS_TOL and rot_err_rad < rot_tol)

        return IKResult(
            q_deg=q_deg,
            converged=converged,
            iters=iters,
            pos_err_mm=pos_err_m * 1000.0,
            rot_err_deg=float(np.degrees(rot_err_rad)),
            clamped=clamped,
            usable=usable,
            position_priority_used=position_priority_used,
            boundary_clamped=boundary_clamped,
        )


# ---------------------------------------------------------------------------
# Helpers for working with SE(3) pose from/to (x,y,z, roll, pitch, yaw).
# ---------------------------------------------------------------------------
def pose_from_xyzrpy(
    x: float, y: float, z: float,
    roll: float, pitch: float, yaw: float,
) -> pin.SE3:
    """Build an SE(3) pose from translation (m) and intrinsic RPY (rad).

    Rotation convention: ZYX intrinsic (yaw then pitch then roll) — common
    in robotics and what MoveIt's pose_stamped uses by default. If you
    change this, change pose_to_xyzrpy accordingly.
    """
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    # R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    R = np.array([
        [cy*cp,  cy*sp*sr - sy*cr,  cy*sp*cr + sy*sr],
        [sy*cp,  sy*sp*sr + cy*cr,  sy*sp*cr - cy*sr],
        [-sp,    cp*sr,             cp*cr            ],
    ])
    return pin.SE3(R, np.array([x, y, z]))


def pose_to_xyzrpy(pose: pin.SE3) -> tuple[float, float, float, float, float, float]:
    """Inverse of pose_from_xyzrpy. Returns (x, y, z, roll, pitch, yaw) in m / rad.

    Assumes the rotation is not at gimbal lock (pitch != ±π/2). If it is,
    roll+yaw become ambiguous and we pick an arbitrary decomposition.
    """
    R = pose.rotation
    t = pose.translation
    # Extract roll/pitch/yaw matching pose_from_xyzrpy's ZYX convention
    sp = -R[2, 0]
    if abs(sp) > 0.9999:
        # Gimbal lock
        pitch = np.pi / 2 * np.sign(sp)
        roll = 0.0
        yaw = float(np.arctan2(-R[0, 1], R[1, 1]))
    else:
        pitch = float(np.arcsin(sp))
        roll = float(np.arctan2(R[2, 1], R[2, 2]))
        yaw = float(np.arctan2(R[1, 0], R[0, 0]))
    return float(t[0]), float(t[1]), float(t[2]), roll, pitch, yaw
