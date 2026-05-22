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

import numpy as np
import pinocchio as pin

logger = logging.getLogger(__name__)

_DAMPING = 1e-6
_MAX_ITERS = 40
_POS_TOL = 1e-4      # 0.1 mm  — strict 6-DOF convergence
_ROT_TOL = 1e-3      # ~0.06°  — strict 6-DOF convergence
_STEP_ALPHA = 1.0    # full step per iteration

# "Usable" tolerances — looser than strict convergence. A solution
# whose position error is below this is good enough to send to the
# motors, even if it didn't reach the strict convergence target.
# Beyond _USABLE_POS_TOL we declare "freeze" and warn the caller.
_USABLE_POS_TOL = 5e-3   # 5 mm position error — anything beyond and
                         # we say the target is genuinely unreachable.
_USABLE_ROT_TOL = math.radians(15.0)  # 15° — fairly loose, since
                                      # position-priority deliberately
                                      # lets rotation drift.


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

    def solve(self, target_pose: pin.SE3, q_seed_deg_7: list[float]) -> IKResult:
        """Solve IK for a single target with a strict-then-relaxed fallback.

        Strategy:
          1. Run strict 6-DOF damped least squares (existing behaviour). If
             it converges to the tight (_POS_TOL, _ROT_TOL) thresholds,
             return the result.
          2. Otherwise, re-run from the strict pass's final q with
             position-priority weighting — rotation rows of the error
             vector are scaled down by _POSPRI_ROT_WEIGHT so the solver
             trades wrist orientation for position match. This is what
             handles "user nudged X by 1 cm but the requested orientation
             can't be achieved at that X" — the wrist drifts off the
             requested orientation but X is honoured.
          3. Whichever pass produced the lower **position** error wins.
             We then mark the result ``usable`` if the position error is
             below _USABLE_POS_TOL — caller can act on the relaxed
             solution rather than freezing the arm.

        ``converged`` retains its old meaning (strict 6-DOF). Use
        ``usable`` to decide whether to drive the motors.
        """
        q = np.zeros(self.model.nq)
        for i, v in enumerate(q_seed_deg_7):
            q[self._q_indices[i]] = np.radians(v)

        # Pass 1: strict 6-DOF.
        q_strict, pos_err_strict, rot_err_strict, iters_strict = self._iterate(
            q.copy(), target_pose, position_only=False,
        )
        strict_converged = (pos_err_strict < _POS_TOL
                            and rot_err_strict < _ROT_TOL)

        if strict_converged:
            return self._finalize_result(
                q_strict, pos_err_strict, rot_err_strict, iters_strict,
                converged=True, position_priority_used=False,
                target_pose=target_pose,
            )

        # Pass 2: position-priority fallback, seeded from the original
        # user seed (NOT from q_strict — empirically the strict pass
        # can wander into poor local minima when the target is over-
        # extended, and seeding the relaxed pass from there inherits
        # the bad starting point).
        q_seed_arr = np.zeros(self.model.nq)
        for i, v in enumerate(q_seed_deg_7):
            q_seed_arr[self._q_indices[i]] = np.radians(v)
        q_relaxed, pos_err_relaxed, rot_err_relaxed, iters_relaxed = self._iterate(
            q_seed_arr, target_pose, position_only=True,
        )

        # The strict pass didn't converge. Prefer the relaxed result
        # whenever it's at least as good on position (with a small
        # tolerance for floating-point ties), because the relaxed pass
        # is the one designed to handle "rotation can't be fully
        # honoured" situations — it's the right answer for the user's
        # primary use case ("push X, let other axes drift"). Only fall
        # back to the strict pass if the relaxed pass actually did
        # significantly worse on position, which would mean the
        # relaxed solver wandered.
        relaxed_better = pos_err_relaxed <= pos_err_strict + 1e-4
        if relaxed_better:
            return self._finalize_result(
                q_relaxed, pos_err_relaxed, rot_err_relaxed,
                iters_strict + iters_relaxed,
                converged=False, position_priority_used=True,
                target_pose=target_pose,
            )
        return self._finalize_result(
            q_strict, pos_err_strict, rot_err_strict, iters_strict,
            converged=False, position_priority_used=False,
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

    def _finalize_result(
        self,
        q: np.ndarray,
        pos_err_m: float,
        rot_err_rad: float,
        iters: int,
        converged: bool,
        position_priority_used: bool,
        target_pose: pin.SE3,
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
        if clamped:
            q_full = np.zeros(self.model.nq)
            for i, deg in enumerate(q_deg):
                q_full[self._q_indices[i]] = np.radians(deg)
            pin.framesForwardKinematics(self.model, self.data, q_full)
            current = self.data.oMf[self._tcp_fid]
            err_local = pin.log(current.actInv(target_pose)).vector
            pos_err_m = float(np.linalg.norm(err_local[:3]))
            rot_err_rad = float(np.linalg.norm(err_local[3:]))
            # Strict 6-DOF "converged" no longer applies after a clamp
            # that increased error. Recompute against the strict
            # thresholds.
            converged = (pos_err_m < _POS_TOL and rot_err_rad < _ROT_TOL)

        usable = (pos_err_m < _USABLE_POS_TOL
                  and rot_err_rad < _USABLE_ROT_TOL)

        return IKResult(
            q_deg=q_deg,
            converged=converged,
            iters=iters,
            pos_err_mm=pos_err_m * 1000.0,
            rot_err_deg=float(np.degrees(rot_err_rad)),
            clamped=clamped,
            usable=usable,
            position_priority_used=position_priority_used,
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
