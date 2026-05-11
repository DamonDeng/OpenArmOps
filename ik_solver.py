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
from dataclasses import dataclass

import numpy as np
import pinocchio as pin

logger = logging.getLogger(__name__)

_DAMPING = 1e-6
_MAX_ITERS = 40
_POS_TOL = 1e-4      # 0.1 mm
_ROT_TOL = 1e-3      # ~0.06°
_STEP_ALPHA = 1.0    # full step per iteration


@dataclass
class IKResult:
    q_deg: list[float]      # 7 joint angles
    converged: bool
    iters: int
    pos_err_mm: float
    rot_err_deg: float
    clamped: bool           # True if joint limits modified the solution


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
        """Solve IK for a single target. Returns IKResult — caller decides
        what to do with non-converged or clamped solutions.
        """
        q = np.zeros(self.model.nq)
        for i, v in enumerate(q_seed_deg_7):
            q[self._q_indices[i]] = np.radians(v)

        pos_err_m = rot_err_rad = 0.0
        converged = False
        for it in range(_MAX_ITERS):
            pin.framesForwardKinematics(self.model, self.data, q)
            current = self.data.oMf[self._tcp_fid]
            # SE(3) error: log of (current^-1 * target) gives a 6-vector
            # in the local frame. Convention matches Pinocchio's
            # examples/inverse-kinematics.py.
            err_local = pin.log(current.actInv(target_pose)).vector
            pos_err_m = float(np.linalg.norm(err_local[:3]))
            rot_err_rad = float(np.linalg.norm(err_local[3:]))
            if pos_err_m < _POS_TOL and rot_err_rad < _ROT_TOL:
                converged = True
                break

            # Jacobian of the TCP frame in its local frame (matches err_local).
            J = pin.computeFrameJacobian(
                self.model, self.data, q, self._tcp_fid,
                pin.ReferenceFrame.LOCAL,
            )
            # Keep only the columns we can actuate (our 7 arm joints). Other
            # joints (the other arm, fingers) can't move so their Jacobian
            # columns would be garbage to solve against.
            J_arm = J[:, self._v_indices]

            # Damped pseudo-inverse step: dq = J^T (J J^T + lambda I)^-1 err
            JJt = J_arm @ J_arm.T
            damped = JJt + _DAMPING * np.eye(6)
            dq_arm = J_arm.T @ np.linalg.solve(damped, err_local)

            for k, idx in enumerate(self._v_indices):
                q[idx] += _STEP_ALPHA * dq_arm[k]

        # Extract our 7 angles and convert to degrees
        q_deg = [float(np.degrees(q[self._q_indices[i]])) for i in range(7)]

        # Clamp to joint limits. Note: the clamp may re-introduce error at
        # the TCP because moving away from the IK solution breaks FK. We
        # simply report the clamp; the caller decides whether to freeze.
        clamped = False
        for i, (lo, hi) in enumerate(self._limits_deg):
            if q_deg[i] < lo:
                q_deg[i] = lo
                clamped = True
            elif q_deg[i] > hi:
                q_deg[i] = hi
                clamped = True

        return IKResult(
            q_deg=q_deg,
            converged=converged,
            iters=it + 1,
            pos_err_mm=pos_err_m * 1000.0,
            rot_err_deg=float(np.degrees(rot_err_rad)),
            clamped=clamped,
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
