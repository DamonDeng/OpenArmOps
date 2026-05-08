"""
Pinocchio-based gravity compensation for OpenArm.

Loads the OpenArm URDF once at startup and provides per-cycle computation of
gravity feedforward torques (N·m) given the current joint pose.

Usage:
    gc = GravityCompensator("urdf/openarm.urdf")
    tau = gc.compute_tau_gravity([j1_deg, j2_deg, ..., j7_deg])
"""

import numpy as np
import pinocchio as pin


class GravityCompensator:
    """URDF joint naming:
       - single-arm URDF (bimanual:=false): "openarm_joint1..7"
       - bimanual URDF (bimanual:=true):    "openarm_right_joint1..7" / "openarm_left_joint1..7"
       The bimanual URDF is preferred — it includes the +/-90 deg base rotation
       that reflects how each arm is physically mounted to the body.
    """

    def __init__(self, urdf_path, arm_side="right"):
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()

        # Try bimanual naming first, then fall back to single-arm naming
        urdf_names = [f"openarm_{arm_side}_joint{i+1}" for i in range(7)]
        if self.model.getJointId(urdf_names[0]) >= self.model.njoints:
            urdf_names = [f"openarm_joint{i+1}" for i in range(7)]

        self._q_indices = []
        self._v_indices = []
        for urdf_name in urdf_names:
            joint_id = self.model.getJointId(urdf_name)
            if joint_id >= self.model.njoints:
                raise ValueError(f"Joint {urdf_name!r} not found in URDF")
            self._q_indices.append(self.model.idx_qs[joint_id])
            self._v_indices.append(self.model.idx_vs[joint_id])
        self._urdf_names = urdf_names

        # Pre-allocate vectors — re-used each cycle to avoid GC churn
        self._q = np.zeros(self.model.nq)
        self._zero_v = np.zeros(self.model.nv)
        self._zero_a = np.zeros(self.model.nv)

    def compute_tau_gravity(self, positions_deg):
        """Compute gravity feedforward torques.

        Args:
            positions_deg: iterable of 7 joint positions in degrees (LeRobot j1..j7 order).

        Returns:
            list of 7 torques in N·m (LeRobot j1..j7 order).
        """
        self._q.fill(0.0)
        for i in range(7):
            self._q[self._q_indices[i]] = np.radians(positions_deg[i])

        tau = pin.rnea(self.model, self.data, self._q, self._zero_v, self._zero_a)
        return [float(tau[self._v_indices[i]]) for i in range(7)]
