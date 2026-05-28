"""Entry point for the OpenArm Controller UI (M1 skeleton).

Lifecycle: construct the window → robot.connect() (with torque OFF) →
event loop → on close, robot.disconnect(). Nothing else in M1. Run as a
module from the repository root so relative imports resolve:

    python -m openarm_controller_ui_lerobot.app
"""

from __future__ import annotations

import logging
import sys

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QMessageBox,
    QStatusBar,
    QTabWidget,
)

from . import config
from .key_bindings import load_bindings
from .motion_worker import MotionWorker
from .robot_service import RobotService
from .runtime_state import RuntimeState
from .session_config import load_session
from .tab_cartesian import CartesianTab
from .tab_controller import ControllerTab
from .tab_system import SystemTab
from .tab_vr import VRTab
from .tab_vr_control import VRControlTab
from .vr_input import VRInputReceiver

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    def __init__(
        self,
        robot: RobotService,
        state: RuntimeState,
        worker: MotionWorker,
        vr_receiver: VRInputReceiver,
    ) -> None:
        super().__init__()
        self.robot = robot
        self.state = state
        self.worker = worker
        self.vr_receiver = vr_receiver
        self.setWindowTitle("OpenArm Controller (LeRobot direct)")
        self.resize(1200, 800)

        tabs = QTabWidget()
        self.controller_tab = ControllerTab(robot, state, worker)
        self.cartesian_tab = CartesianTab(robot, worker)
        self.vr_info_tab = VRTab(vr_receiver)
        self.vr_control_tab = VRControlTab(worker, vr_receiver)
        # System tab gets a reference to the controller tab so its
        # "Reload key bindings" button can reach into ControllerTab's
        # bindings dict.
        self.system_tab = SystemTab(robot, state, self.controller_tab, vr_receiver)
        tabs.addTab(self.controller_tab, "Controller (movej)")
        tabs.addTab(self.cartesian_tab, "Cartesian (movel)")
        tabs.addTab(self.vr_info_tab, "VR Info")
        tabs.addTab(self.vr_control_tab, "VR Control")
        tabs.addTab(self.system_tab, "System")
        self.setCentralWidget(tabs)

        # Wire the cross-tab callbacks. The Controller tab's keyboard
        # filter hands cartesian bindings to the Cartesian tab; the
        # Cartesian tab hands gripper nudges (which are 1-DOF and not
        # part of the pose target) back to the Controller tab's
        # existing gripper-slider path.
        self.controller_tab.cartesian_nudge_callback = self.cartesian_tab.handle_cartesian_nudge
        self.cartesian_tab.set_gripper_nudge_callback(self.controller_tab.nudge_gripper_target)

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self._connection_label = QLabel("disconnected")
        self.status.addWidget(self._connection_label)
        self._warning_label = QLabel("")
        self._warning_label.setStyleSheet("color: #c44; font-weight: bold;")
        self.status.addPermanentWidget(self._warning_label)
        self.controller_tab.warning_changed.connect(self._on_warning_changed)
        self._refresh_status()

    def _on_warning_changed(self, msg: str) -> None:
        self._warning_label.setText(msg)

    def _refresh_status(self) -> None:
        text = "connected, torque OFF" if self.robot.connected else "disconnected"
        self._connection_label.setText(text)

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt signature)
        logger.info("Window close: stopping VR receiver…")
        self.vr_receiver.stop()
        self.vr_receiver.wait(1000)
        logger.info("Window close: stopping motion worker…")
        self.worker.stop()
        self.worker.wait(2000)  # 2 s to let current tick drain
        logger.info("Window close: disconnecting robot…")
        self.robot.disconnect()
        super().closeEvent(event)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Load bindings once at startup to fail fast on malformed JSON. The
    # UI will later expose a "reload" button that re-calls load_bindings.
    try:
        bindings = load_bindings()
        logger.info(f"Loaded {len(bindings)} key binding(s) from {config.DEFAULT_KEY_BINDINGS_PATH}")
    except Exception as e:
        logger.error(f"Could not load key bindings: {e}")
        bindings = {}

    app = QApplication(sys.argv)
    # Use Qt's default high-DPI handling if available
    app.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    robot = RobotService()
    try:
        robot.connect()
    except Exception as e:
        logger.exception("Failed to connect to robot")
        QMessageBox.critical(
            None,
            "Connection failed",
            f"Could not connect to the bimanual follower:\n\n{e}\n\n"
            "Check that CAN interfaces are up and cameras are accessible.",
        )
        return 2

    state = RuntimeState()
    # Auto-load persisted motion settings if a config file exists. Missing
    # file: silent pass-through (defaults apply). Malformed file: log and
    # keep defaults so startup isn't blocked.
    if config.SESSION_CONFIG_PATH.exists():
        try:
            load_session(state)
        except Exception as e:
            logger.error(f"Failed to load session config: {e}. Using defaults.")
    # VR receiver is constructed before the motion worker so the worker
    # can hold a reference to it (Phase 2b-β reads controller state from
    # it each tick). We start() it after the worker starts so its logs
    # appear after the "MotionWorker started" banner.
    vr_receiver = VRInputReceiver()

    worker = MotionWorker(robot, state, vr_receiver=vr_receiver)
    worker.start()
    vr_receiver.start()

    window = MainWindow(robot, state, worker, vr_receiver)
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
