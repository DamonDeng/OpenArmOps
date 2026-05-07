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
from .robot_service import RobotService
from .tab_controller import ControllerTab
from .tab_system import SystemTab

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    def __init__(self, robot: RobotService) -> None:
        super().__init__()
        self.robot = robot
        self.setWindowTitle("OpenArm Controller (LeRobot direct)")
        self.resize(1200, 800)

        tabs = QTabWidget()
        self.controller_tab = ControllerTab(robot)
        self.system_tab = SystemTab(robot)
        tabs.addTab(self.controller_tab, "Controller")
        tabs.addTab(self.system_tab, "System")
        self.setCentralWidget(tabs)

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self._connection_label = QLabel("disconnected")
        self.status.addWidget(self._connection_label)
        self._refresh_status()

    def _refresh_status(self) -> None:
        text = "connected, torque OFF" if self.robot.connected else "disconnected"
        self._connection_label.setText(text)

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt signature)
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

    window = MainWindow(robot)
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
