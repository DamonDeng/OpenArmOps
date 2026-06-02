"""XRoboToolkit (PXREASDK) backend for the VR-controller receiver.

Attaches to Pico's RoboticsServiceProcess via libPXREARobotSDK.so on
``127.0.0.1:60061`` and consumes ``PXREADeviceStateJson`` callbacks.
Each callback delivers one frame of *both* controllers in a single
JSON payload (so dual-arm has no extra cost over single-arm), at a
sustained 90 Hz on hardware tested 2026-06-02.

See ``docs/vr_packet_rate_investigation.md`` for the full diagnosis
of why this replaces the legacy UDP path. The SDK header lives at
``/opt/apps/roboticsservice/SDK/include/PXREARobotSDK.h``; the
relevant entry points are:

    int PXREAInit(void* ctx, callback, unsigned mask);
    int PXREADeinit();

with callback signature ``(ctx, type, status, userData)`` and
``userData`` for ``PXREADeviceStateJson`` pointing at:

    struct { char devID[32]; char stateJson[16352]; }

Inner JSON shape (from a real callback, 2026-06-02):

    {
      "functionName": "Tracking",
      "value": "{ ... escaped JSON with predictTime, Controller, Hand,
                  timeStampNs, Input ... }"
    }
"""

from __future__ import annotations

import ctypes
import json
import logging
import time
from typing import Optional

from . import config
from .vr_input import ControllerState, VRInputReceiver

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# C API mirror — kept minimal; we only need init/deinit and the one
# callback type that delivers PXREADeviceStateJson payloads.
# ---------------------------------------------------------------------------

# enum PXREAClientCallbackType bits we care about
_CB_SERVER_CONNECT      = 1 << 2
_CB_SERVER_DISCONNECT   = 1 << 3
_CB_DEVICE_FIND         = 1 << 4
_CB_DEVICE_MISSING      = 1 << 5
_CB_DEVICE_CONNECT      = 1 << 9
_CB_DEVICE_STATE_JSON   = 1 << 25
_CB_FULL_MASK           = 0xFFFFFFFF


class _PXREADevStateJson(ctypes.Structure):
    _fields_ = [
        ("devID", ctypes.c_char * 32),
        ("stateJson", ctypes.c_char * 16352),
    ]


# Callback prototype: void cb(void* ctx, int type, int status, void* userData)
_CB_PROTO = ctypes.CFUNCTYPE(
    None,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_void_p,
)


def _load_sdk(path: str) -> ctypes.CDLL:
    """Load libPXREARobotSDK.so. Tries the configured absolute path
    first, then falls back to a bare basename so LD_LIBRARY_PATH /
    /etc/ld.so.conf.d entries also work.
    """
    try:
        return ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
    except OSError as e:
        logger.warning(
            f"Direct dlopen of {path} failed: {e}; trying basename via "
            "the dynamic linker."
        )
        return ctypes.CDLL("libPXREARobotSDK.so", mode=ctypes.RTLD_GLOBAL)


class PXREASDKVRReceiver(VRInputReceiver):
    """Default VR receiver: XRoboToolkit broker via PXREARobotSDK.

    The SDK runs its own internal worker thread which invokes our
    callback on every device data frame. Our ``run()`` body just owns
    init/deinit and a sleep loop that drives the 10 Hz UI emit.
    """

    def __init__(self) -> None:
        super().__init__()
        self._sdk: Optional[ctypes.CDLL] = None
        # Strong ref to the CFUNCTYPE wrapper; if we drop this the SDK
        # will call into freed memory on the next callback.
        self._cb_holder: Optional[ctypes.CFUNCTYPE] = None
        # Set by the SERVER_CONNECT callback so the System tab can
        # show whether the broker handshake succeeded. Surface via
        # last_source for now — same field the UI already renders.
        self._connected_to_broker = False

    # ------------------------------------------------------------------
    # Thread body
    # ------------------------------------------------------------------
    def run(self) -> None:
        try:
            self._sdk = _load_sdk(config.VR_PXREASDK_LIB)
        except OSError as e:
            logger.error(
                f"PXREASDK: failed to load libPXREARobotSDK.so ({e}). "
                "Make sure RoboticsService is installed; the receiver "
                "will sit idle until restart."
            )
            return

        self._sdk.PXREAInit.argtypes = [ctypes.c_void_p, _CB_PROTO, ctypes.c_uint]
        self._sdk.PXREAInit.restype = ctypes.c_int
        self._sdk.PXREADeinit.argtypes = []
        self._sdk.PXREADeinit.restype = ctypes.c_int

        self._cb_holder = _CB_PROTO(self._on_sdk_callback)
        rc = self._sdk.PXREAInit(None, self._cb_holder, _CB_FULL_MASK)
        logger.info(f"PXREASDK: PXREAInit returned {rc}")

        try:
            # The SDK runs its own thread for the network + callbacks.
            # All we need to do here is keep the QThread alive, drive
            # the 10 Hz UI emit, and check for shutdown.
            while not self._stop:
                self._maybe_emit()
                time.sleep(0.05)
        finally:
            try:
                self._sdk.PXREADeinit()
            except Exception as e:
                logger.warning(f"PXREASDK: PXREADeinit raised: {e}")
            self._cb_holder = None
            logger.info("PXREASDK receiver stopped")

    # ------------------------------------------------------------------
    # SDK callback — runs on the SDK's worker thread.
    # ------------------------------------------------------------------
    def _on_sdk_callback(
        self,
        ctx: int,
        cb_type: int,
        status: int,
        user_data: int,
    ) -> None:
        try:
            if cb_type == _CB_SERVER_CONNECT:
                self._connected_to_broker = True
                with self._lock:
                    # Re-purpose last_source so the existing Stream panel
                    # shows something meaningful for this backend.
                    self._stats.last_source = "PXREASDK broker @ 127.0.0.1:60061"
                logger.info("PXREASDK: server connect")
            elif cb_type == _CB_SERVER_DISCONNECT:
                self._connected_to_broker = False
                logger.warning("PXREASDK: server disconnect")
            elif cb_type == _CB_DEVICE_FIND:
                logger.info("PXREASDK: device find")
            elif cb_type == _CB_DEVICE_CONNECT:
                logger.info(f"PXREASDK: device connect (status={status})")
            elif cb_type == _CB_DEVICE_MISSING:
                logger.warning("PXREASDK: device missing")
            elif cb_type == _CB_DEVICE_STATE_JSON:
                if user_data:
                    payload = ctypes.cast(
                        user_data, ctypes.POINTER(_PXREADevStateJson),
                    ).contents
                    self._handle_state_json(payload)
        except Exception as e:
            logger.exception(f"PXREASDK callback raised: {e}")

    # ------------------------------------------------------------------
    # Payload parser
    # ------------------------------------------------------------------
    def _handle_state_json(self, payload: _PXREADevStateJson) -> None:
        # The SDK fills both fields with zero-terminated strings.
        raw = bytes(payload.stateJson).rstrip(b"\x00")
        dev_id = bytes(payload.devID).rstrip(b"\x00").decode(errors="replace")
        if not raw:
            return

        now = time.monotonic()
        # Recorder hook: stash the wrapper JSON so replays can
        # reproduce the exact bytes the SDK handed us. Source is the
        # device ID; useful for distinguishing multi-headset captures.
        self._record_raw_packet(now, dev_id, raw)

        with self._lock:
            self._stats.record(now, len(raw), dev_id)

        try:
            outer = json.loads(raw)
        except Exception as e:
            with self._lock:
                self._stats.parse_errors += 1
            logger.debug(f"PXREASDK: outer JSON parse error: {e}")
            return

        if outer.get("functionName") != "Tracking":
            with self._lock:
                self._stats.unknown_messages += 1
            return

        try:
            inner = json.loads(outer.get("value", ""))
        except Exception as e:
            with self._lock:
                self._stats.parse_errors += 1
            logger.debug(f"PXREASDK: inner JSON parse error: {e}")
            return

        ctrl = inner.get("Controller") or {}
        ts_ns = int(inner.get("timeStampNs", 0))
        if "left" in ctrl:
            self._apply_controller_json(self._left, ctrl["left"], ts_ns, now)
        if "right" in ctrl:
            self._apply_controller_json(self._right, ctrl["right"], ts_ns, now)

    def _apply_controller_json(
        self,
        target: ControllerState,
        side: dict,
        ts_ns: int,
        now: float,
    ) -> None:
        pose_str = side.get("pose", "")
        # Pose comes through as a comma-separated string of 7 floats:
        # tx,ty,tz,qx,qy,qz,qw. This matches the legacy UDP wire
        # ordering exactly so the downstream remap math is unchanged.
        try:
            pose = [float(x) for x in pose_str.split(",")]
        except ValueError:
            with self._lock:
                self._stats.parse_errors += 1
            return
        if len(pose) < 7:
            with self._lock:
                self._stats.parse_errors += 1
            return

        with self._lock:
            if target.last_rx > 0.0 and target.ts_ns > target.last_consumed_ts_ns:
                target.unread_overwrites += 1
            target.tx, target.ty, target.tz = pose[0], pose[1], pose[2]
            target.qx, target.qy, target.qz, target.qw = pose[3], pose[4], pose[5], pose[6]
            target.trigger = float(side.get("trigger", 0.0))
            target.grip = float(side.get("grip", 0.0))
            # XRoboToolkit-Unity-Client convention:
            #   right.primary = A, right.secondary = B
            #   left.primary  = X, left.secondary  = Y
            # We don't know which side ``target`` is here without an
            # explicit flag, so we map both buttons onto A/B for the
            # right hand and X/Y for the left by checking identity
            # against the receiver's own state objects.
            primary = bool(side.get("primaryButton", False))
            secondary = bool(side.get("secondaryButton", False))
            if target is self._right:
                target.a = int(primary)
                target.b = int(secondary)
                target.x = 0
                target.y = 0
            else:
                target.a = 0
                target.b = 0
                target.x = int(primary)
                target.y = int(secondary)
            # No "rate" field in this protocol — slow/fast toggle is
            # the operator's responsibility on the headset UI. Default
            # to 1.0 so downstream limit code uses the fast cap.
            target.rate = 1.0
            target.ts_ns = ts_ns
            target.last_rx = now
