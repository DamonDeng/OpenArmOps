"""Replay a recorded VR log into the live app's UDP receiver.

The recorder writes one JSON object per UDP packet, with a ``raw`` field
containing the exact ASCII datagram body. Replay just sends those bodies
back to the configured UDP port at the original inter-packet timing
(scaled by --speed if requested), making the live motion worker behave
as if the user were physically moving the controllers.

Usage
-----
    python3 -m openarm_controller_ui_lerobot.tools.replay_vr_log \\
        path/to/vr_log_*.jsonl

By default the script targets ``127.0.0.1:5100`` (the receiver's bind
addr/port from config.py). It prompts for confirmation before sending
because **the real arm will move**. Pass ``--yes`` to skip the prompt
(only do this once you've verified the destination is correct).

Safety
------
* On Ctrl-C, exit, or any unhandled error the script sends a "safe-state"
  packet for each controller it has touched: pos=(0,0,0), quat=identity,
  trigger=0, grip=0, all buttons=0. The motion worker reads grip=0 as
  "clutch released" and freezes the arm in place.
* ``--dry-run`` prints what would be sent without opening a socket.
* ``--speed 0.5`` plays back at half speed; useful for the first
  attempt at a new sequence.
* ``--right-only`` / ``--left-only`` skip packets from the other
  controller. Useful when you want to replay just one arm's motion
  without disturbing the other.

Pre-flight checklist (the script cannot verify these for you):
1. Live app is running and connected to the robot.
2. The arm(s) you want driven are in VR-enabled cartesian mode.
3. Arm pose roughly matches the start pose of the recording, so the
   first grip-engage snapshot lines up.
4. You can reach the e-stop button if the arm misbehaves.
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import socket
import sys
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Defaults match config.VR_UDP_BIND_ADDR / VR_UDP_PORT — but the
# receiver binds to 0.0.0.0, so we send to localhost. We do NOT import
# config here so the script stays usable without the rest of the app.
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 5100


@dataclass
class Packet:
    t: float
    raw: str
    kind: str   # "LEFT" | "RIGHT" | "HEAD" | other


def parse_log(path: Path) -> list[Packet]:
    packets: list[Packet] = []
    with path.open("r", encoding="utf-8") as f:
        for ln_no, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f"line {ln_no}: bad JSON ({e}); skipping")
                continue
            raw = str(rec.get("raw", ""))
            kind = raw.split(" ", 1)[0].upper() if raw else ""
            packets.append(Packet(t=float(rec.get("t", 0.0)), raw=raw, kind=kind))
    packets.sort(key=lambda p: p.t)
    return packets


def safe_state_datagram(kind: str) -> str:
    """Build a 'controller released, no input' datagram in the same
    on-wire format the APK uses. The motion worker treats this as
    clutch-disengaged and freezes the arm.

    Layout (matches openarmx_teleop_vr_apk):
        kind tx ty tz qx qy qz qw trigger grip A B X Y rate ts_ns
    """
    # 15 fields after the kind. ts_ns of 0 is fine — the worker doesn't
    # use it for control decisions.
    return f"{kind} 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0"


class Replayer:
    def __init__(
        self,
        sock: socket.socket | None,
        host: str,
        port: int,
        dry_run: bool,
    ):
        self.sock = sock
        self.host = host
        self.port = port
        self.dry_run = dry_run
        self.touched_kinds: set[str] = set()
        self.sent_count = 0
        self.sent_bytes = 0

    def send_raw(self, raw: str) -> None:
        kind = raw.split(" ", 1)[0].upper()
        if kind in ("LEFT", "RIGHT", "HEAD"):
            self.touched_kinds.add(kind)
        if self.dry_run or self.sock is None:
            return
        data = raw.encode("ascii")
        self.sock.sendto(data, (self.host, self.port))
        self.sent_count += 1
        self.sent_bytes += len(data)

    def send_safe_state(self) -> None:
        """Send grip=0 packets for every controller we drove during the
        replay so the motion worker knows the user "released" everything
        and freezes the arm.
        """
        for kind in sorted(self.touched_kinds):
            if kind in ("LEFT", "RIGHT"):
                raw = safe_state_datagram(kind)
                logger.info(f"safe-state: sending {kind} grip=0 release")
                self.send_raw(raw)


def replay(
    packets: list[Packet],
    host: str,
    port: int,
    speed: float,
    dry_run: bool,
    only_kinds: set[str] | None,
) -> int:
    """Stream packets to (host, port) at original timing × ``speed``.
    Returns 0 on clean completion, non-zero on early exit.
    """
    if not packets:
        logger.error("nothing to replay (no packets)")
        return 1

    sock: socket.socket | None = None
    if not dry_run:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # No bind needed for sendto. SO_REUSEADDR is irrelevant for clients.

    rep = Replayer(sock, host, port, dry_run)
    interrupted = False

    def _handle_sigint(signum, frame):
        nonlocal interrupted
        interrupted = True
        logger.warning("Ctrl-C received; finishing safe-state and exiting…")

    prev_handler = signal.signal(signal.SIGINT, _handle_sigint)

    log_t0 = packets[0].t
    log_duration = packets[-1].t - log_t0
    eff_duration = log_duration / speed
    logger.info(
        f"Replaying {len(packets)} packets, log duration "
        f"{log_duration:.2f}s, effective {eff_duration:.2f}s @ "
        f"speed={speed:g}× → {host}:{port}"
        + (" [DRY RUN]" if dry_run else "")
    )
    if only_kinds is not None:
        logger.info(f"kind filter: {sorted(only_kinds)}")

    wall_t0 = time.monotonic()
    try:
        for i, pkt in enumerate(packets):
            if interrupted:
                break
            if only_kinds is not None and pkt.kind not in only_kinds:
                continue
            # Wait until our wall-clock has advanced to the packet's
            # log-time-since-start divided by speed.
            target_wall = wall_t0 + (pkt.t - log_t0) / speed
            now = time.monotonic()
            sleep_for = target_wall - now
            if sleep_for > 0:
                # Shorter sleeps are more responsive to Ctrl-C; we cap
                # each sleep at 50 ms and re-check the interrupt flag.
                end = now + sleep_for
                while not interrupted:
                    remaining = end - time.monotonic()
                    if remaining <= 0:
                        break
                    time.sleep(min(remaining, 0.05))
            if interrupted:
                break
            rep.send_raw(pkt.raw)

            # Light progress log every 1 second of wall time
            if i % 100 == 0:
                elapsed = time.monotonic() - wall_t0
                logger.debug(
                    f"  {i+1}/{len(packets)}  log_t={pkt.t-log_t0:.2f}s  "
                    f"wall={elapsed:.2f}s"
                )
    finally:
        # Always restore the signal handler.
        signal.signal(signal.SIGINT, prev_handler)
        # Always send the safe-state release, even on clean completion.
        # The recording itself may end on a held-grip frame; without
        # this the motion worker would be left thinking the user is
        # still gripping.
        try:
            rep.send_safe_state()
        except Exception:
            logger.exception("safe-state send failed; arm may not freeze!")
        if sock is not None:
            sock.close()

    elapsed = time.monotonic() - wall_t0
    logger.info(
        f"Replay done in {elapsed:.2f}s. Sent {rep.sent_count} packets, "
        f"{rep.sent_bytes} bytes. interrupted={interrupted}"
    )
    return 0 if not interrupted else 130


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s: %(message)s"
    )
    parser = argparse.ArgumentParser(
        description="Replay a recorded VR log to the live app's UDP receiver."
    )
    parser.add_argument("input", type=Path,
                        help="Path to the .jsonl recording to replay.")
    parser.add_argument("--host", default=DEFAULT_HOST,
                        help=f"UDP target host (default: {DEFAULT_HOST}).")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help=f"UDP target port (default: {DEFAULT_PORT}).")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Playback speed multiplier (default 1.0). "
                             "0.5 = half speed; 2.0 = double speed.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Parse and report timing but do not send any UDP.")
    parser.add_argument("--right-only", action="store_true",
                        help="Replay only RIGHT controller packets.")
    parser.add_argument("--left-only", action="store_true",
                        help="Replay only LEFT controller packets.")
    parser.add_argument("--no-head", action="store_true", default=True,
                        help="Skip HEAD packets (default: skipped — they're "
                             "not used by the motion worker today).")
    parser.add_argument("--include-head", dest="no_head", action="store_false",
                        help="Include HEAD packets in the replay.")
    parser.add_argument("-y", "--yes", action="store_true",
                        help="Skip the confirmation prompt.")
    args = parser.parse_args(argv)

    if not args.input.exists():
        logger.error(f"input file not found: {args.input}")
        return 2
    if args.speed <= 0:
        logger.error("--speed must be > 0")
        return 2
    if args.right_only and args.left_only:
        logger.error("--right-only and --left-only are mutually exclusive")
        return 2

    packets = parse_log(args.input)
    if not packets:
        logger.error("input contained no packets")
        return 1

    only_kinds: set[str] | None = None
    if args.right_only:
        only_kinds = {"RIGHT"}
    elif args.left_only:
        only_kinds = {"LEFT"}
    else:
        only_kinds = {"LEFT", "RIGHT"}
    if not args.no_head:
        only_kinds = (only_kinds or set()) | {"HEAD"}

    log_duration = packets[-1].t - packets[0].t
    eff_duration = log_duration / args.speed
    print()
    print(f"  File:       {args.input}")
    print(f"  Packets:    {len(packets)}")
    print(f"  Duration:   {log_duration:.2f}s log,  "
          f"{eff_duration:.2f}s effective at speed {args.speed:g}×")
    print(f"  Kinds:      {sorted(only_kinds) if only_kinds else 'all'}")
    print(f"  Target:     udp://{args.host}:{args.port}")
    print(f"  Mode:       {'DRY RUN (no UDP)' if args.dry_run else 'LIVE'}")
    print()

    if not args.dry_run and not args.yes:
        print("⚠  This will drive the connected robot via the live app.")
        print("   Make sure the e-stop is reachable. Continue? [y/N] ", end="")
        sys.stdout.flush()
        try:
            ans = input().strip().lower()
        except EOFError:
            ans = ""
        if ans not in ("y", "yes"):
            print("aborted by user")
            return 0

    return replay(
        packets,
        host=args.host,
        port=args.port,
        speed=args.speed,
        dry_run=args.dry_run,
        only_kinds=only_kinds,
    )


if __name__ == "__main__":
    sys.exit(main())
