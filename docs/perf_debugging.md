# Motion-worker perf debugging

Two pieces work together:

1. **Per-tick perf log** — every motion-worker tick writes one CSV row.
2. **VR replay** — re-streams a saved session over UDP at the original
   cadence so you can drive the live worker without wearing the headset.

## Where the perf log lives

`~/.openarm_ui_config/perf_logs/motion_perf_<YYYYMMDD_HHMMSS>.csv`

One row per tick. Columns:

```
t_rel_s, tick,
drain_cmd_ms, read_state_ms,
vr_left_ms, vr_right_ms,
cart_left_ms, cart_right_ms,
action_build_ms, send_ms,
total_ms,
vr_left_on, vr_right_on, cart_left_on, cart_right_on
```

`total_ms ≥ 33` means we missed 30 Hz that tick. The `*_on` flags let
you slice by mode without correlating against another log.

A summary line lands in the standard logger every
`PERF_LOG_SUMMARY_INTERVAL_SEC` seconds (default 5):

```
perf n=148 total= 11.5/ 14.2/ 28.1 ... send=  6.3/  8.1/ 18.4  (med/p95/max ms)
```

## Replaying a recorded session

```
python3 -m openarm_controller_ui_lerobot.tools.replay_vr_log \
    ~/.openarm_ui_config/vr_recordings/vr_log_<...>.jsonl
```

Sends each packet at its recorded `t` (monotonic seconds). Defaults to
`127.0.0.1:5100`. Useful flags:

- `--speed 0.5` half-speed
- `--right-only` / `--left-only`
- `--dry-run` parse + report timing without sending

Pre-flight:
1. Live app running, robot connected.
2. Arm(s) you want driven are in VR-enabled cartesian mode.
3. Arm pose roughly matches the start of the recording.
4. E-stop reachable.

On Ctrl-C / completion the replayer sends a grip=0 release packet for
every controller it touched, so the worker freezes the arm cleanly.

## Typical investigation

1. Start the app, connect, enable both arms in VR + cartesian mode.
2. Find the matching `vr_recordings/*.jsonl` for the session you care
   about (or capture a fresh one from System tab).
3. Run the replayer; let it finish.
4. Open the matching `perf_logs/*.csv`. Look for:
   - `total_ms` distribution by `vr_left_on + vr_right_on`
   - which stage's median jumps when both arms are on (likely `cart_*`
     from IK, or `send` if CAN parallelisation isn't actually
     overlapping)
5. The summary lines in the standard log are usually enough to spot
   the culprit at a glance; the CSV is for the deep-dive.
