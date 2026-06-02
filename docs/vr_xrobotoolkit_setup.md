# VR teleop with XRoboToolkit (PXREASDK backend)

This is the new default VR path. It replaces the closed-source
`openarmx-vr-pico.apk` (UDP, ~5 Hz aggregate dual-arm) with Pico's own
MIT-licensed XRoboToolkit stack (TCP, sustained 90 Hz dual-arm). See
`vr_packet_rate_investigation.md` for the full diagnosis trail.

## Architecture at a glance

```
                            Wi-Fi (TCP)
  Pico 4 Ultra            ─────────────────►       Spark
  XRoboToolkit-Unity-Client                   ┌──────────────────────────┐
  (the headset APK)                           │ RoboticsServiceProcess   │
                                              │   :63901 (device side)   │
                                              │   :60061 (SDK side)      │
                                              └──────┬───────────────────┘
                                                     │ libPXREARobotSDK.so
                                                     ▼
                                              ┌──────────────────────────┐
                                              │ openarm_controller_ui    │
                                              │ (PXREASDKVRReceiver)     │
                                              └──────────────────────────┘
```

Three pieces talk to each other. All three must be up.

## One-time setup (already done on this machine)

1. Pico install — `XRoboToolkit-Unity-Client.apk` v1.1.1 sideloaded on
   the headset (developer mode enabled, no developer account needed).
2. Spark install — `XRoboToolkit-PC-Service` v1.0.0 .deb installed at
   `/opt/apps/roboticsservice/`. Required dependency `libicu70`
   sideloaded from Ubuntu 22.04 ports alongside the host's libicu74.
3. Network — Pico on the same Wi-Fi as the Spark, no special routing.
   The headset auto-discovers the broker via local broadcast; no IP
   needs to be entered manually.

## Per-session checklist

Run **in this order**:

### 1. Start the broker (once per boot)

```bash
cd /opt/apps/roboticsservice && ./runService.sh
```

Confirm it's listening:

```bash
ss -tlnp | grep -E '63901|60061'
# Should show two LISTEN lines, both owned by RoboticsService.
```

The terminal will print "release mode" and stay attached. Leave it
running. To stop it later: `Ctrl-C` in that terminal.

### 2. Start the controller UI

In a separate terminal, from the repo root:

```bash
cd ~/workspace/openarm_space
python -m openarm_controller_ui_lerobot.app
```

Module form (`-m … .app`) is required so the package's relative
imports resolve.

The startup log should include a line like:

```
INFO ... VR receiver backend: 'pxreasdk' (PXREASDKVRReceiver)
```

If it says `(UDPVRReceiver)` instead, the SDK lib failed to load and
the app silently fell back to UDP. Check the lines just above for the
real reason — usually a missing service install or a stale
`config.VR_PXREASDK_LIB` path.

### 3. On the Pico

1. Put on the headset. Launch **XRoboToolkit-Unity-Client**.
2. Service IP field — leave blank. The app discovers the Spark via
   broadcast.
3. Toggle data streaming **on** in the headset UI.

Within ~1 s, the controller UI's **Stream** panel should show:

- `from: PXREASDK broker @ 127.0.0.1:60061`
- `rate: ~90 Hz`
- `packets:` increasing
- last seen: a few ms ago, no STALE warning

## First-time safety check (do this before enabling arms)

Open the **VR** tab (Phase 2a debug dashboard) and verify **before any
motor command goes out**:

1. **Pose updates live** — wiggle each controller, watch the
   Position / Quaternion fields update.
2. **Trigger sweeps 0 → 1** — pull each trigger slowly.
3. **Grip sweeps 0 → 1** — squeeze each controller's grip slowly.
   This is the dead-man. If the bar stays stuck at 0 even while
   squeezing hard, **stop**: the JSON field name in this APK build
   may differ from what the receiver looks for. Don't enable the arms
   until grip is verified to move.
4. **Buttons fire** — press each of A / B / X / Y in turn and watch
   the dots fill on the right and left panels.
   - right.primary → A, right.secondary → B
   - left.primary  → X, left.secondary  → Y
   - If they're inverted on your APK build, edit the right/left arm
     branch in `vr_input_pxreasdk.py::_apply_controller_json`.

Only after all four pass should you switch the arms into VR mode.

## Switching backends

In **System tab → VR receiver backend**:

| label                          | when to use                                           |
|---                             |---                                                    |
| XRoboToolkit (PXREASDK, 90 Hz) | default; you're running the new client + service      |
| Legacy UDP APK (5 Hz dual-arm) | only if you've reverted the headset to the old APK    |

Changing the dropdown stages the choice. To make it active:
1. Click **Save motion config** in the Motion settings group so the
   choice survives restart.
2. Quit the app and relaunch.

The currently-active receiver is shown below the dropdown
(`PXREASDKVRReceiver` or `UDPVRReceiver`); a "restart required"
message appears if your selection differs from what's running.

## Diagnostics

### "rate: 0 Hz, packets: 0" in the Stream panel

Walk through the chain.

1. Is the broker up?
   ```bash
   pgrep -af RoboticsServiceProcess
   ```
   Empty output → start it (step 1 above).

2. Is our consumer attached?
   ```bash
   ss -tnp | grep 60061
   ```
   Should show one connection between the Spark and itself
   (`127.0.0.1:60061 ↔ 127.0.0.1:<ephemeral>`) owned by `python` /
   our app. If not, the receiver thread didn't connect — check the
   app log for `PXREASDK: PXREAInit returned …`.

3. Is the headset connected to the broker?
   ```bash
   ss -tnp | grep 63901
   ```
   If empty, the Pico isn't reaching the Spark. Check Wi-Fi, then
   re-toggle data streaming in the headset app.

4. Last resort — bypass our app and confirm the broker is delivering
   data to *any* consumer:
   ```bash
   ./openarm_controller_ui_lerobot/scripts/run_consoledemo.sh
   ```
   This launches the bundled reference SDK consumer. If it prints
   `device data {…}` lines and our app sees nothing, the bug is in
   our Python code. If it also prints nothing, the headset isn't
   feeding the broker. **Stop `ConsoleDemo` before relaunching the
   UI** — only one SDK consumer should be attached at a time.

### "STALE" appearing on the controllers

`VR_STALE_SEC = 1.0` in `config.py`. With a healthy XRoboToolkit feed
the inter-frame median is ~11 ms, so STALE means we missed ~90
consecutive frames — almost always a broker death or Wi-Fi drop, not
in-app jitter.

### Recording packets for offline analysis

System tab → **VR packet recording (diagnostic)** → toggle on, do
the test, click **Save & clear**. Output lands in
`~/.openarm_ui_config/vr_recordings/vr_log_<ts>.jsonl`. One JSON line
per delivered frame; the `raw` field is the full Tracking JSON the
SDK handed us.

## Files involved

| file                                          | role                                          |
|---                                            |---                                            |
| `vr_input.py`                                 | base class + UDP backend + factory            |
| `vr_input_pxreasdk.py`                        | new XRoboToolkit backend (ctypes)             |
| `config.py` (`VR_RECEIVER_BACKEND`, `VR_PXREASDK_LIB`) | defaults                              |
| `tab_system.py` (VR receiver backend group)   | runtime selector                              |
| `app.py` (`make_vr_receiver(...)`)            | startup wiring                                |
| `scripts/run_consoledemo.sh`                  | diagnostic: bundled reference SDK consumer    |
| `/opt/apps/roboticsservice/`                  | broker install (system, not in repo)          |
