# VR packet-rate investigation (2026-06-01 → 2026-06-02)

## Resolution (2026-06-02): switched APK + transport, problem gone

**The dual-arm rate cliff was on the closed-source APK side, not on our
host or our code.** We replaced the closed-source `openarmx-vr-pico.apk`
with Pico's own MIT-licensed [XRoboToolkit-Unity-Client](https://github.com/PICO-XR/XRoboToolkit-Unity-Client)
(installed via sideload) and run its companion
[XRoboToolkit-PC-Service](https://github.com/PICO-XR/XRoboToolkit-PC-Service)
on the Spark. Result on 2026-06-02:

| metric                   | old APK (dual arm) | XRoboToolkit (dual arm) |
|---                       |---                 |---                      |
| sustained rate           | 5.3 Hz             | **90.3 Hz**             |
| median inter-frame gap   | 200 ms             | **11.1 ms**             |
| p95 gap                  | ~1 s               | **12.2 ms**             |
| max gap (3.6 s window)   | 2.51 s             | **14.8 ms**             |
| transport               | UDP datagrams      | TCP single stream       |
| both controllers per pkt | no (separate LEFT/RIGHT) | yes (one JSON) |

XRoboToolkit's transport is a three-tier broker: the headset opens a
**TCP** stream to `RoboticsServiceProcess` on `:63901`; local SDK
consumers attach via `libPXREARobotSDK.so` to `:60061` and receive
`PXREADeviceStateJson` callbacks. Because it's TCP we don't see the
Wi-Fi UDP cliff at all. Both controllers ship in a single JSON
payload, so dual-arm is no different from single-arm in cost.

Original investigation below is preserved for context — the original
APK is still bundled with OpenArm hardware, so anyone debugging that
path may want the trail.

### Wire format (PXREASDK consumer side)

`device data` callback `userData` is a `PXREADevStateJson` with one
JSON object per frame:

```json
{
  "functionName": "Tracking",
  "value": "{ ... escaped JSON ... }"
}
```

Inner payload (parsed from the escaped `value` string):

```jsonc
{
  "predictTime": 928650628.147,           // float, ms
  "appState": {"focus": true},
  "Controller": {
    "left":  {
      "axisX": 0.0, "axisY": 0.0, "axisClick": false,
      "grip": 0.0,  "trigger": 0.0,
      "primaryButton": false, "secondaryButton": false, "menuButton": false,
      "pose": "tx,ty,tz,qx,qy,qz,qw"      // 7 floats, comma-separated
    },
    "right": { ... same shape ... }
  },
  "Hand": {
    "leftHand":  {"isActive":0,"count":26,"scale":1.0,"HandJointLocations":[...]},
    "rightHand": {"isActive":0,"count":26,"scale":1.0,"HandJointLocations":[...]}
  },
  "timeStampNs": 1780369062464280064,     // int, APK monotonic ns
  "Input": 1
}
```

A/B/X/Y mapping (XRoboToolkit-Unity-Client convention):
- right.primaryButton = A, right.secondaryButton = B
- left.primaryButton  = X, left.secondaryButton  = Y

There is no separate HEAD message in this format; head pose is not
exposed via the Tracking callback. (If we end up needing head, it'd
come through a different `functionName` value or via the gRPC side
of the service.)

### Architecture diagram

```
                             Wi-Fi (TCP)
  Pico 4 Ultra            ─────────────────►       Spark (Linux aarch64)
  (XRoboToolkit-Unity                              ┌──────────────────────────┐
   -Client APK)                                    │ RoboticsServiceProcess   │
                                                   │   listens :63901 (device)│
                                                   │   listens :60061 (SDK)   │
                                                   └──────┬───────────────────┘
                                                          │ libPXREARobotSDK.so
                                                          │ (PXREAInit + cb)
                                                          ▼
                                                   ┌──────────────────────────┐
                                                   │ our process              │
                                                   │ (UI / motion worker)     │
                                                   └──────────────────────────┘
```

### Diagnosis trail (why this is the right reading)

We initially thought the broker had broken framing because its log
showed thousands of `NoPadding data size less than msg` lines at
`tcpconnectionmodel.cpp:124`. Wrong reading: **those lines mean
"frames piled up and no SDK consumer was attached to drain them"**.
Once we ran the bundled `ConsoleDemo` (the reference SDK consumer),
the broker happily delivered 328 device-data callbacks in 3.63 s with
zero gaps over 50 ms.

Smoke test artefact: `/tmp/consoledemo_20260602_105703/consoledemo.log`.

---

# Original investigation (2026-06-01)

## TL;DR

- Two-arm "trembling" we hit on 2026-06-01 was not caused by our motion
  worker. Motion loop sustained 30 Hz with no missed ticks
  (`>33 ms = 0` across 4,967 ticks; `total_ms` p95 = 8.2 ms).
- The Pico APK was only delivering **~5.3 Hz total** (LEFT 2.6 Hz +
  RIGHT 2.7 Hz) during the dual-arm session. With 30 Hz motor ticks
  and 200 ms median packet gaps (max 2.5 s), the arm has nothing new
  to track for ~6 ticks at a time → it freezes, then jumps when a new
  pose finally arrives.
- Single-arm recordings from the same APK historically hit 30–72 Hz on
  the active hand. The drop only shows up when both controllers are
  streaming.
- **Loss is on the wireless path, not in our code.** Confirmed via
  three-way count (APK sent / tcpdump on host NIC / our app socket):
  single-arm test delivered 1602/1612 (~0% host-side loss), dual-arm
  test delivered 66/2000 (~97% loss). NIC == app socket in both cases,
  i.e. our process and the kernel UDP buffer are not the bottleneck.
  The earlier ts_ns ≈ recv_ts comparison only measured gaps between
  *surviving* packets and was silent about the missing ones —
  retracted.
- The "rate" field in the wire format is **NOT** a send-rate
  ⇒ it's a `0.1 = slow / 1.0 = fast` speed-mode selector consumed by
  the receiver to pick between `max_step_deg_*` and
  `fast_max_step_deg_*`. Earlier comments calling it a send rate
  were wrong.
- The APK is closed-source. Disassembly of `libBasicDemo.so` strongly
  suggests an internal pacing loop (one `std::this_thread::sleep_for`
  with a runtime-computed `chrono::nanoseconds` between the per-hand
  send lambdas). Effective per-hand cadence is APK-internal and not
  configurable from our side.

## Test artefacts

Backed up under `local_data/backup_20260601_1205/`:
- `motion_perf_20260601_120244.csv` — 4,967 motion-worker ticks (~166 s)
- `vr_log_20260601_120436.jsonl` — 145 VR packets (~27 s window)

## Motion-worker side: clean

By the bucket of `(vr_left_on, vr_right_on, cart_left_on, cart_right_on)`:

| bucket            | n    | total med | total p95 | total max | >33 ms |
|---                |------|-----------|-----------|-----------|--------|
| all-off (idle)    |  497 |  2.94     |  4.56     | 22.61     | 0      |
| left only         |   26 |  3.13     |  4.09     |  4.58     | 0      |
| **both arms VR**  | 4444 |  4.05     |  8.22     | 12.02     | 0      |

Read/send stages with parallel CAN dispatch:

| stage      | both-arms median | both-arms p95 |
|---         |---               |---            |
| read_state | 1.18 ms          | 2.05 ms       |
| send       | 1.36 ms          | 2.38 ms       |

Single-arm `read_state` was 1.15 ms — i.e. parallel dispatch is
overlapping cleanly (no observable serialisation cost).

One genuine asymmetry to note: `cart_right` (1.54 ms median) ran
~10× slower than `cart_left` (0.13 ms median). This is *not* the cause
of the trembling but is the next thing to look at if we ever get
tighter on the budget. Likely candidates: IK seed quality, or
sequential pinocchio data sharing on the worker thread.

## VR-stream side: the actual problem

### Packet rate over time, this session vs prior

| recording                  | duration | total Hz | LEFT pkts | RIGHT pkts |
|---                         |----------|----------|-----------|------------|
| `vr_log_20260526_100341`   |  91.0 s  |   34.6   |   559     |  2588      |
| `vr_log_20260526_130622`   | 125.9 s  |   29.7   |    58     |  3681      |
| `vr_log_20260527_155249`   |  34.0 s  |   52.8   |     7     |  1788      |
| `vr_log_20260528_140228`   |  31.4 s  |   71.8   |  2251     |     8      |
| **`vr_log_20260601_120436`** | **27.3 s** | **5.3** | **71** | **74** |

All older recordings were single-arm (one hand was barely held). Today's
was the first dual-arm test. Per-hand rate collapsed to ~2.6 Hz.

### Inter-packet gap distribution (today's recording)

| range          | LEFT | RIGHT |
|---             |------|-------|
| < 50 ms        |   8  |    9  |
| 50–200 ms      |  26  |   28  |
| 200–500 ms     |  14  |   14  |
| 500–1000 ms    |  15  |   15  |
| ≥ 1 s          | **7**| **7** |
| **median gap** | 208 ms | 166 ms |
| **max gap**    | 2.51 s | 2.51 s |

### Inter-packet gaps: APK clock vs receive clock

Comparison of consecutive packets' APK-stamped `ts_ns` field vs our
receive clock:

| arm   | APK clock median | recv clock median |
|---    |---               |---                |
| LEFT  | 200.0 ms         | 207.9 ms          |
| RIGHT | 166.6 ms         | 165.9 ms          |

APK clock and receive clock agree within a few ms on the packets we
*did* see. **This was originally cited as proof that nothing was being
dropped on the wire — that conclusion was wrong.** The comparison is
silent about packets that never arrived: if the APK sent 100 and we
got 5, the 5 we got could still have matching ts_ns gaps. See "Three-
way packet count" below for the corrected picture.

## Three-way packet count (added 2026-06-01 evening)

To find where the missing packets actually go, we instrumented the
receive side and ran two head-to-head tests. Three counts in play:

1. **APK "sent"** — number reported by the Pico-side APK UI on the
   headset. What the Pico claims it transmitted.
2. **Host NIC** — `tcpdump -i any -n 'udp port 5100' -w …` running
   alongside our app. Counts datagrams that reached the host's NIC
   regardless of whether our socket drained them.
3. **Our app** — `StreamStats.total_packets` counter (cumulative
   since process start; subtract baseline for per-test delta).

We also added:
- **`kernel_drops`** — sampled from `/proc/net/udp` column 12 (UDP
  receive-buffer overflows on our bind port), baselined at process
  start. Visible in the Stream panel.
- **`unread_overwrites`** — counts latest-wins overwrites in
  `_apply_controller` for packets the motion worker hadn't yet
  acknowledged via `mark_consumed`. Visible per-arm in Stream panel.
- **`SO_RCVBUF`** bumped to 1 MB (from Linux default ~212 KB).

### Test A — dual-arm

| count           | value | rate over ~12 s | notes |
|---              |-------|-----------------|-------|
| APK sent        | ~2000 | ~167 Hz         | reported on headset UI |
| **NIC tcpdump** | **66**| **~5.5 Hz**     | source 192.168.3.182:38924 → 192.168.3.156:5100, all `wlP9s9 In` |
| our app         |   145 | —               | cumulative; matches NIC delta |
| kernel_drops    |     0 | —               | no UDP-buffer overflow |
| unread_overwrites | L 0  R 0 | —          | no in-process drops |
| parse errors / unknown | 0 | —           | clean stream |

**~97% loss between APK and host NIC.** Kernel buffer is empty,
overwrite counters are zero, errors are zero — every packet that
reached the kernel was delivered to our app and consumed in time.

### Test B — single-arm

```
sudo tcpdump -i any -n 'udp port 5100' -w /tmp/vr_20260601_140710.pcap
…
1602 packets captured
1612 packets received by filter
0 packets dropped by kernel
```

| count           | value  | notes |
|---              |--------|-------|
| APK sent        | (not recorded)| forgot to read headset UI |
| NIC tcpdump     | 1612   | (1602 captured to disk + 10 in tcpdump's userspace ring) |
| **our app**     | **1602** | matches NIC captured count |

**~0% loss host-side**, and the NIC captured 1612 vs the dual-arm
test's 66 over a similar window. The wireless path can carry
1000+ packets when the APK is actually sending them.

### Diagnosis

The cliff is between APK-egress and host-NIC-ingress, and it kicks in
somewhere between single-arm rate (~30–70 Hz) and dual-arm rate
(~167 Hz). Three remaining hypotheses, none cheaply distinguishable
without further testing:

1. **APK-side egress.** The Pico's own UDP send queue overflows at
   high rate. Closed-source, can't fix from our side.
2. **Wi-Fi air saturation.** 100+ Hz unicast UDP triggers retries /
   loss, especially on 2.4 GHz with other traffic.
3. **AP rate-limiting** past some threshold per client.

**Test that distinguishes (1) from (2)/(3):** USB-tether the Pico if
the APK can bind to that interface, or put the host on a wired bridge
with the Pico on a private 5 GHz AP. If dual-arm at 167 Hz delivers
cleanly, the APK is fine and the wireless link is the bottleneck. If
it still drops to ~5 Hz, the APK itself is shedding load.

What's ruled out either way: our motion worker, our UDP socket, our
kernel receive buffer, our parsing, our latest-wins overwrite logic.

## Reference repo evidence (`openarmx_teleop_vr`)

Cloned to `code_reference/openarmx_reference/`. Two relevant facts:

### 1. The "rate" wire field is a speed-mode toggle

`openarmx_teleop_bridge_vr/src/openarmx_teleop_bridge_vr_node.cpp`
parses field 13 (after pose+trigger+grip+buttons) as `rate`:

```cpp
if (temp_value == 0.1 || temp_value == 1.0) {
  out_sample.rate = temp_value;
} else {
  out_sample.rate = 0.1;
  RCLCPP_WARN(get_logger(), "Invalid rate value detected: %.2f, resetting to 0.1", temp_value);
}
```

…and `openarmx_teleop_vr_node.py` consumes it as:

```python
def _absolute_rate_callback(self, msg: Float32):
    self.absolute_is_full_speed = float(msg.data) >= 0.999
```

…where `is_full_speed` selects between `max_step_deg_*` (slow,
4°/joint/tick) and `fast_max_step_deg_*` (8–20°/joint/tick) in
`_limit_joint_step`. So `rate=0.1` in our recordings means
"the user has the slow-mode toggle on", not anything about Hz.

### 2. The reference design point is ~100 Hz packets

`openarmx_teleop_vr/config/teleop_params.yaml`:

```yaml
control_rate: 100.0          # Hz — receiver-side control loop
grip_threshold: 0.5
```

`openarmx_teleop_vr_node.py` constructs the teleop core with
`stream_timeout_sec=0.3`. So the reference assumes packets at 30–100 Hz
and treats >300 ms as a stale stream. We're getting 200–2500 ms gaps
in dual-arm mode — well into "stale" territory.

## APK disassembly notes

Cloned to `code_reference/openarmx_teleop_vr_apk/`. Repo contains only
binaries (`openarmx-vr-pico.apk`, `openarmx-vr-quest.apk`) — no source.

`lib/arm64-v8a/libBasicDemo.so` symbol map:

| symbol                                    | address     | size    |
|---                                        |---          |---      |
| `BasicDemo::updateControllers()`          | 0x407058    | 4948 B  |
| `updateControllersEv::operator()` lambda 1 | 0x408f8c   | 1164 B  |
| `updateControllersEv::operator()` lambda 2 | 0x409418   |  632 B  |
| `frame_count` (static int in update fn)   | 0x8c5f54    |    4 B  |
| `last_udp_button_a / b / x / y` statics   | 0x8c5f58…   |    2 B each |
| **only `sleep_for` callsite in lib**      | **0x408f38**| —       |

Findings:

- `updateControllers` itself does **NOT** call `sendto` and does NOT
  call `sleep_for`. The two `sendto` callsites (0x408cfc, 0x408d2c)
  are inside the lambda bodies, which run on a worker thread.
- The single `sleep_for` callsite at 0x408f38 sits in an unnamed
  helper between the lambdas. Its argument is a runtime-built
  `chrono::nanoseconds` (multiple constructor calls preceding the
  `bl sleep_for`), consistent with a **"sleep until next scheduled
  send"** pacing loop — i.e. the APK enforces a fixed inter-send
  cadence per-hand, by design.
- The `frame_count` static inside `updateControllers` strongly suggests
  an additional "every Nth render frame" gate before the lambda is
  scheduled.

So: APK paces internally. Whatever cadence the APK has chosen is what
we'll receive — there is no setting we can flip on our side that will
make it send faster.

## What we have ruled out

- Our motion worker can absolutely sustain 30 Hz with both arms enabled.
- ~~Network loss between Pico and host~~ — **retracted**. The
  APK-stamped vs receive-clock gap comparison is only valid for
  packets that arrived; the three-way count above shows ~97% loss
  on the wireless path in dual-arm mode.
- Our own recorder buffering / parsing.
- Kernel UDP-buffer overflow on receive (`kernel_drops` = 0 in dual
  arm test even at 1 MB buffer).
- Latest-wins overwrites in our process (`unread_overwrites` = 0).
- The "rate" wire field controlling send Hz (it doesn't).

## What's still possible

1. **APK has a slow-mode UI toggle** (different from the wire `rate`
   field) that the operator can disable on the headset. The next time
   the headset is on, look for one. The wire `rate=0.1` field IS a
   slow-mode flag, so something on the headset is setting it to 0.1
   right now — that something might be the same control that throttles
   the send cadence.
2. **Dual-controller mode in the APK halves per-hand send rate.** If
   the APK shares one outgoing socket between hands and pacers run
   sequentially, going from 1 active hand → 2 active hands could halve
   the per-hand Hz (we'd expect 30+ → ~15, not 30+ → ~2.5; so this is
   probably not the whole story).
3. **Pico Wi-Fi link quality.** Worth re-testing on a known-good
   network, or a Pico → host USB tether, to rule the network out
   completely. The APK's own ts_ns suggests it isn't the network, but
   confirming is cheap.

## Suggested next steps (in cost order)

1. **Check the APK UI on the Pico** for any rate / slow-mode / low-power
   toggle next time you put the headset on. Cheapest possible check.
2. **USB-tether or wired-bridge the Pico** to bypass Wi-Fi entirely
   for one dual-arm test. This is the single test that distinguishes
   "APK egress is the cliff" from "wireless link is the cliff".
   Result determines whether to fix on our side (mitigation) or on
   the APK side (file an issue).
3. **Open an issue / email** at `openarmrobot@gmail.com`
   (their listed contact) — "we observe ~5 Hz aggregate / 2.5 Hz per
   hand in two-hand mode on Pico 4 Ultra; is there a config to bump
   this?". Also cheap.
4. **Implement client-side prediction** — between packets, advance the
   target along the last observed velocity (with a hard cap), so the
   arm doesn't freeze for 200+ ms gaps. Bounded effort, addresses the
   symptom even if APK never improves.
5. **Switch off the EMA filter when packets are sparse.** Current
   `VR_POSE_FILTER_ALPHA = 0.4` assumes ~30 Hz inputs (50 ms time
   constant). At 5 Hz it adds ~1.5 ticks of lag for no gain. Could
   adapt α to inter-arrival time, or bypass entirely below some Hz
   threshold.

## Cleanup TODO in our code

- Update the comment on `vr_input.py`'s `rate` field and the docstring
  in `motion_perf_log.py` to reflect that `rate` is the slow/fast
  speed-mode toggle, not a send-rate measurement.
- The "perf log summary" interpretation in `docs/perf_debugging.md`
  doesn't depend on this, but the `rate` column in saved recordings
  has been mis-labeled in passing comments.
