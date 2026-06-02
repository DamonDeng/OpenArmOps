# Pico 4 Ultra dev stack survey

Starting line for "should we write our own Pico APK" — what dev paths exist,
what the input/networking story looks like, what we can confirm from public
docs vs. what stays open.

Date of survey: 2026-06. Authoritative source per project lead:
https://developer.picoxr.com/document/

> Caveat on sourcing: the Pico developer site (`developer.picoxr.com`,
> `developer-global.pico-interactive.com`) is a client-side rendered SPA. A
> plain HTTP fetch of any inner doc URL returns only the navigation skeleton,
> so most concrete claims below are sourced from Pico's GitHub
> (`github.com/Pico-Developer`, `github.com/XR-Robotics`) and from the
> Khronos OpenXR registry rather than from the Pico HTML pages directly.
> Pages we couldn't load are listed in "Open questions" and should be opened
> in a real browser before we lock in a plan.

---

## TL;DR — recommended path for our use case

**Recommended: fork or vendor `XRoboToolkit-Unity-Client` and strip it down to
the controller/head-pose UDP sender we want.** Single biggest reason: it is
literally a Pico-authored, MIT-licensed, Pico-4-Ultra-targeted teleop client
that already streams 56-byte pose frames at 90 Hz from headset to PC, with
auto-reconnect and the AndroidManifest/permissions story already solved. We
would change the wire transport (TCP -> UDP), re-pack the payload to our
schema, and delete everything else (camera, video receive, UI flows). We do
not have to invent the OpenXR plumbing, the Unity-Pico project settings, or
the signing/sideload process.
- Repo: https://github.com/XR-Robotics/XRoboToolkit-Unity-Client
- Project page: https://xr-robotics.github.io/

If we wanted a smaller surface area, the second-best path is **native C++
with the OpenXR loader + Khronos `hello_xr` as a starting point**. That gives
us a ~few-hundred-line input loop with no Unity dependency, but the build
toolchain (Android Studio + NDK + OpenXR loader from Pico's runtime) is real
work and there is no Pico-published "BasicDemo" we found that would shortcut
this. The closed-source APK we currently use is named `libBasicDemo.so`, but
no public Pico repo named "BasicDemo" surfaced on GitHub — see "Open
questions".

Unity-via-XRoboToolkit beats native here mainly because the input plumbing,
the device picker UI, and the keystore/signing flow are already wired.

---

## Dev frameworks

### Unity Integration SDK (recommended primary path for us)

- Repo: https://github.com/Pico-Developer/PICO-Unity-Integration-SDK
- **Latest version: 3.4.0** (released 2026-03-25, release notes 2026-02-27)
  per the GitHub repo description.
- Distribution: Unity Package Manager (`package.json` present in repo).
- Unity 6 + AR Foundation 6 are supported as of v1.4.0 of the
  `PICO-Unity-OpenXR-SDK` companion package
  (https://github.com/Pico-Developer/PICO-Unity-OpenXR-SDK; min PICO system
  5.13.0).
- "PICO 4 series" (which includes 4 Ultra in Pico's marketing) is referenced
  for Sense Pack features (Spatial Anchor, Spatial Mesh, Scene Capture, MR
  Safeguard).
- Controller-input docs page: https://developer.picoxr.com/document/unity/input-mapping
  — could not be scraped, see Open questions.
- Sample worth knowing about: `BasicSample-Unity`
  (https://github.com/Pico-Developer/BasicSample-Unity) — three .unitypackage
  files: Controller (vibration + buttons), UI, Interaction. README explicitly
  scopes it to "the Pico Unity Integration SDK and the Unity XR Interaction
  Toolkit." Does NOT mention OpenXR or any `libBasicDemo.so`.

### PICO Unity OpenXR SDK (separate package)

- Repo: https://github.com/Pico-Developer/PICO-Unity-OpenXR-SDK
- Latest: **1.4.0** (Apr 2025).
- Requires PICO device system 5.13.0+, supports Unity 6 and AR Foundation 6.
- Includes high-frequency hand tracking ("60 Hz hand tracking" in 1.4.0
  release notes — note that's hand tracking, not controller input).
- Has known limitation: on system 5.13.0, "finger-pinch rays can't interact
  with UI" — irrelevant to us but a marker that the SDK is actively churning.

### Native OpenXR + Android NDK

- No standalone Pico-authored "native OpenXR sample" repo turned up under
  https://github.com/Pico-Developer (we paged through both pages of the
  org's 48 repos). The closest things are Unity-side `BasicSample-Unity`
  and `Getstarted-Unity`. There is no `Native-OpenXR-Sample` or
  `PICO-OpenXR-SDK` repo under that org — both 404.
- Pico's docs reference a "Native" / "OpenXR Mobile SDK" section
  (https://developer.picoxr.com/document/native, /document/native/openxr-mobile-sdk)
  but those pages are SPA-rendered and we could not extract their contents
  via WebFetch.
- For starter native code, the canonical reference is Khronos's `hello_xr`
  in the OpenXR-SDK-Source repo:
  https://github.com/KhronosGroup/OpenXR-SDK-Source/tree/main/src/tests/hello_xr
  — Gradle Android build present (`build.gradle`, `AndroidManifest.xml`,
  `gradlew`), `platformplugin_android.cpp`, `main.cpp`,
  `openxr_program.cpp`. This is what most third-party tutorials build on.

### Unreal plugin (for completeness; we are not using this)

- Repo: https://github.com/Pico-Developer/PICO-Unreal-Integration-SDK
- Confirmed to exist; companion samples include `MRSample-Unreal`,
  `BasicSample-Unreal`, `SpatialAudioSample-Unreal`, `Getstarted-Unreal`,
  `PlatformSample-Unreal`. Version not extracted (SPA pages).

### Other paths

- **WebXR**: Pico maintains
  https://github.com/Pico-Developer/awesome-webxr-development ("Building
  blocks for WebXR apps") — viable for very-low-friction prototypes, but
  WebXR over UDP is not really a thing; we'd be in browser-WebSocket land.
  Not recommended for our throughput target.
- **MRTK**: not surfaced as Pico-supported in the GitHub org listing.
- **PICO-Emulator**: Pico publishes a Qemu/aemu-based emulator
  (`PICO-Emulator-qemu`, `PICO-Emulator-gfxstream`,
  `PICO-Emulator-manifest`, `PICO-Emulator-aemu`,
  `PICO-Emulator-common` — all under https://github.com/Pico-Developer).
  Useful to know exists; we have hardware so probably won't need it.

---

## OpenXR input on Pico 4 Ultra

What we can say from cross-checking sources:

- Pico's modern stack (Unity Integration SDK, Unity OpenXR SDK, Unreal
  plugin) all sit on top of OpenXR per Pico's own positioning
  (https://github.com/XR-Robotics — "Supports PICO 4 Ultra through OpenXR,
  hand and controller interaction, PICO Motion Tracker").
- The standard OpenXR way to read controller pose + buttons is the **action
  system**: define an `XrAction` of `XR_ACTION_TYPE_POSE_INPUT` for grip/aim
  pose, plus float/bool actions for trigger/grip/A/B/X/Y, bind them via
  interaction profile (e.g. `khr/simple_controller` or a vendor profile),
  call `xrSyncActions` once per frame, then `xrLocateSpace` for the action
  space against a reference space (e.g. `XR_REFERENCE_SPACE_TYPE_LOCAL`).
  Reference: `xrSyncActions` and `xrLocateSpace` man pages at
  https://registry.khronos.org/OpenXR/specs/1.0/man/html/openxr.html
  (we got 403 on direct fetch from the harness; the pages are public in a
  browser).
- We could not pull a definitive list of `XR_PICO_*` extensions. Pico's
  /document/openxr/extensions page is SPA-rendered and not scrapeable. The
  Khronos OpenXR registry (https://registry.khronos.org/OpenXR/) hosts the
  authoritative vendor-extension list — open in a browser and grep for
  `XR_PICO`. **For our minimum-viable use case (controller pose + buttons +
  head pose + UDP) we should not need any vendor extension.** Standard
  OpenXR core 1.0/1.1 covers all of it. `XR_EXT_palm_pose` is now
  consolidated into 1.1 core
  (https://www.khronos.org/openxr/ — "OpenXR 1.1 consolidates multiple
  extensions into the core").

---

## Polling rate — what's possible

This is the question we most need a definitive answer on. We did not get
one from the Pico site. Here is what the OpenXR spec says (in principle —
we couldn't fetch the spec HTML, citing from common knowledge of the spec
plus the Khronos landing page https://www.khronos.org/openxr/):

- `xrSyncActions` is conceptually expected to be called **once per frame**,
  paired with `xrWaitFrame`/`xrBeginFrame`/`xrEndFrame`. It updates the
  action state with whatever input the runtime has buffered.
- `xrLocateSpace(space, baseSpace, time, &location)` accepts an arbitrary
  `XrTime`. Runtimes are required to give a pose at that time, with
  prediction/interpolation as needed. The spec does not forbid calling it
  off-thread or at a higher rate than the frame loop. In practice, on Quest
  this is the standard trick for "high-rate controller pose" — you keep the
  frame loop at 72/90/120 Hz but call `xrLocateSpace` at a higher rate from
  a worker thread.
- However: on most runtimes the **underlying tracker IMU rate** is what
  bounds the practical output. Quest controllers are commonly cited at
  ~500 Hz IMU but the OpenXR runtime smooths/predicts; effective unique
  pose samples often top out at 200-500 Hz. Pico has not published an
  equivalent number that we found.
- Compositor frame rate on Pico 4 Ultra: per Pico marketing, the device
  supports 72/90/120 Hz refresh. The /document/discover/xr-devices
  page should confirm; we couldn't scrape it.

**Strong claim we can make:** OpenXR does not architecturally lock input
poll rate to compositor frame rate. `xrLocateSpace` accepts arbitrary
`XrTime` and can be called from a side thread.

**Weak/unknown claim:** what Pico's runtime actually delivers for
controller pose at >90 Hz. We should write a 30-line probe that just spins
on `xrLocateSpace` from a thread and counts unique-pose samples per second.
That would settle it in an afternoon.

We did get one **concrete, Pico-side, datapoint** from XRoboToolkit:
> "Pose Data Channel: 90Hz with low-latency priority"
> "56 bytes/frame"
- https://github.com/XR-Robotics/XRoboToolkit-Unity-Client (README)

So Pico's own teleop reference design ships at **90 Hz pose**, comfortably
above the 5-7 Hz we're seeing today and at exactly the compositor rate.
Whether that's a hard ceiling or just a sensible default, the README does
not say.

---

## Networking & permissions

What we know:

- XRoboToolkit-Unity-Client uses **TCP for pose** and **UDP for video**
  (asynchronous `UdpClient`). Their `Network/` folder confirms both:
  `TcpClient.cs`, `TcpHandler.cs`, `TcpServer.cs`, `UdpReceiver.cs`,
  `NetworkDataProtocol*.cs`. Source listing:
  https://github.com/XR-Robotics/XRoboToolkit-Unity-Client/tree/main/Assets/Scripts/Network
- `TcpClient.cs` is a thin Unity wrapper around a Java helper
  `com.picovr.robotassistantlib.TcpClient` (shipped as
  `robotassistant_lib-i18n-release.aar`). The bare `Send(byte[])` has no
  rate limiter, so the cadence is set by the caller.
- The README does NOT enumerate `AndroidManifest.xml` permissions. For
  generic Android, opening a UDP socket requires the
  `android.permission.INTERNET` permission (free-tier, no runtime grant).
  No background-service permission is needed for an in-app UDP socket
  while the app is foregrounded.
- Pico-specific gotchas mentioned in XRoboToolkit:
  - VST passthrough camera requires `PXR_Enterprise.OpenVSTCamera()` and
    "special enterprise approval" from Pico — irrelevant to us, but a
    reminder that Pico has an enterprise-only feature gate.
  - Serial number readout only works on the Pico 4 Ultra **enterprise**
    variant.
  - Security zone (guardian) can be disabled via
    `PXR_Enterprise.SwitchSystemFunction(...)` — also enterprise.
- "Number of APKs associated with a key exceeds limit" is a documented
  signing failure; Pico links to a help article for it (per XRoboToolkit
  README). Plan to reuse one keystore per project.
- The standard Android caveats apply: targeting modern Android (API 33+)
  requires foreground-service declarations if we want to keep streaming
  with the headset asleep. For "user wears headset, runs app" use we
  don't need that.

What we do NOT know from public docs:
- Whether Pico has any MagicWindow / VR-mode policy that throttles or
  pauses Android networking when the compositor is busy.
- Whether there is a Pico-specific permission to keep CPU/Wi-Fi awake at
  full clocks (Quest has a "boost mode" equivalent).

---

## Build & sideload (Linux host)

What we know:

- Android Studio + NDK is required for either Unity or native paths.
  XRoboToolkit-Unity-Client pins specific versions
  (https://github.com/XR-Robotics/XRoboToolkit-Unity-Client README):
  - Unity 2022.3.16f1+
  - Android Studio 4.2.2+
  - **Android SDK 29**
  - **Android NDK 21.4.7075529**
  - PICO Integration SDK 3.1.2
- Sideload uses `adb` directly (XRoboToolkit instructs `adb pull` /
  `adb push` against `/sdcard/Android/data/com.xrobotoolkit.client/files/`,
  which only works once the device is in developer mode and authorized
  over USB / Wi-Fi adb).
- Pico provides a desktop tool called "PICO Developer Center" (PDC)
  https://developer-global.pico-interactive.com/resources/#pdc — this is
  an alternative to raw adb but is not required.
- A Pico-developer-mode toggle on the headset itself is required before
  adb works. The official instructions live at
  https://developer.picoxr.com/document/unity/enable-developer-mode/ —
  page is SPA-rendered, we could not scrape it. In our experience with
  similar Android-based headsets the toggle is "tap build number 7 times
  in About → developer mode on → enable USB debugging."
- We could NOT confirm from the docs whether PDC has a Linux build. Many
  Android XR vendor desktop tools are Windows-only. **adb itself is
  Linux-native**, so even if PDC is Windows-only we can develop end-to-end
  on Ubuntu 24.04 using Android Studio + adb.
- Unity Editor itself runs on Linux (officially supported since 2019.x).

Bottom line: we can almost certainly do this entirely from Ubuntu 24.04.
Confirm PDC's Linux availability before assuming we can use Pico's GUI
flashing tools.

---

## Prior art / starter samples

Ranked by how close they are to "minimal Pico APK that streams controller
poses":

1. **`XR-Robotics/XRoboToolkit-Unity-Client`** — closest possible match.
   Pico-authored, MIT-licensed, Pico-4-Ultra-targeted, already streams
   pose frames at 90 Hz over the network, has the AndroidManifest /
   keystore / Unity project pre-wired. Companion `XRoboToolkit-PC-Service`
   (https://github.com/XR-Robotics/XRoboToolkit-PC-Service) is the Linux
   receiver side. A Quest variant exists:
   `XRoboToolkit-Unity-Client-Quest`. Paper:
   https://arxiv.org/abs/2508.00097
2. **`Pico-Developer/BasicSample-Unity`**
   (https://github.com/Pico-Developer/BasicSample-Unity) — three small
   `.unitypackage` files (Controller, UI, Interaction) demonstrating the
   Pico Unity Integration SDK + Unity XR Interaction Toolkit. Cleanest
   demo of "read button, vibrate motor." No networking, no OpenXR.
   We did NOT find conclusive evidence that this sample maps to the
   `libBasicDemo.so` we observed in the closed-source APK. It is a likely
   candidate (same naming convention, same author org) but unconfirmed.
3. **`Pico-Developer/Getstarted-Unity`**
   (https://github.com/Pico-Developer/Getstarted-Unity) — full-scene VR
   demo with locomotion + teleport. Older (Unity 2020.3.48, SDK 2.3.0,
   PICO system 5.7.0). Less useful for us.
4. **`KhronosGroup/OpenXR-SDK-Source/src/tests/hello_xr`**
   (https://github.com/KhronosGroup/OpenXR-SDK-Source/tree/main/src/tests/hello_xr)
   — vendor-neutral OpenXR sample with full Android Gradle build. Use
   this as the baseline for the native-C++ alternative path.
5. `Pico-Developer/PICOMotionTrackerSample-Unity`
   (https://github.com/Pico-Developer/PICOMotionTrackerSample-Unity) —
   only relevant if we ever add aux trackers (elbow / waist).

---

## Gotchas

Things that would surprise us a week in:

- **Compositor must run.** Even a "headless input streamer" Pico APK has
  to be a normal VR app — it must call `xrBeginSession` /
  `xrWaitFrame` / `xrEndFrame` or it will be killed by the runtime. The
  OpenXR session will not deliver controller poses outside an active
  frame loop. (Standard OpenXR semantics, not Pico-specific.) We can
  render a black quad and that's fine.
- **Unity's `UnityEngine.Input` is not the right API on OpenXR.** Use
  the OpenXR action system (XR Interaction Toolkit's input actions, or
  raw `InputDevice` from `UnityEngine.XR.InputDevices`). The legacy
  Input class will give stale or zeroed pose data.
- **Pico's own teleop reference ships at 90 Hz, not higher.** That is
  either a deliberate choice (sufficient for IK) or a runtime ceiling.
  We need to verify before committing to "100+ Hz per controller."
- **Signing.** "Number of APKs associated with a key exceeds limit"
  shows up if we sign too many distinct APKs against one Pico keystore
  during dev. Reuse a single dev keystore per project.
- **Enterprise gates.** Several Pico features (VST passthrough camera,
  serial number, security-zone disable) require an enterprise-tier
  approval. None of our base needs do, but if we ever want raw camera
  frames out of the headset, that's a different conversation with Pico.
- **Min Android API.** Not stated in any README we read. Pico devices
  typically run a custom Android 12+ image; assume we can target API 29
  per XRoboToolkit, with min likely API 26-29.
- **Wi-Fi to LAN host.** The headset and the host must be on the same
  reachable network (or the host must run a hotspot the headset
  joins). In our prior diagnosis the wire-level bursting was on the
  Pico side, not the network — so a clean UDP-from-the-headset path
  will likely be MTU-bound, not network-bound.
- **No public Pico "BasicDemo" repo.** The closed-source APK we have
  links `libBasicDemo.so`, but we could not find a Pico-published
  "BasicDemo" sample in any of the 48 Pico-Developer GitHub repos. The
  name is generic enough that it may be the third-party vendor's
  internal naming, not a Pico-published sample.

---

## Open questions we couldn't answer from public docs

1. **Definitive list of `XR_PICO_*` OpenXR vendor extensions** for Pico 4
   Ultra. The /document/openxr/extensions page is SPA-rendered. Open in
   a browser; cross-check against
   https://registry.khronos.org/OpenXR/.
2. **Hard ceiling on Pico 4 Ultra controller pose update rate** through
   OpenXR. Pico's reference uses 90 Hz. Whether `xrLocateSpace` from a
   side thread can deliver unique samples at 200-500 Hz is unverified.
   *Action:* write a probe that spins on `xrLocateSpace` and counts
   sample changes / second. Single afternoon of work.
3. **Latest Pico OpenXR Mobile SDK version + download URL.** The
   /document/native/openxr-mobile-sdk page is SPA-rendered and the
   release-notes URL we tried 403'd. The SDK is referenced in the doc
   tree but we could not get a current version number.
4. **Linux support for PICO Developer Center (PDC).** The PDC resource
   page is gated behind the SPA. adb itself works on Linux, so this is
   only a question of "do we lose any nice-to-have GUI tooling." Confirm
   with a quick check of https://developer-global.pico-interactive.com/resources/#pdc
   in a real browser.
5. **Whether `BasicSample-Unity` is the actual source of the
   `libBasicDemo.so` we see in the third-party APK.** Possible but
   unverified.
6. **Pico-specific network/CPU governor behavior in VR mode.** No public
   doc on whether the Pico runtime throttles networking, the JVM, or
   the CPU when in compositor mode. Worth a forum post
   (https://developer-global.pico-interactive.com/community/) before
   shipping.
7. **AndroidManifest minimums for our use case.** No Pico doc page we
   could reach enumerated the manifest requirements for a non-Store
   sideloaded app. Generic Android needs `INTERNET`; Pico likely needs
   the standard VR/intent-filter declarations
   (`<category android:name="com.oculus.intent.category.VR"/>` analog —
   Pico uses `com.pico.intent.category.VR` historically; verify against
   an XRoboToolkit-Unity-Client manifest, which we could read directly
   from the repo).
