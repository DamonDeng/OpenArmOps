"""Analyze just the vr_tz (forward axis) values in a recorded log.

Wire format: ``kind tx ty tz qx qy qz qw trig grip A B X Y rate ts_ns``
so ``tz`` is field index 3 of the raw datagram (or index 3 of
``raw.split()`` since the kind is at index 0).

For each LEFT/RIGHT controller packet that is grip-engaged and not a
synthetic-zero packet, report:
  - count of negative tz values
  - min, max, mean, median of all tz
  - same stats for negative-only and positive-only subsets
  - histogram by 1-cm buckets

Usage:
    python3 -m openarm_controller_ui_lerobot.tools.analyze_vr_z <log.jsonl>
        [--side LEFT|RIGHT|both]   default: both
        [--engaged-only]           default: only grip>=0.8 packets
        [--include-synthetic]      default: skip pos=(0,0,0)+identity packets
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path


GRIP_THRESHOLD = 0.8


def is_synthetic(tx: float, ty: float, tz: float,
                 qx: float, qy: float, qz: float, qw: float) -> bool:
    return (tx == 0.0 and ty == 0.0 and tz == 0.0
            and qx == 0.0 and qy == 0.0 and qz == 0.0 and qw == 1.0)


def collect_z(path: Path, side: str, engaged_only: bool,
              include_synth: bool) -> list[tuple[int, str, float, float]]:
    """Return list of (line_no, kind, grip, tz) tuples."""
    out: list[tuple[int, str, float, float]] = []
    with path.open("r", encoding="utf-8") as f:
        for ln_no, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            tokens = str(rec.get("raw", "")).split()
            if len(tokens) < 16:
                continue
            kind = tokens[0].upper()
            if kind not in ("LEFT", "RIGHT"):
                continue
            if side != "both" and kind != side:
                continue
            try:
                tx, ty, tz = float(tokens[1]), float(tokens[2]), float(tokens[3])
                qx, qy, qz, qw = (float(tokens[4]), float(tokens[5]),
                                  float(tokens[6]), float(tokens[7]))
                grip = float(tokens[9])
            except ValueError:
                continue
            if not include_synth and is_synthetic(tx, ty, tz, qx, qy, qz, qw):
                continue
            if engaged_only and grip < GRIP_THRESHOLD:
                continue
            out.append((ln_no, kind, grip, tz))
    return out


def stats(label: str, values: list[float]) -> None:
    if not values:
        print(f"  {label:20s}: (empty)")
        return
    print(f"  {label:20s}: n={len(values):4d}  "
          f"min={min(values):+8.4f}  max={max(values):+8.4f}  "
          f"mean={statistics.mean(values):+8.4f}  "
          f"median={statistics.median(values):+8.4f}")


def histogram(values: list[float], bin_width: float = 0.01) -> None:
    if not values:
        return
    lo = min(values)
    hi = max(values)
    # Round to bin edges for readability.
    lo_bin = int(lo / bin_width) * bin_width if lo >= 0 else (int(lo / bin_width) - 1) * bin_width
    hi_bin = (int(hi / bin_width) + 1) * bin_width
    bins: dict[int, int] = {}
    for v in values:
        b = int(v / bin_width)
        bins[b] = bins.get(b, 0) + 1
    max_count = max(bins.values())
    bar_max = 60
    print(f"  Histogram (bin width {bin_width*100:.0f} cm):")
    print(f"  {'range (m)':>16s}  {'count':>6s}  {'%':>5s}")
    total = len(values)
    for b in sorted(bins.keys()):
        lo_e = b * bin_width
        hi_e = lo_e + bin_width
        count = bins[b]
        pct = count / total * 100
        bar = "#" * int(count / max_count * bar_max)
        print(f"  [{lo_e:+.2f}, {hi_e:+.2f})  {count:>6d}  {pct:>5.1f}%  {bar}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--side", choices=("LEFT", "RIGHT", "both"),
                        default="both")
    parser.add_argument("--engaged-only", action="store_true",
                        help="Only consider packets with grip ≥ 0.8.")
    parser.add_argument("--include-synthetic", action="store_true",
                        help="Include synthetic (pos=0, qw=1) packets.")
    parser.add_argument("--bin-cm", type=float, default=2.0,
                        help="Histogram bin width in cm (default 2).")
    args = parser.parse_args(argv)

    if not args.input.exists():
        print(f"input not found: {args.input}", file=sys.stderr)
        return 2

    rows = collect_z(args.input, args.side,
                     args.engaged_only, args.include_synthetic)
    if not rows:
        print("No matching packets found.")
        return 1

    zs = [tz for _, _, _, tz in rows]
    neg = [z for z in zs if z < 0]
    pos = [z for z in zs if z > 0]
    zero = [z for z in zs if z == 0]

    side_str = ("both controllers" if args.side == "both"
                else f"{args.side} controller")
    eng_str = "grip ≥ 0.8" if args.engaged_only else "all grip values"
    synth_str = ("(including synthetic)" if args.include_synthetic
                 else "(synthetic excluded)")

    print(f"\nFile:    {args.input}")
    print(f"Filter:  {side_str}, {eng_str} {synth_str}")
    print(f"Total packets matched: {len(zs)}")
    print()

    print("Statistics for vr_tz (controller forward axis, +Z = forward in Pico frame):")
    stats("all values", zs)
    stats("negative only", neg)
    stats("positive only", pos)
    if zero:
        print(f"  exact zeros        : n={len(zero)}")
    print(f"  fraction negative  : {len(neg)}/{len(zs)} = {len(neg)/len(zs)*100:.1f}%")
    print()
    histogram(zs, bin_width=args.bin_cm / 100.0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
