"""Split a VR recorder .jsonl by ABAB...XYXY sentinel sequences.

Usage:
    python3 -m openarm_controller_ui_lerobot.tools.split_vr_log <log.jsonl>

Each detected start → end range is written to a separate .jsonl in the
same directory as the input, named ``<base>_test_<NN>.jsonl``. The
sentinel trigger packets are *included* in the output (the user asked
for them to be kept so each test segment is self-marking).

Sentinel definition
-------------------
The canonical start sentinel is ``ABAB`` and the canonical end sentinel
is ``XYXY``. Each sentinel matches when its letter sequence appears as
rising-edge button presses on the *same* controller, with consecutive
gaps ≤ ``MAX_GAP_S``.

To tolerate the empirically-observed "dropped first press" behaviour
(when a sentinel attempt starts during a grip-released period the
APK frequently drops the first press), we accept any contiguous
3-of-4 sub-sequence of the canonical sentinel as a match — i.e.
``ABAB`` matches if the controller fires ``ABA``, ``BAB``, or the
full ``ABAB``. This dramatically improves recall without making the
sentinel ambiguous: the start (ABA/BAB) and end (XYX/YXY) letter
sets are disjoint.

A "button press" is an edge: a packet where the named button bit is 1
but the previous packet of the same controller had it at 0 (so a held
button only counts as one press).

Either controller can carry the sentinel; we don't require start and
end to be on the same side. The first start seen opens a segment; the
next end closes it. Subsequent starts without an intervening end are
ignored with a warning (we don't support nested segments).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


# Maximum seconds between consecutive presses to still consider them
# part of the same sentinel. 1.5s gives the user comfortable room
# without tolerating an accidental press from a previous test bleeding
# into the next.
MAX_GAP_S = 1.5

START_SENTINEL = ("A", "B", "A", "B")
END_SENTINEL = ("X", "Y", "X", "Y")


@dataclass
class ParsedRecord:
    """One line of the log, lightly parsed. We keep the original raw
    JSON line for round-tripping into the output files unchanged.
    """
    raw_line: str           # the source .jsonl line including newline-stripping
    t: float
    kind: str               # "LEFT" / "RIGHT" / "HEAD" / ...
    a: int = 0
    b: int = 0
    x: int = 0
    y: int = 0


def parse_line(line: str) -> ParsedRecord | None:
    """Return a ParsedRecord, or None if the line is malformed or not a
    LEFT/RIGHT/HEAD/... record we can read.
    """
    line = line.rstrip("\n")
    if not line.strip():
        return None
    try:
        rec = json.loads(line)
    except json.JSONDecodeError:
        return None
    raw = str(rec.get("raw", ""))
    tokens = raw.split()
    if not tokens:
        return None
    kind = tokens[0].upper()
    pr = ParsedRecord(raw_line=line, t=float(rec.get("t", 0.0)), kind=kind)
    if kind in ("LEFT", "RIGHT") and len(tokens) >= 14:
        # raw.split() keeps the kind at index 0, so the field offsets are
        # +1 vs the ones in vr_input._apply_controller (which strips kind
        # first). Layout: [0]=kind [1..3]=tx ty tz [4..7]=qx qy qz qw
        # [8]=trigger [9]=grip [10..13]=A B X Y [14]=rate [15]=ts_ns.
        try:
            pr.a = int(float(tokens[10]))
            pr.b = int(float(tokens[11]))
            pr.x = int(float(tokens[12]))
            pr.y = int(float(tokens[13]))
        except ValueError:
            return pr  # keep timing, drop button info
    return pr


@dataclass
class ButtonEdge:
    """A rising-edge (0→1) of one button on one controller."""
    line_idx: int     # index into the records list
    t: float
    kind: str         # "LEFT" or "RIGHT"
    letter: str       # "A" / "B" / "X" / "Y"


def find_button_edges(records: list[ParsedRecord]) -> list[ButtonEdge]:
    """Walk records in order; for each LEFT/RIGHT packet, compare button
    bits to the previous packet of the *same* controller. Emit one
    ButtonEdge per rising edge.
    """
    edges: list[ButtonEdge] = []
    last_state = {"LEFT": (0, 0, 0, 0), "RIGHT": (0, 0, 0, 0)}
    for i, r in enumerate(records):
        if r.kind not in ("LEFT", "RIGHT"):
            continue
        cur = (r.a, r.b, r.x, r.y)
        prev = last_state[r.kind]
        for letter, c, p in zip("ABXY", cur, prev):
            if c == 1 and p == 0:
                edges.append(ButtonEdge(line_idx=i, t=r.t,
                                        kind=r.kind, letter=letter))
        last_state[r.kind] = cur
    return edges


def _try_match_at(
    edges: list[ButtonEdge],
    start: int,
    pattern: str,
    max_gap_s: float,
) -> list[int] | None:
    """Try to match ``pattern`` (e.g. "BAB") starting at edges[start].
    Returns the list of matched edge indices on success, or None.
    All matched edges must be on the same controller as edges[start],
    consecutive same-controller edges must be ≤ max_gap_s apart, and
    intervening same-controller edges with the wrong letter abort the
    match.
    """
    if start >= len(edges) or edges[start].letter != pattern[0]:
        return None
    side = edges[start].kind
    matched = [start]
    last_t = edges[start].t
    j = start + 1
    for expected in pattern[1:]:
        while j < len(edges):
            e = edges[j]
            if e.kind != side:
                j += 1
                continue
            if e.t - last_t > max_gap_s:
                return None
            if e.letter == expected:
                matched.append(j)
                last_t = e.t
                j += 1
                break
            return None  # wrong letter on same side aborts the match
        else:
            return None  # ran out of edges
    return matched


def find_sentinel_matches(
    edges: list[ButtonEdge],
    sentinel: tuple[str, ...],
    max_gap_s: float = MAX_GAP_S,
    min_subsequence: int = 3,
) -> list[tuple[int, int]]:
    """Return non-overlapping ``(first_edge_idx, last_edge_idx)`` pairs
    where the sentinel matches.

    The sentinel is the *canonical* sequence (e.g. ``("A","B","A","B")``).
    To tolerate dropped first-press packets, we accept any contiguous
    sub-sequence of length ≥ ``min_subsequence`` as a match. We try the
    full sentinel first and fall back to shorter sub-sequences only if
    the full one didn't match starting at this position.
    """
    full = "".join(sentinel)
    if min_subsequence >= len(full):
        candidates = [full]
    else:
        # Generate all contiguous sub-sequences of length >= min_subsequence,
        # longest-first so we prefer the full match when available.
        candidates = []
        for length in range(len(full), min_subsequence - 1, -1):
            for offset in range(0, len(full) - length + 1):
                sub = full[offset:offset + length]
                if sub not in candidates:
                    candidates.append(sub)

    matches: list[tuple[int, int]] = []
    i = 0
    n = len(edges)
    while i < n:
        chosen = None
        for pat in candidates:
            m = _try_match_at(edges, i, pat, max_gap_s)
            if m is not None:
                chosen = m
                break
        if chosen is not None:
            matches.append((chosen[0], chosen[-1]))
            i = chosen[-1] + 1
        else:
            i += 1
    return matches


def split_log(input_path: Path, output_dir: Path | None = None) -> list[Path]:
    """Read input_path, find ABAB → XYXY pairs, write one file per pair.
    Returns the list of output file paths (in order).
    """
    if output_dir is None:
        output_dir = input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    records: list[ParsedRecord] = []
    with input_path.open("r", encoding="utf-8") as f:
        for ln_no, line in enumerate(f, 1):
            pr = parse_line(line)
            if pr is None:
                continue
            records.append(pr)
    logger.info(f"Loaded {len(records)} parseable records from {input_path}")

    edges = find_button_edges(records)
    logger.info(f"Detected {len(edges)} button rising-edges")

    starts = find_sentinel_matches(edges, START_SENTINEL)
    ends = find_sentinel_matches(edges, END_SENTINEL)
    logger.info(f"Sentinel hits: ABAB={len(starts)}, XYXY={len(ends)}")

    if not starts:
        logger.warning("No ABAB start sentinels found; nothing to split.")
        return []

    # Pair each start with the next end whose first edge is after the
    # start's last edge. Iterate ends with a pointer.
    base = input_path.stem
    out_paths: list[Path] = []
    end_ptr = 0
    test_no = 0
    for s_first, s_last in starts:
        s_first_line = edges[s_first].line_idx
        s_last_line = edges[s_last].line_idx
        # Find the first XYXY whose first edge is strictly after s_last.
        while end_ptr < len(ends) and edges[ends[end_ptr][0]].line_idx <= s_last:
            end_ptr += 1
        if end_ptr < len(ends):
            e_first, e_last = ends[end_ptr]
            e_last_line = edges[e_last].line_idx
            end_ptr += 1
            tail_line = e_last_line  # inclusive upper bound
            kind = "complete"
        else:
            # No XYXY available; segment runs to end of log.
            tail_line = len(records) - 1
            kind = "open-ended (no XYXY found after this ABAB)"

        test_no += 1
        out_path = output_dir / f"{base}_test_{test_no:02d}.jsonl"
        with out_path.open("w", encoding="utf-8") as fout:
            for r in records[s_first_line:tail_line + 1]:
                fout.write(r.raw_line + "\n")
        n_lines = tail_line - s_first_line + 1
        out_paths.append(out_path)
        logger.info(
            f"test_{test_no:02d}: lines {s_first_line + 1}..{tail_line + 1} "
            f"({n_lines} packets, {kind}) → {out_path.name}"
        )

    if end_ptr < len(ends):
        leftover = len(ends) - end_ptr
        logger.warning(
            f"{leftover} XYXY sentinel(s) had no preceding ABAB and were "
            f"ignored."
        )

    return out_paths


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Split a VR log into per-test segments delimited by "
                    "ABAB (start) and XYXY (end) button sentinels."
    )
    parser.add_argument("input", type=Path,
                        help="Path to the .jsonl recorder log to split.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Directory for output files (default: same as input).")
    args = parser.parse_args(argv)

    if not args.input.exists():
        logger.error(f"input file not found: {args.input}")
        return 2

    out_paths = split_log(args.input, args.output_dir)
    if not out_paths:
        return 1
    print(f"\nWrote {len(out_paths)} segment file(s):")
    for p in out_paths:
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
