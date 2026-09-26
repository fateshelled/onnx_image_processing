"""Re-aggregate Sim(3) diagnostic rows: gate sensitivity and robustness.

Reads the diag JSON reports (``rows``) and computes:
- gate sensitivity per sequence (accepted = fit_ok and n_inliers >= gate)
- leave-one-sequence-out (LOSO) gate selection: pick the smallest gate with
  zero false accepts on the remaining sequences, then evaluate it on the
  held-out sequence
- locked-gate (default 6) evaluation on every sequence
- candidate-level Wilson CI vs the sequence-cluster view of false accepts

The source diagnostic uses odometry windows only.  These statistics therefore
describe that offline population, not the default-on online profile with
keyframe-baseline windows; use ``scripts/ab_loop_verifier.py`` for deployment
gate comparisons.

Acceptance is a deterministic function of ``fit_ok``/``n_inliers`` in the
rows.  Re-aggregation is exact only when the diagnostic fit floor is no larger
than the smallest requested gate; incompatible legacy reports are rejected.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def wilson(k, n, z=1.96):
    if n == 0:
        return (None, None)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def load_sequences(paths):
    sequences = []
    seen = set()
    for path in paths:
        for report in json.loads(Path(path).read_text()):
            if report.get("error"):
                continue
            name = report["sequence"]
            if name in seen:
                raise SystemExit(
                    f"duplicate sequence in reports: {name}; pass one "
                    "population (one run) per analysis")
            seen.add(name)
            sequences.append(report)
    return sequences


def validate_fit_floor(reports, requested_gates):
    """Reject reports that discarded candidates needed by a lower gate."""
    minimum_gate = min(requested_gates)
    for report in reports:
        options = report.get("options", {})
        # Reports before --min-tracks used --min-inliers as both fit floor and
        # deployment gate, so that is the only safe legacy fallback.
        floor = int(options.get("min_tracks", options.get("min_inliers", 0)))
        if floor > minimum_gate:
            raise SystemExit(
                f"{report['sequence']}: diagnostic fit floor {floor} exceeds "
                f"requested gate {minimum_gate}; rerun diag_loop_sim3.py "
                f"with --min-tracks <= {minimum_gate}")


def select_loso_gate(reports):
    """Smallest gate with no false accept on ``reports`` (None if unknown)."""
    false_fits = [row.get("n_inliers", 0) for report in reports
                  for row in report["rows"]
                  if not row["label"] and row.get("fit_ok")]
    return None if not false_fits else max(false_fits) + 1


def accepted(row, gate):
    return bool(row.get("fit_ok") and row.get("n_inliers", 0) >= gate)


def false_accepts(report, gate):
    return sum(1 for row in report["rows"]
               if not row["label"] and accepted(row, gate))


def true_accepts(report, gate):
    return sum(1 for row in report["rows"]
               if row["label"] and accepted(row, gate))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+")
    parser.add_argument("--gates", default="6,8,9,10,11,12,15")
    parser.add_argument("--locked-gate", type=int, default=6)
    parser.add_argument("--uncertainty-gate", type=int, default=8,
                        help="gate used for the false-accept uncertainty "
                             "section")
    opts = parser.parse_args()

    gates = [int(value) for value in opts.gates.split(",")]
    sequences = load_sequences(opts.reports)
    validate_fit_floor(
        sequences, gates + [opts.locked_gate, opts.uncertainty_gate])

    print("== gate sensitivity (true/false accepts per sequence)")
    header = "sequence".ljust(35) + "".join(f">={g:<7}" for g in gates)
    print(header)
    for report in sequences:
        cells = [f"{true_accepts(report, g)}/{false_accepts(report, g)}"
                 for g in gates]
        print(report["sequence"].ljust(35)
              + "".join(cell.ljust(9) for cell in cells))
    print()
    for gate in gates:
        n_true = sum(1 for report in sequences for row in report["rows"]
                     if row["label"])
        n_false = sum(1 for report in sequences for row in report["rows"]
                      if not row["label"])
        ta = sum(true_accepts(report, gate) for report in sequences)
        fa = sum(false_accepts(report, gate) for report in sequences)
        lo, hi = wilson(fa, n_false)
        interval = f" Wilson95=[{lo:.4f},{hi:.4f}]" if lo is not None else ""
        print(f"gate>={gate:<3}: true {ta}/{n_true}  false {fa}/{n_false}"
              f"{interval}")

    print()
    print("== leave-one-sequence-out gate selection")
    print("(gate = smallest gate with zero false accepts on the remaining "
          "sequences)")
    failures = 0
    uninformative = 0
    for held_out in sequences:
        rest = [report for report in sequences if report is not held_out]
        selected = select_loso_gate(rest)
        if selected is None:
            uninformative += 1
            held_false_fits = sum(
                1 for row in held_out["rows"]
                if not row["label"] and row.get("fit_ok"))
            print(f"hold out {held_out['sequence']:35s} selected gate= n/a"
                  "  [no_information: no false fits in the training folds;"
                  f" held-out false fits: {held_false_fits}]")
            continue
        held_false = false_accepts(held_out, selected)
        held_true = true_accepts(held_out, selected)
        status = "FAIL" if held_false else "ok"
        if held_false:
            failures += 1
        print(f"hold out {held_out['sequence']:35s} selected gate={selected:2d}"
              f"  held-out true {held_true:2d}"
              f" false {held_false:2d}  [{status}]")
    informative = len(sequences) - uninformative
    print(f"LOSO informative folds failing: {failures}/{informative}"
          f" (no_information folds: {uninformative}/{len(sequences)})")

    print()
    print(f"== locked gate {opts.locked_gate}")
    for report in sequences:
        print(f"{report['sequence']:35s} true {true_accepts(report, opts.locked_gate):2d}"
              f" false {false_accepts(report, opts.locked_gate):2d}")

    print()
    gate = opts.uncertainty_gate
    print(f"== false-accept uncertainty (candidate vs sequence cluster,"
          f" gate>={gate})")
    clusters = [(report["sequence"],
                 sum(1 for row in report["rows"] if not row["label"]),
                 false_accepts(report, gate))
                for report in sequences
                if any(not row["label"] for row in report["rows"])]
    n_false = sum(c[1] for c in clusters)
    n_fa = sum(c[2] for c in clusters)
    if n_false == 0:
        print("candidate-level: no false candidates in the reports (n/a)")
        print("cluster-level: n/a (no false candidates)")
        return
    lo, hi = wilson(n_fa, n_false)
    print(f"candidate-level: {n_fa}/{n_false} Wilson95=[{lo:.4f},{hi:.4f}]"
          " (assumes independent candidates)")
    for name, n_cand, n_acc in clusters:
        print(f"  cluster {name:35s} false candidates {n_cand:3d}"
              f" false accepts {n_acc}")
    k_clusters = sum(1 for c in clusters if c[2])
    cluster_lo, cluster_hi = wilson(k_clusters, len(clusters))
    print(f"cluster-level: {k_clusters}/{len(clusters)} sequences with false"
          f" candidates had >=1 false accept"
          f" (Wilson95=[{cluster_lo:.4f},{cluster_hi:.4f}])"
          " (intervals are dominated by between-sequence variance)")


if __name__ == "__main__":
    main()
