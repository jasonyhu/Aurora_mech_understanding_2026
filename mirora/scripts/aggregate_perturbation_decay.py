"""
Aggregate the per-task CSVs from compute_perturbation_decay.py into a
mean ± std summary table over initializations (SI perturbation-decay table).
Prints a Markdown table and a LaTeX tabular.
"""

import csv
import sys
from collections import defaultdict

import numpy as np

import consts

OUT_DIR = consts.NOX_ANALYSIS_PATH / "perturbation_decay"

REPORT_LEAD_HOURS = [12, 24, 36, 60, 120]


def main() -> None:
    rows = []
    for path in sorted(OUT_DIR.glob("decay_partial_*.csv")):
        with open(path, newline="") as f:
            rows.extend(csv.DictReader(f))
    if not rows:
        print(f"No partial CSVs found in {OUT_DIR}", file=sys.stderr)
        sys.exit(1)
    print(f"Loaded {len(rows)} rows from {OUT_DIR}")

    # bucket: (ppb, lead_hours) -> list of fractions across init_times
    buckets: dict[tuple[float, int], list[float]] = defaultdict(list)
    for r in rows:
        buckets[(float(r["ppb"]), int(r["lead_hours"]))].append(float(r["fraction_remaining"]))

    # init-time count per ppb (for reporting in caption)
    n_inits = {ppb: len({r["init_time"] for r in rows if float(r["ppb"]) == ppb})
               for ppb in consts.NOX_MAGNITUDES_PPB}

    # (mean, std) in % per magnitude and lead
    stats = {}
    for ppb in consts.NOX_MAGNITUDES_PPB:
        for lead in REPORT_LEAD_HOURS:
            vals = np.asarray(buckets[(ppb, lead)]) * 100.0
            stats[(ppb, lead)] = vals.mean(), vals.std(ddof=1)

    print("\n## Markdown table\n")
    print("| Perturbation | " + " | ".join(f"+{h}h" for h in REPORT_LEAD_HOURS) + " |")
    print("|" + "---|" * (1 + len(REPORT_LEAD_HOURS)))
    for ppb in consts.NOX_MAGNITUDES_PPB:
        cells = [f"{mean:.1f} ± {std:.1f}%" for mean, std in (stats[(ppb, h)] for h in REPORT_LEAD_HOURS)]
        print(f"| +{ppb} ppb | " + " | ".join(cells) + " |")
    print()
    print("Per-magnitude initialization count:", n_inits)

    print("\n## LaTeX tabular\n")
    print(r"\begin{tabular}{lccccc}")
    print(r"\toprule")
    print("Perturbation & " + " & ".join(rf"+{h}\,h" for h in REPORT_LEAD_HOURS) + r" \\")
    print(r"\midrule")
    for ppb in consts.NOX_MAGNITUDES_PPB:
        cells = [rf"${mean:.1f} \pm {std:.1f}$\%" for mean, std in (stats[(ppb, h)] for h in REPORT_LEAD_HOURS)]
        print(rf"$+{ppb}$~ppb" + " & " + " & ".join(cells) + r" \\")
    print(r"\bottomrule")
    print(r"\end{tabular}")


if __name__ == "__main__":
    main()
