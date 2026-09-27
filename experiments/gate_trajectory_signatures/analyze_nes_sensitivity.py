#!/usr/bin/env python3
"""X13: sensitivity of the matched-objective result to the NES configuration (predictions P16a-e).

For every variant: the separability analysis of the main corpus (analyze_specificity_workloads.py,
written to <variant>/analysis_r2), the AUROC of the acceptance rate alone (orientation fixed on the
fit split), alarms at the released operating points, and the validity of each benign client
(median acceptance rate on the FIT split of at least 0.10 in the variants with an acceptance test;
amendment of 2026-09-27 15:58). The main corpus is included as the reference configuration.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.gate_trajectory_signatures import blacklight_rule  # noqa: E402

A = ROOT / "analysis_outputs"
BASE = A / "specificity_nes_sensitivity_20260927"
VARIANTS = {
    "main": (A / "stateful_specificity_workloads_20260925", 0.25, "improve"),
    "v1_step1": (BASE / "v1_step1", 1.0, "improve"),
    "v2_step2": (BASE / "v2_step2", 2.0, "improve"),
    "v3_step2_always": (BASE / "v3_step2_always", 2.0, "always"),
}
BENIGN = ("restore", "confidence_boost")
DETECTORS = ("logreg:gwad", "logreg:gwad_plus", "logreg:blacklight")
VALID_ACCEPTANCE = 0.10


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-separability", action="store_true")
    args = parser.parse_args()
    rows, rates = [], []
    for name, (root, step, accept) in VARIANTS.items():
        if name != "main" and not args.skip_separability:
            subprocess.run([sys.executable, str(Path(__file__).with_name("analyze_specificity_workloads.py")), "--root", str(root)],
                           check=True, stdout=subprocess.DEVNULL)
        sep = pd.read_csv(root / "analysis_r2/separability.csv")
        s = pd.DataFrame(blacklight_rule.load_sessions(root))
        s = s[s.objective.isin(("attack",) + BENIGN)]
        s["accept"] = [float((np.asarray(a) == 1).mean()) for a in s.accepted]
        s["bl"] = [d["blacklight"]["first_alarm"] for d in s.detectors]
        s["gp"] = [d["gwad_plus"]["first_alarm"] for d in s.detectors]
        for start in ("denoise", "deblur"):
            g = s[s.workload == start]
            attack = g[g.objective == "attack"]
            ev_attack = attack[attack.split == "evaluation"]
            succ = ev_attack[ev_attack.first_success > 0]
            rates.append({"variant": name, "step_255": step, "acceptance": accept, "start": start, "client": "attack",
                          "n_eval": len(ev_attack), "acceptance_median_eval": float(ev_attack.accept.median()),
                          "blacklight_alarm": float((ev_attack.bl > 0).mean()), "gwad_plus_alarm": float((ev_attack.gp > 0).mean()),
                          "blacklight_first_median": float(ev_attack.bl[ev_attack.bl > 0].median()),
                          "success_rate": float((ev_attack.first_success > 0).mean()),
                          "median_first_success": float(succ.first_success.median()) if len(succ) else np.nan})
            for objective in BENIGN:
                b = g[g.objective == objective]
                fit = pd.concat([attack[attack.split == "fit"].assign(y=1), b[b.split == "fit"].assign(y=0)])
                ev = pd.concat([ev_attack.assign(y=1), b[b.split == "evaluation"].assign(y=0)])
                sign = 1.0 if roc_auc_score(fit.y, fit.accept) >= 0.5 else -1.0
                fit_median = float(b[b.split == "fit"].accept.median())
                valid = True if accept == "always" else bool(fit_median >= VALID_ACCEPTANCE)
                eb = b[b.split == "evaluation"]
                rates.append({"variant": name, "step_255": step, "acceptance": accept, "start": start, "client": objective,
                              "n_eval": len(eb), "acceptance_median_eval": float(eb.accept.median()),
                              "blacklight_alarm": float((eb.bl > 0).mean()), "gwad_plus_alarm": float((eb.gp > 0).mean()),
                              "blacklight_first_median": float(eb.bl[eb.bl > 0].median()),
                              "success_rate": float((eb.first_success > 0).mean()), "median_first_success": np.nan})
                row = {"variant": name, "step_255": step, "acceptance": accept, "start": start, "benign": objective,
                       "benign_acceptance_median_fit": fit_median, "valid": valid,
                       "attack_acceptance_median": float(ev_attack.accept.median()), "benign_acceptance_median": float(eb.accept.median()),
                       "acceptance_only_auroc": float(roc_auc_score(ev.y, sign * ev.accept))}
                for det in DETECTORS:
                    c = sep[(sep.workload == start) & (sep.negative == objective) & (sep.prefix == "1024") & (sep.detector == det)].iloc[0]
                    key = det.split(":")[1]
                    row[f"{key}_auroc"], row[f"{key}_lo"], row[f"{key}_hi"] = float(c.auroc), float(c.lo), float(c.hi)
                rows.append(row)
    cells = pd.DataFrame(rows)
    alarm = pd.DataFrame(rates)
    out = BASE / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    cells.to_csv(out / "cells.csv", index=False)
    alarm.to_csv(out / "alarms.csv", index=False)

    new = cells[cells.variant != "main"]
    valid = new[new.valid]
    tested = valid[valid.acceptance == "improve"]
    always = new[new.acceptance == "always"]
    reference = cells[(cells.variant == "main") & (cells.benign == "restore")].set_index("start").gwad_plus_auroc
    checks = {
        "P16a_alarm_ge_99pct": {"minimum_blacklight": float(alarm[alarm.variant != "main"].blacklight_alarm.min()),
                                "minimum_gwad_plus": float(alarm[alarm.variant != "main"].gwad_plus_alarm.min()),
                                "held": bool(alarm[alarm.variant != "main"][["blacklight_alarm", "gwad_plus_alarm"]].min().min() >= 0.99)},
        "P16b_blacklight_le_0.62": {"maximum": float(valid.blacklight_auroc.max()), "cells": len(valid),
                                    "held": bool(valid.blacklight_auroc.max() <= 0.62)},
        "P16c_gwad_plus_minus_acceptance_le_0.05": {
            "maximum_difference": float((tested.gwad_plus_auroc - tested.acceptance_only_auroc).max()) if len(tested) else None,
            "cells": len(tested), "held": bool((tested.gwad_plus_auroc - tested.acceptance_only_auroc).max() <= 0.05) if len(tested) else None},
        "P16d_no_acceptance_test_gwad_plus_le_0.62": {"maximum": float(always.gwad_plus_auroc.max()), "cells": len(always),
                                                      "held": bool(always.gwad_plus_auroc.max() <= 0.62)},
    }
    for variant in ("v2_step2", "v1_step1"):
        r = valid[(valid.variant == variant) & (valid.benign == "restore")]
        if len(r):
            gain = (r.set_index("start").gwad_plus_auroc - reference).dropna()
            checks["P16e_gwad_plus_vs_restore_rises_by_0.10"] = {"variant": variant, "gain_by_start": {k: float(v) for k, v in gain.items()},
                                                                 "held": bool(gain.max() >= 0.10)}
            break
    else:
        checks["P16e_gwad_plus_vs_restore_rises_by_0.10"] = {"variant": None, "held": None, "note": "restoration invalid in V1 and V2: not testable"}
    (out / "predictions.json").write_text(json.dumps(checks, indent=2))
    pd.set_option("display.width", 260)
    print(cells.round(3).to_string(index=False))
    print(alarm.round(3).to_string(index=False))
    print(json.dumps(checks, indent=1))


if __name__ == "__main__":
    main()
