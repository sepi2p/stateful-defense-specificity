#!/usr/bin/env python3
"""X16: stateful detectors under benign-alarm constraints (procedure in operating_profile.py).

Settings
  imagenet   confirmatory. ResNet-50 at 224 px, tiled NES. Calibration split 'calibration' (images never
             run before X16), held-out split 'confirmation' (fresh images). Declared in
             docs/specificity_workloads_preregistration.md (X16) before the run.
  cifar10    retrospective. ResNet-18 (seed 0) at 32 px, NES, main corpus. Calibration split
             'calibration', held-out split 'evaluation'. The released-threshold results of the held-out
             split were known before this procedure was defined; the library clients at 32 px exist on the
             evaluation images only and are reported next to the profile, not in it.

Profiles
  explanation   G = unrelated images and the explanation clients of the setting
  diagnostic    G plus the matched benign optimizers (Tier B), a controlled diagnostic
Out-of-profile workloads are reported at the selected configuration and are not constrained.

Output: <output-dir>/<setting>/{selection.json, profile.csv, released.csv}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.gate_trajectory_signatures.operating_profile import (  # noqa: E402
    evaluate, select, sessions_from_logs, to_json)

A = Path("analysis_outputs")
X16 = A / "operating_profile_imagenet_20260928"
SETTINGS = {
    "imagenet": {
        "calibration": "calibration", "held_out": "confirmation",
        "sources": [
            (X16 / "workloads", {"attack": "attack", "confidence_boost": "confidence raising"}),
            (X16 / "libraries_resnet50", {"lime_package": "LIME (lime)", "captum_kernelshap": "KernelSHAP (Captum)",
                                          "captum_occlusion": "Occlusion (Captum)"}),
            (X16 / "controls", {"shuffled": "unrelated images", "noise": "noise copies", "sweep": "JPEG/brightness sweep"}),
        ],
        "explanation": ("unrelated images", "LIME (lime)", "KernelSHAP (Captum)", "Occlusion (Captum)"),
        "tier_b": ("confidence raising",),
        "reported": ("noise copies", "JPEG/brightness sweep"),
    },
    "cifar10": {
        "calibration": "calibration", "held_out": "evaluation",
        "sources": [
            (A / "stateful_specificity_workloads_20260925", {"attack": "attack", "restore": "restoration",
                                                            "confidence_boost": "confidence raising",
                                                            "boundary_probe": "boundary probing",
                                                            "counterfactual": "counterfactual search"}),
            (A / "explanation_clients_20260926", {"lime": "LIME (ours)", "kernelshap": "KernelSHAP (ours)",
                                                  "occlusion": "Occlusion (ours)", "rise": "RISE (ours)"}),
            (A / "explanation_libraries_20260927/cifar10", {"captum_kernelshap": "KernelSHAP (Captum)",
                                                            "captum_occlusion": "Occlusion (Captum)"}),
            (A / "stateful_specificity_controls_20260925", {"shuffled": "unrelated images", "noise": "noise copies",
                                                            "sweep": "JPEG/brightness sweep"}),
        ],
        "explanation": ("unrelated images", "LIME (ours)", "KernelSHAP (ours)", "Occlusion (ours)", "RISE (ours)"),
        "tier_b": ("restoration", "confidence raising"),
        "reported": ("KernelSHAP (Captum)", "Occlusion (Captum)", "noise copies", "JPEG/brightness sweep",
                     "boundary probing", "counterfactual search"),
    },
}
DETECTORS = ("blacklight", "gwad_plus", "gwad")
RELEASED = {"blacklight": 25, "gwad_plus": "native", "gwad": "native"}
ALPHAS = (0.01, 0.05)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setting", choices=tuple(SETTINGS), required=True)
    parser.add_argument("--output-dir", type=Path, default=A / "operating_profile_r1")
    args = parser.parse_args()
    cfg = SETTINGS[args.setting]
    out = args.output_dir / args.setting
    out.mkdir(parents=True, exist_ok=True)
    sessions = sessions_from_logs(cfg["sources"], {cfg["calibration"], cfg["held_out"]})
    cal = [s for s in sessions if s.split == cfg["calibration"]]
    held = [s for s in sessions if s.split == cfg["held_out"]]
    counts = pd.Series([f"{s.split}/{s.workload}" for s in sessions]).value_counts().sort_index()
    print(counts.to_string())
    everything = tuple(dict.fromkeys(cfg["explanation"] + cfg["tier_b"] + cfg["reported"]))
    selection, rows, released = {}, [], []
    for detector in DETECTORS:
        ev = evaluate(held, detector, RELEASED[detector], (), 1.0, extra_workloads=everything)
        for g, b in ev["benign"].items():
            released.append({"detector": detector, "workload": g, **{k: b[k] for k in ("alarms", "sessions", "rate")},
                             "ci_lo": b["ci"][0], "ci_hi": b["ci"][1]})
        released.append({"detector": detector, "workload": "attack (timely)", "alarms": ev["attack"]["timely"],
                         "sessions": ev["attack"]["successful"], "rate": ev["attack"]["R"], "ci_method": ev["attack"]["R_ci_method"],
                         "ci_lo": ev["attack"]["R_ci"][0], "ci_hi": ev["attack"]["R_ci"][1],
                         "success_rate": ev["attack"]["success_rate"]})
        for profile in ("explanation", "diagnostic"):
            workloads = cfg["explanation"] + (cfg["tier_b"] if profile == "diagnostic" else ())
            extra = tuple(g for g in everything if g not in workloads)
            for alpha in ALPHAS:
                sel = select(cal, detector, workloads, alpha)
                key = f"{detector}|{profile}|{alpha}"
                chosen = next((t for t in sel["table"] if t["theta"] == sel["theta"]), None)
                row = {"detector": detector, "profile": profile, "alpha": alpha, "status": sel["status"],
                       "theta": sel["theta"], "calibration_R": chosen["R"] if chosen else None,
                       "calibration_max_F": chosen["max_F"] if chosen else None,
                       "feasible_candidates": sum(t["feasible"] for t in sel["table"])}
                if sel["theta"] is not None:
                    ev = evaluate(held, detector, sel["theta"], workloads, alpha, extra)
                    row |= {"held_out_status": ev["status"], "violated": ";".join(ev["violated"]),
                            "R": ev["attack"]["R"], "R_lo": ev["attack"]["R_ci"][0], "R_hi": ev["attack"]["R_ci"][1],
                            "R_ci_method": ev["attack"]["R_ci_method"],
                            "timely": ev["attack"]["timely"], "successful": ev["attack"]["successful"],
                            "attacks": ev["attack"]["sessions"], "success_rate": ev["attack"]["success_rate"],
                            "timely_of_all": ev["attack"]["timely_of_all"]}
                    for g, b in ev["benign"].items():
                        row[f"F[{g}]"] = f"{b['alarms']}/{b['sessions']}"
                        row[f"Fci[{g}]"] = f"{b['ci'][0]:.4f}-{b['ci'][1]:.4f}"
                    selection[key] = {"selection": {k: v for k, v in sel.items() if k != "table"},
                                      "calibration_table": [{k: v for k, v in t.items() if k != "rates"} | {"benign": t["rates"]["benign"],
                                                             "attack": t["rates"]["attack"]} for t in sel["table"]],
                                      "held_out": ev}
                else:
                    selection[key] = {"selection": {k: v for k, v in sel.items() if k != "table"},
                                      "calibration_table": [{k: v for k, v in t.items() if k != "rates"} | {"benign": t["rates"]["benign"],
                                                             "attack": t["rates"]["attack"]} for t in sel["table"]]}
                rows.append(row)
    (out / "selection.json").write_text(json.dumps(to_json(selection), indent=1))
    pd.DataFrame(rows).to_csv(out / "profile.csv", index=False)
    pd.DataFrame(released).to_csv(out / "released.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print(pd.DataFrame(rows).drop(columns=[c for c in pd.DataFrame(rows).columns if c.startswith("F[")]).to_string())


if __name__ == "__main__":
    main()
