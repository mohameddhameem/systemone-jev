"""Check that the published routers reproduce the saved test scores: route.py's P(large) for every test question must
match the judge's `A-test.parquet` score (from training) for both routers and all three seeds.

The saved scores were computed in bf16 on a GPU. On a GPU the check holds to 1e-4. On a CPU (fp32) it holds to 1e-2, and
the check also counts how many routing decisions flip at the published threshold. Results go to results/score_check.json,
which push_hf.py puts on the model card.

Needs the router folders in models/ (from the Hub), the pair's split cache in models/run/, and the HotpotQA distractor
parquet files (the passages are rebuilt from the dataset, as in training):

    uv run python scripts/check_scores.py [--data-dir data/hotpotqa] [--device cuda|cpu]
"""

import argparse
import json
import os
import sys
import tomllib
from pathlib import Path

import pandas as pd

# Set SYSTEMONE_PARQUET_ENGINE=fastparquet where pyarrow's Parquet DLL is blocked (Windows App Control).
if os.environ.get("SYSTEMONE_PARQUET_ENGINE"):
    pd.set_option("io.parquet.engine", os.environ["SYSTEMONE_PARQUET_ENGINE"])

from systemone_jev.evaluate_routing import expert_params, with_cost_unit  # noqa: E402
from systemone_jev.judge import Splits  # noqa: E402
from systemone_jev.metrics import metric_view  # noqa: E402
from systemone_jev.route import Router  # noqa: E402
from push_hf import threshold  # noqa: E402  (the same calib-picked threshold the cards publish)

ROOT = Path(__file__).resolve().parents[1]
ROUTERS = {"qp": "qp-large-helps-nanojev-A", "qpa": "qpa-large-helps-nanojev-A"}
TOLERANCE = {"cuda": 1e-4, "cpu": 1e-2}  # the saved scores are bf16-on-GPU; CPU fp32 differs by up to ~7.6e-3 observed


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=ROOT / "data" / "hotpotqa")
    p.add_argument("--run", type=Path, default=ROOT / "models" / "run", help="the pair's test/calib + config")
    p.add_argument("--device", default=None, help="cuda or cpu (default: cuda if available)")
    args = p.parse_args()

    plan = tomllib.loads((ROOT / "configs" / "system-one-slr.toml").read_text())
    cfg = tomllib.loads((args.run / "config.toml").read_text())
    params, models = expert_params(cfg), {e["name"]: e.get("model", "") for e in cfg["experts"]}
    calib = with_cost_unit(metric_view(pd.read_parquet(args.run / "calib.parquet"), plan["metric"]), plan["cost"],
                           params, models)
    splits = Splits({"A": args.run}, args.data_dir, plan.get("split_data"))
    test = splits.get("A", "test")
    print(f"test questions: {len(test)}")

    rows, device = [], None
    for inputs, name in ROUTERS.items():
        for seed in (0, 1, 2):
            model = ROOT / "models" / name / f"seed{seed}" / "model"
            router = Router.from_pretrained(str(model), device=args.device, inputs=inputs)
            device = router.judge.device.split(":")[0]
            ours = router.p_escalate_batch(test.to_dict("records"))
            saved = pd.read_parquet(ROOT / "models" / name / f"seed{seed}" / "A-test.parquet")
            merged = pd.DataFrame({"id": test["id"].to_numpy(), "ours": ours}).merge(saved, on="id", validate="1:1")
            diff = float((merged["ours"] - merged["score"]).abs().max())
            thr = threshold(name, seed, calib, plan["max_drop"])
            flips = int(((merged["ours"] >= thr) != (merged["score"] >= thr)).sum())
            rows.append({"router": inputs, "seed": seed, "max_abs_diff": diff, "threshold": thr,
                         "decisions_flipped": flips, "questions": len(merged)})
            status = "ok" if diff < TOLERANCE[device] else "FAIL"
            print(f"{inputs} seed{seed}: max |route.py - saved| = {diff:.2e}, flips at threshold {flips}  {status}")

    tol = TOLERANCE[device]
    worst = max(r["max_abs_diff"] for r in rows)
    summary = {"device": device, "tolerance": tol, "max_abs_diff": worst,
               "max_decisions_flipped": max(r["decisions_flipped"] for r in rows), "questions": len(test),
               "runs": rows}
    (ROOT / "results" / "score_check.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    if worst >= tol:
        sys.exit(f"score check failed on {device}: max diff {worst:.2e} >= {tol:g}")
    print(f"score check passed on {device}: max diff {worst:.2e} < {tol:g}; "
          f"at most {summary['max_decisions_flipped']} of {len(test)} decisions flip")


if __name__ == "__main__":
    main()
