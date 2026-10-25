"""Prepare (and, with --push, upload) the published routers to Hugging Face.

For each router (qp, qpa), seed 0 goes to the repo root and seeds 1-2 to `seed1/`, `seed2/`. Each folder also gets
`systemone.json`: its input set and the operating threshold, picked on calib with the same rule as the evaluation. The
root gets README.md, rendered from model-cards/template.md with the numbers from results/report.json (all 1,481 test
questions) and results/report-clean.json (the 1,303 that nano-jev did not see), and figures/ from make_figures.py.

    uv run python scripts/push_hf.py --org <hf-org>            # dry run: stages into build/hf/<repo>/ and prints it
    uv run python scripts/push_hf.py --org <hf-org> --push     # upload (needs `hf auth login`)
    uv run python scripts/push_hf.py --org <hf-org> --cards-only [--push]   # only README.md, to fix a card's text

--run is the pair's run folder (calib.parquet + config.toml), copied to models/run.
"""

import argparse
import os
import json
import shutil
import tomllib
from pathlib import Path

import pandas as pd

# Set SYSTEMONE_PARQUET_ENGINE=fastparquet where pyarrow's Parquet DLL is blocked (Windows App Control).
if os.environ.get("SYSTEMONE_PARQUET_ENGINE"):
    pd.set_option("io.parquet.engine", os.environ["SYSTEMONE_PARQUET_ENGINE"])

from systemone_jev.evaluate_routing import expert_params, pick_threshold, with_cost_unit
from systemone_jev.metrics import metric_view

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs" / "system-one-slr.toml"
ROUTERS = {"qp": "qp-large-helps-nanojev-A", "qpa": "qpa-large-helps-nanojev-A"}


def threshold(name: str, seed: int, calib: pd.DataFrame, max_drop: float) -> float:
    scores = pd.read_parquet(ROOT / "models" / name / f"seed{seed}" / "A-calib.parquet")
    merged = calib.merge(scores, on="id", validate="1:1")
    return float(pick_threshold(merged, merged["score"].to_numpy(), max_drop))


# The strongest logistic-regression router that decides at the same point (before) or after the small answer (cascade).
BASELINES = {"qp": ["router:question/large-helps", "router:question+evidence/large-helps", "router:evidence/large-helps"],
             "qpa": ["router:evidence+small/large-helps", "router:question+evidence+small/large-helps"]}
BASELINE_NAMES = {"qp": "Best logistic-regression router, same decision point (before the small model)",
                  "qpa": "Best logistic-regression cascade router (after the small model)"}
BASELINE_SHORT = {"qp": "best logistic-regression router at the same decision point",
                  "qpa": "best logistic-regression cascade router"}


def ci_text(ci: list[float]) -> str:
    return f"[{ci[0]:.3f}, {ci[1]:.3f}]"


def render_card(inputs: str, name: str, org: str, report: dict, clean: dict, plan: dict) -> str:
    judge_key = f"router:judge/{name}"
    judge = report["judges"][f"judge/{name}"]
    aiq, boot = report["aiq"], report["bootstrap"]
    base_key = max(BASELINES[inputs], key=lambda k: aiq[k])

    # Paired bootstrap intervals of judge minus each baseline. The report stores "A - B" for each pair,
    # so the judge-minus-baseline interval is the baseline-minus-judge one with its sign flipped.
    to_entropy = boot["aiq_diff"][f"{judge_key} - small-entropy (post)"]["ci95"]
    lo, hi = boot["aiq_diff"][f"{base_key} - {judge_key}"]["ci95"]
    to_base = (-hi, -lo)
    for ci in (to_entropy, to_base):
        if ci[0] > 0 or ci[1] < 0:  # the card says neither difference is significant; refuse otherwise
            raise SystemExit(f"{name}: a judge difference is significant; update model-cards/template.md first")

    check = json.loads((ROOT / "results" / "score_check.json").read_text())
    runs = [r for r in check["runs"] if r["router"] == inputs]  # this router's CPU check, one entry per seed
    seed0 = judge["aiq_per_seed"][0]  # the root folder's own model, the one its systemone.json thresholds
    fields = {"inputs": inputs, "answer_arg": ", answer=small_answer" if inputs == "qpa" else "",
              "inputs_text": {"qp": "the question and the small model's passages (before the small model answers)",
                              "qpa": "the question, the small model's passages and its answer"}[inputs],
              "repo": f"{org}/systemone-jev-{inputs}", "metric": report["metric"],
              "cost_unit": "AgentTTS Table 5 price per answering call", "tau": plan["tau"],
              "cpu_max_diff": f"{max(r['max_abs_diff'] for r in runs):.1e}",
              "cpu_flips": str(next(r["decisions_flipped"] for r in runs if r["seed"] == 0)),
              "cpu_flips_max": str(max(r["decisions_flipped"] for r in runs)), "n_test": f"{check['questions']:,}",
              "aiq_seed0": f"{seed0:.3f}",
              "aiq": f"{judge['aiq']:.3f}", "judge_ci": ci_text(boot["aiq_ci95"][judge_key]),
              "aiq_with_cost": f"{judge['aiq_with_judge_cost']:.3f}", "judge_cost": f"{judge['judge_cost']:.3f}",
              "aiq_clean": f"{clean['judges'][f'judge/{name}']['aiq']:.3f}",
              "aiq_seeds": f"{judge['aiq_seed_mean']:.3f} ± {judge['aiq_seed_sd']:.3f}",
              "entropy_aiq": f"{aiq['small-entropy (post)']:.3f}",
              "entropy_ci": ci_text(boot["aiq_ci95"]["small-entropy (post)"]),
              "baseline_name": BASELINE_NAMES[inputs], "baseline_short": BASELINE_SHORT[inputs],
              "baseline_aiq": f"{aiq[base_key]:.3f}", "baseline_ci": ci_text(boot["aiq_ci95"][base_key]),
              "oracle": f"{aiq['oracle']:.3f}",
              "vs_entropy": _diff_text(aiq[judge_key] - aiq["small-entropy (post)"], *to_entropy),
              "vs_baseline": _diff_text(aiq[judge_key] - aiq[base_key], *to_base)}
    return (ROOT / "model-cards" / "template.md").read_text(encoding="utf-8").format(**fields)


def _diff_text(mean: float, lo: float, hi: float) -> str:
    return f"{mean:+.3f} (95% CI {lo:+.3f} to {hi:+.3f}, not significant)"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--org", required=True)
    p.add_argument("--run", type=Path, default=ROOT / "models" / "run", help="the pair's calib.parquet + config.toml")
    p.add_argument("--push", action="store_true")
    p.add_argument("--cards-only", action="store_true",
                   help="stage (and with --push upload) only each repo's README.md; needs no models/")
    args = p.parse_args()
    plan = tomllib.loads(PLAN.read_text())
    report = json.loads((ROOT / "results" / "report.json").read_text())
    clean = json.loads((ROOT / "results" / "report-clean.json").read_text())
    for r in (report, clean):  # the numbers on the card must come from the plan that trained the uploaded judges
        if r["metric"] != plan["metric"] or r["cost_unit"] != plan["cost"]:
            raise SystemExit(f"results report is for {r['metric']} / {r['cost_unit']}, not the plan's "
                             f"{plan['metric']} / {plan['cost']}: refetch it")
    if not (ROOT / "results" / "score_check.json").exists():
        raise SystemExit("results/score_check.json is missing: run scripts/check_scores.py --device cpu first")
    if not all((ROOT / "figures" / f).exists() for f in ("aiq.png", "cost_accuracy.png")):
        raise SystemExit("figures/ is missing: run scripts/make_figures.py first")
    if args.cards_only:
        for inputs, name in ROUTERS.items():
            repo = f"{args.org}/systemone-jev-{inputs}"
            card = ROOT / "build" / "hf-cards" / repo.replace("/", "__") / "README.md"
            card.parent.mkdir(parents=True, exist_ok=True)
            card.write_text(render_card(inputs, name, args.org, report, clean, plan), encoding="utf-8")
            print(f"{repo}: card staged in {card}")
            if args.push and "license: TODO" in card.read_text(encoding="utf-8"):
                raise SystemExit("set the licence in model-cards/template.md before pushing")
            if args.push:
                from huggingface_hub import HfApi

                HfApi().upload_file(path_or_fileobj=card, path_in_repo="README.md", repo_id=repo,
                                    commit_message="Model card: per-router CPU check, ensemble use")
                print(f"  pushed https://huggingface.co/{repo}")
        return
    cfg = tomllib.loads((args.run / "config.toml").read_text())
    params = expert_params(cfg)
    models = {e["name"]: e.get("model", "") for e in cfg["experts"]}
    calib = with_cost_unit(metric_view(pd.read_parquet(args.run / "calib.parquet"), plan["metric"]), plan["cost"],
                           params, models)
    for inputs, name in ROUTERS.items():
        repo = f"{args.org}/systemone-jev-{inputs}"
        stage = ROOT / "build" / "hf" / repo.replace("/", "__")
        shutil.rmtree(stage, ignore_errors=True)
        for seed in (0, 1, 2):
            dst = stage if seed == 0 else stage / f"seed{seed}"
            shutil.copytree(ROOT / "models" / name / f"seed{seed}" / "model", dst, dirs_exist_ok=True)
            shutil.copy(ROOT / "models" / name / f"seed{seed}" / "done.json", dst / "training.json")
            card = {"inputs": inputs, "threshold": threshold(name, seed, calib, plan["max_drop"]),
                    "options": ["small model", "large model"], "seed": seed, "metric": plan["metric"], "tau": plan["tau"]}
            (dst / "systemone.json").write_text(json.dumps(card, indent=1))
        (stage / "README.md").write_text(render_card(inputs, name, args.org, report, clean, plan), encoding="utf-8")
        shutil.copytree(ROOT / "figures", stage / "figures", dirs_exist_ok=True)
        print(f"{repo}: staged in {stage}")
        for f in sorted(stage.rglob("*")):
            if f.is_file():
                print(f"  {f.relative_to(stage)}  {f.stat().st_size / 1e6:.1f} MB")
        if args.push and "license: TODO" in (stage / "README.md").read_text(encoding="utf-8"):
            raise SystemExit("set the licence in model-cards/template.md before pushing")
        if args.push:
            from huggingface_hub import HfApi

            api = HfApi()
            api.create_repo(repo, exist_ok=True)
            api.upload_folder(folder_path=stage, repo_id=repo, commit_message="SystemOne-Jev router")
            print(f"  pushed https://huggingface.co/{repo}")


if __name__ == "__main__":
    main()
