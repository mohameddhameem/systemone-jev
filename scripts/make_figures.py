"""Two figures for the model card, from the evaluation reports in results/ (report.json and curves.csv).

    uv run python scripts/make_figures.py                       # writes figures/*.png
    uv run python scripts/make_figures.py --results <dir> --out <dir>

Figures: aiq.png (each policy's AIQ with its 95% paired-bootstrap interval) and cost_accuracy.png (the accuracy-vs-cost
curves the AIQ is the area under). Colours: the two judges use the first two categorical slots of the palette;
baselines are grey. Requires matplotlib (in the `plot` extra of stair-common).
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#9b9a94", "#e3e2dd"
BLUE, ORANGE = "#2a78d6", "#eb6834"  # categorical slots 1 and 2 (light surface)
ENTROPY = "small-entropy (post)"
JUDGE = {"qp": "router:judge/qp-large-helps-nanojev-A", "qpa": "router:judge/qpa-large-helps-nanojev-A"}
LR_BEST = {"qp": "router:evidence/large-helps", "qpa": "router:evidence+small/large-helps"}  # as push_hf.py


def style(ax):
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=INK2, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def aiq_figure(report: dict, out: Path) -> None:
    aiq, ci, n = report["aiq"], report["bootstrap"]["aiq_ci95"], report["bootstrap"]["n"]
    rows = [("Random", "random", MUTED),
            ("Small-model entropy threshold", ENTROPY, MUTED),
            ("Best logistic-regression router, before", LR_BEST["qp"], MUTED),
            ("Best logistic-regression cascade, after", LR_BEST["qpa"], MUTED),
            ("Judge qp, before the small model", JUDGE["qp"], BLUE),
            ("Judge qpa, after the small model", JUDGE["qpa"], ORANGE),
            ("Oracle (upper bound)", "oracle", MUTED)]
    fig, ax = plt.subplots(figsize=(8, 3.6), dpi=200)
    fig.patch.set_facecolor("white")
    for y, (label, key, color) in enumerate(reversed(rows)):
        value = aiq[key]
        lo, hi = ci[key]
        ax.hlines(y, lo, hi, color=color, linewidth=2.2)
        ax.plot(value, y, "o", color=color, markersize=7, zorder=3)
        ax.annotate(f"{value:.3f}", (hi, y), xytext=(10, 0), textcoords="offset points", va="center",
                    color=INK, fontsize=9)
    ax.set_yticks(range(len(rows)), [r[0] for r in reversed(rows)], color=INK, fontsize=10)
    ax.set_xlim(0.62, 0.78)
    ax.set_xlabel("AIQ (area under the accuracy-vs-cost curve, higher is better)", color=INK2, fontsize=9)
    style(ax)
    fig.suptitle("On the SLR split, the judges tie with the baselines", x=0.02, ha="left", color=INK,
                 fontsize=13, fontweight="bold", y=0.98)
    fig.text(0.02, 0.02, f"HotpotQA slr-test, 1,481 questions. Dots: AIQ. Bars: 95% paired-bootstrap interval "
             f"({n:,} resamples). Judges: 3-seed ensemble.", color=INK2, fontsize=8)
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    fig.savefig(out / "aiq.png", facecolor="white")
    plt.close(fig)


def cost_figure(curves: pd.DataFrame, out: Path) -> None:
    series = [("Judge qp, before", JUDGE["qp"], BLUE, 2.4),
              ("Judge qpa, after", JUDGE["qpa"], ORANGE, 2.4),
              ("Entropy threshold", ENTROPY, INK2, 1.4),
              ("Best LR cascade", LR_BEST["qpa"], MUTED, 1.4),
              ("Oracle", "oracle", MUTED, 1.0)]
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=200)
    fig.patch.set_facecolor("white")
    for label, key, color, width in series:
        s = curves[curves["policy"] == key].sort_values("cost")
        ax.plot(s["cost"], s["f1"], color=color, linewidth=width, label=label, drawstyle="default")
    ax.set_xlabel("Cost per question (AgentTTS Table 5 units; always-small 0.5, always-large 10)", color=INK2, fontsize=9)
    ax.set_ylabel("Accuracy (lenient)", color=INK2, fontsize=9)
    style(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, loc="upper right", fontsize=9, labelcolor=INK)
    fig.suptitle("Accuracy against cost: the judges follow the baselines", x=0.02, ha="left",
                 color=INK, fontsize=13, fontweight="bold", y=0.98)
    fig.text(0.02, 0.02, "HotpotQA slr-test, 1,481 questions. Each curve sends a growing share of questions to the large "
             "model,\nin order of its score. After-the-small-model policies also pay for the small answer (they end at 10.5).",
             color=INK2, fontsize=8)
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    fig.savefig(out / "cost_accuracy.png", facecolor="white")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", type=Path, default=ROOT / "results")
    p.add_argument("--out", type=Path, default=ROOT / "figures")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    report = json.loads((args.results / "report.json").read_text())
    curves = pd.read_csv(args.results / "curves.csv")
    aiq_figure(report, args.out)
    cost_figure(curves, args.out)
    print(f"wrote {args.out / 'aiq.png'} and {args.out / 'cost_accuracy.png'}")


if __name__ == "__main__":
    main()
