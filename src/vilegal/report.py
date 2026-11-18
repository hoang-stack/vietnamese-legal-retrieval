"""Figures and reports/results.md from results/runs.json (and the Kaggle timings)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
ACCENT = "#2a78d6"
CURVE_STYLE = {  # fixed colour per system (never by rank)
    "BM25": ("#2a78d6", "-"),
    "multilingual-e5-base (zero-shot)": ("#eb6834", "-"),
    "bge-m3 (zero-shot)": ("#1baf7a", "-"),
    "e5-base fine-tuned": ("#eda100", "-"),
    "Hybrid (BM25 + fine-tuned e5, RRF)": ("#e87ba4", "-"),
    "Hybrid + bge-reranker-v2-m3": ("#6250d6", "-"),
}


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.size": 10,
            "text.color": INK,
            "axes.labelcolor": INK_2,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.titlepad": 10,
            "axes.edgecolor": BASELINE,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_2,
            "ytick.labelcolor": INK_2,
            "legend.frameon": False,
            "lines.linewidth": 1.6,
        }
    )


def _save(fig, path: Path) -> Path:
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_test(rows, out: Path) -> Path | None:
    test = sorted([r for r in rows if r["split"] == "test"], key=lambda r: r["mrr@10"])
    if not test:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 0.48 * len(test) + 1.4), sharey=True)
    for ax, key in zip(axes, ("mrr@10", "recall@10"), strict=True):
        for y, r in enumerate(test):
            lo, hi = r[f"{key}_ci"]
            ax.plot([lo, hi], [y, y], color=ACCENT, alpha=0.45)
            ax.plot(r[key], y, "o", ms=7, color=ACCENT, mec=SURFACE, mew=1.5, zorder=3)
            ax.annotate(
                f"{r[key]:.3f}",
                (hi, y),
                xytext=(6, 0),
                textcoords="offset points",
                va="center",
                fontsize=8.5,
                color=INK_2,
            )
        ax.set_title(key.upper().replace("@", "@"))
        ax.set_yticks(range(len(test)), [r["system"] for r in test])
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
        ax.set_xlim(min(r[f"{key}_ci"][0] for r in test) - 0.03, 1.0)
    fig.suptitle(
        f"Test questions (n = {test[0]['queries']}), mean and 95% bootstrap CI",
        x=0.01,
        ha="left",
        fontsize=11.5,
        fontweight="bold",
    )
    fig.tight_layout()
    return _save(fig, out / "test_metrics.png")


def plot_recall_curves(rows, out: Path) -> Path | None:
    test = {r["system"]: r for r in rows if r["split"] == "test" and "recall_curve" in r}
    shown = [s for s in CURVE_STYLE if s in test]
    if not shown:
        return None
    fig, ax = plt.subplots(figsize=(8.6, 4.0))
    ks = np.arange(1, 101)
    for s in shown:
        color, ls = CURVE_STYLE[s]
        ax.plot(ks, test[s]["recall_curve"], ls, color=color, label=s)
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 5, 10, 20, 50, 100], ["1", "2", "5", "10", "20", "50", "100"])
    ax.set_xlabel("k (number of articles returned)")
    ax.set_ylabel("Recall@k on test questions")
    ax.set_ylim(0.5, 1.0)
    ax.legend(loc="lower right", fontsize=8.5)
    ax.set_title("How many questions find their article in the top k?")
    return _save(fig, out / "recall_at_k.png")


def plot_bm25(rows, out: Path) -> Path | None:
    grid = [r for r in rows if r["stage"] == "bm25_grid"]
    if not grid:
        return None
    k1s = sorted({r["k1"] for r in grid})
    bs = sorted({r["b"] for r in grid})
    mat = np.array(
        [[next(r["mrr@10"] for r in grid if r["k1"] == k and r["b"] == b) for b in bs] for k in k1s]
    )
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list(
        "blues", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"]
    )
    ax.imshow(mat, cmap=cmap, aspect="auto", origin="lower")
    for (i, j), v in np.ndenumerate(mat):
        ax.text(
            j,
            i,
            f"{v:.3f}",
            ha="center",
            va="center",
            fontsize=8.5,
            color=INK if v < mat.mean() else "#ffffff",
        )
    ax.set_xticks(range(len(bs)), [str(b) for b in bs])
    ax.set_yticks(range(len(k1s)), [str(k) for k in k1s])
    ax.set_xlabel("b (length normalisation)")
    ax.set_ylabel("k1 (term-frequency saturation)")
    ax.grid(False)
    ax.set_title("BM25 tuning on dev questions (MRR@10)")
    return _save(fig, out / "bm25_grid.png")


def results_markdown(rows, results: Path, kaggle_dir: Path) -> str:
    lines = ["# Results", ""]
    test = sorted([r for r in rows if r["split"] == "test"], key=lambda r: -r["mrr@10"])
    if test:
        lines += [
            f"## Test questions (n = {test[0]['queries']})",
            "",
            "| System | MRR@10 (95% CI) | nDCG@10 | R@1 | R@5 | R@10 (95% CI) | R@100 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for r in test:
            lo, hi = r["mrr@10_ci"]
            rlo, rhi = r["recall@10_ci"]
            lines.append(
                f"| {r['system']} | {r['mrr@10']:.3f} ({lo:.3f}–{hi:.3f}) | {r['ndcg@10']:.3f} | "
                f"{r['recall@1']:.3f} | {r['recall@5']:.3f} | {r['recall@10']:.3f} "
                f"({rlo:.3f}–{rhi:.3f}) | {r['recall@100']:.3f} |"
            )
        sig = results / "significance.json"
        if sig.exists():
            s = json.loads(sig.read_text())
            lines += [
                "",
                f"Paired bootstrap on MRR@10 against **{s['best']}**:",
                "",
                "| vs. | Δ MRR@10 | 95% CI | p |",
                "| --- | ---: | ---: | ---: |",
            ]
            for name, v in sorted(s["vs"].items(), key=lambda kv: kv[1]["diff"]):
                p = "< 0.001" if v["p_value"] < 0.001 else f"{v['p_value']:.3f}"
                lines.append(
                    f"| {name} | {v['diff']:+.3f} | {v['ci_low']:+.3f} to "
                    f"{v['ci_high']:+.3f} | {p} |"
                )
    dev = [r for r in rows if r["split"] == "dev" and r["stage"] != "bm25_grid"]
    if dev:
        lines += [
            "",
            "## Dev questions (used for every choice)",
            "",
            "| Stage | System | MRR@10 | R@10 | R@100 |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
        lines += [
            f"| {r['stage']} | {r['system']} | {r['mrr@10']:.3f} | {r['recall@10']:.3f} | "
            f"{r['recall@100']:.3f} |"
            for r in dev
        ]
    tpath = kaggle_dir / "timings.json"
    if tpath.exists():
        t = json.loads(tpath.read_text())
        lines += ["", "## GPU timings (Kaggle T4)", ""]
        for k, v in t.items():
            if isinstance(v, (int, float)):
                lines.append(f"- {k}: {v:.0f} s")
    return "\n".join(lines) + "\n"


def build_report(results_dir="results", report_dir="reports", kaggle_dir="kaggle/output/main"):
    results, out = Path(results_dir), Path(report_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    _style()
    rows = json.loads((results / "runs.json").read_text())
    plot_test(rows, out / "figures")
    plot_recall_curves(rows, out / "figures")
    plot_bm25(rows, out / "figures")
    md = out / "results.md"
    md.write_text(results_markdown(rows, results, Path(kaggle_dir)), encoding="utf-8")
    return md
