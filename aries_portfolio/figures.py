from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd

from .geo import crossing_count, gps_to_local_xy
from .io import load_frozen_instances


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 8,
        "axes.linewidth": 0.8,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
)

COLORS = {
    "Raw": "#9A9A9A",
    "NN+2opt": "#A8C4E4",
    "Insertion+2opt": "#78A6D2",
    "MultiStart2opt": "#356FA8",
    "MultiStart2opt-RT": "#153F6B",
    "ILS": "#79B8A9",
    "ILS-RT": "#2F7F72",
    "ACO": "#B8A6D9",
    "ACO×3": "#7A68A6",
    "Portfolio": "#B44747",
}
LABELS = {
    "Raw": "Raw",
    "NN+2opt": "NN + 2-opt",
    "Insertion+2opt": "Insertion + 2-opt",
    "MultiStart2opt": "Multi-start 2-opt",
    "MultiStart2opt-RT": "Multi-start 2-opt (RT)",
    "ILS": "ILS",
    "ILS-RT": "ILS (RT)",
    "ACO": "ACO",
    "ACO×3": "ACO×3",
    "Portfolio": "Five-candidate Portfolio",
}


def _save(fig: plt.Figure, root: Path, stem: str) -> None:
    output = root / "figures"
    output.mkdir(exist_ok=True)
    fig.savefig(output / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(
        output / f"{stem}.tiff",
        dpi=600,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.13, 1.05, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")


def figure1_architecture(root: Path) -> None:
    """System architecture; shaded region marks the controlled subsystem."""
    fig, ax = plt.subplots(figsize=(7.2, 6.8))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    def box(x, y, w, h, text, face="#F2F4F6", edge="#52616D", size=7):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
            boxstyle="round,pad=0.006,rounding_size=0.008",
            linewidth=0.8, edgecolor=edge, facecolor=face))
        ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=size)

    def arrow(start, end):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>",
            mutation_scale=9, linewidth=0.85, color="#52616D"))

    ax.text(0.015, 0.985, "ARIES-Mission  |  semantic-to-geometric mission generation",
            va="top", fontsize=9, fontweight="bold")
    box(0.22, 0.889, 0.35, 0.054, "Operator instruction", "#E7EEF5", "#356FA8")
    box(0.015, 0.792, 0.155, 0.064, "Aerial image +\ngeographic extent", "#E7EEF5", "#356FA8", 6.5)
    box(0.22, 0.797, 0.35, 0.054, "Vision-language semantic grounding", "#E7EEF5", "#356FA8")
    box(0.22, 0.705, 0.35, 0.054, "Geographic targets + Home\n(original emission order retained)", "#E7EEF5", "#356FA8")
    arrow((0.395, 0.889), (0.395, 0.855))
    arrow((0.17, 0.824), (0.215, 0.824))
    arrow((0.395, 0.797), (0.395, 0.763))
    ax.plot([0.015, 0.965], [0.674, 0.674], color="#356FA8", lw=1.1, ls="--")
    ax.text(0.66, 0.682, "Semantic / geometric boundary", ha="center", va="bottom", fontsize=7, color="#244D77")
    ax.add_patch(FancyBboxPatch((0.012, 0.046), 0.62, 0.608,
        boxstyle="round,pad=0.007", facecolor="#F7FAFC", edgecolor="#B5C6D5", linewidth=0.8))
    ax.text(0.027, 0.635, "AUDITABLE GEOMETRIC MISSION LAYER", fontsize=7, fontweight="bold")
    ax.text(0.027, 0.607, "Controlled evaluation: semantic output held fixed", fontsize=6.8, color="#356FA8")
    arrow((0.395, 0.700), (0.395, 0.563))
    xs = [0.028, 0.148, 0.268, 0.388, 0.508]
    names = ["Raw\nfallback", "NN + 2-opt\ndeterministic", "ACO\nstochastic", "DE\nstochastic", "PSO\nstochastic"]
    ax.plot([0.082, 0.562], [0.563, 0.563], color="#52616D", lw=0.8)
    for x, name, color in zip(xs, names, ["#ECECEC", "#E5EFF8", "#EEE9F6", "#EEE9F6", "#EEE9F6"]):
        arrow((x+0.054, 0.563), (x+0.054, 0.531))
        box(x, 0.459, 0.108, 0.068, name, color, size=6.5)
        arrow((x+0.054, 0.455), (x+0.054, 0.423))
    ax.plot([0.082, 0.562], [0.423, 0.423], color="#52616D", lw=0.8)
    ax.text(0.322, 0.405, "Five independently generated complete route candidates", ha="center", fontsize=6.5)
    arrow((0.322, 0.394), (0.322, 0.37))
    box(0.11, 0.318, 0.424, 0.047, "Validity audit: complete target permutation")
    box(0.11, 0.236, 0.424, 0.047, "Common physical rescoring: closed-loop Haversine")
    box(0.11, 0.154, 0.424, 0.047, "Terminal arbitration: shortest valid candidate", "#F6EDED", "#A46060")
    box(0.11, 0.072, 0.424, 0.047, "Selected UAV mission route", "#E5F2EC", "#2F7F72")
    for y in [0.318, 0.236, 0.154]:
        arrow((0.322, y), (0.322, y-0.031))

    ax.add_patch(FancyBboxPatch((0.68, 0.177), 0.282, 0.418,
        boxstyle="round,pad=0.009", facecolor="#FFFBEF", edgecolor="#A38B55", linestyle="--", linewidth=1))
    ax.text(0.821, 0.572, "EXTERNAL EVALUATION", ha="center", fontsize=7, fontweight="bold")
    ax.text(0.821, 0.543, "Same frozen targets and objective", ha="center", fontsize=6.3)
    ax.text(0.697, 0.506,
        "Nearest neighbour; insertion\nInsertion + 2-opt\nMulti-start 2-opt\nILS; ACO×3\nRuntime-calibrated controls\nConcorde exact reference\nHeld–Karp cross-check", va="top", fontsize=6.7, linespacing=1.75)
    ax.text(0.821, 0.212, "Never enter terminal selection", ha="center", fontsize=6.7, fontweight="bold", color="#785D27")
    ax.text(0.015, 0.014, "Geographic route output; perception accuracy and physical flight execution are not evaluated here.", fontsize=6.5)
    pd.DataFrame([
        {"element": "semantic stage", "role": "operator instruction; vision-language grounding; geographic targets"},
        {"element": "experimental boundary", "role": "freeze semantic output; evaluate geometric mission construction"},
        {"element": "candidate redundancy", "role": "Raw; NN+2-opt; ACO; DE; PSO"},
        {"element": "terminal decision", "role": "permutation validity; common Haversine rescoring; arbitration"},
        {"element": "external controls", "role": "NN; insertion; insertion+2-opt; Multi-start; ILS; ACOx3; runtime controls; Concorde; Held-Karp; never selectable"},
    ]).to_csv(root / "figures" / "figure1_source.csv", index=False)
    _save(fig, root, "figure1_architecture")


def figure2_route_quality(root: Path) -> None:
    aggregate = pd.read_csv(root / "results" / "aggregate.csv").set_index("method")
    task = pd.read_csv(root / "results" / "task_method_means.csv")
    gaps = pd.read_csv(root / "results" / "optimality_gaps.csv")

    methods_a = ["Raw", "NN+2opt", "Insertion+2opt", "MultiStart2opt", "ILS", "ACO", "ACO×3", "Portfolio", "MultiStart2opt-RT"]
    fig = plt.figure(figsize=(7.2, 5.25))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.12, 1.0], height_ratios=[1.0, 1.0], hspace=0.42, wspace=0.48)
    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 1])

    y = np.arange(len(methods_a))[::-1]
    means = np.array([aggregate.loc[m, "mean_km"] for m in methods_a])
    lo = np.array([aggregate.loc[m, "ci95_low_km"] for m in methods_a])
    hi = np.array([aggregate.loc[m, "ci95_high_km"] for m in methods_a])
    ax_a.errorbar(means, y, xerr=np.vstack([means - lo, hi - means]), fmt="none", ecolor="#777777", elinewidth=0.9, capsize=2, zorder=1)
    for yi, method, value in zip(y, methods_a, means):
        marker = "D" if method == "Portfolio" else "o"
        ax_a.scatter(value, yi, s=34, marker=marker, color=COLORS[method], edgecolor="white", linewidth=0.5, zorder=3)
        ax_a.text(value + 0.018, yi, f"{value:.3f}", va="center", fontsize=6)
    ax_a.set_yticks(y, [LABELS[m] for m in methods_a])
    ax_a.set_xlabel("Mean closed-loop distance (km)\n(task-bootstrap 95% CI)")
    ax_a.set_xlim(min(lo) - 0.04, max(hi) + 0.10)
    ax_a.set_title("Route quality under strong controls", loc="left", fontweight="bold")
    _panel_label(ax_a, "a")
    pd.DataFrame(
        {
            "method": methods_a,
            "mean_km": means,
            "ci95_low_km": lo,
            "ci95_high_km": hi,
            "total_km": [aggregate.loc[m, "total_km"] for m in methods_a],
        }
    ).to_csv(root / "figures" / "figure2a_source.csv", index=False)

    comparators = ["MultiStart2opt", "ILS", "ACO×3", "MultiStart2opt-RT"]
    portfolio = task[task.method == "Portfolio"].set_index("task_id").length_km
    paired_rows = []
    for pos, method in enumerate(comparators):
        comparator = task[task.method == method].set_index("task_id").length_km
        common = portfolio.index.intersection(comparator.index)
        diff_m = 1000.0 * (comparator.loc[common] - portfolio.loc[common])
        order = np.argsort(diff_m.to_numpy())
        jitter = np.linspace(-0.13, 0.13, len(diff_m))
        ax_b.scatter(diff_m.to_numpy()[order], pos + jitter, s=12, color=COLORS[method], alpha=0.78, edgecolor="none")
        ax_b.plot([np.median(diff_m), np.median(diff_m)], [pos - 0.20, pos + 0.20], color="#202020", lw=1.5)
        for task_id, value in diff_m.items():
            paired_rows.append({"task_id": task_id, "comparator": method, "comparator_minus_portfolio_m": value})
    ax_b.axvline(0, color="#777777", lw=0.8, ls="--")
    ax_b.set_yticks(range(len(comparators)), [LABELS[m] for m in comparators])
    ax_b.set_xlabel("Comparator − Portfolio (m)\npositive favours Portfolio")
    ax_b.set_title("Paired task differences", loc="left", fontweight="bold")
    _panel_label(ax_b, "b")
    pd.DataFrame(paired_rows).to_csv(root / "figures" / "figure2b_source.csv", index=False)

    methods_c = ["NN+2opt", "Insertion+2opt", "MultiStart2opt", "ILS", "ACO×3", "Portfolio", "MultiStart2opt-RT"]
    gap_rows = gaps[gaps.method.isin(methods_c)].copy()
    for pos, method in enumerate(methods_c):
        values = gap_rows.loc[gap_rows.method == method, "gap_pct"].to_numpy()
        ordered = np.sort(values)
        jitter = np.linspace(-0.12, 0.12, len(values))
        ax_c.scatter(np.full(len(values), pos) + jitter, ordered, s=11, color=COLORS.get(method, "#777777"), alpha=0.75, edgecolor="none")
        ax_c.plot([pos - 0.20, pos + 0.20], [np.median(values), np.median(values)], color="#202020", lw=1.3)
    ax_c.set_yscale("symlog", linthresh=1e-4, linscale=0.6)
    ax_c.set_xticks(range(len(methods_c)), ["NN+2", "Ins.+2", "MS-2", "ILS", "ACO×3", "Portfolio", "MS-2 RT"], rotation=38, ha="right")
    ax_c.set_ylabel("Gap to Concorde reference (%)")
    ax_c.set_title("All-instance exact gaps", loc="left", fontweight="bold")
    _panel_label(ax_c, "c")
    gap_rows[["task_id", "method", "gap_pct", "reference_km", "length_km"]].to_csv(root / "figures" / "figure2c_source.csv", index=False)

    fig.subplots_adjust(left=0.20, right=0.98, bottom=0.12, top=0.93)
    _save(fig, root, "figure2_route_quality")


def _plot_route(ax: plt.Axes, image: np.ndarray, points: list[list[float]], order: list[int], title: str, color: str) -> None:
    ax.imshow(image)
    height, width = image.shape[:2]
    targets = np.asarray([(x / 100.0 * width, y / 100.0 * height) for x, y in points])
    home = np.asarray((0.10 * width, 0.10 * height))
    nodes = np.vstack([home, targets[np.asarray(order, dtype=int)], home])
    ax.plot(nodes[:, 0], nodes[:, 1], color=color, lw=1.7, alpha=0.92, zorder=2)
    ax.scatter(targets[:, 0], targets[:, 1], s=24, facecolor="white", edgecolor="#222222", linewidth=0.8, zorder=3)
    for index, (x, y) in enumerate(targets):
        ax.text(x, y - 7, str(index + 1), fontsize=5.5, ha="center", va="bottom", color="#111111", zorder=4)
    ax.scatter([home[0]], [home[1]], marker="*", s=95, color="#E0A328", edgecolor="#222222", linewidth=0.6, zorder=5)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def figure3_routes(root: Path, task_id: int = 8, seed: int = 0) -> None:
    instances = {item.task_id: item for item in load_frozen_instances(root / "results" / "frozen_instances_gps.json", root / "data" / "nano30" / "images")}
    instance = instances[task_id]
    records = pd.read_csv(root / "results" / "formal_source_data.csv")
    selected = records[(records.task_id == task_id) & (records.seed == seed) & records.method.isin(["Raw", "Portfolio"])].copy()
    if len(selected) != 2:
        raise RuntimeError("Representative route records are missing")
    image = plt.imread(instance.image_path)
    points = [list(map(float, point)) for point in instance.metadata["points_percent"]]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.35))
    source_rows = []
    for ax, method, label, panel in zip(axes, ["Raw", "Portfolio"], ["Raw recognition-order reference", "Five-candidate Portfolio"], ["a", "b"]):
        row = selected[selected.method == method].iloc[0]
        order = list(map(int, json.loads(row.route_indices)))
        crossings = crossing_count(order, gps_to_local_xy(instance.home_gps, instance.targets_gps))
        title = f"{label}\n{row.route_length_km:.3f} km; {crossings} crossings"
        _plot_route(ax, image, points, order, title, COLORS[method])
        _panel_label(ax, panel)
        for sequence, target in enumerate(order):
            source_rows.append(
                {
                    "task_id": task_id,
                    "seed": seed,
                    "method": method,
                    "sequence_position": sequence,
                    "target_index": target,
                    "route_length_km": row.route_length_km,
                    "crossings": crossings,
                }
            )
    fig.subplots_adjust(left=0.03, right=0.99, top=0.86, bottom=0.02, wspace=0.08)
    pd.DataFrame(source_rows).to_csv(root / "figures" / "figure3_source.csv", index=False)
    _save(fig, root, "figure3_route_comparison")


def figure4_budget_runtime(root: Path) -> None:
    sensitivity = pd.read_csv(root / "results" / "aco_sensitivity_summary.csv")
    runtime = pd.read_csv(root / "results" / "runtime_summary.csv").set_index("method")
    aggregate = pd.read_csv(root / "results" / "aggregate.csv").set_index("method")
    base_total = float(sensitivity.loc[sensitivity.config_id == "base", "total_route_km"].iloc[0])
    sens = sensitivity.copy()
    sens["delta_total_m"] = 1000.0 * (sens.total_route_km - base_total)
    sens = sens[sens.config_id != "base"].copy()

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.58), gridspec_kw={"width_ratios": [1.10, 0.88, 1.15]})
    ax_a, ax_b, ax_c = axes
    labels = []
    for row in sens.itertuples():
        labels.append(f"budget={row.value:g}" if row.factor == "evaluation_budget" else f"{row.factor}={row.value:g}")
    y = np.arange(len(sens))[::-1]
    colors = np.where(sens.delta_total_m.to_numpy() <= 0, "#6AAE9F", "#C98C8C")
    ax_a.barh(y, sens.delta_total_m, color=colors, height=0.68)
    ax_a.axvline(0, color="#666666", lw=0.8)
    ax_a.set_yticks(y, labels)
    ax_a.set_xlabel("Change from base ACO total (m)")
    ax_a.set_title("ACO sensitivity", loc="left", fontweight="bold")
    _panel_label(ax_a, "a")
    sens[["config_id", "factor", "value", "total_route_km", "delta_total_m", "mean_runtime_s"]].to_csv(root / "figures" / "figure4a_source.csv", index=False)

    runtime_methods = ["MultiStart2opt", "ILS", "MultiStart2opt-RT", "ILS-RT", "Portfolio", "ACO×3"]
    values = [runtime.loc[m, "mean_runtime_s"] for m in runtime_methods]
    y2 = np.arange(len(runtime_methods))[::-1]
    ax_b.barh(y2, values, color=[COLORS.get(m, "#777777") for m in runtime_methods], height=0.66)
    ax_b.set_xscale("log")
    ax_b.set_yticks(y2, [LABELS[m] for m in runtime_methods])
    ax_b.set_xlabel("Mean serial runtime (s, log scale)")
    ax_b.set_title("Measured computation", loc="left", fontweight="bold")
    _panel_label(ax_b, "b")
    pd.DataFrame({"method": runtime_methods, "mean_runtime_s": values}).to_csv(root / "figures" / "figure4b_source.csv", index=False)

    quality_methods = ["NN+2opt", "Insertion+2opt", "MultiStart2opt", "ILS", "ACO", "ACO×3", "Portfolio", "MultiStart2opt-RT", "ILS-RT"]
    for method in quality_methods:
        x = float(runtime.loc[method, "mean_runtime_s"])
        yv = float(aggregate.loc[method, "total_km"])
        short = {"Insertion+2opt": "Ins.+2", "MultiStart2opt": "MS-2", "MultiStart2opt-RT": "MS-2 RT", "Portfolio": "Portfolio"}.get(method, LABELS.get(method, method))
        ax_c.scatter(
            x,
            yv,
            s=28,
            color=COLORS.get(method, "#777777"),
            edgecolor="white",
            linewidth=0.5,
            label=short,
        )
    ax_c.set_xscale("log")
    ax_c.set_xlabel("Mean serial runtime (s, log scale)")
    ax_c.set_ylabel("Total route distance (km)")
    ax_c.set_title("Quality–runtime boundary", loc="left", fontweight="bold")
    ax_c.legend(
        loc="upper center",
        bbox_to_anchor=(0.50, -0.27),
        ncol=3,
        fontsize=5.8,
        handletextpad=0.25,
        columnspacing=0.55,
    )
    _panel_label(ax_c, "c")
    pd.DataFrame(
        {
            "method": quality_methods,
            "mean_runtime_s": [runtime.loc[m, "mean_runtime_s"] for m in quality_methods],
            "total_km": [aggregate.loc[m, "total_km"] for m in quality_methods],
        }
    ).to_csv(root / "figures" / "figure4c_source.csv", index=False)

    for ax in axes:
        ax.tick_params(axis="both", labelsize=7.8)
        ax.xaxis.label.set_size(8.0)
        ax.yaxis.label.set_size(8.0)
        ax.title.set_fontsize(9.0)
    fig.subplots_adjust(left=0.145, right=0.985, bottom=0.31, top=0.89, wspace=0.70)
    _save(fig, root, "figure4_budget_runtime")


def supplementary_convergence(root: Path) -> None:
    records = json.loads((root / "results" / "routes_stochastic.json").read_text())
    baseline = pd.read_csv(root / "results" / "routes_baselines.csv")
    reference = baseline[baseline.method == "ReferenceExact"].set_index("task_id").length_km.to_dict()
    progress = np.linspace(0.0, 1.0, 31)
    rows = []
    for record in records:
        if record["method"] not in {"ACO", "DE", "PSO"} or not record.get("history_km"):
            continue
        history = np.asarray(record["history_km"], dtype=float)
        sampled = np.interp(progress, np.linspace(0.0, 1.0, len(history)), history)
        gap = 100.0 * np.maximum(0.0, sampled - reference[int(record["task_id"])]) / reference[int(record["task_id"])]
        for fraction, value in zip(progress, gap):
            rows.append({"task_id": int(record["task_id"]), "seed": int(record["seed"]), "method": record["method"], "budget_fraction": fraction, "gap_pct": value})
    frame = pd.DataFrame(rows)
    task_means = frame.groupby(["task_id", "method", "budget_fraction"], as_index=False).gap_pct.mean()
    summary = task_means.groupby(["method", "budget_fraction"], as_index=False).gap_pct.agg(
        median="median", q25=lambda values: values.quantile(0.25), q75=lambda values: values.quantile(0.75)
    )
    fig, ax = plt.subplots(figsize=(4.6, 3.15))
    for method, color in [("ACO", COLORS["ACO"]), ("DE", "#6687A8"), ("PSO", "#B07A78")]:
        sub = summary[summary.method == method]
        x = 7500.0 * sub.budget_fraction.to_numpy()
        ax.plot(x, sub["median"], label=method, color=color, lw=1.5)
        ax.fill_between(x, sub.q25, sub.q75, color=color, alpha=0.16, linewidth=0)
    ax.set_yscale("symlog", linthresh=1e-3)
    ax.set_xlabel("Objective evaluations")
    ax.set_ylabel("Gap to Concorde reference (%)")
    ax.legend(ncol=3, loc="upper right")
    ax.set_title("Stochastic-solver convergence", loc="left", fontweight="bold")
    fig.tight_layout(pad=0.8)
    summary.to_csv(root / "figures" / "figureS_convergence_source.csv", index=False)
    _save(fig, root, "figureS_convergence")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_manifest(root: Path) -> None:
    paths = sorted(path for path in (root / "figures").iterdir() if path.is_file() and path.name != "figure_manifest.json")
    manifest = {
        "backend": "Python/matplotlib only",
        "outputs": {
            path.name: {"sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in paths
        },
    }
    (root / "figures" / "figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    figure1_architecture(root)
    figure2_route_quality(root)
    figure3_routes(root)
    figure4_budget_runtime(root)
    supplementary_convergence(root)
    write_manifest(root)


if __name__ == "__main__":
    main()
