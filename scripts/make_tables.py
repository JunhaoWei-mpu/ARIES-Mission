from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "manuscript"

METHODS = [
    "Raw",
    "NearestNeighbor",
    "CheapestInsertion",
    "NN+2opt",
    "Insertion+2opt",
    "MultiStart2opt",
    "ILS",
    "ACO",
    "ACO×3",
    "DE",
    "PSO",
    "Portfolio",
]
LABELS = {
    "Raw": "Raw recognition order",
    "NearestNeighbor": "Nearest neighbour",
    "CheapestInsertion": "Cheapest insertion",
    "NN+2opt": "NN + 2-opt",
    "Insertion+2opt": "Insertion + 2-opt",
    "MultiStart2opt": "Multi-start 2-opt",
    "ILS": "ILS",
    "ACO": "ACO",
    "ACO×3": r"ACO$\times$3",
    "DE": "DE",
    "PSO": "PSO",
    "Portfolio": "Five-candidate Portfolio",
    "MultiStart2opt-RT": "Multi-start 2-opt (runtime-matched)",
    "ILS-RT": "ILS (runtime-matched)",
    "ReferenceExact": "Concorde exact reference",
}


def table1_route_quality() -> None:
    frame = pd.read_csv(ROOT / "results" / "aggregate.csv").set_index("method")
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Method & Mean (km) & 95\% CI & Total (km) & Reduction vs. Raw \\",
        r"\midrule",
    ]
    for method in METHODS:
        row = frame.loc[method]
        text = (
            f"{LABELS[method]} & {row.mean_km:.3f} & [{row.ci95_low_km:.3f}, {row.ci95_high_km:.3f}]"
            f" & {row.total_km:.3f} & {row.improvement_vs_raw_pct:.2f}\\% \\\\"
        )
        if method == "Portfolio":
            text = r"\bfseries " + text
        lines.append(text)
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (OUT / "generated_table1.tex").write_text("\n".join(lines) + "\n")


def table2_exact_runtime() -> None:
    gaps = pd.read_csv(ROOT / "results" / "exact_gap_summary.csv").set_index("method")
    runtime = pd.read_csv(ROOT / "results" / "runtime_summary.csv").set_index("method")
    methods = [
        "NearestNeighbor",
        "CheapestInsertion",
        "NN+2opt",
        "Insertion+2opt",
        "MultiStart2opt",
        "ILS",
        "ACO",
        "ACO×3",
        "DE",
        "PSO",
        "Portfolio",
        "MultiStart2opt-RT",
        "ILS-RT",
    ]
    lines = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Method & Mean exact gap (\%) & Maximum exact gap (\%) & Serial runtime (s) \\",
        r"\midrule",
    ]
    for method in methods:
        lines.append(
            f"{LABELS[method]} & {gaps.loc[method, 'mean_gap_pct']:.4f}"
            f" & {gaps.loc[method, 'max_gap_pct']:.4f} & {runtime.loc[method, 'mean_runtime_s']:.4f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (OUT / "generated_table2.tex").write_text("\n".join(lines) + "\n")


def supplementary_per_task() -> None:
    means = pd.read_csv(ROOT / "results" / "task_method_means.csv")
    wide = means.pivot(index="task_id", columns="method", values="length_km")
    features = pd.read_csv(ROOT / "results" / "task_features.csv").set_index("task_id")
    methods = ["Raw", "NN+2opt", "Insertion+2opt", "MultiStart2opt", "ILS", "ACO×3", "Portfolio", "ReferenceExact"]
    lines = [
        r"\begin{longtable}{rr" + "r" * len(methods) + "}",
        r"\caption{Per-task closed-loop route lengths in kilometres. Stochastic methods are averaged over seeds 0--19.}\label{tab:per_task}\\",
        r"\toprule",
        r"Task & $N$ & Raw & NN+2opt & Ins.+2opt & MS-2opt & ILS & ACO$\times$3 & Portfolio & Exact \\",
        r"\midrule\endfirsthead",
        r"\toprule Task & $N$ & Raw & NN+2opt & Ins.+2opt & MS-2opt & ILS & ACO$\times$3 & Portfolio & Exact \\",
        r"\midrule\endhead",
    ]
    for task in wide.index:
        values = " & ".join(f"{wide.loc[task, method]:.3f}" for method in methods)
        lines.append(f"{task} & {int(features.loc[task, 'num_targets'])} & {values} \\\\ ")
    lines.extend([r"\bottomrule", r"\end{longtable}"])
    (OUT / "generated_table_s1.tex").write_text("\n".join(lines) + "\n")


def supplementary_statistics() -> None:
    frame = pd.read_csv(ROOT / "results" / "statistical_tests.csv")
    lines = [
        r"\begin{longtable}{lrrrrrr}",
        r"\caption{Paired task-level comparisons against the five-candidate Portfolio. Positive gains favour the Portfolio. All 13 comparisons form one Holm family. Comparisons with selectable Raw, NN+2-opt, ACO, DE and PSO measure incremental candidate-pool benefit, not independent algorithm superiority.}\label{tab:statistics}\\",
        r"\toprule",
        r"Comparator & W/T/L & Total gain (m) & Median [95\% CI] (m) & $r_{rb}$ & Raw $p$ & Holm $p$ \\",
        r"\midrule\endfirsthead",
        r"\toprule Comparator & W/T/L & Total gain (m) & Median [95\% CI] (m) & $r_{rb}$ & Raw $p$ & Holm $p$ \\",
        r"\midrule\endhead",
    ]
    for row in frame.itertuples():
        method = row.comparison.replace("Portfolio vs ", "")
        label = LABELS.get(method, method)
        lines.append(
            f"{label} & {row.wins}/{row.ties}/{row.losses} & {1000 * row.total_gain_km:.2f}"
            f" & {1000 * row.median_gain_km:.3f} [{1000 * row.median_gain_ci95_low_km:.3f},"
            f" {1000 * row.median_gain_ci95_high_km:.3f}] & {row.rank_biserial:.3f}"
            f" & {row.p_raw:.4g} & {row.p_holm:.4g} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{longtable}"])
    (OUT / "generated_table_s2.tex").write_text("\n".join(lines) + "\n")


def supplementary_search_accounting() -> None:
    frame = pd.read_csv(ROOT / "results" / "search_accounting_summary.csv").set_index("method")
    methods = ["Insertion+2opt", "MultiStart2opt", "ILS", "MultiStart2opt-RT", "ILS-RT"]
    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Method & Initials & Candidate moves & Accepted moves & LS passes & Perturbations & Full-route evals \\",
        r"\midrule",
    ]
    for method in methods:
        row = frame.loc[method]
        lines.append(
            f"{LABELS[method]} & {row.mean_initial_solutions:.1f} & {row.mean_candidate_moves_evaluated:.1f}"
            f" & {row.mean_accepted_2opt_moves:.1f} & {row.mean_completed_local_search_passes:.1f}"
            f" & {row.mean_perturbations:.1f} & {row.mean_full_route_objective_evaluations:.1f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (OUT / "generated_table_s3.tex").write_text("\n".join(lines) + "\n")


def sensitivity_table() -> None:
    frame = pd.read_csv(ROOT / "results" / "aco_sensitivity_summary.csv")
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"ACO setting & Total (km) & Mean gap (\%) & Maximum gap (\%) & Runtime (s) \\",
        r"\midrule",
    ]
    for row in frame.itertuples():
        if row.config_id == "base":
            label = "base"
        elif row.factor == "evaluation_budget":
            label = f"budget={row.value:g}"
        else:
            label = f"{row.factor}={row.value:g}"
        lines.append(
            f"{label} & {row.total_route_km:.3f} & {row.mean_gap_pct:.3f}"
            f" & {row.max_gap_pct:.3f} & {row.mean_runtime_s:.3f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (OUT / "generated_table_sensitivity.tex").write_text("\n".join(lines) + "\n")


def selection_table() -> None:
    frame = pd.read_csv(ROOT / "results" / "selection_frequency.csv")
    lines = [r"\begin{tabular}{lrr}", r"\toprule", r"Selected candidate & Count & Frequency (\%) \\", r"\midrule"]
    for row in frame.itertuples():
        lines.append(f"{LABELS.get(row.selected_method, row.selected_method)} & {int(row.count)} & {100 * row.frequency:.1f} \\\\ ")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (OUT / "generated_table_selection.tex").write_text("\n".join(lines) + "\n")


def main() -> None:
    table1_route_quality()
    table2_exact_runtime()
    supplementary_per_task()
    supplementary_statistics()
    supplementary_search_accounting()
    sensitivity_table()
    selection_table()


if __name__ == "__main__":
    main()
