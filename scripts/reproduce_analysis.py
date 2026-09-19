"""Recompute summaries from archived runs in an isolated output directory.

No route search, VLM generation, exact solve or timing experiment is executed.
The archived records in the release remain unchanged.
"""
from __future__ import annotations
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
import pandas as pd
from pandas.testing import assert_frame_equal

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, default=ROOT/'reproduced_analysis')
    parser.add_argument('--figures', action='store_true', help='Also regenerate figures that do not require third-party imagery')
    args=parser.parse_args()
    root=args.root.resolve(); out=args.output.resolve()
    if out == root or root in out.parents and out.relative_to(root).parts[0] in {'results','configs','aries_portfolio','scripts','tests','figures','manuscript'}:
        raise ValueError('Output must not overwrite release source directories')
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use a new or empty output directory')
    out.mkdir(parents=True, exist_ok=True)
    for name in ['results','configs']:
        shutil.copytree(root/name, out/name)
    sys.path.insert(0,str(root))
    from aries_portfolio.statistics import build_statistics
    from aries_portfolio.audit import audit_results
    build_statistics(out)
    # Numeric tolerance only covers CSV parser/roundtrip arithmetic, not scientific changes.
    compared=[]
    for name in ['aggregate.csv','task_method_means.csv','statistical_tests.csv','optimality_gaps.csv','exact_gap_summary.csv','exact_crosscheck.csv','selection_frequency.csv','task_gains.csv','gain_correlations.csv']:
        assert_frame_equal(pd.read_csv(root/'results'/name),pd.read_csv(out/'results'/name),check_exact=False,rtol=1e-12,atol=1e-12)
        compared.append(name)
    assert json.loads((root/'results/holm_family.json').read_text()) == json.loads((out/'results/holm_family.json').read_text())
    audit=audit_results(out)
    (out/'scripts').mkdir(); (out/'manuscript').mkdir()
    shutil.copy2(root/'scripts/make_tables.py',out/'scripts/make_tables.py')
    subprocess.run([sys.executable,str(out/'scripts/make_tables.py')],check=True)
    if args.figures:
        from aries_portfolio.figures import figure1_architecture,figure2_route_quality,figure4_budget_runtime,supplementary_convergence,write_manifest
        (out/'figures').mkdir()
        for fn in [figure1_architecture,figure2_route_quality,figure4_budget_runtime,supplementary_convergence]: fn(out)
        write_manifest(out)
    report={'status':'pass','summaries_compared':compared,'holm_family_unchanged':True,'audit':audit,'new_experiments':False}
    (out/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__ == '__main__': main()
