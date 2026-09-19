# Reproduce the reported analysis

Python 3.13 with `requirements-lock.txt` records the routing-analysis environment.
The dependency lock is not a record of the historical VLM acquisition environment.

## Verify the frozen evidence without rerunning experiments

```bash
sha256sum -c SHA256SUMS
python -m pip install -r requirements-lock.txt
python -m pytest -q
python scripts/reproduce_analysis.py --output reproduced_analysis --figures
```

The output directory must be absent or empty. Summary tables and the 13-test
Holm family are compared with archived outputs. The audit checks 6,747 route
records, 600 five-candidate selections, all-instance exact-reference metadata
and the 27 Held–Karp cross-checks. No exact solver is needed for this audit.
Certification provenance is checked from archived records; this is not a fresh
Concorde certification run. Figure 3 is omitted by this command because it uses
third-party imagery; its source table and plotting implementation are included.

## Optional routing reruns (not required for this revision)

Run full experiments in a separate copy of the archive: the original drivers
write results in place. The budgets in `configs/experiment.json` must remain
unchanged. In particular, runtime-calibrated controls use 639 starts and 2,131
ILS perturbations. Do not run `runtime_calibration` to replace these reported
budgets. Its synthetic development script and original report are supplied for
provenance; calibration and fresh timing depend on hardware and are not expected
to reproduce the historical seconds exactly.

```bash
# Five-candidate convenience CLI, writes reproduced_results/ by default:
python -m aries_portfolio.reproduce --tasks 1 --seeds 0 --workers 1
# Full stochastic candidates, ACOx3 and exact references (requires Concorde):
python -m aries_portfolio.experiment --root .
# Existing classical and runtime-calibrated control implementations:
python -m aries_portfolio.formal_controls --root .
```

`baselines.py` implements Multi-start 2-opt and ILS; `solvers.py` implements the
ACO×3 terminal selection, called by `experiment.py` with three independent streams.
`formal_controls.py` applies both the primary and frozen runtime-calibrated
local-search budgets. `latency.py`, `local_search_latency.py` and
`runtime_reporting.py` contain the measurement/reporting drivers. Their commands
perform fresh timing and must not overwrite the historical submission records.
`statistics.py`, `source_data.py`, `audit.py`, `scripts/make_tables.py` and
`figures.py` provide the existing analysis and figure workflows.

## Exact solver and third-party imagery

See `external_tools/README.md` for Concorde 03.12.19 with QSopt, binary checksum,
installation location and the small-instance wrapper. Do not redistribute its
binary under the authors' MIT licence. Held–Karp is implemented in `baselines.py`.
The derived exact-reference metadata and integerisation scale are included.

Benchmark sources:
- https://github.com/Sautenich/UAV-VLPA
- https://github.com/sautenich/uav-vla

To regenerate Figure 3, obtain the original Task 8 satellite image under its
source terms and place it at `data/nano30/images/8.jpg`. Retain the frozen point percentages in the provided instance
metadata. Then call `figure3_routes(Path('.'))` from `aries_portfolio.figures`.
The frozen geographic coordinates are sufficient for all routing and quantitative
analysis; no image, GPU, API key or model weight is required for those steps.
The original raw HumanPlan files are excluded; derived HumanPlan summaries remain
available as unmatched context, and are not part of the routing superiority tests.

## Optional grounding adapter

`vlm.py` retains the existing acquisition adapter. Running it requires the
third-party model, PyTorch, Transformers, Accelerate and bitsandbytes, plus suitable
hardware. These optional dependencies and the missing historical model revision
are not reconstructed by this release. New grounding outputs would define new
instances and must not replace the frozen inputs used in the manuscript.
