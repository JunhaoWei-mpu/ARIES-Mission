# ARIES-Mission: A Vision-Language System for Auditable UAV Mission Generation

ARIES-Mission separates semantic target generation from geometric mission
construction through an explicit, auditable interface. Multiple complete route
candidates remain available until permutation validation, common Haversine
rescoring and terminal arbitration. The controlled experiments freeze the
semantic output to isolate geometric routing from perception errors.

## Version 3.1.0

This revision expands the public code coverage using the existing implementations.
It adds no new experiments and changes no reported route, statistical family,
calibrated operation budget or original timing measurement. Version 3.1.0 is archived at https://doi.org/10.5281/zenodo.22843586.
The published ZIP was downloaded and verified against the checked release archive.
The preceding archive is https://doi.org/10.5281/zenodo.22044422.
Source repository: https://github.com/JunhaoWei-mpu/ARIES-Mission.

Included code covers:

- Raw, NN+2-opt, ACO, DE and PSO; common evaluator and terminal selector;
- nearest neighbour, cheapest insertion, insertion+2-opt, Multi-start 2-opt,
  ILS and ACO×3, including runtime-calibrated control drivers;
- runtime calibration and latency scripts, frozen calibration report and budgets;
- statistics, source-data assembly, route audit, table and figure generation;
- Concorde wrapper and exact-reference metadata, Held–Karp cross-check code;
- the existing optional grounding adapter, tests and reproduction instructions.

The geographic routing output is not an autopilot command or a flight-safety
certificate. Perception accuracy and physical flight execution were not evaluated.
The optional grounding adapter does not recover the missing historical immutable
model revision; frozen inputs are the source of truth for the reported study.

## Evidence and interpretation

Recognition order totals 79.655 km; the five-candidate implementation totals
60.348 km. Primary Multi-start, ILS and ACO×3 total 60.336–60.364 km. Runtime-calibrated
Multi-start totals 60.333 km, with 4.876 s mean serial runtime versus 7.307 s for
Portfolio. Portfolio records 0/23/7 wins/ties/losses and Holm-adjusted
p=0.0499609093097315; the control reaches the integerisation bound on all task means.
This is a backend deployment trade-off, not evidence of heterogeneous-search
superiority. Comparisons against selectable candidates measure incremental
candidate-pool benefit, not independent algorithm superiority.

## Reproduction

Use Python 3.13 and `requirements-lock.txt`. See `REPRODUCE.md` for verification
without route search, optional experimental reruns and third-party dependencies.

```bash
python -m pip install -r requirements-lock.txt
python -m pytest -q
python scripts/reproduce_analysis.py --output reproduced_analysis --figures
```

The last command recalculates summaries and audits all archived routes in a new
directory. It does not perform new searches, recalibration, VLM calls or timing
experiments, and leaves archived records unchanged.

## Licence and exclusions

The authors' original code and generated data are MIT licensed. Third-party
imagery, raw Mission Planner files, model weights and the Concorde binary are
excluded and retain their respective terms. Unreported exploratory work and
private review correspondence are not distributed. SHA256SUMS identifies every
archived file. Releasing independently implemented classical controls does not
redistribute third-party solver source code.
