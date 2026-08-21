# Reproduction protocol

## Frozen protocol

- Input: `results/frozen_instances_gps.json` (30 tasks).
- Seeds: integers 0--19.
- Candidate set: Raw, NN+2-opt, ACO, DE and PSO.
- Stochastic budget: 7,500 complete-route objective evaluations for each of
  ACO, DE and PSO per task and seed.
- Population: 50.
- Objective: closed-loop double-precision Haversine distance with Earth radius
  6,371.0088 km.
- Validity: every target index must occur exactly once.
- Tie tolerance: 1e-12 km, resolved in the fixed order Raw, NN+2-opt, ACO, DE,
  PSO.

The code never reads exact optima or external-control outputs during search or
terminal selection.

## Commands

Install and test:

```bash
python -m pip install -r requirements-lock.txt
python -m pytest -q
```

Fast smoke run on seed 0:

```bash
python -m aries_portfolio.reproduce --root . --tasks 1 --seeds 0 --workers 1
```

Complete run:

```bash
python -m aries_portfolio.reproduce --root . --workers 20
```

Outputs are written to `reproduced_results/portfolio_routes.csv` and
`reproduced_results/portfolio_routes.json`. If
`results/reported_portfolio_routes.csv` is present, the program checks method,
route indices, selected candidate and length (tolerance 1e-12 km) for every
requested task and seed. Runtime is recorded but is not used as a reproducibility
criterion.

The complete run should contain 3,600 rows: six records (five candidates plus
Portfolio) for each of 30 tasks and 20 seeds. The Portfolio total is computed by
averaging its 20 seeded lengths within each task and then summing over tasks.
