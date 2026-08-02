# Isolated topology manual review

This is a user-directed experimental exception to the normal production/review
directory split. Nothing in production imports this package, and all generated
material stays beneath `review_assets/generated/`.

`topology.py` began as a byte-for-byte copy of the production topology module.
The accepted baseline is retained in the generated review. The local copy now
contains the first candidate change: a one-child standard must place its child
aperture center at no less than 80% of the peak distance in its exact filled
parent contour. The replay helper always evaluates the unchanged production
classifier first, then evaluates this candidate on the same frame evidence.

From the `deterministic_v3` directory:

```bash
python3.13 -m src.manual_review.topology.run_review \
  ../../../../../logs/run-20260801T031401Z
```

Baseline mode requires the copied source and every decision to match
production. Because the local copy now contains an intentional candidate
change, use:

```bash
python3.13 -m src.manual_review.topology.run_review RUN_PATH --mode candidate
```

Each execution creates a new timestamped directory containing a manifest,
one JSONL record and one five-panel visualization per component, plus static
HTML indexes grouped by predicted topology label.
