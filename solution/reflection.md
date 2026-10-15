# Reflection (≤1 page)

**Results:** practice 48.28 (TPR 1.00, FPR 0.057) · public 46.69 (TPR 1.00, FPR 0.050) ·
**private 30.87 (TPR 0.63, FPR 0.021, no cost overage)**.

**Approach.** Each event gets exactly one metered call, because every tool already
returns all of its fields. Full coverage costs about 1.5 credits per event, which fits
private's 320 budget. Exact structural checks alert unconditionally: contract
`violations`, a freshness SLA breach, a lineage edge the job normally has but is now
missing, and `downstream_count == 0`. Numeric signals become z-scores against an online
per-metric Gaussian model. The published mean ± 3σ baseline is the prior. For one-sided
bounds (`max = μ + 3σ`), I derive σ as `(max − observed clean mean)/3`, and this matched
the practice data for all six such metrics. Only events judged clean update the model,
so a fault can't drag the reference toward itself. An event alerts at 2.4σ on one
metric, or when two metrics are both at 2.0σ.

**Which fault types were hardest to catch, and why?**
- *Subtle distribution faults on `data_batch`* (volume, mean_amount, nulls, staleness).
  Each batch has four noisy metrics, so a 2–3σ shift on one of them overlaps the clean
  tail. Lowering the threshold catches more of them, but false alarms grow with the
  number of metrics.
- *Subtle runtime anomalies and corpus staleness* close to normal variance. These are
  one-sided metrics whose clean mean is not published, so the first few events of a
  stream are judged against a rough prior until enough clean samples arrive.
- *False positives were the real trap.* My first version also scored `std_amount`,
  which has no published baseline. Its σ came from a handful of online samples, and it
  produced about half of public's false alarms without catching a single fault.
  Dropping it cut public FPR from 9% to 5% at the same TPR.
- Structural faults (schema/type violations, missing upstream, orphan outputs) were
  easy once I noticed that a clean run's `actual_upstream` holds *more* than the
  declared inputs (`raw.customers`). Comparing against the declared inputs alone would
  miss the dropped edge, so I compare against the job's learned normal edge set.

**What would you change about your cost/coverage tradeoff, if you had another pass?**
- Cost was never the binding constraint: one call per event left headroom. Coverage is
  what pays, since skipping an event forfeits its fault, while overspending is only a
  small linear penalty.
- Private FPR (2%) came in well below public (5%) while TPR fell to 63%, so on the
  subtle-heavy stream I was too conservative. With the 0.5·TPR vs 0.3·FPR weighting and
  roughly one fault per 3–4 events, catching a near-threshold fault is worth about four
  times what a false alarm costs. My simulation (synthetic 1.5–3.5σ faults on the
  fitted clean distributions) kept improving down to about 2.0–2.2σ. I'd move the
  single-metric threshold toward 2.2σ and lean harder on combined weak signals.
- To make a lower threshold safe, I'd stabilise the early-stream estimates with a
  warm-up phase (a stricter threshold for the first few events of each type) and use
  robust statistics (median/MAD) so that undetected subtle faults don't inflate σ.
- I'd look for cross-event signals as well. For example, a contract checkpoint and its
  `checkpoint_batch_id` could be checked against the matching batch's null rate, since
  the contract declares `null_rate_max`.
