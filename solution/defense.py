"""
Your defense. Implement register(ctx) and a handler per event type.
See ../README.md for the full interface + toolkit reference, and
../RULES.md before you start.

Strategy
--------
* One metered call per event (full coverage fits the budget; a second call
  never adds information because each tool already returns every field).
* Hard structural checks first — contract violations, missing lineage edges,
  orphaned outputs. These are exact, so they alert with no tolerance.
* Numeric signals are scored as z-scores against an adaptive per-metric
  model: the published baseline (mean ± 3σ) acts as a prior, and every event
  we judge clean refines the mean/σ online. Faulty-looking values are never
  folded back in, so a run of faults can't drag the model toward them.
* The alert threshold (Z_ALERT = 2.4σ) sits deliberately inside the 3σ
  baseline bound: with the score's 0.5·TPR vs 0.3·FPR weights and ~1 fault per
  3-4 events, catching a near-normal fault is worth far more than the rare
  clean event out past 2.6σ costs.
"""
import math

from api import Verdict

Z_ALERT = 2.4          # single-metric alert threshold (in σ)
Z_COMBINED = 2.0       # two metrics both past this also alert (weak signals combined)
PRIOR_WEIGHT = 6       # pseudo-observations the baseline-derived prior is worth
MIN_ONLINE = 3         # clean samples needed before a one-sided prior re-centres
BUDGET_SLACK = 1.25    # stop calling tools only if we'd blow far past budget

COST = {"data_batch": 1.0, "contract_checkpoint": 1.5, "lineage_run": 1.0,
        "feature_materialization": 2.0, "embedding_batch": 2.0}


# --------------------------------------------------------------------------
# adaptive per-metric model
# --------------------------------------------------------------------------

class Metric:
    """Gaussian model for one metric: baseline prior + Welford updates on
    values from events judged clean. `side` is "hi", "lo" or "both".

    Two prior shapes, both read off the published mean ± 3σ calibration:
      * two-sided (lo, hi): prior mean (lo+hi)/2, prior σ (hi-lo)/6.
      * one-sided (hi only): the bound is mean + 3σ, so once we've seen a few
        clean values the prior σ is (hi - observed mean)/3. Before that we
        fall back to a rough mean ≈ 3σ guess (spread over 0..hi).
    """

    def __init__(self, side="both", lo=None, hi=None, sd=None, mean=None):
        self.side, self.lo, self.hi = side, lo, hi
        self.fixed_sd, self.fixed_mean = sd, mean
        self.n, self.mean, self.m2 = 0, 0.0, 0.0

    def _prior(self):
        if self.fixed_sd is not None:
            return self.fixed_mean, self.fixed_sd
        if self.lo is not None and self.hi is not None:
            return (self.lo + self.hi) / 2.0, (self.hi - self.lo) / 6.0
        if self.hi is not None:
            if self.n >= MIN_ONLINE and self.mean < self.hi:
                return self.mean, (self.hi - self.mean) / 3.0
        return self.hi / 2.0, self.hi / 6.0

    def params(self):
        mu0, sd0 = self._prior()
        w = PRIOR_WEIGHT
        mu = (w * mu0 + self.n * self.mean) / (w + self.n)
        # pooled σ: prior spread + observed within-sample spread
        return mu, math.sqrt((w * sd0 ** 2 + self.m2) / (w + self.n)) or 1e-9

    def z(self, x):
        p = self.params()
        if p is None or x is None:
            return 0.0
        mu, sd = p
        d = (x - mu) / sd
        if self.side == "hi":
            return max(d, 0.0)
        if self.side == "lo":
            return max(-d, 0.0)
        return abs(d)

    def update(self, x):
        if x is None:
            return
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (x - self.mean)


def _models(ctx):
    m = ctx.state.get("models")
    if m is not None:
        return m
    b = ctx.baseline
    m = {
        "row_count": Metric("both", lo=b["row_count_min"], hi=b["row_count_max"]),
        "mean_amount": Metric("both", lo=b["mean_amount_min"], hi=b["mean_amount_max"]),
        "null_rate": Metric("hi", hi=b["null_rate_max"]),
        "staleness": Metric("hi", hi=b["staleness_min_max"]),
        "freshness": Metric("hi", hi=b["freshness_delay_max_min"]),
        "duration": Metric("hi", hi=b["lineage_duration_ms_max"]),
        # baseline bounds |shift| (half-normal): mean + 3sd = 2.607·s for the signed shift
        "feature_shift": Metric("both", mean=0.0, sd=b["feature_mean_shift_sigma_max"] / 2.607),
        "centroid": Metric("hi", hi=b["embedding_centroid_shift_max"]),
        "doc_age": Metric("hi", hi=b["corpus_avg_doc_age_days_max"]),
    }
    ctx.state["models"] = m
    return m


def _judge(ctx, pillar, signals, hard_reasons=()):
    """signals: {metric_name: value}. Alerts on any hard reason, any single
    z ≥ Z_ALERT, or two z's ≥ Z_COMBINED. Clean verdicts feed the models."""
    models = _models(ctx)
    zs = {k: models[k].z(v) for k, v in signals.items()}
    strong = [k for k, z in zs.items() if z >= Z_ALERT]
    medium = [k for k, z in zs.items() if z >= Z_COMBINED]
    reasons = list(hard_reasons)
    if strong:
        reasons.append("outlier:" + ",".join("%s=%.1fσ" % (k, zs[k]) for k in strong))
    elif len(medium) >= 2:
        reasons.append("combined:" + ",".join("%s=%.1fσ" % (k, zs[k]) for k in medium))
    alert = bool(reasons)
    if not alert:
        for k, v in signals.items():
            models[k].update(v)
    conf = min(1.0, max(zs.values(), default=0.0) / (2 * Z_ALERT)) if not hard_reasons else 1.0
    return Verdict(alert=alert, confidence=conf if alert else 1.0 - conf,
                   reason="; ".join(reasons), pillar=pillar)


def _can_afford(ctx, etype):
    budget_left = ctx.tools.budget_remaining()
    spent = ctx.tools.spend_so_far()
    budget = budget_left + spent
    return spent + COST[etype] <= budget * BUDGET_SLACK


def _num(x):
    return x if isinstance(x, (int, float)) else None


# --------------------------------------------------------------------------
# handlers
# --------------------------------------------------------------------------

def register(ctx):
    ctx.on("data_batch", check_data_batch)
    ctx.on("contract_checkpoint", check_contract_checkpoint)
    ctx.on("lineage_run", check_lineage_run)
    ctx.on("feature_materialization", check_feature_materialization)
    ctx.on("embedding_batch", check_embedding_batch)


def check_data_batch(payload, ctx):
    if not _can_afford(ctx, "data_batch"):
        return Verdict(alert=False, confidence=0.0, reason="budget", pillar="checks")
    r = ctx.tools.batch_profile(payload["batch_id"])
    if "error" in r:
        return Verdict(alert=False, confidence=0.0, reason=r["error"], pillar="checks")
    b = ctx.baseline
    nulls = (r.get("null_rate") or {}).get("customer_id")
    hard = []
    # published 3σ bounds are always violations, independent of the online model
    rc, ma, st = _num(r.get("row_count")), _num(r.get("mean_amount")), _num(r.get("staleness_min"))
    if rc is not None and not (b["row_count_min"] <= rc <= b["row_count_max"]):
        hard.append("row_count out of baseline")
    if ma is not None and not (b["mean_amount_min"] <= ma <= b["mean_amount_max"]):
        hard.append("mean_amount out of baseline")
    if _num(nulls) is not None and nulls > b["null_rate_max"]:
        hard.append("null_rate over baseline")
    if st is not None and st > b["staleness_min_max"]:
        hard.append("staleness over baseline")
    return _judge(ctx, "checks", {
        # std_amount is deliberately not scored: it has no published baseline,
        # so its σ comes from a handful of online samples and false-alarms a lot.
        "row_count": rc, "mean_amount": ma, "null_rate": _num(nulls), "staleness": st,
    }, hard)


def check_contract_checkpoint(payload, ctx):
    if not _can_afford(ctx, "contract_checkpoint"):
        return Verdict(alert=False, confidence=0.0, reason="budget", pillar="contracts")
    r = ctx.tools.contract_diff(payload["contract_id"], payload["checkpoint_batch_id"])
    if "error" in r:
        return Verdict(alert=False, confidence=0.0, reason=r["error"], pillar="contracts")
    hard = ["violation:" + str(v) for v in (r.get("violations") or [])]
    fresh = _num(r.get("freshness_delay_min"))
    sla = (payload.get("declared_sla") or {}).get("freshness_min")
    if fresh is not None and (fresh > ctx.baseline["freshness_delay_max_min"]
                              or (_num(sla) is not None and fresh > sla)):
        hard.append("freshness SLA breach")
    return _judge(ctx, "contracts", {"freshness": fresh}, hard)


def check_lineage_run(payload, ctx):
    if not _can_afford(ctx, "lineage_run"):
        return Verdict(alert=False, confidence=0.0, reason="budget", pillar="lineage")
    r = ctx.tools.lineage_graph_slice(payload["run_id"])
    if "error" in r:
        return Verdict(alert=False, confidence=0.0, reason=r["error"], pillar="lineage")

    job = payload.get("job", "?")
    hist = ctx.state.setdefault("lineage_hist", {}).setdefault(job, {"runs": 0, "edges": {}})
    actual = set(r.get("actual_upstream") or [])
    declared = {i.get("name") for i in (payload.get("inputs") or []) if isinstance(i, dict)}

    hard = []
    missing = declared - actual
    # edges this job normally has (seen in most previous runs) but lost now
    if hist["runs"] >= 2:
        missing |= {e for e, c in hist["edges"].items() if c / hist["runs"] >= 0.6} - actual
    if missing:
        hard.append("missing upstream:" + ",".join(sorted(missing)))
    down = _num(r.get("actual_downstream_count"))
    n_out = len(payload.get("outputs") or [])
    if down is not None and (down == 0 or down < n_out):
        hard.append("orphan output (downstream=%s)" % down)
    if _num(r.get("duration_ms")) is not None and r["duration_ms"] > ctx.baseline["lineage_duration_ms_max"]:
        hard.append("runtime over baseline")

    v = _judge(ctx, "lineage", {"duration": _num(r.get("duration_ms"))}, hard)
    if not missing:                     # learn the job's normal edge set from intact graphs
        hist["runs"] += 1
        for e in actual:
            hist["edges"][e] = hist["edges"].get(e, 0) + 1
    return v


def check_feature_materialization(payload, ctx):
    if not _can_afford(ctx, "feature_materialization"):
        return Verdict(alert=False, confidence=0.0, reason="budget", pillar="ai_infra")
    r = ctx.tools.feature_drift(payload["feature_view"], payload["batch_id"])
    if "error" in r:
        return Verdict(alert=False, confidence=0.0, reason=r["error"], pillar="ai_infra")
    serve, train, sd = _num(r.get("serve_mean")), _num(r.get("train_mean")), _num(r.get("train_std"))
    if None not in (serve, train, sd) and sd > 0:
        shift = (serve - train) / sd            # signed, so drift in either direction scores
    else:
        shift = _num(r.get("mean_shift_sigma"))
    hard = []
    if shift is not None and abs(shift) > ctx.baseline["feature_mean_shift_sigma_max"]:
        hard.append("feature skew over baseline")
    return _judge(ctx, "ai_infra", {"feature_shift": shift}, hard)


def check_embedding_batch(payload, ctx):
    if not _can_afford(ctx, "embedding_batch"):
        return Verdict(alert=False, confidence=0.0, reason="budget", pillar="ai_infra")
    r = ctx.tools.embedding_drift(payload["corpus"], payload["chunk_batch_id"])
    if "error" in r:
        return Verdict(alert=False, confidence=0.0, reason=r["error"], pillar="ai_infra")
    cs, age = _num(r.get("centroid_shift")), _num(r.get("avg_doc_age_days"))
    hard = []
    if cs is not None and cs > ctx.baseline["embedding_centroid_shift_max"]:
        hard.append("embedding drift over baseline")
    if age is not None and age > ctx.baseline["corpus_avg_doc_age_days_max"]:
        hard.append("corpus stale over baseline")
    return _judge(ctx, "ai_infra", {"centroid": cs, "doc_age": age}, hard)
