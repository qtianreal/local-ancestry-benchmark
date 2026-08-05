"""Fit and validate the identifiability index.

Inside an ancestry tract a likelihood classifier accumulates per-site
log-likelihood ratios. For m sites the sum is approximately Gaussian, so
accuracy is Phi(sqrt(kappa m Dbar)) with Dbar the mean per-site discrimination
between the two source frequency distributions. Two substitutions make that
computable from quantities a practitioner has before running anything:

    Dbar ~ F_ST        the per-site KL divergence between two Bernoulli
                       sources reduces, for small frequency differences, to
                       (p1-p2)^2 / 2 pbar(1-pbar), whose expectation over
                       segregating sites is Hudson's estimator up to scale
    m    = delta / g   sites per tract: segregating sites per Morgan divided
                       by generations since admixture, tract length under a
                       single pulse being ~1/g Morgans

leaving one constant, kappa, absorbing the site frequency spectrum, the MAF
filter and the efficiency of the method being calibrated.

The fit is on real panels, which is the case a practitioner faces. The
simulated arm is fitted separately and reported, not pooled: the two require
different constants, and the ratio is a measure of how far the simulator
flatters the methods run on it.

Writes results/identifiability.json. No new simulation; every input already
exists in results/.
"""

import glob
import json
import math
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE / "results"

RHO = 1e-8   # per base per generation, the rate used in both arms
G = 30       # generations since the admixture pulse, fixed by the design
METHODS = ("naive_bayes", "hmm", "rfmix", "flare", "cnn")


def phi(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def inv_phi(p, k):
    """Index x at which the fitted curve reaches accuracy p."""
    lo, hi = 0.0, 1e6
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if phi(math.sqrt(k * mid)) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def collect():
    rows = []
    for r in json.loads((RES / "external_results.json").read_text()):
        acc = {k: r[k] for k in METHODS if k in r}
        rows.append(dict(arm="sim", label=f"T={r['split_time']}", fst=r["fst"],
                         n_sites=r["n_sites"], span=10e6, acc=acc))
    for f in sorted(glob.glob(str(RES / "real_*.json"))):
        d = json.loads(Path(f).read_text())
        if "n_sites" not in d or "fst" not in d:
            continue
        acc = {k: d[k] for k in METHODS if k in d}
        tag = "_".join(d["pops"])
        ext = RES / f"realext_{tag}.json"
        if ext.exists():
            e = json.loads(ext.read_text())
            acc.update({k: e[k] for k in ("rfmix", "flare") if k in e})
        rows.append(dict(arm="real", label="/".join(d["pops"]), fst=d["fst"],
                         n_sites=d["n_sites"], span=d["end"] - d["start"], acc=acc))
    for r in rows:
        r["delta"] = r["n_sites"] / (r["span"] * RHO)   # sites per Morgan
        r["m"] = r["delta"] / G                          # sites per tract
        r["x"] = r["m"] * r["fst"]                       # identifiability index
        r["best"] = max(r["acc"].values())
    return rows


def fit(sample, key):
    """Least squares for kappa on a log grid; one parameter, so this suffices."""
    lo, hi, n = math.log(1e-8), math.log(1e2), 8000
    grid = [math.exp(lo + i * (hi - lo) / n) for i in range(n + 1)]
    return min(((sum((phi(math.sqrt(k * r[key])) - r["best"]) ** 2 for r in sample), k)
                for k in grid))[1]


def loo(sample, key):
    """Leave-one-out: refit without each point, predict it."""
    errs = []
    for i, held in enumerate(sample):
        k = fit([r for j, r in enumerate(sample) if j != i], key)
        errs.append(phi(math.sqrt(k * held[key])) - held["best"])
    n = len(errs)
    return dict(rmse=math.sqrt(sum(e * e for e in errs) / n),
                mae=sum(abs(e) for e in errs) / n,
                max=max(abs(e) for e in errs),
                errors=errs)


def main():
    rows = collect()
    real = [r for r in rows if r["arm"] == "real"]
    sim = [r for r in rows if r["arm"] == "sim"]

    k_real, k_sim = fit(real, "x"), fit(sim, "x")
    lo_index = loo(real, "x")
    lo_plain = loo(real, "fst")      # same curve, F_ST alone, no density term

    fold = lambda v: max(v) / min(v)
    out = dict(
        n_real=len(real), n_sim=len(sim), g=G, rho=RHO,
        kappa_real=k_real, kappa_sim=k_sim, kappa_ratio=k_sim / k_real,
        loo_index=lo_index, loo_fst_only=lo_plain,
        # the density term cannot be identified from a design in which density
        # barely varies; report both spans so the text can say why
        fold_density=fold([r["delta"] for r in real]),
        fold_fst=fold([r["fst"] for r in real]),
        thresholds={f"{p:.2f}": dict(x=inv_phi(p, k_real),
                                     fst=inv_phi(p, k_real) / (sum(r["m"] for r in real) / len(real)))
                    for p in (0.60, 0.70, 0.80, 0.90, 0.95)},
        median_m=sorted(r["m"] for r in real)[len(real) // 2],
        pairs=[dict(arm=r["arm"], label=r["label"], fst=r["fst"], n_sites=r["n_sites"],
                    delta=r["delta"], m=r["m"], x=r["x"], best=r["best"],
                    pred=phi(math.sqrt((k_real if r["arm"] == "real" else k_sim) * r["x"])))
               for r in rows],
    )
    (RES / "identifiability.json").write_text(json.dumps(out, indent=2))

    print(f"real pairs {len(real)}, simulated levels {len(sim)}")
    print(f"kappa real {k_real:.4g}   sim {k_sim:.4g}   ratio {out['kappa_ratio']:.2f}")
    print(f"LOO index    RMSE {lo_index['rmse']:.4f}  MAE {lo_index['mae']:.4f}  max {lo_index['max']:.3f}")
    print(f"LOO F_ST only RMSE {lo_plain['rmse']:.4f}  (density spans {out['fold_density']:.2f}x, "
          f"F_ST spans {out['fold_fst']:.0f}x)")
    for p, t in out["thresholds"].items():
        print(f"  accuracy {p}: F_ST >= {t['fst']:.4f} at median 1000G density")
    print(f"wrote {RES/'identifiability.json'}")


if __name__ == "__main__":
    main()
