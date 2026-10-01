#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

import numpy as np


def compute_tst_rates(
    energies: "np.ndarray",
    T: float,
    R: float,
    kB: float,
    h: float,
    complex_model: bool,
) -> "np.ndarray":
    E = np.asarray(energies, dtype=float)

    nstates = 6 if complex_model else 4
    if E.ndim != 2 or E.shape[1] != nstates or E.shape[0] < 1 or not np.all(np.isfinite(E)):
        raise ValueError("energies have invalid shape or values")
    if not all(np.isfinite(v) and v > 0 for v in (T, R, kB, h)):
        raise ValueError("physical constants must be finite and positive")
    A = kB * T / h
    if complex_model:
        barriers = np.column_stack((E[:, 1] - E[:, 0], E[:, 1] - E[:, 2], E[:, 3] - E[:, 2], E[:, 3] - E[:, 4], E[:, 5] - E[:, 4]))
    else:
        barriers = np.column_stack((E[:, 1] - E[:, 0], E[:, 1] - E[:, 2], E[:, 3] - E[:, 2]))
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        out = A * np.exp(-barriers / (R * T))
    if not np.all(np.isfinite(out)) or np.any(out <= 0):
        raise ValueError("rates are not finite positive")
    return out

import numpy as np


def compute_kinetic_observables(
    rate_states: "np.ndarray",
    complex_model: bool,
) -> "np.ndarray":
    r = np.asarray(rate_states, dtype=float)
    nr = 5 if complex_model else 3

    if (
        r.ndim != 2
        or r.shape[1] != nr
        or r.shape[0] < 1
        or not np.all(np.isfinite(r))
        or np.any(r <= 0)
    ):
        raise ValueError("invalid rate states")

    if complex_model:
        k1, km1, k2, km2, k3 = r.T
        D = k2 + km2 + k3
        kcat = k2 * k3 / D
        km = (
            k2 * k3
            + km1 * km2
            + km1 * k3
        ) / (k1 * D)
    else:
        k1, km1, k2 = r.T
        kcat = k2
        km = (km1 + k2) / k1

    eta = kcat / km
    kd = km1 / k1

    out = np.column_stack((kcat, km, eta, kd))

    if not np.all(np.isfinite(out)) or np.any(out <= 0):
        raise ValueError("invalid observables")

    return out

import numpy as np


def build_additive_double_energies(
    wild_type_energies: "np.ndarray",
    mutation_deltas: "np.ndarray",
    pair_indices: "np.ndarray",
) -> "np.ndarray":
    wt = np.asarray(wild_type_energies, dtype=float)
    d = np.asarray(mutation_deltas, dtype=float)
    p = np.asarray(pair_indices)
    if wt.ndim != 1 or d.ndim != 2 or d.shape[1] != wt.size or wt.size not in (4, 6) or not np.all(np.isfinite(wt)) or not np.all(np.isfinite(d)):
        raise ValueError("invalid energies")
    if p.ndim != 2 or p.shape[1] != 2 or p.shape[0] < 1 or not np.issubdtype(p.dtype, np.integer) or np.any(p < 0) or np.any(p >= d.shape[0]):
        raise ValueError("invalid pairs")
    out = wt[None, :] + d[p[:, 0]] + d[p[:, 1]]
    return out.astype(float, copy=False)

import numpy as np


def compute_single_fold_changes(
    observables: "np.ndarray",
    anchor_index: int,
) -> "np.ndarray":
    x = np.asarray(observables, dtype=float)
    if x.ndim != 2 or x.shape[1] != 4 or x.shape[0] < 1 or not np.all(np.isfinite(x)) or np.any(x <= 0):
        raise ValueError("invalid observables")
    if isinstance(anchor_index, (bool, np.bool_)) or not isinstance(anchor_index, (int, np.integer)) or anchor_index < 0 or anchor_index >= x.shape[0]:
        raise ValueError("invalid anchor")
    out = x / x[int(anchor_index)]
    if not np.all(np.isfinite(out)) or np.any(out <= 0):
        raise ValueError("invalid folds")
    return out

import numpy as np


def compute_pair_interactions(
    single_folds: "np.ndarray",
    anchor_observable: float,
    double_observables: "np.ndarray",
    pair_indices: "np.ndarray",
    anchor_index: int,
    observable_index: int,
) -> "np.ndarray":
    f = np.asarray(single_folds, dtype=float)
    dob = np.asarray(double_observables, dtype=float)
    p = np.asarray(pair_indices)
    if f.ndim != 2 or f.shape[1] != 4 or f.shape[0] < 2 or not np.all(np.isfinite(f)) or np.any(f <= 0):
        raise ValueError("invalid folds")
    if dob.ndim != 2 or dob.shape[1] != 4 or dob.shape[0] != p.shape[0] or not np.all(np.isfinite(dob)) or np.any(dob <= 0):
        raise ValueError("invalid doubles")
    if p.ndim != 2 or p.shape[1] != 2 or not np.issubdtype(p.dtype, np.integer) or np.any(p < 0) or np.any(p >= f.shape[0]):
        raise ValueError("invalid pairs")
    if isinstance(anchor_index, (bool, np.bool_)) or not isinstance(anchor_index, (int, np.integer)) or anchor_index < 0 or anchor_index >= f.shape[0]:
        raise ValueError("invalid anchor")
    if isinstance(observable_index, (bool, np.bool_)) or not isinstance(observable_index, (int, np.integer)) or observable_index < 0 or observable_index >= 4:
        raise ValueError("invalid observable index")
    a = int(observable_index)
    wt_fold = f[int(anchor_index), a]
    if not np.isfinite(anchor_observable) or anchor_observable <= 0:
        raise ValueError("anchor_observable must be finite and positive")
    null = anchor_observable * (f[p[:, 0], a] * f[p[:, 1], a]) / wt_fold
    out = dob[:, a] / null
    if not np.all(np.isfinite(out)) or np.any(out <= 0):
        raise ValueError("invalid interactions")
    return out.astype(float, copy=False)

import numpy as np


def classify_epistasis(
    interactions: "np.ndarray",
    threshold: float,
) -> "np.ndarray":
    e = np.asarray(interactions, dtype=float)
    if e.ndim != 1 or e.size < 1 or not np.all(np.isfinite(e)) or np.any(e <= 0):
        raise ValueError("invalid interactions")
    if not np.isfinite(threshold) or threshold <= 1:
        raise ValueError("threshold must exceed one")
    pos = e > threshold
    neg = e < 1.0 / threshold
    sig = pos | neg
    return np.array([sig.sum(), pos.sum(), neg.sum(), sig.mean()], dtype=float)

import numpy as np


def compute_complexity_ratio(
    simple_stats: "np.ndarray",
    complete_stats: "np.ndarray",
) -> float:
    s = np.asarray(simple_stats, dtype=float)
    c = np.asarray(complete_stats, dtype=float)
    if s.shape != (4,) or c.shape != (4,) or not np.all(np.isfinite(s)) or not np.all(np.isfinite(c)) or s[0] <= 0 or c[0] < 0:
        raise ValueError("invalid statistics")
    if s[3] <= 0:
        raise ValueError("simple prevalence must be positive")
    out = float(c[3] / s[3])
    if not np.isfinite(out) or out <= 0:
        raise ValueError("invalid ratio")
    return out

def resolve_kinetic_complexity_ratio(mutation_deltas: "np.ndarray") -> float:
    import numpy as np

    d = np.asarray(mutation_deltas, dtype=float)

    if d.shape != (16, 6) or not np.all(np.isfinite(d)):
        raise ValueError("mutation_deltas must have shape (16,6)")

    T = 298.15
    R = 1.98720425864083e-3
    kB = 1.380649e-23
    h = 6.62607015e-34

    wt4 = np.array([0.0, 10.0, -5.0, 11.0])
    wt6 = np.array([0.0, 10.0, -5.0, 11.0, -9.0, 9.0])

    pairs = np.array(
        [(i, j) for i in range(16) for j in range(i + 1, 16)],
        dtype=int
    )

    def _run(wt, complex_model):
        deltas = d if complex_model else d[:, :4]

        single_E = np.vstack(
            (wt, wt[None, :] + deltas)
        )

        single_rates = compute_tst_rates(
            single_E,
            T,
            R,
            kB,
            h,
            complex_model
        )

        single_obs = compute_kinetic_observables(
            single_rates,
            complex_model
        )

        folds = compute_single_fold_changes(
            single_obs,
            0
        )

        double_E = build_additive_double_energies(
            wt,
            deltas,
            pairs
        )

        double_rates = compute_tst_rates(
            double_E,
            T,
            R,
            kB,
            h,
            complex_model
        )

        double_obs = compute_kinetic_observables(
            double_rates,
            complex_model
        )

        interactions = compute_pair_interactions(
            folds,
            single_obs[0, 2],
            double_obs,
            pairs + 1,
            0,
            2
        )

        return classify_epistasis(
            interactions,
            1.5
        )

    simple = _run(wt4, False)
    complete = _run(wt6, True)

    return compute_complexity_ratio(
        simple,
        complete
    )
SCICODE_GOLD_EOF
