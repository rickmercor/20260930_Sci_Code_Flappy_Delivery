#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

def site_table(records: "list[tuple]") -> "np.ndarray":
    """Reference implementation for site_table."""
    import numpy as np

    out = []
    for entry in records:
        if not (isinstance(entry, (tuple, list)) and len(entry) == 5):
            raise ValueError("each site record must be a 5-tuple "
                             "(label, E_dis, k, instrument, replicas)")
        label, E, k, instrument, replicas = entry
        if not isinstance(label, str):
            raise ValueError("site label must be a string")
        E = float(E); k = float(k)
        if not (np.isfinite(E) and np.isfinite(k)):
            raise ValueError("E_dis and k must be finite")
        if E <= 0.0:
            raise ValueError("E_dis must be positive")
        if k <= 0.0:
            raise ValueError("force constant must be positive")
        if instrument not in ("A", "B"):
            raise ValueError("instrument must be 'A' or 'B'")
        beta = float(np.sqrt(k / (2.0 * E)))
        unit = 1.0 if instrument == "A" else 0.1

        trajectories = list(replicas)
        if len(trajectories) == 0:
            raise ValueError("each site needs at least one replica")
        replica_forces = []
        for trajectory in trajectories:
            readings = np.asarray(trajectory, dtype=float)
            if readings.ndim != 1 or readings.size == 0:
                raise ValueError("each replica must be a non-empty 1-D sequence")
            if not np.all(np.isfinite(readings)):
                raise ValueError("extension readings must be finite")
            failures = np.flatnonzero(readings == 0.0)
            start = int(failures[-1]) + 1 if failures.size else 0
            production = readings[start:]
            if production.size == 0:
                raise ValueError("a replica has no production reading after its last "
                                 "non-converged reading")
            x = production * unit
            u = np.exp(-beta * x)
            frame_force = 2.0 * E * beta * u * (1.0 - u) * 1.66054     # pN, signed
            replica_forces.append(float(np.mean(frame_force)))
        median = float(np.median(replica_forces))
        surviving = [f for f in replica_forces if abs(f - median) <= 0.10 * abs(median)]
        if not surviving:
            raise ValueError("every replica is contaminated")
        bond_force = float(np.mean(surviving))
        out.append([E, k, float(np.round(bond_force / 1000.0, 2))])
    if not out:
        raise ValueError("site table must not be empty")
    return np.asarray(out, dtype=float)

def stationary_points(site_array: "np.ndarray") -> "np.ndarray":
    """Reference implementation for stationary_points."""
    import numpy as np

    arr = np.asarray(site_array, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3 or arr.shape[0] == 0:
        raise ValueError("site array must have shape (n, 3)")
    if not np.all(np.isfinite(arr)):
        raise ValueError("site entries must be finite")
    if np.any(arr[:, 0] <= 0.0) or np.any(arr[:, 1] <= 0.0):
        raise ValueError("E_dis and k must be positive")
    if np.any(arr[:, 2] <= 0.0):
        raise ValueError("F must be positive")

    out = np.empty((arr.shape[0], 2), dtype=float)
    for j, (E, k, F) in enumerate(arr):
        beta = np.sqrt(k / (2.0 * E))
        g = 602.2 * F
        # dV_eff/dx = 0 is a quadratic in u = exp(-beta x): u^2 - u + g/(2 E beta) = 0.
        rad = 1.0 - 2.0 * g / (E * beta)
        if rad <= 0.0:
            x = np.log(2.0) / beta          # the merged double root, u = 1/2
            out[j, 0] = x
            out[j, 1] = x
            continue
        s = np.sqrt(rad)
        u_min = (1.0 + s) / 2.0             # larger root: the shifted well minimum
        u_max = (g / (E * beta)) / (1.0 + s)  # smaller root, written without cancellation
        out[j, 0] = -np.log(u_min) / beta
        out[j, 1] = -np.log(u_max) / beta
    return out

def site_rates(site_array: "np.ndarray") -> "tuple[np.ndarray, np.ndarray]":
    """Reference implementation for site_rates."""
    import numpy as np

    arr = np.asarray(site_array, dtype=float)
    xs = stationary_points(arr)          # validates the array as well

    NU, RGAS, T, TH = 2.88e11, 8.31446261815324e-3, 300.0, 294.15
    r_hom, r_hyd = [], []
    for j, (E, k, F) in enumerate(arr):
        beta = np.sqrt(k / (2.0 * E))
        g = 602.2 * F
        v = lambda x: E * (1.0 - np.exp(-beta * x)) ** 2 - g * x
        dV = float(max(v(xs[j, 1]) - v(xs[j, 0]), 0.0))
        r_hom.append(float(NU * np.exp(-dV / (RGAS * T))))

        low = 26.26 * F - 19.77
        high = -20.342988 + 0.070648 * TH + 1.605233 * F
        if F <= 0.65:
            lnk = low
        elif F > 0.75:
            lnk = high
        else:
            frac = (F - 0.65) / 0.10
            lnk = (1.0 - frac) * low + frac * high
        r_hyd.append(float(np.exp(lnk)))
    return np.asarray(r_hom), np.asarray(r_hyd)

def selection_probability(r_target: "np.ndarray", r_other: "np.ndarray") -> float:
    """Reference implementation for selection_probability."""
    import numpy as np

    rh = np.asarray(r_target, dtype=float)
    ry = np.asarray(r_other, dtype=float)
    if rh.ndim != 1 or ry.ndim != 1 or rh.shape != ry.shape or rh.size == 0:
        raise ValueError("rates must be non-empty equal-length one-dimensional arrays")
    if not (np.all(np.isfinite(rh)) and np.all(np.isfinite(ry))):
        raise ValueError("rates must be finite")
    if np.any(rh < 0.0) or np.any(ry < 0.0):
        raise ValueError("rates must be non-negative")
    tot = float(np.sum(rh) + np.sum(ry))
    if tot <= 0.0:
        raise ValueError("total rate must be positive")
    return float(np.sum(rh) / tot)

def branch_probabilities(r_target: "np.ndarray", r_other: "np.ndarray") -> "np.ndarray":
    """Reference implementation for branch_probabilities."""
    import numpy as np

    rh = np.asarray(r_target, dtype=float)
    ry = np.asarray(r_other, dtype=float)
    if rh.ndim != 1 or ry.ndim != 1 or rh.shape != ry.shape or rh.size == 0:
        raise ValueError("rates must be non-empty equal-length one-dimensional arrays")
    if not (np.all(np.isfinite(rh)) and np.all(np.isfinite(ry))):
        raise ValueError("rates must be finite")
    if np.any(rh < 0.0) or np.any(ry < 0.0):
        raise ValueError("rates must be non-negative")
    tot = float(np.sum(rh) + np.sum(ry))
    if tot <= 0.0:
        raise ValueError("total rate must be positive")
    return rh / tot

def relax_forces(site_array: "np.ndarray",
                         broken_index: int,
                         transfer_fraction: float) -> "np.ndarray":
    """Reference implementation for relax_forces."""
    import numpy as np

    arr = np.asarray(site_array, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3 or arr.shape[0] == 0:
        raise ValueError("site array must have shape (n, 3)")
    if not np.all(np.isfinite(arr)):
        raise ValueError("site entries must be finite")
    if np.any(arr[:, 0] <= 0.0) or np.any(arr[:, 1] <= 0.0):
        raise ValueError("E_dis and k must be positive")
    if np.any(arr[:, 2] <= 0.0):
        raise ValueError("F must be positive")
    t = float(transfer_fraction)
    if not np.isfinite(t):
        raise ValueError("transfer fraction must be finite")
    if t < 0.0:
        raise ValueError("transfer fraction must be non-negative")
    i = int(broken_index)
    n = arr.shape[0]
    if i < 0 or i >= n:
        raise ValueError("broken index out of range")

    keep = [j for j in range(n) if j != i]
    if not keep:
        return np.empty(0, dtype=float)

    E = arr[keep, 0]
    k = arr[keep, 1]
    beta = np.sqrt(k / (2.0 * E))
    g = 602.2 * arr[keep, 2]                     # kJ/mol/nm

    # Each survivor sits at its own shifted well minimum, u = exp(-beta x) = (1 + s) / 2.
    rad = 1.0 - 2.0 * g / (E * beta)
    if np.any(rad <= 0.0):
        raise ValueError("a survivor is already at or beyond the force it can sustain")
    u = (1.0 + np.sqrt(rad)) / 2.0

    def _total(d):
        """Total force the survivors carry when all are stretched by a further d."""
        v = u * np.exp(-beta * d)
        return float(np.sum(2.0 * E * beta * v * (1.0 - v)))

    target = float(np.sum(g)) + t * 602.2 * float(arr[i, 2])
    # Beyond this extension the first survivor passes its own force maximum, so the total
    # stops rising; the balance is monotone on [0, cap] and is bracketed there.
    cap = float(np.min(np.log(2.0 * u) / beta))
    if _total(cap) < target:
        raise ValueError("the survivors cannot take up the transferred load")
    lo, hi = 0.0, cap
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if _total(mid) < target:
            lo = mid
        else:
            hi = mid
    v = u * np.exp(-beta * (0.5 * (lo + hi)))
    return np.round(2.0 * E * beta * v * (1.0 - v) / 602.2, 4)

def apply_break(site_array: "np.ndarray",
                        broken_index: int,
                        pathway: str,
                        transfer_fraction: float,
                        weakening_fraction: float) -> "np.ndarray":
    """Reference implementation for apply_break."""
    import numpy as np

    arr = np.asarray(site_array, dtype=float)
    if pathway not in ("hom", "hyd"):
        raise ValueError("pathway must be 'hom' or 'hyd'")
    r = float(weakening_fraction)
    if not np.isfinite(r):
        raise ValueError("weakening fraction must be finite")
    if r < 0.0 or r >= 1.0:
        raise ValueError("weakening fraction must lie in [0, 1)")

    # The relaxation validates the array, the broken index and the transfer fraction.
    forces = relax_forces(arr, broken_index, transfer_fraction)

    keep = [j for j in range(arr.shape[0]) if j != int(broken_index)]
    out = arr[keep].copy()
    if out.shape[0] == 0:
        return out
    out[:, 2] = forces
    if pathway == "hom":
        r_hom, _ = site_rates(out)
        # argmax keeps the earliest row when two survivors are equally fast.
        tgt = int(np.argmax(r_hom))
        out[tgt, 0] = float(np.round(out[tgt, 0] * (1.0 - r), 4))
    return out

def sequence_probability(site_array: "np.ndarray",
                                 pathways: "list[str]",
                                 transfer_fraction: float,
                                 weakening_fraction: float) -> float:
    """Reference implementation for sequence_probability."""
    import numpy as np

    arr = np.asarray(site_array, dtype=float)
    seq = list(pathways)
    if len(seq) == 0:
        raise ValueError("pathway sequence must be non-empty")
    if any(s not in ("hom", "hyd") for s in seq):
        raise ValueError("each pathway must be 'hom' or 'hyd'")
    if arr.ndim != 2 or arr.shape[1] != 3 or arr.shape[0] == 0:
        raise ValueError("site array must have shape (n, 3)")
    if len(seq) > arr.shape[0]:
        raise ValueError("sequence longer than the number of bonds in the pool")

    t = float(transfer_fraction)
    if not np.isfinite(t):
        raise ValueError("transfer fraction must be finite")
    if t < 0.0:
        raise ValueError("transfer fraction must be non-negative")
    r = float(weakening_fraction)
    if not np.isfinite(r):
        raise ValueError("weakening fraction must be finite")
    if r < 0.0 or r >= 1.0:
        raise ValueError("weakening fraction must lie in [0, 1)")

    r_hom, r_hyd = site_rates(arr)
    target, other = (r_hom, r_hyd) if seq[0] == "hom" else (r_hyd, r_hom)
    if len(seq) == 1:
        return float(selection_probability(target, other))
    w = branch_probabilities(target, other)
    cond = []
    for i in range(arr.shape[0]):
        pool = apply_break(arr, i, seq[0], t, r)
        cond.append(sequence_probability(
            pool, seq[1:], t, r))
    return float(np.sum(w * np.asarray(cond)))

def four_event_mixed_cascade(records: "list[tuple]",
                                     transfer_fraction: float,
                                     weakening_fraction: float) -> float:
    """Reference implementation for four_event_mixed_cascade."""
    import numpy as np

    triples = site_table(records)
    if triples.shape[0] < 4:
        raise ValueError("the four-event cascade requires at least four sites")

    t = float(transfer_fraction)
    if not np.isfinite(t):
        raise ValueError("transfer fraction must be finite")
    if t < 0.0:
        raise ValueError("transfer fraction must be non-negative")
    r = float(weakening_fraction)
    if not np.isfinite(r):
        raise ValueError("weakening fraction must be finite")
    if r < 0.0 or r >= 1.0:
        raise ValueError("weakening fraction must lie in [0, 1)")

    r_hom, r_hyd = site_rates(triples)
    w1 = branch_probabilities(r_hyd, r_hom)
    cond = []
    for i in range(triples.shape[0]):
        pool = apply_break(triples, i, "hyd", t, r)
        cond.append(sequence_probability(
            pool, ("hom", "hom", "hom"), t, r))
    return float(np.sum(w1 * np.asarray(cond)))
SCICODE_GOLD_EOF
