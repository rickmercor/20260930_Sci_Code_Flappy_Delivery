"""
Compose the phase-sector taxonomy diversity index.

Call every preceding public function. Generate the common RAMBO ensemble, build all three assignment tables, their 3-by-12 population table, and their three ordering-disagreement fractions. For each taxonomy compute normalized Shannon entropy -sum_k p_k log(p_k)/log(K), using only positive entries and K=(12,12,5). Return mean(disagreement fractions)+mean(normalized entropies), without rounding. This equal-weight index is in [0,2]: it is large only when the antenna classes often select different colour chains and each paper-specific sectorization broadly occupies its available cases. Do not reproduce the sector rules or RAMBO generator inside this function.

Returns
-------
mapping_index : float Mean ordering disagreement plus mean normalized sector entropy.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def compute_sector_mapping_index(seed: int, n_events: int) -> float:
    '''Return the unrounded mapping-taxonomy diversity index.

    Returns
    -------
    mapping_index : float
        Mean ordering disagreement plus mean normalized sector entropy.'''
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _momenta(momenta):
    try:
        raw = np.asarray(momenta)
        if np.iscomplexobj(raw):
            raise ValueError("momenta must be real")
        p = raw.astype(float, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("momenta must be a finite real (N,5,4) array with N >= 1") from exc
    if (
        p.ndim != 3
        or p.shape[1:] != (5, 4)
        or p.shape[0] < 1
        or not np.all(np.isfinite(p))
    ):
        raise ValueError("momenta must be a finite real (N,5,4) array with N >= 1")
    if np.any(p[:, :, 0] <= 0.0):
        raise ValueError("all momenta must be future directed")
    mass2 = p[:, :, 0] ** 2 - np.sum(p[:, :, 1:] ** 2, axis=2)
    scale = np.maximum(p[:, :, 0] ** 2, 1.0)
    if np.any(np.abs(mass2) > 2.0e-10 * scale):
        raise ValueError("all momenta must be massless")
    return p

def _invariants(momenta):
    p = _momenta(momenta)
    metric = np.array([1.0, -1.0, -1.0, -1.0])
    return 2.0 * np.einsum("nia,nja,a->nij", p, p, metric)

def _c_assign_unordered_antenna_sectors(momenta):
    """Section 3.2.1: three unordered emissions, twelve sectors."""
    s_all = _invariants(momenta)
    out = np.empty((len(s_all), 6), dtype=np.int64)
    for r, S in enumerate(s_all):
        def s(a, b):
            return S[a - 1, b - 1]

        values = [s(1, 2), s(1, 3), s(1, 4), s(2, 5), s(3, 5), s(4, 5)]
        which = int(np.argmin(values))
        if which == 0:
            first = s(1, 3) * s(4, 5) <= s(1, 4) * s(3, 5)
            sector, middle = ((1, (2, 3, 4)) if first else (9, (2, 4, 3)))
        elif which == 1:
            first = s(1, 2) * s(4, 5) <= s(1, 4) * s(2, 5)
            sector, middle = ((5, (3, 2, 4)) if first else (11, (3, 4, 2)))
        elif which == 2:
            first = s(1, 3) * s(2, 5) <= s(1, 2) * s(3, 5)
            sector, middle = ((3, (4, 3, 2)) if first else (7, (4, 2, 3)))
        elif which == 3:
            first = s(1, 4) * s(3, 5) <= s(1, 3) * s(4, 5)
            sector, middle = ((4, (4, 3, 2)) if first else (12, (3, 4, 2)))
        elif which == 4:
            first = s(1, 4) * s(2, 5) <= s(1, 2) * s(4, 5)
            sector, middle = ((8, (4, 2, 3)) if first else (10, (2, 4, 3)))
        else:
            first = s(1, 2) * s(3, 5) <= s(1, 3) * s(2, 5)
            sector, middle = ((2, (2, 3, 4)) if first else (6, (3, 2, 4)))
        out[r] = (sector, 1, *middle, 5)
    return out

def _c_assign_ordered_pair_antenna_sectors(momenta):
    """Section 3.2.2: one unordered emission and an ordered pair."""
    s_all = _invariants(momenta)
    out = np.empty((len(s_all), 6), dtype=np.int64)
    for r, S in enumerate(s_all):
        def s(a, b):
            return S[a - 1, b - 1]

        values = [s(1, 2), s(1, 3), s(1, 4), s(2, 5), s(3, 5), s(4, 5)]
        which = int(np.argmin(values))
        s_hi = values[which]
        if s_hi >= s(3, 4):
            first = ((s(1, 3) + s(1, 4)) * s(2, 5)
                     <= s(1, 2) * (s(3, 5) + s(4, 5)))
            sector, middle = ((1, (3, 4, 2)) if first else (2, (2, 3, 4)))
        elif which == 0:
            sector, middle = 3, (2, 3, 4)
        elif which == 3:
            sector, middle = 4, (3, 4, 2)
        elif which == 1:
            first = s(1, 4) * s(2, 5) > s(1, 2) * s(4, 5)
            sector, middle = ((5, (3, 2, 4)) if first else (6, (3, 4, 2)))
        elif which == 2:
            first = s(1, 3) * s(2, 5) > s(1, 2) * s(3, 5)
            sector, middle = ((7, (2, 3, 4)) if first else (8, (3, 4, 2)))
        elif which == 4:
            first = s(1, 2) * s(4, 5) > s(1, 4) * s(2, 5)
            sector, middle = ((9, (3, 4, 2)) if first else (10, (2, 3, 4)))
        else:
            first = s(1, 2) * s(3, 5) > s(1, 3) * s(2, 5)
            sector, middle = ((11, (3, 2, 4)) if first else (12, (2, 3, 4)))
        out[r] = (sector, 1, *middle, 5)
    return out

def _c_assign_multidipole_antenna_sectors(momenta):
    """Section 3.2.3: a gluon emitted between multiple dipoles."""
    s_all = _invariants(momenta)
    out = np.empty((len(s_all), 6), dtype=np.int64)
    for r, S in enumerate(s_all):
        def s(a, b):
            return S[a - 1, b - 1]

        values = [s(1, 2), s(2, 3), s(3, 4) + s(3, 5) + s(4, 5),
                  s(2, 4), s(2, 5)]
        sector = int(np.argmin(values)) + 1
        middle = (2, 3, 4) if sector <= 3 else (3, 4, 2)
        out[r] = (sector, 1, *middle, 5)
    return out

def _c_generate_five_parton_events(seed, n_events):
    """Generate unit-COM-energy massless five-particle events by RAMBO."""
    if not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not isinstance(n_events, (int, np.integer)) or not 1 <= int(n_events) <= 200000:
        raise ValueError("n_events must be an integer in [1,200000]")
    rng = np.random.default_rng(int(seed))
    u = rng.random((int(n_events), 5, 4))
    c = 2.0 * u[:, :, 0] - 1.0
    phi = 2.0 * np.pi * u[:, :, 1]
    energy = -np.log(np.maximum(u[:, :, 2] * u[:, :, 3], np.finfo(float).tiny))
    st = np.sqrt(np.maximum(0.0, 1.0 - c * c))
    q = np.stack((energy, energy * st * np.cos(phi),
                  energy * st * np.sin(phi), energy * c), axis=2)
    total = np.sum(q, axis=1)
    mass = np.sqrt(total[:, 0] ** 2 - np.sum(total[:, 1:] ** 2, axis=1))
    boost = -total[:, 1:] / mass[:, None]
    gamma = total[:, 0] / mass
    a = 1.0 / (1.0 + gamma)
    dot = np.einsum("ni,nji->nj", boost, q[:, :, 1:])
    p0 = (gamma[:, None] * q[:, :, 0] + dot) / mass[:, None]
    pvec = (q[:, :, 1:] + boost[:, None, :] * q[:, :, 0, None]
            + a[:, None, None] * dot[:, :, None] * boost[:, None, :]) / mass[:, None, None]
    return np.concatenate((p0[:, :, None], pvec), axis=2)

def _c_tabulate_sector_populations(unordered, ordered_pair, multidipole):
    """Return three zero-padded sector-frequency rows."""
    tables = [np.asarray(unordered), np.asarray(ordered_pair), np.asarray(multidipole)]
    if any(t.ndim != 2 or t.shape[1] != 6 for t in tables):
        raise ValueError("each assignment table must have shape (N,6)")
    if not tables[0].shape[0] or len({len(t) for t in tables}) != 1:
        raise ValueError("assignment tables must have the same positive row count")
    out = np.zeros((3, 12), dtype=float)
    for row, (table, n_sector) in enumerate(zip(tables, (12, 12, 5))):
        sector = np.asarray(table[:, 0], dtype=np.int64)
        if np.any(sector < 1) or np.any(sector > n_sector):
            raise ValueError("sector identifier outside the expected range")
        out[row, :n_sector] = np.bincount(sector, minlength=n_sector + 1)[1:] / len(table)
    return out

def _c_compute_ordering_disagreements(unordered, ordered_pair, multidipole):
    """Fractions of events on which each pair chooses a different ordering."""
    tables = [np.asarray(unordered), np.asarray(ordered_pair), np.asarray(multidipole)]
    if any(t.ndim != 2 or t.shape[1] != 6 for t in tables):
        raise ValueError("each assignment table must have shape (N,6)")
    if not tables[0].shape[0] or len({len(t) for t in tables}) != 1:
        raise ValueError("assignment tables must have the same positive row count")
    return np.array([
        np.mean(np.any(tables[0][:, 1:] != tables[1][:, 1:], axis=1)),
        np.mean(np.any(tables[1][:, 1:] != tables[2][:, 1:], axis=1)),
        np.mean(np.any(tables[0][:, 1:] != tables[2][:, 1:], axis=1)),
    ], dtype=float)

def _c_compute_sector_mapping_index(seed, n_events):
    """Compose all steps into an equal-weight disagreement/diversity index."""
    momenta = _c_generate_five_parton_events(seed, n_events)
    unordered = _c_assign_unordered_antenna_sectors(momenta)
    ordered_pair = _c_assign_ordered_pair_antenna_sectors(momenta)
    multidipole = _c_assign_multidipole_antenna_sectors(momenta)
    populations = _c_tabulate_sector_populations(unordered, ordered_pair, multidipole)
    disagreements = _c_compute_ordering_disagreements(unordered, ordered_pair, multidipole)
    entropy = []
    for row, count in enumerate((12, 12, 5)):
        prob = populations[row, :count]
        prob = prob[prob > 0.0]
        entropy.append(-np.sum(prob * np.log(prob)) / np.log(count))
    return float(np.mean(disagreements) + np.mean(entropy))

def _oracle_compute_sector_mapping_index(seed, n_events):
    import numpy as np
    momenta = _oracle_generate_five_parton_events(seed, n_events)
    unordered = _oracle_assign_unordered_antenna_sectors(momenta)
    ordered_pair = _oracle_assign_ordered_pair_antenna_sectors(momenta)
    multidipole = _oracle_assign_multidipole_antenna_sectors(momenta)
    populations = _oracle_tabulate_sector_populations(unordered, ordered_pair, multidipole)
    disagreements = _oracle_compute_ordering_disagreements(
        unordered, ordered_pair, multidipole
    )
    entropy = []
    for row, count in enumerate((12, 12, 5)):
        prob = populations[row, :count]
        prob = prob[prob > 0.0]
        entropy.append(-np.sum(prob * np.log(prob)) / np.log(count))
    return float(np.mean(disagreements) + np.mean(entropy))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': '# case: normal', 'call': 'compute_sector_mapping_index(17,1024)', 'gold_call': '_oracle_compute_sector_mapping_index(17,1024)'},
        {'setup': '# case: normal', 'call': 'compute_sector_mapping_index(260829,4096)', 'gold_call': '_oracle_compute_sector_mapping_index(260829,4096)'},
        {'setup': '# case: normal', 'call': 'compute_sector_mapping_index(20260829,8192)', 'gold_call': '_oracle_compute_sector_mapping_index(20260829,8192)'},
        {'setup': '# case: boundary', 'call': 'compute_sector_mapping_index(20260829,1)', 'gold_call': '_oracle_compute_sector_mapping_index(20260829,1)'}
    ]
