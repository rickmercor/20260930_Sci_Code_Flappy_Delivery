"""
Measure pairwise disagreement of the selected colour chains.

Compare only the five particle-label columns of the three aligned assignment tables; sector identifiers are taxonomy-local and are not comparable. An event disagrees when any particle label differs. Return the three event fractions in fixed order (unordered versus ordered-pair, ordered-pair versus multidipole, unordered versus multidipole). Raise ValueError for malformed, empty, or unequal-length inputs. Values are dimensionless and lie in [0,1].

Returns
-------
disagreements : np.ndarray, shape (3,), float Fractions in (unordered/ordered-pair, ordered-pair/multidipole, unordered/multidipole) order.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def compute_ordering_disagreements(unordered: np.ndarray, ordered_pair: np.ndarray, multidipole: np.ndarray) -> np.ndarray:
    '''Return pairwise mapping-order disagreement fractions.

    Returns
    -------
    disagreements : np.ndarray, shape (3,), float
        Fractions in (unordered/ordered-pair, ordered-pair/multidipole, unordered/multidipole) order.'''
    return np.zeros(3, dtype=float)

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

def _oracle_compute_ordering_disagreements(unordered, ordered_pair, multidipole):
    return _c_compute_ordering_disagreements(unordered, ordered_pair, multidipole)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': '# case: normal\np=_c_generate_five_parton_events(26082901,240)\nu=_c_assign_unordered_antenna_sectors(p)\no=_c_assign_ordered_pair_antenna_sectors(p)\nc=_c_assign_multidipole_antenna_sectors(p)', 'call': 'compute_ordering_disagreements(u,o,c)', 'gold_call': '_oracle_compute_ordering_disagreements(u,o,c)'},
        {'setup': '# case: normal\np=_c_generate_five_parton_events(26082902,1200)\nu=_c_assign_unordered_antenna_sectors(p)\no=_c_assign_ordered_pair_antenna_sectors(p)\nc=_c_assign_multidipole_antenna_sectors(p)', 'call': 'compute_ordering_disagreements(u,o,c)', 'gold_call': '_oracle_compute_ordering_disagreements(u,o,c)'},
        {'setup': '# case: normal\np=_c_generate_five_parton_events(26082903,4096)\nu=_c_assign_unordered_antenna_sectors(p)\no=_c_assign_ordered_pair_antenna_sectors(p)\nc=_c_assign_multidipole_antenna_sectors(p)', 'call': 'compute_ordering_disagreements(u,o,c)', 'gold_call': '_oracle_compute_ordering_disagreements(u,o,c)'},
        {'setup': '# case: boundary\np=_c_generate_five_parton_events(0,1)\nu=_c_assign_unordered_antenna_sectors(p)\no=_c_assign_ordered_pair_antenna_sectors(p)\nc=_c_assign_multidipole_antenna_sectors(p)', 'call': 'compute_ordering_disagreements(u,o,c)', 'gold_call': '_oracle_compute_ordering_disagreements(u,o,c)'},
        {'setup': '# case: edge\ndef capture_value_error(fn):\n    try:\n        fn()\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n    return 0\nu=np.zeros((1,5), dtype=int)\no=np.zeros((1,6), dtype=int)\nc=np.zeros((1,6), dtype=int)', 'call': 'capture_value_error(lambda: compute_ordering_disagreements(u,o,c))', 'gold_call': 'capture_value_error(lambda: _oracle_compute_ordering_disagreements(u,o,c))'}
    ]
