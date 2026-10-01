"""
Assign sectors for a gluon emitted between multiple dipoles.

Implement the exact five-case construction in Section 3.2.3 of the source paper. Particle 2 is the gluon, particles 3 and 4 are the quark/antiquark pair, and particles 1 and 5 are hard radiators. Return the 1-based position of the selected listed case plus the exact five-label order supplied to the corresponding 5-to-2 mapping. The section mixes different invariant types and fixes a paper-specific association between cases and the two available chains; reproduce that case-to-chain association exactly. Raise ValueError if momenta is not a finite real array of shape (N,5,4) with N at least 1. The momenta supplied are future-directed, massless, and free of exact ties, so no further validation is needed. Use s_ij=2 p_i dot p_j and metric (+,-,-,-).

Returns
-------
assignment : np.ndarray, shape (N,6), int64 Sector, then the five 1-based particle labels in mapping order.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def assign_multidipole_antenna_sectors(momenta: np.ndarray) -> np.ndarray:
    '''Return the Section 3.2.3 sector and mapping order.

    Returns
    -------
    assignment : np.ndarray, shape (N,6), int64
        Sector, then the five 1-based particle labels in mapping order.'''
    return np.empty((0, 6), dtype=np.int64)

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

def _oracle_assign_multidipole_antenna_sectors(momenta):
    return _c_assign_multidipole_antenna_sectors(momenta)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': '# case: normal\np=_c_generate_five_parton_events(26082901,240)', 'call': 'assign_multidipole_antenna_sectors(p)', 'gold_call': '_oracle_assign_multidipole_antenna_sectors(p)'},
        {'setup': '# case: normal\np=_c_generate_five_parton_events(26082902,1200)', 'call': 'assign_multidipole_antenna_sectors(p)', 'gold_call': '_oracle_assign_multidipole_antenna_sectors(p)'},
        {'setup': '# case: normal\np=_c_generate_five_parton_events(26082903,4096)', 'call': 'assign_multidipole_antenna_sectors(p)', 'gold_call': '_oracle_assign_multidipole_antenna_sectors(p)'},
        {'setup': '# case: boundary\np=_c_generate_five_parton_events(0,1)', 'call': 'assign_multidipole_antenna_sectors(p)', 'gold_call': '_oracle_assign_multidipole_antenna_sectors(p)'},
        {'setup': '# case: edge\ndef capture_value_error(fn):\n    try:\n        fn()\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n    return 0\np=np.zeros((1,5,3), dtype=float)', 'call': 'capture_value_error(lambda: assign_multidipole_antenna_sectors(p))', 'gold_call': 'capture_value_error(lambda: _oracle_assign_multidipole_antenna_sectors(p))'},
        {'setup': '# case: edge\ndef capture_value_error(fn):\n    try:\n        fn()\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n    return 0\np=np.zeros((0,5,4), dtype=float)', 'call': 'capture_value_error(lambda: assign_multidipole_antenna_sectors(p))', 'gold_call': 'capture_value_error(lambda: _oracle_assign_multidipole_antenna_sectors(p))'}
    ]
