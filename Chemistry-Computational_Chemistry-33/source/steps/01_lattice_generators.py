"""
The setup is a lattice model with N sites. Each configuration is a binary occupation vector: a site is 1 if occupied and 0 if empty. Configuration i is encoded as the integer whose bit s is 1 when site s is occupied, so there are m = 2^N configurations and configuration i and the integer i are the same object. The probability vector follows the master equation dp/dt = W p in the column convention: W[j, i] is the rate from configuration i to configuration j for j != i, and W[i, i] is the negative sum of the outgoing rates, so every column of W sums to zero.

Lattice structure

- The surface is channelled, as on a missing-row reconstructed fcc(110) face.

- Sites form an nrow x ncol lattice with periodic boundaries in both directions, holding at most 15 sites.

- Each site is indexed as s = r * ncol + c.

- A bond joining two sites of the same row runs along a channel.

- A bond joining two sites of the same column runs across channels, over the missing row.

- Periodicity needs care on small lattices: wrapping can send a site to itself, which is not a bond at all, and it can produce the same unordered pair twice, which is still one bond. A lattice one row deep therefore has no cross-channel bonds, and a lattice one column wide has no in-channel bonds.

Energy

- P(i) is the number of doubly occupied bonds of either orientation in configuration i.

- Configuration i carries the lateral-interaction energy J * P(i).

- Both orientations enter P(i), so every rate that feels the energy feels bonds of both kinds.

Three distinctions, one per role

1. Interaction: both orientations enter P(i).

2. Reaction: both orientations react and both empty their two sites at once, but a pair across channels is the further apart of the two and carries its own rate constant. A reaction fires at its bare rate constant; the interaction factor acts on hops only.

3. Diffusion: only motion along a channel is fast; crossing between channels is slow and stays in the slow generator.

Hopping rules

- Every hop carries the symmetric-barrier factor exp(-(J / 2) * (P(j) - P(i))) for the move i -> j: the barrier is raised by half of whatever energy the hop must pay and lowered by half of whatever it releases.

- Along a channel the hop is also driven. Writing the two in-channel neighbours of a site as its downstream neighbour, the one at the next column index, and its upstream neighbour, the one at the previous column index, the downstream hop carries the prefactor kappa * (1 + drive_bias) and the upstream hop kappa * (1 - drive_bias). The drive multiplies the interaction factor; it does not replace it.

- A channel one site long has no in-channel neighbour at all.

- A channel two sites long is the case in which the downstream and the upstream neighbour are the same site: the two directed moves are then the same move and their prefactors add, so such a channel carries 2 * kappa times the interaction factor and no drive survives, as it cannot on a ring with no sense of direction.

- Cross-channel hops feel the same interaction factor with their own prefactor kappa_cross, and the drive along the channels does not act across them, so both of their directions carry the same prefactor.

- The two-site ring is treated differently across channels than it is along one. On a lattice two rows deep the neighbour above and the neighbour below a site are the same site, and that cross-channel hop is carried once at kappa_cross, not twice: the cross-channel move is taken from the deduplicated bond set rather than from a pair of directed moves. Only the in-channel prefactors add on a two-site ring.

Adsorption and desorption

- Both are environment independent: adsorption fires on every empty site at k_ads and desorption on every occupied site at k_des.

Generator split

- The generator is split as W = F / eps + S.

- F holds the driven in-channel hops only, built from the normalised prefactor kappa.

- S holds adsorption, desorption, both reactions and the cross-channel hops.

- The physical in-channel prefactor is kappa / eps, and the small parameter enters only later.

Size

- A lattice of 15 sites has m = 2^15 = 32768 configurations. A dense m x m array of float64 then takes 8.6 GB, while a column of F or S holds only a few dozen nonzero rates, because a configuration changes by one elementary event at a time.

- F and S are therefore assembled and returned as scipy.sparse matrices in CSR format, and no dense m x m array is formed at any point, whatever the lattice size.

Observable

- What an experiment on such a surface records is not a configuration but the rate at which product leaves it.

- For each configuration, the total reaction propensity is the rate at which a reaction event of either orientation fires in it.

- It is a per-configuration rate, not a probability, and it is the observable the rest of the pipeline averages.

Returns

tuple (F, S, event_rate):

- F: fast generator of driven in-channel hopping, a scipy.sparse CSR matrix of shape (2 ** (nrow * ncol), 2 ** (nrow * ncol)), column convention with zero column sums.

- S: slow generator holding adsorption, desorption, both reactions and the cross-channel hops, same format, shape and convention.

- event_rate: np.ndarray of shape (2 ** (nrow * ncol),), the total rate at which a reaction event of either orientation fires in each configuration.

The setup is a lattice model with N sites. Each configuration is a binary occupation vector: a site is 1 if occupied and 0 if empty. Configuration i is encoded as the integer whose bit s is 1 when site s is occupied, so there are m = 2^N configurations and configuration i and the integer i are the same object. The probability vector follows the master equation dp/dt = W p in the column convention: W[j, i] is the rate from configuration i to configuration j for j != i, and W[i, i] is the negative sum of the outgoing rates, so every column of W sums to zero.

Returns
-------
tuple `(F, S, event_rate)`: fast and slow sparse generators in column convention, plus the total reaction propensity for every lattice configuration.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def lattice_generators(nrow: int, ncol: int, k_ads: float, k_des: float, k_rxn: float, k_rxn_cross: float, kappa: float, drive_bias: float, kappa_cross: float, interaction: float) -> tuple: """Build the fast and slow generators of a driven channelled lattice. Raises ValueError if dimensions are invalid, nrow*ncol exceeds 15, a rate or hopping prefactor is negative or non-finite, drive_bias is non-finite or outside [-1, 1], or interaction is non-finite."""; return F, S, event_rate

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
import scipy.sparse


def _oracle_lattice_generators(nrow: int, ncol: int, k_ads: float, k_des: float,
                               k_rxn: float, k_rxn_cross: float, kappa: float,
                               drive_bias: float, kappa_cross: float,
                               interaction: float) -> tuple:
    import scipy.sparse as sp
    for name, v in (("nrow", nrow), ("ncol", ncol)):
        if isinstance(v, bool) or not isinstance(v, (int, np.integer)):
            raise ValueError(name + " must be an integer")
        if v < 1:
            raise ValueError(name + " must be >= 1")
    nrow, ncol = int(nrow), int(ncol)
    nsites = nrow * ncol
    if nsites > 15:
        raise ValueError("nrow * ncol must not exceed 15")
    rates = [float(k_ads), float(k_des), float(k_rxn), float(k_rxn_cross),
             float(kappa), float(kappa_cross)]
    if any((not np.isfinite(k)) or k < 0.0 for k in rates):
        raise ValueError("rate constants must be finite and non-negative")
    k_ads, k_des, k_rxn, k_rxn_cross, kappa, kappa_cross = rates
    drive_bias = float(drive_bias)
    if not np.isfinite(drive_bias) or abs(drive_bias) > 1.0:
        raise ValueError("drive_bias must be finite and lie in [-1, 1]")
    interaction = float(interaction)
    if not np.isfinite(interaction):
        raise ValueError("interaction must be finite")
    along, across = set(), set()
    for r in range(nrow):
        for c in range(ncol):
            s = r * ncol + c
            n = r * ncol + (c + 1) % ncol
            if n != s:
                along.add((min(s, n), max(s, n)))
            n = ((r + 1) % nrow) * ncol + c
            if n != s:
                across.add((min(s, n), max(s, n)))
    along, across = sorted(along), sorted(across)
    m = 1 << nsites
    conf = np.arange(m, dtype=np.int64)
    occ = ((conf[:, None] >> np.arange(nsites, dtype=np.int64)) & 1).astype(bool)
    pairs_along = np.zeros(m)
    for a, b in along:
        pairs_along += occ[:, a] & occ[:, b]
    pairs_across = np.zeros(m)
    for a, b in across:
        pairs_across += occ[:, a] & occ[:, b]
    pair_total = pairs_along + pairs_across
    fast, slow = ([], [], []), ([], [], [])

    def _add(store, mask, target, rate):
        src = np.flatnonzero(mask)
        store[0].append(target[src])
        store[1].append(src)
        store[2].append(np.broadcast_to(rate, (m,))[src])

    for r in range(nrow):
        for c in range(ncol):
            s = r * ncol + c
            targets = {}
            down = r * ncol + (c + 1) % ncol
            up = r * ncol + (c - 1) % ncol
            if down != s:
                targets[down] = targets.get(down, 0.0) + kappa * (1.0 + drive_bias)
            if up != s:
                targets[up] = targets.get(up, 0.0) + kappa * (1.0 - drive_bias)
            for t, prefactor in targets.items():
                j = conf ^ (1 << s) ^ (1 << t)
                _add(fast, occ[:, s] & ~occ[:, t], j,
                    prefactor * np.exp(-0.5 * interaction * (pair_total[j] - pair_total)))
    for s in range(nsites):
        _add(slow, ~occ[:, s], conf | (1 << s), k_ads)
        _add(slow, occ[:, s], conf & ~(1 << s), k_des)
    for group, k_react in ((along, k_rxn), (across, k_rxn_cross)):
        if k_react > 0.0:
            for a, b in group:
                _add(slow, occ[:, a] & occ[:, b], conf & ~(1 << a) & ~(1 << b), k_react)
    for a, b in across:
        j = conf ^ (1 << a) ^ (1 << b)
        _add(slow, occ[:, a] != occ[:, b], j,
            kappa_cross * np.exp(-0.5 * interaction * (pair_total[j] - pair_total)))

    def _assemble(store):
        if store[0]:
            rows, cols, vals = (np.concatenate(x) for x in store)
        else:
            rows = cols = np.zeros(0, dtype=np.int64)
            vals = np.zeros(0)
        M = sp.csr_matrix((vals, (rows, cols)), shape=(m, m))
        out = np.asarray(M.sum(axis=0)).ravel()
        return (M - sp.diags(out, format="csr")).tocsr()

    F = _assemble(fast)
    S = _assemble(slow)
    event_rate = k_rxn * pairs_along + k_rxn_cross * pairs_across
    return F, S, event_rate

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    setup_common = """import numpy as np
import scipy.sparse
def _fp(res):
    F, S, R = res
    R = np.asarray(R, dtype=float).ravel()
    m = R.size
    i = np.arange(m, dtype=float)
    X = np.cos(np.outer(i, [0.37, 1.13]) + 0.2)
    w = np.sin(0.71 * i + 0.3)
    out = 1e-3 * m + 5.0 * float(R @ np.cos(0.53 * i))
    for c, M in ((1.0, F), (2.0, S)):
        if tuple(M.shape) != (m, m):
            return float("nan")
        out += c * float(w @ np.asarray(M @ X) @ np.array([1.0, 0.5]))
    return out
"""
    def _raises(args):
        return setup_common + """
def run_model():
    try:
        lattice_generators(%s)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_lattice_generators(%s)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""" % (args, args)
    def _case(args):
        return {"setup": setup_common,
                "call": "_fp(lattice_generators(%s))" % args,
                "gold_call": "_fp(_oracle_lattice_generators(%s))" % args}
    return [
        # --- Normal: the task's driven 3 x 5 channelled lattice, 32768
        # configurations ---
        _case("3, 5, 8.5, 4.0, 10.0, 4.0, 1.0, 0.5, 0.4, 1.5"),
        # --- Normal: five short channels at full size, a strong reversed drive
        # and an attractive interaction ---
        _case("5, 3, 2.0, 3.0, 5.0, 1.5, 1.2, -0.8, 0.3, -0.6"),
        # --- Normal: a 3 x 3 lattice with the same constants as the task ---
        _case("3, 3, 8.5, 4.0, 10.0, 4.0, 1.0, 0.5, 0.4, 1.5"),
        # --- Normal: attractive interaction, strong drive, unequal channels,
        # and a cross-channel reaction faster than the in-channel one ---
        _case("2, 4, 3.0, 1.5, 2.0, 6.0, 1.3, 0.9, 0.25, -0.8"),
        # --- Normal: the drive reversed ---
        _case("3, 3, 8.5, 4.0, 10.0, 4.0, 1.0, -0.5, 0.4, 1.5"),
        # --- Boundary: an undriven channel, where the two senses of motion
        # carry the same prefactor ---
        _case("3, 3, 8.5, 4.0, 10.0, 4.0, 1.0, 0.0, 0.4, 1.5"),
        # --- Boundary: a fully driven single channel fifteen sites long, where
        # the upstream hop is shut off and no bond crosses channels ---
        _case("1, 15, 2.0, 1.0, 3.0, 5.0, 1.0, 1.0, 0.7, 1.2"),
        # --- Boundary: two-site channels, where the downstream and upstream
        # neighbour coincide and the drive cannot survive ---
        _case("3, 2, 8.5, 4.0, 10.0, 4.0, 1.0, 0.7, 0.4, 1.5"),
        # --- Boundary: a single column, so no hop is an in-channel hop and
        # every bond crosses channels ---
        _case("4, 1, 2.0, 1.0, 3.0, 5.0, 1.0, 0.5, 0.7, 1.2"),
        # --- Edge: no interaction, so every hop keeps its bare prefactor ---
        _case("3, 3, 8.5, 4.0, 10.0, 4.0, 1.0, 0.5, 0.4, 0.0"),
        # --- Edge: an inert surface, where no bond of either orientation
        # reacts and the propensity vanishes everywhere ---
        _case("3, 3, 8.5, 4.0, 0.0, 0.0, 1.0, 0.5, 0.4, 1.5"),
        # --- Edge: a 1 x 1 lattice, which has no bonds of either orientation ---
        _case("1, 1, 2.0, 1.0, 3.0, 5.0, 1.0, 0.5, 0.7, 1.5"),
        # --- Invalid: a bias outside the physical range must raise ---
        {"setup": _raises("3, 3, 8.5, 4.0, 10.0, 4.0, 1.0, 1.4, 0.4, 1.5"),
         "call": "run_model()", "gold_call": "run_gold()"},
        # --- Invalid: a negative cross-channel reaction rate must raise ---
        {"setup": _raises("3, 3, 8.5, 4.0, 10.0, -4.0, 1.0, 0.5, 0.4, 1.5"),
         "call": "run_model()", "gold_call": "run_gold()"},
        # --- Invalid: a lattice larger than 15 sites must raise ---
        {"setup": _raises("4, 4, 8.5, 4.0, 10.0, 4.0, 1.0, 0.5, 0.4, 1.5"),
         "call": "run_model()", "gold_call": "run_gold()"},
    ]
