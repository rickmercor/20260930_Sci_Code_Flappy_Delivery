"""
Transition intensities of product states corrected to first order in the off-diagonal pure X-H anharmonic coupling, which borrows amplitude from other local-bond configurations that share the same bath quantum numbers.

The off-diagonal pure X-H cubic and quartic terms mix product states that differ only in their local bond quantum numbers, while the bath is a spectator. In the doorway treatment this mixing corrects the intensity of every non-resonant product state by first-order perturbation theory: the amplitude of a state picks up the zero-order amplitudes of all other local configurations with the same bath quantum numbers, weighted by the coupling matrix element and divided by the zero-order energy difference, and the intensity is then kept to first order in the coupling. Pairs of states closer in energy than a resonance cutoff are left out of the correction, and the zero-order energies are not changed.

The zero-order amplitude of a state is M_XH + M_B from zero_order_amplitudes. The coupling is the sum of the pure X-H rows (class 0) of force_terms, evaluated with the exact Morse matrix elements of morse_moment_matrices; every local configuration with levels 0 to n_max enters the sum, whether or not it lies inside the energy window. Zero-order energies are the Morse excitation energies of the bonds (the bath energy is the same on both sides of every pair). All vibrational coordinates are dimensionless. For a harmonic mode of frequency omega (cm^-1) the Hamiltonian is (omega / 2)(-d^2/dq^2 + q^2), and a local X-H bond of harmonic frequency omega and anharmonicity x has the Hamiltonian omega [ -(1/2) d^2/dxi^2 + (1 / (4x)) (1 - exp(-sqrt(2x) xi))^2 ] in its own dimensionless coordinate xi.

Returns
-------
numpy.ndarray of shape (K,): first-order corrected intensities in cm^-2
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import itertools

import numpy as np


def first_order_intensities(states: np.ndarray, force_terms: np.ndarray, C: np.ndarray, p_xh: np.ndarray, p_bath: np.ndarray, omega_loc: np.ndarray, x_loc: np.ndarray, delta_loc: np.ndarray, delta_bath: np.ndarray, n_max: int, resonance_cutoff: float) -> np.ndarray:
    '''First-order corrected intensities of product states.

    Parameters
    ----------
    states : numpy.ndarray
        Product states in the layout returned by window_states: shape (K, N + 1),
        columns 0..N_x-1 the local Morse quantum numbers (bond order of omega_loc),
        columns N_x..N-1 the bath quantum numbers (order of omega_bath), last column
        the zero-order excitation energy in cm^-1 (not used here).
    force_terms : numpy.ndarray
        Local-coordinate anharmonic terms in the layout returned by
        local_force_field: shape (L, 7), rows (r, a1, a2, a3, a4, G, cls) with
        r = 3 or 4, coordinate indices a1 <= ... <= ar (0..N_x-1 local bonds,
        N_x..N-1 bath modes, a4 = -1 when r = 3), G the coefficient in cm^-1 of
        the monomial prod xi_a in the potential, and cls = 0 (pure X-H),
        1 (mixed X-H and bath) or 2 (pure bath).
    C, p_xh, p_bath, x_loc, delta_loc, delta_bath :
        As in zero_order_amplitudes.
    omega_loc : numpy.ndarray
        Harmonic frequencies of the N_x bonds in cm^-1.
    n_max : int
        Highest Morse level kept for every bond, in the states and in the sum.
    resonance_cutoff : float
        Pairs with |E_v - E_w| < resonance_cutoff (cm^-1) are left out of the
        correction; non-negative.

    Returns
    -------
    intensities : numpy.ndarray
        Shape (K,), the first-order intensities in cm^-2 (they may be negative for
        weak states). An empty states array gives shape (0,).

    Raises
    ------
    ValueError
        If the array shapes are inconsistent, resonance_cutoff is negative, or
        n_max lies too close to the dissociation limit of a bond (the condition of
        morse_moment_matrices).
    '''
    return intensities

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import itertools

import numpy as np


def _morse_ladder(omega, x, n):
    n = np.asarray(n, dtype=float)
    return omega * (n - x * (n * n + n))


def _oracle_first_order_intensities(states: np.ndarray, force_terms: np.ndarray, C: np.ndarray, p_xh: np.ndarray, p_bath: np.ndarray, omega_loc: np.ndarray, x_loc: np.ndarray, delta_loc: np.ndarray, delta_bath: np.ndarray, n_max: int, resonance_cutoff: float) -> np.ndarray:
    st = np.array(states, dtype=float)
    om_l = np.array(omega_loc, dtype=float).ravel()
    x_l = np.array(x_loc, dtype=float).ravel()
    n_x = om_l.size
    n_b = np.array(delta_bath, dtype=float).size
    if st.ndim != 2 or st.shape[1] != n_x + n_b + 1:
        raise ValueError("inconsistent array shapes")
    if resonance_cutoff < 0.0:
        raise ValueError("resonance_cutoff must be non-negative")
    if st.shape[0] == 0:
        return np.zeros(0)
    terms = np.array(force_terms, dtype=float).reshape(-1, 7)
    xx = [row for row in terms if int(row[6]) == 0]
    mom = [_oracle_morse_moment_matrices(x_l[a], n_max) for a in range(n_x)]
    size = n_max + 1
    grid = np.array(list(itertools.product(range(size), repeat=n_x)), dtype=float)
    e_grid = np.zeros(grid.shape[0])
    for a in range(n_x):
        e_grid += _morse_ladder(om_l[a], x_l[a], grid[:, a])
    e_grid = e_grid.reshape((size,) * n_x)
    occ = np.rint(st[:, :n_x + n_b]).astype(int)
    out = np.empty(st.shape[0])
    cache = {}
    for i in range(st.shape[0]):
        nl = tuple(occ[i, :n_x])
        nb = tuple(occ[i, n_x:])
        if nb not in cache:
            rows = np.column_stack([grid, np.tile(np.array(nb, dtype=float), (grid.shape[0], 1)), np.zeros(grid.shape[0])])
            amp = _oracle_zero_order_amplitudes(rows, C, p_xh, p_bath, x_loc, delta_loc, delta_bath, n_max)
            cache[nb] = (amp[:, 0] + amp[:, 1]).reshape((size,) * n_x)
        m_grid = cache[nb]
        coupling = np.zeros((size,) * n_x)
        for row in xx:
            r = int(row[0])
            idx = [int(v) for v in row[1:1 + r]]
            factors = []
            for a in range(n_x):
                p = idx.count(a)
                if p == 0:
                    e = np.zeros(size)
                    e[nl[a]] = 1.0
                    factors.append(e)
                else:
                    factors.append(mom[a][p - 1][:, nl[a]])
            block = factors[0]
            for f in factors[1:]:
                block = np.multiply.outer(block, f)
            coupling += row[5] * block
        de = e_grid[nl] - e_grid
        keep = np.abs(de) >= resonance_cutoff
        keep[nl] = False
        dm = float(np.sum(coupling[keep] * m_grid[keep] / de[keep]))
        m0 = m_grid[nl]
        out[i] = m0 * m0 + 2.0 * m0 * dm
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: the two-bond model at 7000 cm^-1 with a 10 cm^-1 resonance cutoff ---
        {
            "setup": """import numpy as np
omega_loc = np.array([3050.0, 3600.0])
x_loc = np.array([0.0205, 0.0220])
delta_loc = np.array([0.15, -0.10])
C = np.array([[0.90, 0.30], [-0.25, 0.95]])
p_xh = np.array([300.0, -180.0])
omega_bath = np.array([1500.0, 800.0])
delta_bath = np.array([0.70, 1.00])
p_bath = np.array([400.0, 150.0])
cubic = np.array([[0, 0, 1, -150.0], [0, 1, 1, 80.0], [1, 1, 1, 90.0], [0, 2, 2, 180.0], [1, 2, 3, 60.0], [0, 3, 3, 40.0],
                  [2, 3, 3, 25.0], [2, 2, 3, -12.0]])
quartic = np.array([[0, 0, 1, 1, 30.0], [0, 0, 2, 2, 20.0], [2, 2, 3, 3, 44.0], [0, 1, 2, 3, 15.0]])
thresholds = np.array([0.5, 0.5, 2.0, 2.0, 2.0, 5.0])
states = np.array([
                  [0.0, 0.0, 4.0, 1.0, 6800.0],
                  [1.0, 0.0, 1.0, 3.0, 6824.95],
                  [0.0, 0.0, 3.0, 3.0, 6900.0],
                  [1.0, 0.0, 0.0, 5.0, 6924.95],
                  [0.0, 0.0, 2.0, 5.0, 7000.0],
                  [0.0, 0.0, 1.0, 7.0, 7100.0],
                  [1.0, 1.0, 0.0, 1.0, 7166.549999999999],
                  [0.0, 0.0, 0.0, 9.0, 7200.0]])
terms = np.array([
                  [3.0, 0.0, 0.0, 1.0, -1.0, -38.97523114736362, 0.0],
                  [3.0, 0.0, 1.0, 1.0, -1.0, 88.61736766137423, 0.0],
                  [3.0, 0.0, 2.0, 2.0, -1.0, 91.93548387096772, 1.0],
                  [3.0, 0.0, 2.0, 3.0, -1.0, 16.129032258064527, 1.0],
                  [3.0, 0.0, 3.0, 3.0, -1.0, 20.430107526881716, 1.0],
                  [3.0, 1.0, 2.0, 2.0, -1.0, -29.032258064516117, 1.0],
                  [3.0, 1.0, 2.0, 3.0, -1.0, 58.06451612903226, 1.0],
                  [3.0, 1.0, 3.0, 3.0, -1.0, -6.451612903225804, 1.0],
                  [3.0, 2.0, 2.0, 3.0, -1.0, -6.0, 2.0],
                  [3.0, 2.0, 3.0, 3.0, -1.0, 12.5, 2.0],
                  [4.0, 0.0, 0.0, 0.0, 1.0, 3.7146481298807026, 0.0],
                  [4.0, 0.0, 0.0, 1.0, 1.0, 4.814003507590332, 0.0],
                  [4.0, 0.0, 0.0, 2.0, 2.0, 5.217366169499362, 1.0],
                  [4.0, 0.0, 0.0, 2.0, 3.0, 4.118973291710026, 1.0],
                  [4.0, 0.0, 1.0, 1.0, 1.0, -4.222968400285427, 0.0],
                  [4.0, 0.0, 1.0, 2.0, 2.0, -3.2951786333680175, 1.0],
                  [4.0, 0.0, 1.0, 2.0, 3.0, 13.527575442247656, 1.0],
                  [4.0, 1.0, 1.0, 2.0, 3.0, -4.682622268470342, 1.0],
                  [4.0, 2.0, 2.0, 3.0, 3.0, 11.0, 2.0]])
""",
            "call": "first_order_intensities(states.copy(), terms.copy(), C.copy(), p_xh.copy(), p_bath.copy(), omega_loc.copy(), x_loc.copy(), delta_loc.copy(), delta_bath.copy(), 8, 10.0)",
            "gold_call": "_oracle_first_order_intensities(states.copy(), terms.copy(), C.copy(), p_xh.copy(), p_bath.copy(), omega_loc.copy(), x_loc.copy(), delta_loc.copy(), delta_bath.copy(), 8, 10.0)",
            "tol": 1e-07,
        },
        # --- Normal: benchmark window states at 9000 cm^-1 ---
        {
            "setup": """import numpy as np
omega_loc = np.array([3050.0, 3420.0, 3660.0])
x_loc = np.array([0.0205, 0.0215, 0.0225])
delta_loc = np.array([0.12, -0.08, 0.05])
C = np.array([[0.93, 0.25, 0.12], [-0.21, 0.95, 0.20], [0.10, -0.24, 0.97]])
p_xh = np.array([20.0, -12.0, 8.0])
omega_bath = np.array([1605.0, 1460.0, 1180.0, 760.0])
delta_bath = np.array([0.62, 0.35, 0.48, 0.85])
p_bath = np.array([45.0, 20.0, -15.0, 10.0])
cubic = np.array([[0, 0, 1, -180.0], [0, 1, 2, 95.0], [1, 1, 2, -120.0], [0, 2, 2, 60.0], [0, 4, 4, 195.0], [1, 4, 4, -140.0],
                  [2, 3, 4, 105.0], [0, 3, 5, 68.0], [1, 5, 6, 45.0], [2, 6, 6, -38.0], [0, 5, 5, 60.0], [3, 4, 6, 35.0],
                  [4, 5, 6, 18.0], [5, 6, 6, -22.0], [3, 6, 6, 5.5], [4, 6, 6, 5.5], [3, 4, 5, 2.5], [5, 5, 6, 5.0]])
quartic = np.array([[0, 0, 1, 1, 40.0], [0, 1, 2, 2, -25.0], [0, 0, 4, 4, 27.0], [1, 1, 3, 3, -18.0], [0, 1, 4, 6, 14.0],
                    [2, 2, 5, 5, 10.0], [3, 3, 4, 4, 36.0], [4, 4, 4, 4, 60.0], [5, 5, 6, 6, 48.0], [3, 4, 5, 6, 14.0], [4, 4, 4, 6, 66.0]])
thresholds = np.array([1.0, 1.0, 3.0, 3.0, 3.0, 10.0])
states = np.array([
                  [0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 7.0, 8860.0],
                  [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 8.0, 8865.0],
                  [0.0, 0.0, 0.0, 0.0, 1.0, 5.0, 2.0, 8880.0],
                  [0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 4.0, 8880.0],
                  [1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 4.0, 8884.95],
                  [0.0, 0.0, 0.0, 1.0, 1.0, 3.0, 3.0, 8885.0],
                  [0.0, 0.0, 0.0, 2.0, 1.0, 1.0, 4.0, 8890.0],
                  [0.0, 1.0, 0.0, 0.0, 2.0, 1.0, 2.0, 8892.94],
                  [0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 4.0, 8895.3],
                  [1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 8897.89],
                  [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 5.0, 8900.3],
                  [0.0, 0.0, 0.0, 1.0, 5.0, 0.0, 0.0, 8905.0],
                  [1.0, 0.0, 0.0, 1.0, 3.0, 0.0, 0.0, 8909.95],
                  [0.0, 0.0, 0.0, 3.0, 2.0, 1.0, 0.0, 8915.0],
                  [1.0, 0.0, 0.0, 3.0, 0.0, 1.0, 0.0, 8919.95],
                  [0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 0.0, 8920.3],
                  [0.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 8925.3],
                  [2.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 8934.85],
                  [0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 4.0, 8940.0],
                  [0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 6.0, 8940.0],
                  [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 6.0, 8944.95],
                  [0.0, 0.0, 0.0, 1.0, 0.0, 3.0, 5.0, 8945.0],
                  [0.0, 0.0, 0.0, 2.0, 0.0, 1.0, 6.0, 8950.0],
                  [0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 4.0, 8952.94],
                  [0.0, 0.0, 0.0, 0.0, 4.0, 2.0, 1.0, 8960.0],
                  [1.0, 0.0, 0.0, 0.0, 2.0, 2.0, 1.0, 8964.95],
                  [0.0, 0.0, 0.0, 1.0, 1.0, 5.0, 0.0, 8965.0],
                  [0.0, 0.0, 0.0, 1.0, 4.0, 0.0, 2.0, 8965.0],
                  [1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 2.0, 8969.95],
                  [0.0, 0.0, 0.0, 2.0, 1.0, 3.0, 1.0, 8970.0],
                  [0.0, 0.0, 0.0, 3.0, 1.0, 1.0, 2.0, 8975.0],
                  [0.0, 0.0, 1.0, 0.0, 0.0, 4.0, 1.0, 8975.3],
                  [0.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 8977.94],
                  [0.0, 0.0, 1.0, 1.0, 0.0, 2.0, 2.0, 8980.3],
                  [1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 8982.89],
                  [0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 3.0, 8985.3],
                  [0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 8988.24],
                  [2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 8997.79],
                  [0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 8.0, 9000.0],
                  [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 8.0, 9004.95],
                  [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 6.0, 9012.94],
                  [0.0, 0.0, 0.0, 0.0, 0.0, 7.0, 1.0, 9020.0],
                  [0.0, 0.0, 0.0, 0.0, 3.0, 2.0, 3.0, 9020.0],
                  [1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 9024.95],
                  [0.0, 0.0, 0.0, 1.0, 0.0, 5.0, 2.0, 9025.0],
                  [0.0, 0.0, 0.0, 1.0, 3.0, 0.0, 4.0, 9025.0],
                  [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 4.0, 9029.95],
                  [0.0, 0.0, 0.0, 2.0, 0.0, 3.0, 3.0, 9030.0],
                  [0.0, 1.0, 0.0, 0.0, 1.0, 3.0, 1.0, 9032.94],
                  [0.0, 0.0, 0.0, 3.0, 0.0, 1.0, 4.0, 9035.0],
                  [0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 2.0, 9037.94],
                  [0.0, 2.0, 0.0, 0.0, 1.0, 1.0, 0.0, 9038.82],
                  [0.0, 0.0, 2.0, 0.0, 1.0, 0.0, 1.0, 9045.9],
                  [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 3.0, 9048.24],
                  [0.0, 0.0, 0.0, 2.0, 4.0, 0.0, 0.0, 9050.0],
                  [1.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 9054.95],
                  [0.0, 0.0, 1.0, 0.0, 3.0, 1.0, 0.0, 9055.3],
                  [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 10.0, 9060.0],
                  [0.0, 0.0, 0.0, 4.0, 1.0, 1.0, 0.0, 9060.0],
                  [1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 9060.25],
                  [0.0, 0.0, 1.0, 2.0, 0.0, 2.0, 0.0, 9065.3],
                  [0.0, 0.0, 1.0, 3.0, 0.0, 0.0, 1.0, 9070.3],
                  [0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 5.0, 9080.0],
                  [1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 5.0, 9084.95],
                  [0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 6.0, 9085.0],
                  [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 6.0, 9089.95],
                  [0.0, 1.0, 0.0, 0.0, 0.0, 3.0, 3.0, 9092.94],
                  [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 4.0, 9097.94],
                  [0.0, 2.0, 0.0, 0.0, 0.0, 1.0, 2.0, 9098.82],
                  [0.0, 0.0, 0.0, 0.0, 3.0, 4.0, 0.0, 9100.0],
                  [1.0, 0.0, 0.0, 0.0, 1.0, 4.0, 0.0, 9104.95],
                  [0.0, 0.0, 0.0, 1.0, 3.0, 2.0, 1.0, 9105.0],
                  [0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0, 9105.9],
                  [1.0, 0.0, 0.0, 1.0, 1.0, 2.0, 1.0, 9109.95],
                  [0.0, 0.0, 0.0, 2.0, 0.0, 5.0, 0.0, 9110.0],
                  [0.0, 0.0, 0.0, 2.0, 3.0, 0.0, 2.0, 9110.0],
                  [0.0, 1.0, 0.0, 0.0, 4.0, 0.0, 0.0, 9112.94],
                  [1.0, 0.0, 0.0, 2.0, 1.0, 0.0, 2.0, 9114.95],
                  [0.0, 0.0, 0.0, 3.0, 0.0, 3.0, 1.0, 9115.0],
                  [0.0, 0.0, 1.0, 0.0, 2.0, 1.0, 2.0, 9115.3],
                  [1.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 9117.89],
                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 12.0, 9120.0],
                  [0.0, 0.0, 0.0, 4.0, 0.0, 1.0, 2.0, 9120.0],
                  [1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 2.0, 9120.25],
                  [0.0, 1.0, 0.0, 2.0, 1.0, 1.0, 0.0, 9122.94],
                  [2.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 9124.85],
                  [0.0, 1.0, 1.0, 0.0, 0.0, 2.0, 0.0, 9128.24],
                  [0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 9133.24],
                  [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 7.0, 9140.0],
                  [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 8.0, 9145.0]])
terms = np.array([
                  [3.0, 0.0, 0.0, 1.0, -1.0, -73.35350784405455, 0.0],
                  [3.0, 0.0, 0.0, 2.0, -1.0, 40.63416190779917, 0.0],
                  [3.0, 0.0, 1.0, 1.0, -1.0, 68.06009745978568, 0.0],
                  [3.0, 0.0, 1.0, 2.0, -1.0, 66.81431142773758, 0.0],
                  [3.0, 0.0, 2.0, 2.0, -1.0, 10.527165798048404, 0.0],
                  [3.0, 0.0, 3.0, 4.0, -1.0, -4.918043116629973, 1.0],
                  [3.0, 0.0, 3.0, 5.0, -1.0, 69.2348730529464, 1.0],
                  [3.0, 0.0, 4.0, 4.0, -1.0, 82.82565363870077, 1.0],
                  [3.0, 0.0, 5.0, 5.0, -1.0, 30.54479693512341, 1.0],
                  [3.0, 0.0, 5.0, 6.0, -1.0, 10.57174482886092, 1.0],
                  [3.0, 1.0, 1.0, 2.0, -1.0, -76.03132880064283, 0.0],
                  [3.0, 1.0, 2.0, 2.0, -1.0, 15.182695521824739, 0.0],
                  [3.0, 1.0, 3.0, 4.0, -1.0, 27.36902021407088, 1.0],
                  [3.0, 1.0, 3.0, 5.0, -1.0, -19.37433837984978, 1.0],
                  [3.0, 1.0, 4.0, 4.0, -1.0, -93.2136150925008, 1.0],
                  [3.0, 1.0, 5.0, 5.0, -1.0, -8.547502226404315, 1.0],
                  [3.0, 1.0, 5.0, 6.0, -1.0, 42.06486397929864, 1.0],
                  [3.0, 1.0, 6.0, 6.0, -1.0, -4.9524893720699685, 1.0],
                  [3.0, 2.0, 3.0, 4.0, -1.0, 103.21274343420767, 1.0],
                  [3.0, 2.0, 3.0, 5.0, -1.0, -4.570429990086205, 1.0],
                  [3.0, 2.0, 4.0, 4.0, -1.0, 8.972829465830984, 1.0],
                  [3.0, 2.0, 5.0, 6.0, -1.0, -9.981012551879418, 1.0],
                  [3.0, 2.0, 6.0, 6.0, -1.0, -18.676591669047102, 1.0],
                  [3.0, 3.0, 4.0, 6.0, -1.0, 35.0, 2.0],
                  [3.0, 4.0, 5.0, 6.0, -1.0, 18.0, 2.0],
                  [3.0, 5.0, 6.0, 6.0, -1.0, -11.0, 2.0],
                  [4.0, 0.0, 0.0, 0.0, 1.0, 4.281591526960507, 0.0],
                  [4.0, 0.0, 0.0, 1.0, 1.0, 6.629102199905845, 0.0],
                  [4.0, 0.0, 0.0, 1.0, 2.0, -4.861463289536317, 0.0],
                  [4.0, 0.0, 0.0, 2.0, 2.0, -2.512413347407801, 0.0],
                  [4.0, 0.0, 0.0, 4.0, 4.0, 6.997384648559434, 1.0],
                  [4.0, 0.0, 0.0, 4.0, 6.0, 3.3487149530851625, 1.0],
                  [4.0, 0.0, 1.0, 1.0, 1.0, -5.545854523846756, 0.0],
                  [4.0, 0.0, 1.0, 1.0, 2.0, -4.475441272939001, 0.0],
                  [4.0, 0.0, 1.0, 2.0, 2.0, -8.872173417347467, 0.0],
                  [4.0, 0.0, 1.0, 4.0, 4.0, -3.916225797120526, 1.0],
                  [4.0, 0.0, 1.0, 4.0, 6.0, 12.387414831447678, 1.0],
                  [4.0, 0.0, 2.0, 2.0, 2.0, 2.8634043270520615, 0.0],
                  [4.0, 0.0, 2.0, 4.0, 6.0, -3.382654180443272, 1.0],
                  [4.0, 1.0, 1.0, 1.0, 2.0, 1.7037371136268105, 0.0],
                  [4.0, 1.0, 1.0, 2.0, 2.0, 3.12237797172809, 0.0],
                  [4.0, 1.0, 1.0, 3.0, 3.0, -3.932117292437548, 1.0],
                  [4.0, 1.0, 1.0, 4.0, 6.0, -3.7286616735039244, 1.0],
                  [4.0, 3.0, 4.0, 5.0, 6.0, 14.0, 2.0],
                  [4.0, 4.0, 4.0, 4.0, 6.0, 11.0, 2.0],
                  [4.0, 5.0, 5.0, 6.0, 6.0, 12.0, 2.0]])
""",
            "call": "first_order_intensities(states.copy(), terms.copy(), C.copy(), p_xh.copy(), p_bath.copy(), omega_loc.copy(), x_loc.copy(), delta_loc.copy(), delta_bath.copy(), 10, 10.0)",
            "gold_call": "_oracle_first_order_intensities(states.copy(), terms.copy(), C.copy(), p_xh.copy(), p_bath.copy(), omega_loc.copy(), x_loc.copy(), delta_loc.copy(), delta_bath.copy(), 10, 10.0)",
            "tol": 1e-07,
        },
        # --- Boundary: two nearly degenerate bonds and a zero resonance cutoff, so the 5.8 cm^-1 pair enters the correction ---
        {
            "setup": """import numpy as np
omega_loc = np.array([3050.0, 3056.0])
x_loc = np.array([0.02, 0.02])
delta_loc = np.array([0.15, -0.10])
C = np.array([[0.90, 0.30], [-0.25, 0.95]])
p_xh = np.array([300.0, -180.0])
omega_bath = np.array([1500.0, 800.0])
delta_bath = np.array([0.70, 1.00])
p_bath = np.array([400.0, 150.0])
cubic = np.array([[0, 0, 1, -150.0], [0, 1, 1, 80.0], [1, 1, 1, 90.0], [0, 2, 2, 180.0], [1, 2, 3, 60.0], [0, 3, 3, 40.0],
                  [2, 3, 3, 25.0], [2, 2, 3, -12.0]])
quartic = np.array([[0, 0, 1, 1, 30.0], [0, 0, 2, 2, 20.0], [2, 2, 3, 3, 44.0], [0, 1, 2, 3, 15.0]])
thresholds = np.array([0.5, 0.5, 2.0, 2.0, 2.0, 5.0])
states = np.array([
                  [2.0, 0.0, 0.0, 0.0, 5734.0],
                  [0.0, 2.0, 0.0, 0.0, 5745.28],
                  [1.0, 1.0, 0.0, 0.0, 5861.76],
                  [1.0, 0.0, 2.0, 0.0, 5928.0],
                  [0.0, 1.0, 2.0, 0.0, 5933.76],
                  [0.0, 0.0, 4.0, 0.0, 6000.0],
                  [1.0, 0.0, 1.0, 2.0, 6028.0],
                  [0.0, 1.0, 1.0, 2.0, 6033.76],
                  [0.0, 0.0, 3.0, 2.0, 6100.0]])
terms = np.array([
                  [3.0, 0.0, 0.0, 1.0, -1.0, -38.97523114736362, 0.0],
                  [3.0, 0.0, 1.0, 1.0, -1.0, 88.61736766137423, 0.0],
                  [3.0, 0.0, 2.0, 2.0, -1.0, 91.93548387096772, 1.0],
                  [3.0, 0.0, 2.0, 3.0, -1.0, 16.129032258064527, 1.0],
                  [3.0, 0.0, 3.0, 3.0, -1.0, 20.430107526881716, 1.0],
                  [3.0, 1.0, 2.0, 2.0, -1.0, -29.032258064516117, 1.0],
                  [3.0, 1.0, 2.0, 3.0, -1.0, 58.06451612903226, 1.0],
                  [3.0, 1.0, 3.0, 3.0, -1.0, -6.451612903225804, 1.0],
                  [3.0, 2.0, 2.0, 3.0, -1.0, -6.0, 2.0],
                  [3.0, 2.0, 3.0, 3.0, -1.0, 12.5, 2.0],
                  [4.0, 0.0, 0.0, 0.0, 1.0, 3.7146481298807026, 0.0],
                  [4.0, 0.0, 0.0, 1.0, 1.0, 4.814003507590332, 0.0],
                  [4.0, 0.0, 0.0, 2.0, 2.0, 5.217366169499362, 1.0],
                  [4.0, 0.0, 0.0, 2.0, 3.0, 4.118973291710026, 1.0],
                  [4.0, 0.0, 1.0, 1.0, 1.0, -4.222968400285427, 0.0],
                  [4.0, 0.0, 1.0, 2.0, 2.0, -3.2951786333680175, 1.0],
                  [4.0, 0.0, 1.0, 2.0, 3.0, 13.527575442247656, 1.0],
                  [4.0, 1.0, 1.0, 2.0, 3.0, -4.682622268470342, 1.0],
                  [4.0, 2.0, 2.0, 3.0, 3.0, 11.0, 2.0]])
""",
            "call": "first_order_intensities(states.copy(), terms.copy(), C.copy(), p_xh.copy(), p_bath.copy(), omega_loc.copy(), x_loc.copy(), delta_loc.copy(), delta_bath.copy(), 8, 0.0)",
            "gold_call": "_oracle_first_order_intensities(states.copy(), terms.copy(), C.copy(), p_xh.copy(), p_bath.copy(), omega_loc.copy(), x_loc.copy(), delta_loc.copy(), delta_bath.copy(), 8, 0.0)",
            "tol": 1e-07,
        },
        # --- Edge: the same bonds with a 10 cm^-1 cutoff, which removes the nearly degenerate pair ---
        {
            "setup": """import numpy as np
omega_loc = np.array([3050.0, 3056.0])
x_loc = np.array([0.02, 0.02])
delta_loc = np.array([0.15, -0.10])
C = np.array([[0.90, 0.30], [-0.25, 0.95]])
p_xh = np.array([300.0, -180.0])
omega_bath = np.array([1500.0, 800.0])
delta_bath = np.array([0.70, 1.00])
p_bath = np.array([400.0, 150.0])
cubic = np.array([[0, 0, 1, -150.0], [0, 1, 1, 80.0], [1, 1, 1, 90.0], [0, 2, 2, 180.0], [1, 2, 3, 60.0], [0, 3, 3, 40.0],
                  [2, 3, 3, 25.0], [2, 2, 3, -12.0]])
quartic = np.array([[0, 0, 1, 1, 30.0], [0, 0, 2, 2, 20.0], [2, 2, 3, 3, 44.0], [0, 1, 2, 3, 15.0]])
thresholds = np.array([0.5, 0.5, 2.0, 2.0, 2.0, 5.0])
states = np.array([
                  [2.0, 0.0, 0.0, 0.0, 5734.0],
                  [0.0, 2.0, 0.0, 0.0, 5745.28],
                  [1.0, 1.0, 0.0, 0.0, 5861.76],
                  [1.0, 0.0, 2.0, 0.0, 5928.0],
                  [0.0, 1.0, 2.0, 0.0, 5933.76],
                  [0.0, 0.0, 4.0, 0.0, 6000.0],
                  [1.0, 0.0, 1.0, 2.0, 6028.0],
                  [0.0, 1.0, 1.0, 2.0, 6033.76],
                  [0.0, 0.0, 3.0, 2.0, 6100.0]])
terms = np.array([
                  [3.0, 0.0, 0.0, 1.0, -1.0, -38.97523114736362, 0.0],
                  [3.0, 0.0, 1.0, 1.0, -1.0, 88.61736766137423, 0.0],
                  [3.0, 0.0, 2.0, 2.0, -1.0, 91.93548387096772, 1.0],
                  [3.0, 0.0, 2.0, 3.0, -1.0, 16.129032258064527, 1.0],
                  [3.0, 0.0, 3.0, 3.0, -1.0, 20.430107526881716, 1.0],
                  [3.0, 1.0, 2.0, 2.0, -1.0, -29.032258064516117, 1.0],
                  [3.0, 1.0, 2.0, 3.0, -1.0, 58.06451612903226, 1.0],
                  [3.0, 1.0, 3.0, 3.0, -1.0, -6.451612903225804, 1.0],
                  [3.0, 2.0, 2.0, 3.0, -1.0, -6.0, 2.0],
                  [3.0, 2.0, 3.0, 3.0, -1.0, 12.5, 2.0],
                  [4.0, 0.0, 0.0, 0.0, 1.0, 3.7146481298807026, 0.0],
                  [4.0, 0.0, 0.0, 1.0, 1.0, 4.814003507590332, 0.0],
                  [4.0, 0.0, 0.0, 2.0, 2.0, 5.217366169499362, 1.0],
                  [4.0, 0.0, 0.0, 2.0, 3.0, 4.118973291710026, 1.0],
                  [4.0, 0.0, 1.0, 1.0, 1.0, -4.222968400285427, 0.0],
                  [4.0, 0.0, 1.0, 2.0, 2.0, -3.2951786333680175, 1.0],
                  [4.0, 0.0, 1.0, 2.0, 3.0, 13.527575442247656, 1.0],
                  [4.0, 1.0, 1.0, 2.0, 3.0, -4.682622268470342, 1.0],
                  [4.0, 2.0, 2.0, 3.0, 3.0, 11.0, 2.0]])
""",
            "call": "first_order_intensities(states.copy(), terms.copy(), C.copy(), p_xh.copy(), p_bath.copy(), omega_loc.copy(), x_loc.copy(), delta_loc.copy(), delta_bath.copy(), 8, 10.0)",
            "gold_call": "_oracle_first_order_intensities(states.copy(), terms.copy(), C.copy(), p_xh.copy(), p_bath.copy(), omega_loc.copy(), x_loc.copy(), delta_loc.copy(), delta_bath.copy(), 8, 10.0)",
            "tol": 1e-07,
        },
        # --- Edge: the same bonds with a 30 cm^-1 cutoff, which also removes the next pair ---
        {
            "setup": """import numpy as np
omega_loc = np.array([3050.0, 3056.0])
x_loc = np.array([0.02, 0.02])
delta_loc = np.array([0.15, -0.10])
C = np.array([[0.90, 0.30], [-0.25, 0.95]])
p_xh = np.array([300.0, -180.0])
omega_bath = np.array([1500.0, 800.0])
delta_bath = np.array([0.70, 1.00])
p_bath = np.array([400.0, 150.0])
cubic = np.array([[0, 0, 1, -150.0], [0, 1, 1, 80.0], [1, 1, 1, 90.0], [0, 2, 2, 180.0], [1, 2, 3, 60.0], [0, 3, 3, 40.0],
                  [2, 3, 3, 25.0], [2, 2, 3, -12.0]])
quartic = np.array([[0, 0, 1, 1, 30.0], [0, 0, 2, 2, 20.0], [2, 2, 3, 3, 44.0], [0, 1, 2, 3, 15.0]])
thresholds = np.array([0.5, 0.5, 2.0, 2.0, 2.0, 5.0])
states = np.array([
                  [2.0, 0.0, 0.0, 0.0, 5734.0],
                  [0.0, 2.0, 0.0, 0.0, 5745.28],
                  [1.0, 1.0, 0.0, 0.0, 5861.76],
                  [1.0, 0.0, 2.0, 0.0, 5928.0],
                  [0.0, 1.0, 2.0, 0.0, 5933.76],
                  [0.0, 0.0, 4.0, 0.0, 6000.0],
                  [1.0, 0.0, 1.0, 2.0, 6028.0],
                  [0.0, 1.0, 1.0, 2.0, 6033.76],
                  [0.0, 0.0, 3.0, 2.0, 6100.0]])
terms = np.array([
                  [3.0, 0.0, 0.0, 1.0, -1.0, -38.97523114736362, 0.0],
                  [3.0, 0.0, 1.0, 1.0, -1.0, 88.61736766137423, 0.0],
                  [3.0, 0.0, 2.0, 2.0, -1.0, 91.93548387096772, 1.0],
                  [3.0, 0.0, 2.0, 3.0, -1.0, 16.129032258064527, 1.0],
                  [3.0, 0.0, 3.0, 3.0, -1.0, 20.430107526881716, 1.0],
                  [3.0, 1.0, 2.0, 2.0, -1.0, -29.032258064516117, 1.0],
                  [3.0, 1.0, 2.0, 3.0, -1.0, 58.06451612903226, 1.0],
                  [3.0, 1.0, 3.0, 3.0, -1.0, -6.451612903225804, 1.0],
                  [3.0, 2.0, 2.0, 3.0, -1.0, -6.0, 2.0],
                  [3.0, 2.0, 3.0, 3.0, -1.0, 12.5, 2.0],
                  [4.0, 0.0, 0.0, 0.0, 1.0, 3.7146481298807026, 0.0],
                  [4.0, 0.0, 0.0, 1.0, 1.0, 4.814003507590332, 0.0],
                  [4.0, 0.0, 0.0, 2.0, 2.0, 5.217366169499362, 1.0],
                  [4.0, 0.0, 0.0, 2.0, 3.0, 4.118973291710026, 1.0],
                  [4.0, 0.0, 1.0, 1.0, 1.0, -4.222968400285427, 0.0],
                  [4.0, 0.0, 1.0, 2.0, 2.0, -3.2951786333680175, 1.0],
                  [4.0, 0.0, 1.0, 2.0, 3.0, 13.527575442247656, 1.0],
                  [4.0, 1.0, 1.0, 2.0, 3.0, -4.682622268470342, 1.0],
                  [4.0, 2.0, 2.0, 3.0, 3.0, 11.0, 2.0]])
""",
            "call": "first_order_intensities(states.copy(), terms.copy(), C.copy(), p_xh.copy(), p_bath.copy(), omega_loc.copy(), x_loc.copy(), delta_loc.copy(), delta_bath.copy(), 8, 30.0)",
            "gold_call": "_oracle_first_order_intensities(states.copy(), terms.copy(), C.copy(), p_xh.copy(), p_bath.copy(), omega_loc.copy(), x_loc.copy(), delta_loc.copy(), delta_bath.copy(), 8, 30.0)",
            "tol": 1e-07,
        },
        # --- Edge: no pure X-H terms, so the intensities are the squared zero-order amplitudes ---
        {
            "setup": """import numpy as np
omega_loc = np.array([3050.0, 3600.0])
x_loc = np.array([0.0205, 0.0220])
delta_loc = np.array([0.15, -0.10])
C = np.array([[0.90, 0.30], [-0.25, 0.95]])
p_xh = np.array([300.0, -180.0])
omega_bath = np.array([1500.0, 800.0])
delta_bath = np.array([0.70, 1.00])
p_bath = np.array([400.0, 150.0])
cubic = np.array([[0, 0, 1, -150.0], [0, 1, 1, 80.0], [1, 1, 1, 90.0], [0, 2, 2, 180.0], [1, 2, 3, 60.0], [0, 3, 3, 40.0],
                  [2, 3, 3, 25.0], [2, 2, 3, -12.0]])
quartic = np.array([[0, 0, 1, 1, 30.0], [0, 0, 2, 2, 20.0], [2, 2, 3, 3, 44.0], [0, 1, 2, 3, 15.0]])
thresholds = np.array([0.5, 0.5, 2.0, 2.0, 2.0, 5.0])
states = np.array([
                  [0.0, 0.0, 4.0, 1.0, 6800.0],
                  [1.0, 0.0, 1.0, 3.0, 6824.95],
                  [0.0, 0.0, 3.0, 3.0, 6900.0],
                  [1.0, 0.0, 0.0, 5.0, 6924.95],
                  [0.0, 0.0, 2.0, 5.0, 7000.0],
                  [0.0, 0.0, 1.0, 7.0, 7100.0],
                  [1.0, 1.0, 0.0, 1.0, 7166.549999999999],
                  [0.0, 0.0, 0.0, 9.0, 7200.0]])
terms = np.array([
                  [3.0, 0.0, 2.0, 2.0, -1.0, 91.93548387096772, 1.0],
                  [3.0, 0.0, 2.0, 3.0, -1.0, 16.129032258064527, 1.0],
                  [3.0, 0.0, 3.0, 3.0, -1.0, 20.430107526881716, 1.0],
                  [3.0, 1.0, 2.0, 2.0, -1.0, -29.032258064516117, 1.0],
                  [3.0, 1.0, 2.0, 3.0, -1.0, 58.06451612903226, 1.0],
                  [3.0, 1.0, 3.0, 3.0, -1.0, -6.451612903225804, 1.0],
                  [3.0, 2.0, 2.0, 3.0, -1.0, -6.0, 2.0],
                  [3.0, 2.0, 3.0, 3.0, -1.0, 12.5, 2.0],
                  [4.0, 0.0, 0.0, 2.0, 2.0, 5.217366169499362, 1.0],
                  [4.0, 0.0, 0.0, 2.0, 3.0, 4.118973291710026, 1.0],
                  [4.0, 0.0, 1.0, 2.0, 2.0, -3.2951786333680175, 1.0],
                  [4.0, 0.0, 1.0, 2.0, 3.0, 13.527575442247656, 1.0],
                  [4.0, 1.0, 1.0, 2.0, 3.0, -4.682622268470342, 1.0],
                  [4.0, 2.0, 2.0, 3.0, 3.0, 11.0, 2.0]])
terms = terms[terms[:, 6] != 0]
""",
            "call": "first_order_intensities(states.copy(), terms.copy(), C.copy(), p_xh.copy(), p_bath.copy(), omega_loc.copy(), x_loc.copy(), delta_loc.copy(), delta_bath.copy(), 8, 10.0)",
            "gold_call": "_oracle_first_order_intensities(states.copy(), terms.copy(), C.copy(), p_xh.copy(), p_bath.copy(), omega_loc.copy(), x_loc.copy(), delta_loc.copy(), delta_bath.copy(), 8, 10.0)",
            "tol": 1e-07,
        },
        # --- Edge: an empty list of states ---
        {
            "setup": """import numpy as np
omega_loc = np.array([3050.0, 3600.0])
x_loc = np.array([0.0205, 0.0220])
delta_loc = np.array([0.15, -0.10])
C = np.array([[0.90, 0.30], [-0.25, 0.95]])
p_xh = np.array([300.0, -180.0])
omega_bath = np.array([1500.0, 800.0])
delta_bath = np.array([0.70, 1.00])
p_bath = np.array([400.0, 150.0])
cubic = np.array([[0, 0, 1, -150.0], [0, 1, 1, 80.0], [1, 1, 1, 90.0], [0, 2, 2, 180.0], [1, 2, 3, 60.0], [0, 3, 3, 40.0],
                  [2, 3, 3, 25.0], [2, 2, 3, -12.0]])
quartic = np.array([[0, 0, 1, 1, 30.0], [0, 0, 2, 2, 20.0], [2, 2, 3, 3, 44.0], [0, 1, 2, 3, 15.0]])
thresholds = np.array([0.5, 0.5, 2.0, 2.0, 2.0, 5.0])
terms = np.array([
                  [3.0, 0.0, 0.0, 1.0, -1.0, -38.97523114736362, 0.0],
                  [3.0, 0.0, 1.0, 1.0, -1.0, 88.61736766137423, 0.0],
                  [3.0, 0.0, 2.0, 2.0, -1.0, 91.93548387096772, 1.0],
                  [3.0, 0.0, 2.0, 3.0, -1.0, 16.129032258064527, 1.0],
                  [3.0, 0.0, 3.0, 3.0, -1.0, 20.430107526881716, 1.0],
                  [3.0, 1.0, 2.0, 2.0, -1.0, -29.032258064516117, 1.0],
                  [3.0, 1.0, 2.0, 3.0, -1.0, 58.06451612903226, 1.0],
                  [3.0, 1.0, 3.0, 3.0, -1.0, -6.451612903225804, 1.0],
                  [3.0, 2.0, 2.0, 3.0, -1.0, -6.0, 2.0],
                  [3.0, 2.0, 3.0, 3.0, -1.0, 12.5, 2.0],
                  [4.0, 0.0, 0.0, 0.0, 1.0, 3.7146481298807026, 0.0],
                  [4.0, 0.0, 0.0, 1.0, 1.0, 4.814003507590332, 0.0],
                  [4.0, 0.0, 0.0, 2.0, 2.0, 5.217366169499362, 1.0],
                  [4.0, 0.0, 0.0, 2.0, 3.0, 4.118973291710026, 1.0],
                  [4.0, 0.0, 1.0, 1.0, 1.0, -4.222968400285427, 0.0],
                  [4.0, 0.0, 1.0, 2.0, 2.0, -3.2951786333680175, 1.0],
                  [4.0, 0.0, 1.0, 2.0, 3.0, 13.527575442247656, 1.0],
                  [4.0, 1.0, 1.0, 2.0, 3.0, -4.682622268470342, 1.0],
                  [4.0, 2.0, 2.0, 3.0, 3.0, 11.0, 2.0]])
states = np.zeros((0, 5))
""",
            "call": "first_order_intensities(states.copy(), terms.copy(), C.copy(), p_xh.copy(), p_bath.copy(), omega_loc.copy(), x_loc.copy(), delta_loc.copy(), delta_bath.copy(), 8, 10.0)",
            "gold_call": "_oracle_first_order_intensities(states.copy(), terms.copy(), C.copy(), p_xh.copy(), p_bath.copy(), omega_loc.copy(), x_loc.copy(), delta_loc.copy(), delta_bath.copy(), 8, 10.0)",
            "tol": 1e-07,
        },
    ]
