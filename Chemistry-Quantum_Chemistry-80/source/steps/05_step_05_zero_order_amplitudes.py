"""
Zero-order internal-conversion amplitudes of product final states, split into the X-H-active channel, where the promoting derivative acts on a local bond coordinate, and the bath-active channel, where it acts on a bath mode.

Without Duschinsky rotation, Herzberg-Teller terms or thermal excitation, the nonadiabatic matrix element between the vibrationless S1 level and a product state of S0 is a sum over promoting coordinates: the electronic coupling coefficient of one coordinate multiplies the derivative integral of that coordinate and the Franck-Condon overlaps of all the others. The doorway treatment evaluates the X-H part in the local bond coordinates. Its local coefficients d_a are chosen so that the bond coordinates reproduce the normal-mode coupling vector of the X-H modes, sum_a C_ai d_a = P_i, taken as the least-squares solution given by the Moore-Penrose inverse when C is not square, and the bath modes keep their own coefficients. Every configuration carries the full set of local and bath quantum numbers in both channels.

The two channel amplitudes are returned separately and in cm^-1. They are real, and their relative sign matters because later steps add them. The integrals are those of promoting_integrals: Morse bonds with their S1 minima displaced by delta_loc and harmonic bath modes displaced by delta_bath. Every vibrational eigenfunction is taken real and positive for large positive values of its coordinate (for the Morse functions that is the dissociative side).

Returns
-------
numpy.ndarray of shape (K, 2): X-H-active and bath-active zero-order amplitudes in cm^-1
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def zero_order_amplitudes(states: np.ndarray, C: np.ndarray, p_xh: np.ndarray, p_bath: np.ndarray, x_loc: np.ndarray, delta_loc: np.ndarray, delta_bath: np.ndarray, n_max: int) -> np.ndarray:
    '''X-H-active and bath-active zero-order amplitudes of product states.

    Parameters
    ----------
    states : numpy.ndarray
        Product states in the layout returned by window_states: shape (K, N + 1),
        columns 0..N_x-1 the local Morse quantum numbers (bond order of omega_loc),
        columns N_x..N-1 the bath quantum numbers (order of omega_bath), last column
        the zero-order excitation energy in cm^-1 (not used here).
    C : numpy.ndarray
        Local-bond transformation, shape (N_x, M_x), xi_a = sum_i C[a, i] q_i.
    p_xh : numpy.ndarray
        Electronic coupling coefficients P_i of the M_x X-H normal modes, cm^-1.
    p_bath : numpy.ndarray
        Coupling coefficients of the N_b bath modes, cm^-1.
    x_loc : numpy.ndarray
        Anharmonicities of the N_x bonds.
    delta_loc : numpy.ndarray
        Positions of the S1 minima of the bonds on their S0 coordinates.
    delta_bath : numpy.ndarray
        Positions of the S1 minima of the bath modes on their S0 coordinates.
    n_max : int
        Highest Morse level kept for every bond.

    Returns
    -------
    amplitudes : numpy.ndarray
        Shape (K, 2); column 0 is the X-H-active amplitude M_XH and column 1 the
        bath-active amplitude M_B of each state, in cm^-1. An empty states array
        (K = 0) gives shape (0, 2).

    Raises
    ------
    ValueError
        If the array shapes are inconsistent, a quantum number is negative, a local
        quantum number exceeds n_max, or n_max lies too close to the dissociation
        limit of a bond (the condition of promoting_integrals).
    '''
    return amplitudes

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_zero_order_amplitudes(states: np.ndarray, C: np.ndarray, p_xh: np.ndarray, p_bath: np.ndarray, x_loc: np.ndarray, delta_loc: np.ndarray, delta_bath: np.ndarray, n_max: int) -> np.ndarray:
    st = np.array(states, dtype=float)
    x_l = np.array(x_loc, dtype=float).ravel()
    d_l = np.array(delta_loc, dtype=float).ravel()
    d_b = np.array(delta_bath, dtype=float).ravel()
    p_b = np.array(p_bath, dtype=float).ravel()
    n_x, n_b = x_l.size, d_b.size
    if st.ndim != 2 or st.shape[1] != n_x + n_b + 1 or d_l.size != n_x or p_b.size != n_b:
        raise ValueError("inconsistent array shapes")
    if st.shape[0] == 0:
        return np.zeros((0, 2))
    d_loc = np.linalg.pinv(np.array(C, dtype=float)).T @ np.array(p_xh, dtype=float).ravel()
    occ = np.rint(st[:, :n_x + n_b]).astype(int)
    if np.any(occ < 0) or np.any(occ[:, :n_x] > n_max):
        raise ValueError("occupations out of range")
    nb_top = int(occ[:, n_x:].max()) if n_b else 0
    g = np.ones((n_x + n_b, st.shape[0]))
    t = np.zeros((n_x + n_b, st.shape[0]))
    for a in range(n_x):
        gt = _oracle_promoting_integrals(x_l[a], d_l[a], n_max)
        g[a] = gt[0][occ[:, a]]
        t[a] = gt[1][occ[:, a]]
    for b in range(n_b):
        gt = _oracle_promoting_integrals(0.0, d_b[b], nb_top)
        g[n_x + b] = gt[0][occ[:, n_x + b]]
        t[n_x + b] = gt[1][occ[:, n_x + b]]
    g_loc = np.prod(g[:n_x], axis=0)
    g_bath = np.prod(g[n_x:], axis=0)
    m_xh = np.zeros(st.shape[0])
    for a in range(n_x):
        others = np.prod(np.delete(g[:n_x], a, axis=0), axis=0)
        m_xh += d_loc[a] * t[a] * others
    m_b = np.zeros(st.shape[0])
    for j in range(n_b):
        others = np.prod(np.delete(g[n_x:], j, axis=0), axis=0)
        m_b += p_b[j] * t[n_x + j] * others
    return np.column_stack([m_xh * g_bath, m_b * g_loc])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: the window states of the two-bond model ---
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
""",
            "call": "zero_order_amplitudes(states.copy(), C.copy(), p_xh.copy(), p_bath.copy(), x_loc.copy(), delta_loc.copy(), delta_bath.copy(), 8)",
            "gold_call": "_oracle_zero_order_amplitudes(states.copy(), C.copy(), p_xh.copy(), p_bath.copy(), x_loc.copy(), delta_loc.copy(), delta_bath.copy(), 8)",
            "tol": 1e-09,
        },
        # --- Normal: window states of the benchmark model at 9000 cm^-1 ---
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
""",
            "call": "zero_order_amplitudes(states.copy(), C.copy(), p_xh.copy(), p_bath.copy(), x_loc.copy(), delta_loc.copy(), delta_bath.copy(), 10)",
            "gold_call": "_oracle_zero_order_amplitudes(states.copy(), C.copy(), p_xh.copy(), p_bath.copy(), x_loc.copy(), delta_loc.copy(), delta_bath.copy(), 10)",
            "tol": 1e-09,
        },
        # --- Boundary: the vibrationless S0 level alone ---
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
states = np.zeros((1, 5))
""",
            "call": "zero_order_amplitudes(states.copy(), C.copy(), p_xh.copy(), p_bath.copy(), x_loc.copy(), delta_loc.copy(), delta_bath.copy(), 8)",
            "gold_call": "_oracle_zero_order_amplitudes(states.copy(), C.copy(), p_xh.copy(), p_bath.copy(), x_loc.copy(), delta_loc.copy(), delta_bath.copy(), 8)",
            "tol": 1e-09,
        },
        # --- Edge: two bonds built from three X-H normal modes, so the local couplings are the least-squares solution ---
        {
            "setup": """import numpy as np
omega_loc = np.array([3050.0, 3600.0])
x_loc = np.array([0.0205, 0.0220])
delta_loc = np.array([0.15, -0.10])
C = np.array([[0.80, 0.45, 0.20], [-0.30, 0.70, 0.55]])
p_xh = np.array([300.0, -180.0, 120.0])
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
""",
            "call": "zero_order_amplitudes(states.copy(), C.copy(), p_xh.copy(), p_bath.copy(), x_loc.copy(), delta_loc.copy(), delta_bath.copy(), 8)",
            "gold_call": "_oracle_zero_order_amplitudes(states.copy(), C.copy(), p_xh.copy(), p_bath.copy(), x_loc.copy(), delta_loc.copy(), delta_bath.copy(), 8)",
            "tol": 1e-09,
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
states = np.zeros((0, 5))
""",
            "call": "zero_order_amplitudes(states.copy(), C.copy(), p_xh.copy(), p_bath.copy(), x_loc.copy(), delta_loc.copy(), delta_bath.copy(), 8)",
            "gold_call": "_oracle_zero_order_amplitudes(states.copy(), C.copy(), p_xh.copy(), p_bath.copy(), x_loc.copy(), delta_loc.copy(), delta_bath.copy(), 8)",
            "tol": 1e-09,
        },
    ]
