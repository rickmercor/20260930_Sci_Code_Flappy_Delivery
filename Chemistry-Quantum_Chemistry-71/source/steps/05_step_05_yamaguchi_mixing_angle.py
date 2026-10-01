"""
Step 05: Yamaguchi spin-projected mixing angle from unrestricted Hartree-Fock. Spin-projected unrestricted Hartree-Fock estimate of diradical character and the mixing angle it implies.

Before multiconfigurational methods became routine, the diradical character of a stretched bond or a forbidden transition state was estimated from broken-symmetry unrestricted Hartree-Fock theory. Beyond a critical point the spin-restricted closed-shell solution becomes unstable and the lowest mean-field determinant lets electrons of opposite spin localise on different centres; its natural orbitals then show a highest occupied and a lowest unoccupied orbital with occupations moving towards one. Yamaguchi's spin-projection formula turns the occupation of the highest natural orbital into a diradical index between 0 and 1, and the same index can be mapped onto the rotation angle of the two-configuration bond picture. The mean-field estimate needs only a determinant, but it switches on abruptly at the instability and it ignores the dynamic correlation that a correlated treatment already builds into the reactant, so it differs systematically from the angle obtained from correlated natural occupations.

Returns
-------
numpy.ndarray [E_uhf in eV, n_H, Yamaguchi y, Theta_puhf in degrees] of the lowest unrestricted Hartree-Fock determinant
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def yamaguchi_mixing_angle(h: "np.ndarray", U: float) -> "np.ndarray":
    """Lowest unrestricted Hartree-Fock solution of a half-filled Hubbard model and its source-defined spin-projected mixing descriptors.

    Parameters
    ----------
    h : np.ndarray
        Real symmetric (n, n) one-electron matrix in eV, as in singlet_ground_state.
    U : float
        On-site repulsion in eV, finite and >= 0.

    Returns
    -------
    y : np.ndarray
        Real array (E_uhf, n_H, y_diradical, Theta_puhf), in that order. E_uhf is in eV for the lowest real determinant
        with n/2 spin-up and n/2 spin-down electrons under the energy functional specified in the problem statement;
        n_H is the (n/2)-th largest eigenvalue of P_up + P_down for that determinant; y_diradical and Theta_puhf
        (degrees) are obtained from n_H using the source-defined Yamaguchi spin-projection mapping. Inputs are such that
        all determinants of lowest energy share the same n_H.

    Raises
    ------
    ValueError
        If h or U violates the conditions of singlet_ground_state.
    """
    return y

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _uhf_run(h, U, Da, Db, iters=3000):
    import numpy as np
    k = h.shape[0] // 2
    E_old = np.inf
    for _ in range(iters):
        ea, Ca = np.linalg.eigh(h + U * np.diag(np.diag(Db)))
        eb, Cb = np.linalg.eigh(h + U * np.diag(np.diag(Da)))
        Dan, Dbn = Ca[:, :k] @ Ca[:, :k].T, Cb[:, :k] @ Cb[:, :k].T
        Da, Db = 0.5 * (Da + Dan), 0.5 * (Db + Dbn)
        E = np.sum(h * (Da + Db)) + U * np.sum(np.diag(Da) * np.diag(Db))
        if abs(E - E_old) < 1e-14 and max(np.abs(Dan - Da).max(), np.abs(Dbn - Db).max()) < 1e-11:
            break
        E_old = E
    return E, Da, Db


def _oracle_yamaguchi_mixing_angle(h: "np.ndarray", U: float) -> "np.ndarray":
    import numpy as np
    h, U = _check_h_U(h, U)
    n = h.shape[0]
    k = n // 2
    e, C = np.linalg.eigh(h)
    D0 = C[:, :k] @ C[:, :k].T
    starts = [(D0, D0)]
    grid = np.arange(n * n).reshape(n, n)
    for j in range(1, 20):
        Qa = np.linalg.qr(np.eye(n) + 0.6 * np.sin(1.7 * j + 2.3 * grid + 0.37 * grid ** 2))[0]
        Qb = np.linalg.qr(np.eye(n) + 0.6 * np.cos(2.9 * j - 1.1 * grid + 0.53 * grid ** 2))[0]
        Ca, Cb = C @ Qa, C @ Qb
        starts.append((Ca[:, :k] @ Ca[:, :k].T, Cb[:, :k] @ Cb[:, :k].T))
    best = None
    for Da, Db in starts:
        E, Da, Db = _uhf_run(h, U, Da, Db)
        if max(np.abs(Da @ Da - Da).max(), np.abs(Db @ Db - Db).max()) > 1e-8:
            continue
        if best is None or E < best[0] - 1e-10:
            best = (E, Da, Db)
    E, Da, Db = best
    occ = np.sort(np.linalg.eigvalsh(Da + Db))[::-1]
    T = min(max(occ[k - 1] - 1.0, 0.0), 1.0)
    y = 1.0 - 2.0 * T / (1.0 + T * T)
    return np.array([E, occ[k - 1], y, np.degrees(np.arcsin(np.sqrt(y / 2.0)))])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    base = ("import numpy as np\n"
            "def _ring(n, t, tn):\n    h = np.zeros((n, n))\n    for i in range(n - 1):\n        h[i, i + 1] = h[i + 1, i] = t[i % len(t)]\n"
            "    h[0, n - 1] = h[n - 1, 0] = tn\n    return h\n")
    return [
        # --- Normal: Moebius-like hexatriene ring, broken-symmetry solution ---
        {"setup": base, "call": "np.asarray(yamaguchi_mixing_angle(_ring(6, [-1.9, -2.2], 0.9), 4.0), dtype=float)",
         "gold_call": "np.asarray(_oracle_yamaguchi_mixing_angle(_ring(6, [-1.9, -2.2], 0.9), 4.0), dtype=float)", "tol": 1e-6},
        # --- Normal: stretched two-site bond beyond the restricted-solution instability ---
        {"setup": base, "call": "np.asarray(yamaguchi_mixing_angle(np.array([[0.0, -1.0], [-1.0, 0.0]]), 3.0), dtype=float)",
         "gold_call": "np.asarray(_oracle_yamaguchi_mixing_angle(np.array([[0.0, -1.0], [-1.0, 0.0]]), 3.0), dtype=float)", "tol": 1e-8},
        # --- Normal: eight-site chain with a weak antibonding end coupling ---
        {"setup": base, "call": "np.asarray(yamaguchi_mixing_angle(_ring(8, [-2.4, -2.0], 0.5), 4.5), dtype=float)",
         "gold_call": "np.asarray(_oracle_yamaguchi_mixing_angle(_ring(8, [-2.4, -2.0], 0.5), 4.5), dtype=float)", "tol": 2e-5},
        # --- Boundary: two-site bond below the instability, restricted solution and zero diradical index ---
        {"setup": base, "call": "np.asarray(yamaguchi_mixing_angle(np.array([[0.0, -1.0], [-1.0, 0.0]]), 1.5), dtype=float)",
         "gold_call": "np.asarray(_oracle_yamaguchi_mixing_angle(np.array([[0.0, -1.0], [-1.0, 0.0]]), 1.5), dtype=float)", "tol": 1e-8},
        # --- Boundary: Hueckel-like aromatic ring, restricted solution ---
        {"setup": base, "call": "np.asarray(yamaguchi_mixing_angle(_ring(6, [-2.4, -2.4], -2.4), 3.0), dtype=float)",
         "gold_call": "np.asarray(_oracle_yamaguchi_mixing_angle(_ring(6, [-2.4, -2.4], -2.4), 3.0), dtype=float)", "tol": 1e-8},
        # --- Edge: very strong repulsion, nearly a pure diradical ---
        {"setup": base, "call": "np.asarray(yamaguchi_mixing_angle(_ring(4, [-1.0, -1.0], 0.2), 20.0), dtype=float)",
         "gold_call": "np.asarray(_oracle_yamaguchi_mixing_angle(_ring(4, [-1.0, -1.0], 0.2), 20.0), dtype=float)", "tol": 1e-6},
        # --- Error: non-symmetric matrix ---
        {"setup": base + "def _probe(fn):\n    try:\n        fn(np.array([[0.0, -1.0], [-0.5, 0.0]]), 2.0)\n    except ValueError:\n        return 1\n    return 0\n",
         "call": "_probe(yamaguchi_mixing_angle)", "gold_call": "_probe(_oracle_yamaguchi_mixing_angle)"},
    ]
