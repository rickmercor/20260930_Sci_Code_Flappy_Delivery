"""
Analytic second derivatives of the converged SCF energy with respect to the atom positions.

Unlike the gradient, the curvature of the converged energy depends on how the orbitals respond when an atom moves, because the first-order change of the density enters the second derivative. That response has to be obtained self-consistently from the perturbed Roothaan equations in the moving, non-orthogonal basis, and it feeds both the density and the energy-weighted density that appear in the second derivative alongside the explicit second derivatives of the model matrices.

Returns
-------
np.ndarray, shape (n, n) float matrix of second derivatives of the converged SCF total energy with respect to the atom positions
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def scf_energy_hessian(positions: "np.ndarray", types: "np.ndarray", params: dict) -> "np.ndarray":
    """Return the exact second-derivative matrix of the converged SCF total energy.

    Parameters
    ----------
    positions : np.ndarray
        Shape (n,) atom positions; n is even, and no two atoms coincide.
    types : np.ndarray
        Shape (n,) element types, 0 or 1.
    params : dict
        Model constants.

    Returns
    -------
    hessian : np.ndarray
        Shape (n, n) float array whose entry (a, b) is the second derivative of
        the converged SCF total energy (the energy the reference-energy routine
        returns) with respect to the positions of atoms a and b. The matrix is
        the analytic one, including the response of the orbitals to the
        displacement, evaluated at the SCF solution converged until the largest
        change of a density element between successive diagonalizations is
        below 1e-12, and is checked to 1e-11 in every entry; it is symmetric to
        that precision.

    Raises
    ------
    ValueError
        If the atom count is odd, or if positions and types disagree in length.
    """
    return hessian

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_scf_energy_hessian(positions: "np.ndarray", types: "np.ndarray", params: dict) -> "np.ndarray":
    pos = np.asarray(positions, dtype=float)
    typ = np.asarray(types, dtype=int)
    n = pos.shape[0]
    S, H, gamma, P, eps, C = _converged_scf(pos, typ, params)
    kappa = float(params["kappa"])
    sigma = float(params["sigma"])
    strength = float(params["core_repulsion_strength"])
    rho = float(params["core_repulsion_range"])
    nocc = n // 2
    occ = C[:, :nocc]
    weighted = 2.0 * (occ * eps[:nocc]) @ occ.T
    R = np.abs(pos[:, None] - pos[None, :])
    sign = np.sign(pos[:, None] - pos[None, :])
    off = ~np.eye(n, dtype=bool)
    hdiag = np.diag(H)
    p = np.diag(P)
    pair_weight = np.outer(p, p) - 0.5 * P ** 2
    S1 = S * (-R / sigma ** 2) * off
    S2 = S * (R ** 2 / sigma ** 4 - 1.0 / sigma ** 2) * off
    G1 = -R * gamma ** 3 * off
    G2 = (-gamma ** 3 + 3.0 * R ** 2 * gamma ** 5) * off
    iu = np.triu_indices(n, 1)
    core_second = 2.0 / R[iu] ** 3 + strength / rho ** 2 * np.exp(-R[iu] / rho)
    steps = []
    for a in range(n):
        d = np.zeros((n, n))
        d[a, :] = sign[a, :]
        d[:, a] = -sign[:, a]
        steps.append(d)
    firsts = []
    for a in range(n):
        dS = S1 * steps[a]
        dG = G1 * steps[a]
        dh = -np.sum(dG * off, axis=1)
        dH = np.diag(dh) + kappa * off * (dS * 0.5 * (hdiag[:, None] + hdiag[None, :])
                                          + S * 0.5 * (dh[:, None] + dh[None, :]))
        firsts.append((dS, dG, dh, dH + _two_electron_part(dG, P)))
    responses = [_orbital_response(S, gamma, C, eps, nocc, firsts[b][0], firsts[b][3]) for b in range(n)]
    hessian = np.zeros((n, n))
    for a in range(n):
        dSa, dGa, dha, dFa = firsts[a]
        for b in range(n):
            dSb, dGb, dhb, dFb = firsts[b]
            dd = steps[a] * steps[b]
            S_ab = S2 * dd
            G_ab = G2 * dd
            h_ab = -np.sum(G_ab * off, axis=1)
            H_ab = np.diag(h_ab) + kappa * off * (S_ab * 0.5 * (hdiag[:, None] + hdiag[None, :])
                                                  + dSa * 0.5 * (dhb[:, None] + dhb[None, :])
                                                  + dSb * 0.5 * (dha[:, None] + dha[None, :])
                                                  + S * 0.5 * (h_ab[:, None] + h_ab[None, :]))
            explicit = (np.sum(P * H_ab) + 0.5 * np.sum(pair_weight * G_ab) - np.sum(weighted * S_ab)
                        + np.sum(core_second * steps[a][iu] * steps[b][iu]))
            P_b, W_b = responses[b]
            hessian[a, b] = explicit + np.sum(P_b * dFa) - np.sum(W_b * dSa)
    return hessian

def _two_electron_part(gamma: "np.ndarray", P: "np.ndarray") -> "np.ndarray":
    return np.diag(gamma @ np.diag(P)) - 0.5 * P * gamma

def _orbital_response(S, gamma, C, eps, nocc, dS, dF_explicit):
    """First-order density and energy-weighted density for one displacement, from coupled-perturbed SCF."""
    n = C.shape[0]
    Sm = C.T @ dS @ C
    Fm = C.T @ dF_explicit @ C
    P_occ = np.zeros((n, n))
    for i in range(nocc):
        for j in range(nocc):
            P_occ -= 2.0 * Sm[i, j] * np.outer(C[:, j], C[:, i])
    pairs = [(a, i) for a in range(nocc, n) for i in range(nocc)]
    m = len(pairs)

    def _density(u):
        out = P_occ.copy()
        for k, (a, i) in enumerate(pairs):
            out += 2.0 * u[k] * (np.outer(C[:, a], C[:, i]) + np.outer(C[:, i], C[:, a]))
        return out

    base = C.T @ _two_electron_part(gamma, P_occ) @ C
    rhs = np.array([eps[i] * Sm[a, i] - Fm[a, i] - base[a, i] for a, i in pairs])
    matrix = np.zeros((m, m))
    for col in range(m):
        unit = np.zeros(m)
        unit[col] = 1.0
        coupling = C.T @ _two_electron_part(gamma, _density(unit) - P_occ) @ C
        for k, (a, i) in enumerate(pairs):
            matrix[k, col] = (eps[a] - eps[i]) * (1.0 if k == col else 0.0) + coupling[a, i]
    u = np.linalg.solve(matrix, rhs) if m else np.zeros(0)
    P_b = _density(u)
    F_total = C.T @ (dF_explicit + _two_electron_part(gamma, P_b)) @ C
    U = np.zeros((n, n))
    for i in range(nocc):
        for q in range(n):
            U[q, i] = -0.5 * Sm[i, i] if q == i else (eps[i] * Sm[q, i] - F_total[q, i]) / (eps[q] - eps[i])
    eps_b = np.array([F_total[i, i] - eps[i] * Sm[i, i] for i in range(nocc)])
    C_b = C @ U
    W_b = np.zeros((n, n))
    for i in range(nocc):
        W_b += 2.0 * (eps_b[i] * np.outer(C[:, i], C[:, i])
                      + eps[i] * (np.outer(C_b[:, i], C[:, i]) + np.outer(C[:, i], C_b[:, i])))
    return P_b, W_b

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Test cases for this step."""
    params = ("params = {'onsite_energy': [-1.0, -1.6], 'onsite_repulsion': [0.9, 1.3], "
              "'kappa': 1.75, 'sigma': 1.1, 'core_repulsion_strength': 6.0, 'core_repulsion_range': 0.45}\n")
    return [
        # chain away from its minimum
        {
            "setup": "import numpy as np\nimport copy\n" + params + "positions = np.array([0.0, 1.1, 2.5, 3.9])\ntypes = np.array([0, 1, 0, 1])\n",
            "call": "scf_energy_hessian(*copy.deepcopy((positions, types, params)))",
            "gold_call": "_oracle_scf_energy_hessian(positions, types, params)",
            "tol": 1e-11,
        },
        # chain close to its equilibrium
        {
            "setup": "import numpy as np\nimport copy\n" + params + "positions = np.array([0.0, 1.0399, 2.7489, 3.7827])\ntypes = np.array([0, 1, 0, 1])\n",
            "call": "scf_energy_hessian(*copy.deepcopy((positions, types, params)))",
            "gold_call": "_oracle_scf_energy_hessian(positions, types, params)",
            "tol": 1e-11,
        },
        # stretched heteronuclear diatomic
        {
            "setup": "import numpy as np\nimport copy\n" + params + "positions = np.array([0.0, 1.6])\ntypes = np.array([0, 1])\n",
            "call": "scf_energy_hessian(*copy.deepcopy((positions, types, params)))",
            "gold_call": "_oracle_scf_energy_hessian(positions, types, params)",
            "tol": 1e-11,
        },
        # atoms listed out of spatial order
        {
            "setup": "import numpy as np\nimport copy\n" + params + "positions = np.array([2.4, 0.0, 1.3, 3.5])\ntypes = np.array([1, 0, 1, 0])\n",
            "call": "scf_energy_hessian(*copy.deepcopy((positions, types, params)))",
            "gold_call": "_oracle_scf_energy_hessian(positions, types, params)",
            "tol": 1e-11,
        },
        # odd atom count is rejected
        {
            "setup": "import numpy as np\nimport copy\n" + params + """positions = np.array([0.0, 1.1, 2.2])
types = np.array([0, 1, 0])
def run_model():
    try:
        scf_energy_hessian(positions, types, params)
        return 0
    except ValueError:
        return 1
def run_oracle():
    try:
        _oracle_scf_energy_hessian(positions, types, params)
        return 0
    except ValueError:
        return 1
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
