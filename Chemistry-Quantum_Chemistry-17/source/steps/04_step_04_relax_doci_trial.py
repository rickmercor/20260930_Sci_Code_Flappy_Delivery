"""
Construct a DOCI trial by a fixed number of damped orbital-relaxation updates.

Orbital relaxation changes the paired variational space before auxiliary-field projection. Each accepted rotation changes the orbitals and the variational configuration coefficients together. A specified finite update schedule gives a reproducible partially relaxed trial without assuming that a global orbital minimum has been reached.

Returns
-------
return energy, coefficients, orbitals, energy_history, accepted_scales
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def relax_doci_trial(h: "np.ndarray", factors: "np.ndarray", npair: int, n_updates: int) -> "tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]":
    """Relax a paired trial using the specified deterministic orbital schedule.

    Inputs are finite and real and must not be modified. NumPy and SciPy are
    available. Use the complete paired occupation basis from pair_occupations.
    Start with orbital frame U=I and rediagonalize the projected Hamiltonian
    after each accepted rotation. At every update, E, c, g and B are the
    relaxed orbital energy, coefficients, gradient and Hessian defined by
    doci_orbital_response, including configuration-coefficient response.

    The following numerical schedule defines this task's finite relaxation:
    if ||g||_2 <= 1e-10 Eh, retain U and record scale zero. Otherwise set
    mu=max(0,0.05 Eh-lambda_min(B)), solve (B+mu I)p=-g and shorten p, if
    necessary, to ||p||_2=0.25 radians. If -g.T p <= 1e-14 Eh, also retain U
    and record zero. Otherwise try alpha=2**(-j) for j=0,...,20 in that order.
    Accept the first U_new=U exp(alpha*sum_a p_a K_a) satisfying
    E(U_new) <= E(U)+1e-4*alpha*g.T p, with the generator ordering of
    doci_orbital_response. Each trial energy is rediagonalized. Perform
    exactly n_updates scheduled updates, including any zero-scale updates.
    The constants specify this finite replay, not a convergence criterion
    or a claim that the final orbitals are globally optimal.

    Parameters
    ----------
    h : real ndarray, shape (n,n)
        Symmetric one-electron matrix in Eh, 1 <= n <= 6.
    factors : real ndarray, shape (r,n,n)
        Symmetric two-electron factors in sqrt(Eh), 1 <= r <= 8.
    npair : int
        Number of alpha-beta electron pairs, 1 <= npair <= n.
    n_updates : int
        Number of scheduled orbital updates, 0 <= n_updates <= 6.

    Returns
    -------
    energy : float
        Final DOCI trial energy in Eh.
    coefficients : real ndarray, shape (nd,)
        Normalized final coefficients in lexicographic occupation order,
        with the sign convention of doci_orbital_response.
    orbitals : real ndarray, shape (n,n)
        Final real orthogonal trial orbital frame U.
    energy_history : real ndarray, shape (n_updates+1,)
        Energy before the first update and after each scheduled update.
    accepted_scales : real ndarray, shape (n_updates,)
        Accepted alpha for each update, or zero for a stationary update.

    Raises
    ------
    ValueError
        For incompatible or nonsymmetric integrals, npair or n_updates
        outside its integer range, a projected ground-state gap <= 1e-10 Eh
        at a retained frame with more than one configuration, or failure of
        all 21 trial scales to satisfy the stated decrease condition.
    """
    return energy, coefficients, orbitals, energy_history, accepted_scales

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.linalg import expm


def _oracle_relax_doci_trial(h: "np.ndarray", factors: "np.ndarray", npair: int, n_updates: int) -> "tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]":
    h = np.asarray(h, dtype=float)
    factors = np.asarray(factors, dtype=float)
    if h.ndim != 2 or h.shape[0] != h.shape[1] or not 1 <= len(h) <= 6:
        raise ValueError("h must be square with 1 <= n <= 6")
    if (not isinstance(n_updates, (int, np.integer))
            or not 0 <= n_updates <= 6):
        raise ValueError("n_updates must be an integer from 0 through 6")
    if not isinstance(npair, (int, np.integer)) or not 1 <= npair <= len(h):
        raise ValueError("Invalid pair count")
    occupations = _oracle_pair_occupations(len(h), npair)
    orbitals = np.eye(len(h))
    generators = _oo_generators(len(h))
    energy, coefficients, gradient, hessian = _oracle_doci_orbital_response(
        h, factors, occupations, orbitals
    )
    energies = [energy]
    accepted_scales = np.zeros(n_updates)
    for update in range(n_updates):
        if np.linalg.norm(gradient) > 1e-10:
            shift = max(0.0, 0.05 - np.linalg.eigvalsh(hessian)[0])
            step = -np.linalg.solve(hessian + shift*np.eye(len(gradient)), gradient)
            norm = np.linalg.norm(step)
            if norm > 0.25:
                step *= 0.25/norm
            directional = float(gradient @ step)
            if -directional > 1e-14:
                for power in range(21):
                    alpha = 2.0**(-power)
                    rotation = expm(alpha*np.einsum('a,aij->ij', step, generators))
                    candidate = orbitals @ rotation
                    transformed_h = candidate.T @ h @ candidate
                    transformed_factors = np.asarray([
                        candidate.T @ l @ candidate for l in factors
                    ])
                    matrix = _oracle_doci_matrix(
                        transformed_h, transformed_factors, occupations
                    )
                    candidate_energy = np.linalg.eigvalsh(matrix)[0]
                    if candidate_energy <= energy + 1e-4*alpha*directional:
                        orbitals = candidate
                        accepted_scales[update] = alpha
                        break
                else:
                    raise ValueError("No orbital line-search scale was accepted")
                energy, coefficients, gradient, hessian = _oracle_doci_orbital_response(
                    h, factors, occupations, orbitals
                )
        energies.append(energy)
    return energy, coefficients, orbitals, np.asarray(energies), accepted_scales

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Check finite relaxation, zero updates, stationary frames and input guards."""
    setup = r"""import numpy as np
from scipy.linalg import expm

H = np.array([[-1.084145, 0.12, -0.08, 0.05], [0.12, -0.885553, 0.11, -0.06], [-0.08, 0.11, -0.763931, 0.09], [0.05, -0.06, 0.09, -0.742439]], dtype=float)
L = np.array([[[0.58, 0.196, -0.084, 0.112], [0.196, 0.43, 0.224, -0.056], [-0.084, 0.224, 0.36, 0.168], [0.112, -0.056, 0.168, 0.29]], [[0.16, -0.308, 0.112, 0.084], [-0.308, -0.22, 0.14, 0.196], [0.112, 0.14, 0.19, -0.224], [0.084, 0.196, -0.224, -0.14]], [[-0.12, 0.084, 0.252, -0.14], [0.084, 0.17, -0.168, 0.056], [0.252, -0.168, 0.08, 0.308], [-0.14, 0.056, 0.308, -0.1]]], dtype=float)
OCC = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]], dtype=int)
U = np.eye(4)
"""
    checks = r"""
def _checked(fn, *args):
    copied = [x.copy() if isinstance(x, np.ndarray) else x for x in args]
    before = [x.copy() if isinstance(x, np.ndarray) else x for x in copied]
    try:
        result = fn(*copied)
    finally:
        for a, b in zip(before, copied):
            if isinstance(a, np.ndarray) and not np.array_equal(a, b):
                raise AssertionError("Input arrays must not be modified")
    from math import comb
    n, k, updates = len(args[0]), args[2], args[3]
    shapes = [(), (comb(n,k),), (n,n), (updates+1,), (updates,)]
    if not isinstance(result, tuple) or len(result) != len(shapes):
        raise AssertionError("Expected the documented five-part tuple")
    arrays = [np.asarray(x) for x in result]
    if any(x.shape != shape for x, shape in zip(arrays, shapes)):
        raise AssertionError("Incorrect relaxation output shape")
    if any(np.iscomplexobj(x) or not np.all(np.isfinite(x)) for x in arrays):
        raise AssertionError("Expected finite real output")
    return np.concatenate([x.reshape(-1) for x in arrays])

def _raises(fn, *args):
    try:
        _checked(fn, *args)
    except ValueError:
        return 1
    raise AssertionError("Expected ValueError")
"""
    stationary = """
H0 = np.diag([-1.0, -0.6, 0.2])
L0 = np.zeros((1,3,3))
"""
    return [
        {"setup": setup + checks,
         "call": "_checked(relax_doci_trial, H, L, 2, 4)",
         "gold_call": "_checked(_oracle_relax_doci_trial, H, L, 2, 4)",
         "tol": 2e-7},
        {"setup": setup + checks,
         "call": "_checked(relax_doci_trial, H, L, 2, 0)",
         "gold_call": "_checked(_oracle_relax_doci_trial, H, L, 2, 0)",
         "tol": 2e-7},
        {"setup": setup + checks,
         "call": "_checked(relax_doci_trial, H, L, 2, 6)",
         "gold_call": "_checked(_oracle_relax_doci_trial, H, L, 2, 6)",
         "tol": 2e-7},
        {"setup": setup + stationary + checks,
         "call": "_checked(relax_doci_trial, H0, L0, 1, 4)",
         "gold_call": "_checked(_oracle_relax_doci_trial, H0, L0, 1, 4)",
         "tol": 2e-7},
        {"setup": setup + checks,
         "call": "_checked(relax_doci_trial, np.array([[-1.0,1e-8],[1e-8,0.3]]), np.zeros((1,2,2)), 1, 2)",
         "gold_call": "_checked(_oracle_relax_doci_trial, np.array([[-1.0,1e-8],[1e-8,0.3]]), np.zeros((1,2,2)), 1, 2)",
         "tol": 2e-7},
        {"setup": setup + checks,
         "call": "_checked(relax_doci_trial, H, L, 4, 3)",
         "gold_call": "_checked(_oracle_relax_doci_trial, H, L, 4, 3)",
         "tol": 2e-7},
        {"setup": setup + checks,
         "call": "_checked(relax_doci_trial, np.array([[-0.7]]), np.array([[[0.3]]]), 1, 3)",
         "gold_call": "_checked(_oracle_relax_doci_trial, np.array([[-0.7]]), np.array([[[0.3]]]), 1, 3)",
         "tol": 2e-7},
        {"setup": setup + checks,
         "call": "_raises(relax_doci_trial, H, L, 2, -1)",
         "gold_call": "_raises(_oracle_relax_doci_trial, H, L, 2, -1)",
         "tol": 2e-7},
    ]
