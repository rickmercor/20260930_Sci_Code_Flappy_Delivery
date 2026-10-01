"""
Run the whole pipeline for one configuration and return the quantities the problem statement asks to be reported.

The arguments are the mesh and box dimensions, the material constants, the three retained-mode counts as nd, and the one-based mode index.

The function assembles the partitioned system, derives the four block sizes from the mesh, builds the retained modes, the static interface response and the residual modes for each of the three substructures, combines them into the augmented transformation by supplying the interface identity once, projects, and evaluates the relative deviation for the requested mode.

The function returns a one-dimensional float array of nine entries in this order: the relative deviation; the reduced eigenvalue of the requested mode; the stiffness quadratic form of the base contribution; its mass quadratic form; the number of free equations; the unaugmented dimension; the augmented dimension; the total number of residual-mode columns vanishing across the three substructures; and the number of structural guards satisfied, out of six.

Raises ValueError if nd does not have three entries, or if mode_index is below one or exceeds the number of retained eigenvalues. Errors raised by the steps it calls propagate unchanged.

The individual operations of a substructured reduction are each unremarkable; the difficulty lies in composing them with consistent conventions. The block ordering fixed at assembly has to be respected by every step that indexes into the matrices, the contribution belonging to the interface has to be supplied exactly once rather than once per substructure, and the retained-mode counts have to correspond to the substructures they were computed for.

Because errors in composition produce finite and plausible numbers rather than failures, it is worth returning structural invariants alongside the result. Whether the retained modes of each substructure really do reduce its mass block to the identity, and whether the static interface response really does annihilate the corresponding interior equilibrium residual, are properties that hold by construction when the composition is right and fail silently when it is not. Reporting them as counts rather than as residual norms avoids comparing two near-zero quantities under a relative tolerance, which fails on correct implementations.

The same applies to the vanishing residual-mode columns. Their number is determined entirely by which substructures touch which dividing surface, so it is an integer fingerprint of the partition, and it changes the moment the partition is off by a node layer.

Returns
-------
np.ndarray of shape (9,), dtype float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def reduction_report(nx: int, ny: int, nz: int, Lx: float, Ly: float, Lz: float,
                     E: float, nu: float, rho: float, nd: tuple,
                     mode_index: int) -> 'np.ndarray':
    '''Run the whole pipeline and return the reported quantities.

    Parameters
    ----------
    nx, ny, nz : int
        Element counts along the three coordinate directions.
    Lx, Ly, Lz : float
        Edge lengths of the box.
    E, nu, rho : float
        Young's modulus, Poisson's ratio and density.
    nd : sequence of int
        Three retained-mode counts, one per substructure.
    mode_index : int
        One-based index into the ascending retained eigenvalues.

    Returns
    -------
    np.ndarray
        (9,) float array as described in the step description.
    '''
    return reported_quantities  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_reduction_report(nx: int, ny: int, nz: int, Lx: float, Ly: float,
                             Lz: float, E: float, nu: float, rho: float, nd: tuple,
                             mode_index: int) -> 'np.ndarray':
    import numpy as np
    S = _oracle_assemble_partitioned_system(nx, ny, nz, Lx, Ly, Lz, E, nu, rho)
    Kp, Mp = S[0], S[1]
    nd = np.asarray(nd, dtype=int)
    if nd.size != 3:
        raise ValueError("nd must have three entries")
    per = (ny+1)*(nz+1)*3
    c = nx//3
    counts = np.array([(c-1)*per, (c-1)*per, (nx-2*c)*per, 2*per])
    o = np.concatenate([[0], np.cumsum(counts)])
    nCB = int(nd.sum() + counts[3])
    cols = []
    psi = np.zeros((Kp.shape[0], counts[3]))
    res = np.zeros((Kp.shape[0], counts[3]))
    flags = 0
    vanish = 0
    for k in range(3):
        Phi = _oracle_fixed_interface_modes(Kp, Mp, counts, k, int(nd[k]))
        Psi = _oracle_constraint_modes(Kp, counts, k)
        R = _oracle_residual_modes(Kp, Mp, counts, k, Phi)
        ii = slice(o[k], o[k+1])
        if np.allclose(Phi[ii, :].T @ Mp[ii, ii] @ Phi[ii, :], np.eye(int(nd[k])), atol=1e-9):
            flags += 1
        scale = np.abs(Kp[ii, ii]).max()
        if np.abs(Kp[ii, ii] @ Psi[ii, :] + Kp[ii, o[3]:o[4]]).max() < 1e-6*scale:
            flags += 1
        vanish += int((np.linalg.norm(R[ii, :], axis=0) == 0).sum())
        cols.append(Phi)
        psi += Psi
        res += R
    psi[o[3]:o[4], :] = np.eye(counts[3])
    T = np.hstack(cols + [psi, res])
    lam = _oracle_reduced_eigenvalues(Kp, Mp, T, nCB)
    if not 1 <= mode_index <= lam.size:
        raise ValueError("mode_index out of range")
    u0 = _oracle_base_contribution(Kp, Mp, T, nCB, mode_index)
    val = _oracle_relative_deviation(Kp, Mp, u0, lam[mode_index-1])
    return np.array([val, lam[mode_index-1], float(u0 @ Kp @ u0), float(u0 @ Mp @ u0),
                     float(Kp.shape[0]), float(nCB), float(T.shape[1]),
                     float(vanish), float(flags)])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        # --- production: the configuration of the problem statement ---
        {
            "setup": """import numpy as np
""",
            "call": "reduction_report(24, 12, 1, 1.0, 0.6, 0.01, 210e9, 0.3, 7850.0, (10, 10, 8), 20)",
            "gold_call": "_oracle_reduction_report(24, 12, 1, 1.0, 0.6, 0.01, 210e9, 0.3, 7850.0, (10, 10, 8), 20)",
        },
        # --- surrogate: twelve elements along the length, fifth mode ---
        {
            "setup": """import numpy as np
""",
            "call": "reduction_report(12, 3, 1, 1.0, 0.6, 0.01, 210e9, 0.3, 7850.0, (5, 5, 4), 5)",
            "gold_call": "_oracle_reduction_report(12, 3, 1, 1.0, 0.6, 0.01, 210e9, 0.3, 7850.0, (5, 5, 4), 5)",
        },
        # --- edge: a different material, mesh and aspect ratio ---
        {
            "setup": """import numpy as np
""",
            "call": "reduction_report(12, 4, 1, 2.0, 0.4, 0.02, 70e9, 0.33, 2700.0, (5, 5, 4), 6)",
            "gold_call": "_oracle_reduction_report(12, 4, 1, 2.0, 0.4, 0.02, 70e9, 0.33, 2700.0, (5, 5, 4), 6)",
        },
        # --- invalid: two retained-mode counts instead of three ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        reduction_report(12, 3, 1, 1.0, 0.6, 0.01, 210e9, 0.3, 7850.0, (5, 5), 5)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_reduction_report(12, 3, 1, 1.0, 0.6, 0.01, 210e9, 0.3, 7850.0, (5, 5), 5)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- invalid: mode index beyond the reduced spectrum ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        reduction_report(12, 3, 1, 1.0, 0.6, 0.01, 210e9, 0.3, 7850.0, (5, 5, 4), 100000)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_reduction_report(12, 3, 1, 1.0, 0.6, 0.01, 210e9, 0.3, 7850.0, (5, 5, 4), 100000)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
