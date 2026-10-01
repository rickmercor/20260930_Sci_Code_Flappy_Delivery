"""
Run one full restart cycle of similarity-restoring randomized Krylov-Schur: assemble the sketched instance, start the Krylov pack, expand it for m sketched actions, solve the Euclidean leftover coefficients and restore the last column, check the corrected Ritz value lies in the spectrum, compress to order l, re-expand to order m, solve the leftover coefficients again, restore and check again, and return the Euclidean norm of the second coefficient vector. Require 2 <= m < n, d >= m, 1 <= l < m. Call every earlier public step and use every output.

The orchestrator is the prompt instance. Four sketched steps, one correction, a compression keeping two Ritz values, two more sketched steps, and the cycle-closing correction recover the locked norm; normalizing the carried vector, reorthogonalizing it, compressing the uncorrected matrix, or using classical Krylov-Schur all yield different numbers.

Returns
-------
native Python float: Euclidean norm of the cycle-closing correction vector
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def run_restarted_correction_norm(
    n: int,
    m: int,
    d: int,
    l: int,
    s: np.ndarray,
    data_seed: int,
    sketch_seed: int,
) -> float:
    """Norm of the least-squares correction closing one restart cycle.

    Parameters
    ----------
    n : int
        Dimension, n >= 3.
    m : int
        Expansion order, 2 <= m < n.
    d : int
        Sketch dimension, d >= m.
    l : int
        Restart order, 1 <= l < m.
    s : np.ndarray
        Positive eigenvalues, shape (n,).
    data_seed : int
        RNG seed for the SPD factor draw and start vector.
    sketch_seed : int
        Independent RNG seed for the sketch.

    Returns
    -------
    value : float
        Euclidean norm of the leftover coefficient vector computed on the
        re-expanded order-m basis at the end of the first restart cycle.

    Raises
    ------
    ValueError
        If n, m, d, or l is not an integer, if n < 3, if m is not in
        2, ..., n-1, if d < m, if l is not in 1, ..., m-1, or if a
        corrected Ritz value exceeds max(s) beyond roundoff.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _prior_oracle(public_name):
    """Return a sibling _oracle_* function, never a public stub. """
    oracle_name = "_oracle_" + public_name
    fn = globals().get(oracle_name)
    if not callable(fn):
        raise ValueError(f"missing oracle for {public_name}")
    return fn


def _unpack_restart_instance(pack):
    p = np.asarray(pack, dtype=float).reshape(-1)
    if p.size < 2:
        raise ValueError("instance pack is too short")
    n, d = [int(round(float(v))) for v in p[:2]]
    if min(n, d) < 1:
        raise ValueError("packed shapes must be positive")
    need = 2 + n * n + n + d * n
    if p.size != need:
        raise ValueError("instance pack length does not match header")
    A = p[2 : 2 + n * n].reshape(n, n)
    b = p[2 + n * n : 2 + n * n + n]
    Omega = p[2 + n * n + n :].reshape(d, n)
    return A, b, Omega


def _unpack_restart_state(pack):
    p = np.asarray(pack, dtype=float).reshape(-1)
    if p.size < 3:
        raise ValueError("Krylov pack is too short")
    n, m, n_hess = [int(round(float(v))) for v in p[:3]]
    if n < 2 or m < 1 or n_hess < 0 or n_hess > m:
        raise ValueError("Krylov pack header is invalid")
    need = 3 + n * m + m * m + n + 1
    if p.size != need:
        raise ValueError("Krylov pack length does not match header")
    U = p[3 : 3 + n * m].reshape(n, m)
    return n, m, n_hess, U


def _oracle_run_restarted_correction_norm(n, m, d, l, s, data_seed, sketch_seed):
    for name, val in (("n", n), ("m", m), ("d", d), ("l", l)):
        if not isinstance(val, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
    n, m, d, l = int(n), int(m), int(d), int(l)
    if n < 3:
        raise ValueError("require n >= 3")
    if m < 2 or m >= n:
        raise ValueError("require 2 <= m < n")
    if d < m:
        raise ValueError("require d >= m")
    if l < 1 or l >= m:
        raise ValueError("require 1 <= l < m")

    _oracle_assemble_sketched_instance = _prior_oracle("assemble_sketched_instance")
    _oracle_start_krylov_pack = _prior_oracle("start_krylov_pack")
    _oracle_sketched_action_pack = _prior_oracle("sketched_action_pack")
    _oracle_append_krylov_column = _prior_oracle("append_krylov_column")
    _oracle_leftover_euclidean_coeffs = _prior_oracle("leftover_euclidean_coeffs")
    _oracle_restore_last_column = _prior_oracle("restore_last_column")
    _oracle_dominant_real_eigenvalue = _prior_oracle("dominant_real_eigenvalue")
    _oracle_compress_ritz_pack = _prior_oracle("compress_ritz_pack")
    _oracle_expand_krylov_cycle = _prior_oracle("expand_krylov_cycle")

    s_arr = np.asarray(s, dtype=float).reshape(-1)
    inst = _oracle_assemble_sketched_instance(n, d, s_arr, data_seed, sketch_seed)
    A, _b, Omega = _unpack_restart_instance(inst)
    s_max = float(np.max(s_arr))
    tol = 1e-8 * max(1.0, s_max)

    # First cycle: m sketched Arnoldi steps, then the similarity correction.
    state = _oracle_start_krylov_pack(inst, m)
    for _ in range(m):
        _n, _m, n_hess, U = _unpack_restart_state(state)
        w = A @ U[:, n_hess]
        action = _oracle_sketched_action_pack(Omega, U[:, : n_hess + 1], w)
        state = _oracle_append_krylov_column(state, action)
    hhat = _oracle_leftover_euclidean_coeffs(state)
    Hbar = _oracle_restore_last_column(state, hhat)
    if _oracle_dominant_real_eigenvalue(Hbar) > s_max + tol:
        raise ValueError("corrected Ritz value outside the spectrum of A")

    # Restart: compress to order l, re-expand to order m, correct again.
    decomp = _oracle_compress_ritz_pack(state, hhat, Hbar, l)
    state_new = _oracle_expand_krylov_cycle(inst, decomp, m)
    hhat_new = _oracle_leftover_euclidean_coeffs(state_new)
    Hbar_new = _oracle_restore_last_column(state_new, hhat_new)
    if _oracle_dominant_real_eigenvalue(Hbar_new) > s_max + tol:
        raise ValueError("corrected Ritz value outside the spectrum of A")
    return float(np.linalg.norm(hhat_new))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np
n, m, d, l = 12, 4, 8, 2
s = np.array([8.0, 6.0, 5.0, 1.2, 0.9, 0.7, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15])
data_seed, sketch_seed = 7, 11
""",
            "call": "run_restarted_correction_norm(n, m, d, l, s, data_seed, sketch_seed)",
            "gold_call": "_oracle_run_restarted_correction_norm(n, m, d, l, s, data_seed, sketch_seed)",
        },
        {
            "setup": """import numpy as np
n, m, d, l = 8, 3, 6, 1
s = np.array([5.0, 4.0, 3.0, 0.5, 0.4, 0.3, 0.2, 0.1])
data_seed, sketch_seed = 1, 4
""",
            "call": "run_restarted_correction_norm(n, m, d, l, s, data_seed, sketch_seed)",
            "gold_call": "_oracle_run_restarted_correction_norm(n, m, d, l, s, data_seed, sketch_seed)",
        },
        {
            "setup": """import numpy as np
n, m, d, l = 9, 4, 6, 2
s = np.array([6.0, 4.5, 3.0, 2.0, 0.6, 0.5, 0.4, 0.3, 0.2])
data_seed, sketch_seed = 0, 2
""",
            "call": "run_restarted_correction_norm(n, m, d, l, s, data_seed, sketch_seed)",
            "gold_call": "_oracle_run_restarted_correction_norm(n, m, d, l, s, data_seed, sketch_seed)",
        },
        {
            "setup": """import numpy as np
s = np.array([3.0, 2.0, 0.4, 0.2])
def run_model():
    try:
        run_restarted_correction_norm(4, 3, 4, 3, s, 0, 1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_run_restarted_correction_norm(4, 3, 4, 3, s, 0, 1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
s = np.array([3.0, 2.0, 0.4, 0.2])
def run_model():
    try:
        run_restarted_correction_norm(4, 3, 2, 1, s, 0, 1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_run_restarted_correction_norm(4, 3, 2, 1, s, 0, 1)
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
