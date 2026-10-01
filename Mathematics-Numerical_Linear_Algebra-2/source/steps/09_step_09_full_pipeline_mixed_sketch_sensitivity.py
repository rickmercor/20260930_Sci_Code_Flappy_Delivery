"""
Run the randomized correction, its sketch derivatives, and the generalized spectral sensitivity calculation, returning $\|\Pi_{st}\|_F$.

Keep the source-defined correction $D(s,t)=\widehat{\Delta}(s,t)$ and its sample $W(s,t)$, with the prescribed sine/cosine sketch directions. Form the positive-definite metric $T(s,t)=I_m+W(s,t)W(s,t)^T$. Let $U_r$ span the $r$ largest algebraic generalized eigenvalues of $D u=\lambda T u$, normalized by $U_r^TTU_r=I_r$. Return the Frobenius norm of the mixed derivative of $\Pi=U_rU_r^TT$ in the original coordinates. The selected cluster has a positive gap to the rest of the spectrum for valid benchmark configurations.

Returns
-------
float - $\|\partial_s\partial_t\Pi(0,0)\|_F$ as a native Python float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def full_pipeline_mixed_sketch_sensitivity(
    seed: int,
    p: int,
    m: int,
    k: int,
    eps: float,
    r: int,
) -> float:
    r"""
    Return the final scalar quantity for the deterministic benchmark
    instance.

    Raises
    ------
    ValueError
        If seed is negative, $p<1$, $m<p$, $k<1$, $k>m$,
        $\varepsilon$ is nonpositive or nonfinite, or $r$ is not an
        integer satisfying $1\leq r<m$.
    """
    return mixed_sensitivity

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_full_pipeline_mixed_sketch_sensitivity(
    seed: int,
    p: int,
    m: int,
    k: int,
    eps: float,
    r: int,
) -> float:

    if not (isinstance(seed, (int, np.integer)) and seed >= 0):
        raise ValueError("seed must be a non-negative integer")

    if not (isinstance(p, (int, np.integer)) and p >= 1):
        raise ValueError("p must be positive")

    if not (isinstance(m, (int, np.integer)) and m >= p):
        raise ValueError("m must be >= p")

    if not (isinstance(k, (int, np.integer)) and 1 <= k <= m):
        raise ValueError("k must satisfy 1 <= k <= m")

    if not (isinstance(eps, (int, float)) and np.isfinite(eps) and float(eps) > 0.0):
        raise ValueError("eps must be positive")

    if not isinstance(r, (int, np.integer)) or not 1 <= r < m:
        raise ValueError("r must be an integer satisfying 1 <= r < m")

    # Step 01: generate the deterministic inputs.
    E, C, Omega = _oracle_generate_deterministic_inputs(
        seed,
        p,
        m,
        k,
    )

    # Step 02: consume Step 01 output.
    Y, Y_D = _oracle_solve_diagonal_and_exact_actions(
        E,
        C,
        Omega,
    )

    # Step 03: consume Step 02 output.
    W = _oracle_form_sample_matrix(
        C,
        Y,
        Y_D,
    )

    # Step 04: consume Step 03 output.
    V = _oracle_thin_qr_orthonormal_basis(
        W,
    )

    # Step 05: consume Steps 03, 04, and the original sketch.
    H = _oracle_form_regularized_core(
        W,
        V,
        Omega,
        eps,
    )

    # Step 06: consume Step 04 and Step 05 outputs.
    Delta_hat = _oracle_assemble_low_rank_correction(
        V,
        H,
    )

    ij = np.arange(1, m + 1)[:, None] * np.arange(1, k + 1)[None, :]
    P = np.sin(ij) / np.sqrt(float(m))
    Q = np.cos(ij) / np.sqrt(float(m))
    Y_s, Y_Ds = _oracle_solve_diagonal_and_exact_actions(E, C, P)
    Y_t, Y_Dt = _oracle_solve_diagonal_and_exact_actions(E, C, Q)
    W_s = _oracle_form_sample_matrix(C, Y_s, Y_Ds)
    W_t = _oracle_form_sample_matrix(C, Y_t, Y_Dt)
    D_s, D_t, D_st = _oracle_differentiate_correction_sketch(
        W, Omega, W_s, W_t, P, Q, eps
    )
    T = np.eye(m) + W @ W.T
    T_s = W_s @ W.T + W @ W_s.T
    T_t = W_t @ W.T + W @ W_t.T
    T_st = W_s @ W_t.T + W_t @ W_s.T
    projector_st = _oracle_mixed_generalized_projector_derivative(
        Delta_hat, T, D_s, T_s, D_t, T_t, D_st, T_st, r
    )
    return float(np.linalg.norm(projector_st, ord="fro"))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal scenario: exact benchmark ---
        {
            "setup": """seed, p, m, k, eps, r = 2026, 10, 80, 9, 1e-6, 3
""",
            "call": "full_pipeline_mixed_sketch_sensitivity(seed, p, m, k, eps, r)",
            "gold_call": "_oracle_full_pipeline_mixed_sketch_sensitivity(seed, p, m, k, eps, r)",
        },

        # --- Boundary: one-column sketch ---
        {
            "setup": """seed, p, m, k, eps, r = 1, 3, 7, 1, 0.1, 1
""",
            "call": "full_pipeline_mixed_sketch_sensitivity(seed, p, m, k, eps, r)",
            "gold_call": "_oracle_full_pipeline_mixed_sketch_sensitivity(seed, p, m, k, eps, r)",
        },

        # --- Edge: different undersampled configuration ---
        {
            "setup": """seed, p, m, k, eps, r = 7, 5, 50, 3, 1e-6, 1
""",
            "call": "full_pipeline_mixed_sketch_sensitivity(seed, p, m, k, eps, r)",
            "gold_call": "_oracle_full_pipeline_mixed_sketch_sensitivity(seed, p, m, k, eps, r)",
        },

        # --- Invalid: eps <= 0 ---
        {
            "setup": """seed, p, m, k, eps, r = 2026, 10, 80, 9, 0.0, 3

def run_model():
    try:
        full_pipeline_mixed_sketch_sensitivity(seed, p, m, k, eps, r)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _oracle_full_pipeline_mixed_sketch_sensitivity(seed, p, m, k, eps, r)
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
