"""
Assemble the preceding numerical stages into the complete deterministic two-cycle computation.

The complete benchmark combines the operator construction, randomized embedding, adaptive Krylov computation, reduced matrix-function evaluation, and one restart correction into a single deterministic numerical pipeline.

Returns
-------
float, requested entry of f^[1] as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def sketch_and_restart_pipeline(
    n: int,
    D: float,
    sketch_seed: int,
    t: int,
    tau: float,
    s0: int,
    eta: float,
    mmax0: int,
    entry_index: int,
) -> float:
    """Run the complete two-cycle computation.

    Parameters
    ----------
    n : int
        Grid size.
    D : float
        Diffusion coefficient.
    sketch_seed : int
        Initial random seed.
    t : int
        Truncation parameter.
    tau : float
        Condition-number threshold.
    s0 : int
        Initial and incremental sketch size.
    eta : float
        Sketch-growth parameter.
    mmax0 : int
        Initial maximum Krylov dimension.
    entry_index : int
        Zero-based requested output index.

    Returns
    -------
    float
        Requested entry of the final approximation.
    """
    result = 0.0
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_sketch_and_restart_pipeline(
    n: int,
    D: float,
    sketch_seed: int,
    t: int,
    tau: float,
    s0: int,
    eta: float,
    mmax0: int,
    entry_index: int,
) -> float:
    import numpy as np

    if not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError("n must be positive")
    if float(D) <= 0.0:
        raise ValueError("D must be positive")
    if not isinstance(sketch_seed, (int, np.integer)):
        raise ValueError("sketch_seed must be an integer")
    if t < 1:
        raise ValueError("t must be >= 1")
    if tau <= 1.0:
        raise ValueError("tau must be > 1")
    if s0 < 1:
        raise ValueError("s0 must be positive")
    if eta <= 0.0:
        raise ValueError("eta must be positive")
    if mmax0 < 1:
        raise ValueError("mmax0 must be positive")

    N = n * n

    if not (isinstance(entry_index, (int, np.integer))
            and 0 <= entry_index < N):
        raise ValueError("entry_index out of range")

    state = {}

    # ---------------------------------------------------------------
    # STEP 01
    # ---------------------------------------------------------------
    _oracle_build_convdiff_operator(
        state,
        n,
        D,
    )

    # ---------------------------------------------------------------
    # STEP 02
    # ---------------------------------------------------------------
    _oracle_sparse_sign_sketch(
        state,
        sketch_seed,
        s0,
    )

    # ---------------------------------------------------------------
    # STEP 03: first cycle
    # ---------------------------------------------------------------
    _oracle_truncated_arnoldi_cycle(
        state,
        t,
        tau,
        s0,
        eta,
        mmax0,
    )

    first_m = int(state["cycle_Bm"].shape[1])

    # Preserve first-cycle state under explicit names.
    state["first_Bm"] = state["cycle_Bm"]
    state["first_Hm"] = state["cycle_Hm"]
    state["first_bmp1"] = state["cycle_bmp1"]
    state["first_hmp1_m"] = state["cycle_hmp1_m"]
    state["first_subdiag"] = state["cycle_subdiag"]
    state["first_beta"] = state["beta"]
    state["first_S"] = state["S"]

    # ---------------------------------------------------------------
    # STEP 04: first correction
    # ---------------------------------------------------------------
    _oracle_rank1_harmonic_update(state)

    state["first_corrected_Hm"] = state["corrected_Hm"]
    state["first_corrected_bmp1"] = state["corrected_bmp1"]
    state["first_corrected_hmp1_m"] = state["corrected_hmp1_m"]

    # ---------------------------------------------------------------
    # STEP 05: zero-restart approximation
    # ---------------------------------------------------------------
    _oracle_arnoldi_like_approx(state)

    state["first_f0"] = state["f0"]

    # ---------------------------------------------------------------
    # First-cycle restart data
    # ---------------------------------------------------------------
    theta0 = np.linalg.eigvals(state["first_corrected_Hm"])

    subdiag0 = state["first_subdiag"]

    if subdiag0.size:
        gamma0 = (
            float(np.prod(subdiag0))
            * float(state["first_corrected_hmp1_m"])
        )
    else:
        gamma0 = float(state["first_corrected_hmp1_m"])

    beta0 = float(state["first_beta"])

    state["theta0"] = theta0
    state["gamma0"] = gamma0
    state["beta0"] = beta0

    # ---------------------------------------------------------------
    # STEP 06
    #
    # Call the public/scientific subproblem for one deterministic
    # kernel evaluation. The full matrix application follows in Step 07.
    # ---------------------------------------------------------------
    z_probe = complex(theta0[0])

    _oracle_restart_error_scalar_kernel(
        theta0,
        gamma0,
        beta0,
        z_probe,
        1e-6,
        64,
        4,
    )

    # ---------------------------------------------------------------
    # STEP 03 again: second cycle.
    #
    # The corrected residual direction becomes the second starting
    # vector, and mmax is fixed to the first-cycle dimension.
    # ---------------------------------------------------------------
    second_state = {
        "A": state["A"],
        "b": state["first_corrected_bmp1"],
        "S": state["first_S"].copy(),
        "rng": state["rng"],
    }

    _oracle_truncated_arnoldi_cycle(
        second_state,
        t,
        tau,
        s0,
        eta,
        first_m,
    )

    state["second_Bm"] = second_state["cycle_Bm"]
    state["second_Hm"] = second_state["cycle_Hm"]
    state["second_bmp1"] = second_state["cycle_bmp1"]
    state["second_hmp1_m"] = second_state["cycle_hmp1_m"]
    state["second_subdiag"] = second_state["cycle_subdiag"]
    state["second_beta"] = second_state["beta"]
    state["second_S"] = second_state["S"]
    state["rng"] = second_state["rng"]

    # ---------------------------------------------------------------
    # STEP 04 again: second correction
    # ---------------------------------------------------------------
    update_state = {
        "A": state["A"],
        "S": state["second_S"],
        "cycle_Bm": state["second_Bm"],
        "cycle_Hm": state["second_Hm"],
        "cycle_bmp1": state["second_bmp1"],
        "cycle_hmp1_m": state["second_hmp1_m"],
    }

    _oracle_rank1_harmonic_update(update_state)

    state["second_corrected_Hm"] = update_state["corrected_Hm"]
    state["second_corrected_bmp1"] = update_state["corrected_bmp1"]
    state["second_corrected_hmp1_m"] = update_state["corrected_hmp1_m"]

    # ---------------------------------------------------------------
    # STEP 07
    # ---------------------------------------------------------------
    _oracle_error_kernel_matrix_apply(
        state,
        state["theta0"],
        state["gamma0"],
        state["beta0"],
        1e-6,
        64,
        4,
    )

    # ---------------------------------------------------------------
    # Final restart correction
    # ---------------------------------------------------------------
    H1 = state["second_corrected_Hm"]
    B1 = state["second_Bm"]
    beta1 = float(state["second_beta"])
    errH1 = state["errHm"]

    e1 = np.zeros(H1.shape[0], dtype=complex)
    e1[0] = 1.0

    g1 = B1 @ (errH1 @ (beta1 * e1))

    f0 = np.asarray(state["first_f0"], dtype=complex)
    f1 = f0 + g1

    value = np.real_if_close(f1[entry_index])

    if np.iscomplexobj(value):
        value = np.real(value)

    return float(value)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np
n = 7
D = 1e-3
sketch_seed = 2026
t = 1
tau = 1e4
s0 = 10
eta = 2.0
mmax0 = 12
entry_index = 24
""",
            "call": """sketch_and_restart_pipeline(
    n, D, sketch_seed, t, tau, s0, eta, mmax0, entry_index
)""",
            "gold_call": """_oracle_sketch_and_restart_pipeline(
    n, D, sketch_seed, t, tau, s0, eta, mmax0, entry_index
)""",
        },
        {
            "setup": """import numpy as np
n = 3
D = 1e-2
sketch_seed = 31415
t = 1
tau = 1e4
s0 = 6
eta = 2.0
mmax0 = 5
entry_index = 0
""",
            "call": """sketch_and_restart_pipeline(
    n, D, sketch_seed, t, tau, s0, eta, mmax0, entry_index
)""",
            "gold_call": """_oracle_sketch_and_restart_pipeline(
    n, D, sketch_seed, t, tau, s0, eta, mmax0, entry_index
)""",
        },
        {
            "setup": """import numpy as np
n = 4
D = 2e-3
sketch_seed = 2718
t = 2
tau = 1e4
s0 = 8
eta = 2.0
mmax0 = 8
entry_index = 7
""",
            "call": """sketch_and_restart_pipeline(
    n, D, sketch_seed, t, tau, s0, eta, mmax0, entry_index
)""",
            "gold_call": """_oracle_sketch_and_restart_pipeline(
    n, D, sketch_seed, t, tau, s0, eta, mmax0, entry_index
)""",
        },
    ]
