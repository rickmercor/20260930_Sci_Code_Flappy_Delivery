"""
Form the effective Hamiltonian and norm quadratic forms of the lifted variational state.

If W maps updated orbital coefficients to the many-electron trial state, its energy numerator and norm denominator are W transpose H W and W transpose W. The returned order is Hamiltonian first, metric second.

Returns
-------
Floating array whose first plane is the effective Hamiltonian and second plane is the metric.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def form_eidos_pencil(
    lift_columns: np.ndarray,
    sector_hamiltonian: np.ndarray,
) -> np.ndarray:
    """Form the two effective matrices for one orbital update.

    Parameters
    ----------
    lift_columns : np.ndarray
        Many-electron lift matrix W.
    sector_hamiltonian : np.ndarray
        Symmetric Hamiltonian acting on W's row space.

    Returns
    -------
    np.ndarray
        Array [W.T@H@W, W.T@W] with shape
        (2, n_columns, n_columns).
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_form_eidos_pencil(
    lift_columns,
    sector_hamiltonian,
):
    """Reference effective quadratic forms."""
    import numpy as np

    lift = np.asarray(lift_columns, dtype=float)
    hamiltonian = np.asarray(
        sector_hamiltonian,
        dtype=float,
    )

    if (
        lift.ndim != 2
        or lift.shape[0] < 1
        or lift.shape[1] < 1
    ):
        raise ValueError(
            "lift_columns must be a nonempty matrix"
        )
    if hamiltonian.shape != (
        lift.shape[0],
        lift.shape[0],
    ):
        raise ValueError("Hamiltonian and lift do not align")
    if (
        not np.all(np.isfinite(lift))
        or not np.all(np.isfinite(hamiltonian))
    ):
        raise ValueError("inputs must be finite")
    if not np.allclose(
        hamiltonian,
        hamiltonian.T,
        rtol=0.0,
        atol=1e-11,
    ):
        raise ValueError("Hamiltonian must be symmetric")

    metric = lift.T @ lift
    effective_hamiltonian = (
        lift.T @ hamiltonian @ lift
    )

    return np.stack(
        [
            0.5
            * (
                effective_hamiltonian
                + effective_hamiltonian.T
            ),
            0.5 * (metric + metric.T),
        ]
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    call = (
        "form_eidos_pencil("
        "lift_columns, sector_hamiltonian)"
    )
    gold = (
        "_oracle_form_eidos_pencil("
        "lift_columns, sector_hamiltonian)"
    )

    return [
        {
            "setup": (
                "import numpy as np\n"
                "lift_columns=np.diag([1.,2.]); "
                "sector_hamiltonian=np.diag([3.,5.])"
            ),
            "call": call,
            "gold_call": gold,
        },
        {
            "setup": (
                "import numpy as np\n"
                "lift_columns=np.array([[1.,1.],[0.,0.]]); "
                "sector_hamiltonian=np.eye(2)"
            ),
            "call": call,
            "gold_call": gold,
        },
        {
            "setup": (
                "import numpy as np\n"
                "lift_columns=np.zeros((2,3)); "
                "sector_hamiltonian=np.eye(2)"
            ),
            "call": call,
            "gold_call": gold,
        },
    ]
