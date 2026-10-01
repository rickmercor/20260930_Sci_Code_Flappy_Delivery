"""
Step 4: Assemble the level-3 certifiable LTI generating operator q3.

On three sites, only h_i^2 and the anticommutator {h_i, h_{i+1}} fit inside the truncation [H^2]_3; products with disjoint support are positive semidefinite and may be dropped. The generator is q3 = x1^2 + {x1, x2} - delta x1 + I (x) Z - Z (x) I, with Z = P Y P, and the last two terms telescope to zero when summed over translations.

Returns
-------
# np.ndarray, the (27,27) complex Hermitian generator q3
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

# =============================================================================
# FUNCTION SIGNATURE
# =============================================================================

def level3_generator(
    x: np.ndarray,
    p2_perp: np.ndarray,
    delta: float,
    Y: np.ndarray,
) -> np.ndarray:
    """Construct the 27x27 level-3 certifiable LTI generator q3(delta, Y).

    Parameters
    ----------
    x : np.ndarray
        Hermitian local interaction of shape (9, 9).
    p2_perp : np.ndarray
        Hermitian orthogonal projector of shape (9, 9).
    delta : float
        Finite candidate gap bound.
    Y : np.ndarray
        Hermitian matrix of shape (9, 9).

    Returns
    -------
    q3 : np.ndarray
        Hermitian array of shape (27, 27).

    Raises
    ------
    ValueError
        If x, p2_perp, or Y is not Hermitian of shape (9, 9),
        if p2_perp is not idempotent, or if delta is not finite.
    """
    return q3

# =============================================================================
# GOLD SOLUTION
# =============================================================================

# =============================================================================
# ORACLE SOLUTION
# =============================================================================
import numpy as np

def _oracle_level3_generator(
    x: np.ndarray,
    p2_perp: np.ndarray,
    delta: float,
    Y: np.ndarray,
) -> np.ndarray:
    import numpy as np

    x = np.asarray(x, dtype=complex)
    P = np.asarray(p2_perp, dtype=complex)
    Y = np.asarray(Y, dtype=complex)

    if x.shape != (9, 9) or not np.allclose(
        x, x.conj().T, atol=1e-10, rtol=0.0
    ):
        raise ValueError("x must be Hermitian with shape (9,9)")

    if (
        P.shape != (9, 9)
        or not np.allclose(P, P.conj().T, atol=1e-10, rtol=0.0)
        or not np.allclose(P @ P, P, atol=1e-8, rtol=0.0)
    ):
        raise ValueError(
            "p2_perp must be a Hermitian projector with shape (9,9)"
        )

    if Y.shape != (9, 9) or not np.allclose(
        Y, Y.conj().T, atol=1e-10, rtol=0.0
    ):
        raise ValueError("Y must be Hermitian with shape (9,9)")

    delta = float(delta)

    if not np.isfinite(delta):
        raise ValueError("delta must be finite")

    I3 = np.eye(3, dtype=complex)

    x1 = np.kron(x, I3)
    x2 = np.kron(I3, x)

    Z = P @ Y @ P

    q = (
        x1 @ x1
        + x1 @ x2
        + x2 @ x1
        - delta * x1
        + np.kron(I3, Z)
        - np.kron(Z, I3)
    )

    return 0.5 * (q + q.conj().T)

# =============================================================================
# TEST CASES
# =============================================================================

# =============================================================================
# TEST CASES
# =============================================================================
def test_cases():
    return [
        {
            "setup": (
                "import numpy as np\n"
                "h,_=_oracle_build_local_interaction(0.347,0.783); "
                "P,_=_oracle_excited_subspace_projector(h); "
                "Y=np.diag(np.arange(9.)).astype(complex)"
            ),
            "call": "level3_generator(h,P,0.3,Y)",
            "gold_call": "_oracle_level3_generator(h,P,0.3,Y)",
        },

        # boundary: Y = 0, delta = 0 -> [H^2]_3 restricted to three sites
        {
            "setup": (
                "import numpy as np\n"
                "h,_=_oracle_build_local_interaction(0.347,0.783); "
                "P,_=_oracle_excited_subspace_projector(h); "
                "Y=np.zeros((9,9),complex)"
            ),
            "call": "level3_generator(h,P,0.0,Y)",
            "gold_call": "_oracle_level3_generator(h,P,0.0,Y)",
        },

        # edge: x = projector itself, Y = identity (Z = P), negative delta
        {
            "setup": (
                "import numpy as np\n"
                "h,_=_oracle_build_local_interaction(1.0,1.0); "
                "P,_=_oracle_excited_subspace_projector(h); "
                "Y=np.eye(9,dtype=complex)"
            ),
            "call": "level3_generator(P,P,-0.5,Y)",
            "gold_call": "_oracle_level3_generator(P,P,-0.5,Y)",
        },
    ]
