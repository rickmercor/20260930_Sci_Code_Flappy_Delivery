"""
Impose the two fixed terminal potentials on the superposition of Bloch states and extract the two boundary coefficients belonging to the non-decaying state.

The potential in the branch is a superposition of all the Bloch states, each weighted by an unknown amplitude, and the terminal conditions fix those amplitudes. Holding a point of the branch at a constant potential is a much stronger requirement than fixing one number, because the superposition evaluated at a fixed position is still a periodic function of time through the envelopes: demanding that it be constant forces every nonzero harmonic of the combination to cancel there and the zeroth harmonic alone to take the imposed value. Two such points therefore supply exactly as many equations as there are states. Written out, the system has one block of rows per terminal, each row belonging to one harmonic index and each column to one state, with the entry given by that state's envelope coefficient multiplied by its plane-wave factor evaluated at the terminal. The right-hand side is zero everywhere except in the two rows belonging to the zeroth harmonic, where it carries the two imposed potentials. Because it is that sparse, the amplitude of any single state is a fixed combination of the two imposed potentials with coefficients read from the row of the inverse belonging to that state. Only the state that neither grows nor decays is needed downstream, so only its row is retained.




The decay constants have real parts that grow with the harmonic index, so at a realistic truncation order the plane-wave factors evaluated across the branch span tens of orders of magnitude. The resulting raw condition number is therefore dominated by unequal column scales and is not, by itself, evidence that the boundary coefficients are inaccurate: for the benchmark, direct inversion of the raw matrix recovers the needed row to nearly machine precision. Exact column equilibration remains a useful optional stabilization. Dividing each column by its largest entry and restoring that factor when the inverse row is read reduces the condition number to a small value without changing the physical amplitudes.

Returns
-------
np.ndarray: complex array of shape (2) holding the two boundary coefficients of the non-decaying Bloch state, ordered as the higher-potential terminal first and the lower-potential terminal second.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_boundary_coefficients(alphas: "np.ndarray",
                                  envelopes: "np.ndarray",
                                  position_high: float,
                                  position_low: float) -> "np.ndarray":
    """Extract the boundary coefficients of the non-decaying Bloch state.

    Parameters
    ----------
    alphas : np.ndarray
        Complex spatial decay constants of shape (4 * order + 2).
    envelopes : np.ndarray
        Complex array of shape (2 * order + 1, 4 * order + 2) whose column k
        holds the envelope coefficients of the state with decay constant
        alphas[k], indexed by harmonic from -order to +order.
    position_high : float
        Position in metres at which the higher potential is imposed.
    position_low : float
        Position in metres at which the lower potential is imposed.

    Returns
    -------
    coefficients : np.ndarray
        Complex array of shape (2). The amplitude of the state whose decay
        constant is smallest in magnitude equals coefficients[0] times the
        potential imposed at position_high plus coefficients[1] times the
        potential imposed at position_low.

    Raises
    ------
    ValueError
        If alphas is not one-dimensional, if envelopes is not two-dimensional
        with an odd number of rows >= 3, if alphas and envelopes do not supply
        4 * order + 2 states, if any entry of either is not finite, if either
        terminal position is not a finite number, if the two terminals sit at
        the same position, if some envelope is identically zero, or if the
        stacked boundary matrix is singular.
    """
    return coefficients  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_compute_boundary_coefficients(alphas: "np.ndarray",
                                          envelopes: "np.ndarray",
                                          position_high: float,
                                          position_low: float) -> "np.ndarray":
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    roots = np.asarray(alphas, dtype=complex)
    modes = np.asarray(envelopes, dtype=complex)
    if roots.ndim != 1:
        raise ValueError("alphas must be a one-dimensional array")
    if modes.ndim != 2 or modes.shape[0] < 3 or modes.shape[0] % 2 == 0:
        raise ValueError("envelopes must be 2D with an odd number of rows >= 3")
    order = (modes.shape[0] - 1) // 2
    if roots.size != 4 * order + 2 or modes.shape[1] != 4 * order + 2:
        raise ValueError("alphas and envelopes must supply 4 * order + 2 states")
    if not (np.all(np.isfinite(roots)) and np.all(np.isfinite(modes))):
        raise ValueError("alphas and envelopes must be finite")
    for name, value in (("position_high", position_high), ("position_low", position_low)):
        if not (isinstance(value, (int, float, np.floating, np.integer))
                and np.isfinite(value)):
            raise ValueError(f"{name} must be a finite number")
    if float(position_high) == float(position_low):
        raise ValueError("the two terminals must sit at different positions")

    # Exact column equilibration: rescaling a column rescales the amplitude of
    # that state, so the factor is put back when the inverse is read off.
    shift = np.maximum(roots.real * float(position_high), roots.real * float(position_low))
    scale = np.max(np.abs(modes), axis=0)
    if np.any(scale == 0.0):
        raise ValueError("every envelope must have a nonzero coefficient")
    factor = np.exp(-shift) / scale
    matrix = np.vstack([np.exp(roots * float(position_high) - shift) * modes / scale,
                        np.exp(roots * float(position_low) - shift) * modes / scale])
    try:
        inverse = np.linalg.inv(matrix)
    except np.linalg.LinAlgError as exc:
        raise ValueError("the boundary matrix must be invertible") from exc

    star = int(np.argmin(np.abs(roots)))
    return factor[star] * np.array([inverse[star, order], inverse[star, 3 * order + 1]],
                                   dtype=complex)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: a generic state set with one non-decaying member ---
        {
            "setup": """import numpy as np
order = 2
rng = np.random.default_rng(7)
count = 4 * order + 2
alphas = np.array([-6.0 - 2.0j, -6.0 + 2.0j, -3.5 - 1.0j, -3.5 + 1.0j, -0.4 + 0.0j,
                   0.0 + 0.0j, 2.8 - 0.7j, 2.8 + 0.7j, 5.1 - 3.3j, 5.1 + 3.3j])
envelopes = 0.3 * (rng.normal(size=(2 * order + 1, count)) + 1j * rng.normal(size=(2 * order + 1, count)))
envelopes[order, :] = 1.0
position_high = 0.0
position_low = 1.0

def digest(values):
    grid = np.asarray(values, dtype=complex)
    flat = 1.0e6 * grid.reshape(-1)
    rank = np.arange(1.0, flat.size + 1.0)
    weight = rank / rank.sum()
    root = np.sqrt(rank) / np.sqrt(rank).sum()
    return float(flat.size + 3.0 * grid.shape[0]
                 + np.sqrt(np.sum(weight * np.abs(flat) ** 2))
                 + 0.5 * np.sum(weight * flat.real) + 0.25 * np.sum(root * flat.imag))
""",
            "call": "digest(compute_boundary_coefficients(alphas, envelopes, position_high, position_low))",
            "gold_call": "digest(_oracle_compute_boundary_coefficients(alphas, envelopes, position_high, position_low))",
        },
        # --- Valid: terminals placed away from the origin and in reversed order ---
        {
            "setup": """import numpy as np
order = 3
rng = np.random.default_rng(11)
count = 4 * order + 2
alphas = np.concatenate([np.array([0.0 + 0.0j]),
                         np.linspace(-7.0, 7.0, count - 1) + 1j * np.linspace(-4.0, 4.0, count - 1)])
envelopes = 0.25 * (rng.normal(size=(2 * order + 1, count)) + 1j * rng.normal(size=(2 * order + 1, count)))
envelopes[order, :] = 1.0
position_high = 1.7
position_low = 0.2

def digest(values):
    grid = np.asarray(values, dtype=complex)
    flat = 1.0e6 * grid.reshape(-1)
    rank = np.arange(1.0, flat.size + 1.0)
    weight = rank / rank.sum()
    root = np.sqrt(rank) / np.sqrt(rank).sum()
    return float(flat.size + 3.0 * grid.shape[0]
                 + np.sqrt(np.sum(weight * np.abs(flat) ** 2))
                 + 0.5 * np.sum(weight * flat.real) + 0.25 * np.sum(root * flat.imag))
""",
            "call": "digest(compute_boundary_coefficients(alphas, envelopes, position_high, position_low))",
            "gold_call": "digest(_oracle_compute_boundary_coefficients(alphas, envelopes, position_high, position_low))",
        },
        # --- Boundary: decay constants large enough to overflow an unscaled matrix ---
        {
            "setup": """import numpy as np
order = 2
rng = np.random.default_rng(3)
count = 4 * order + 2
alphas = np.array([-400.0 - 5.0j, -300.0 + 5.0j, -120.0 - 2.0j, -60.0 + 2.0j, -0.9 + 0.0j,
                   0.0 + 0.0j, 55.0 - 1.0j, 130.0 + 1.0j, 290.0 - 6.0j, 420.0 + 6.0j])
envelopes = 0.2 * (rng.normal(size=(2 * order + 1, count)) + 1j * rng.normal(size=(2 * order + 1, count)))
envelopes[order, :] = 1.0
position_high = 0.0
position_low = 1.0

def digest(values):
    grid = np.asarray(values, dtype=complex)
    flat = 1.0e6 * grid.reshape(-1)
    rank = np.arange(1.0, flat.size + 1.0)
    weight = rank / rank.sum()
    root = np.sqrt(rank) / np.sqrt(rank).sum()
    return float(flat.size + 3.0 * grid.shape[0]
                 + np.sqrt(np.sum(weight * np.abs(flat) ** 2))
                 + 0.5 * np.sum(weight * flat.real) + 0.25 * np.sum(root * flat.imag))
""",
            "call": "digest(compute_boundary_coefficients(alphas, envelopes, position_high, position_low))",
            "gold_call": "digest(_oracle_compute_boundary_coefficients(alphas, envelopes, position_high, position_low))",
        },
        # --- Edge: smallest truncation, six states in total ---
        {
            "setup": """import numpy as np
order = 1
alphas = np.array([-4.0 + 0.0j, -1.5 - 2.0j, 0.0 + 0.0j, 1.1 + 0.0j, 2.6 - 1.0j, 3.9 + 1.0j])
envelopes = np.array([[0.4, -0.2j, 0.15, 0.3, -0.05, 0.22],
                      [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                      [-0.1, 0.35, 0.05j, -0.25, 0.4, 0.12]], dtype=complex)
position_high = 0.0
position_low = 0.5

def digest(values):
    grid = np.asarray(values, dtype=complex)
    flat = 1.0e6 * grid.reshape(-1)
    rank = np.arange(1.0, flat.size + 1.0)
    weight = rank / rank.sum()
    root = np.sqrt(rank) / np.sqrt(rank).sum()
    return float(flat.size + 3.0 * grid.shape[0]
                 + np.sqrt(np.sum(weight * np.abs(flat) ** 2))
                 + 0.5 * np.sum(weight * flat.real) + 0.25 * np.sum(root * flat.imag))
""",
            "call": "digest(compute_boundary_coefficients(alphas, envelopes, position_high, position_low))",
            "gold_call": "digest(_oracle_compute_boundary_coefficients(alphas, envelopes, position_high, position_low))",
        },
        # --- Invalid: state count inconsistent with the truncation order ---
        {
            "setup": """import numpy as np
alphas = np.zeros(8, dtype=complex)
envelopes = np.ones((5, 8), dtype=complex)
position_high = 0.0
position_low = 1.0
def run_model():
    try:
        compute_boundary_coefficients(alphas, envelopes, position_high, position_low)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_boundary_coefficients(alphas, envelopes, position_high, position_low)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: both terminals at the same position ---
        {
            "setup": """import numpy as np
order = 1
alphas = np.array([-4.0 + 0.0j, -1.5 - 2.0j, 0.0 + 0.0j, 1.1 + 0.0j, 2.6 - 1.0j, 3.9 + 1.0j])
envelopes = np.ones((3, 6), dtype=complex)
position_high = 0.75
position_low = 0.75
def run_model():
    try:
        compute_boundary_coefficients(alphas, envelopes, position_high, position_low)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_boundary_coefficients(alphas, envelopes, position_high, position_low)
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
