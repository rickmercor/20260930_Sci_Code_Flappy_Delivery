"""
Continue the energy estimate with the determinant form of the moment-Lanczos recurrence through at most $max_order$. Use the task conventions $L_-1 = 1$, $L_0 = 1$, $M_-1 = 0$, and $M_0 = mu_1$; after each tridiagonal energy is formed, terminate before accepting a strict energy increase. After an accepted order below $max_order$, evaluate the next squared off-diagonal coefficient before attempting another order and terminate if it is negative or exactly zero. Do not evaluate a next off-diagonal candidate after accepting $max_order$. Return stop code 0 for completion, 1 for a negative square, 2 for an energy increase, and 3 for exact zero termination.

A finite moment sequence determines the Jacobi matrix of the associated spectral measure through ratios of Hankel-type determinants. Its lowest Ritz value extends the low-order Krylov estimate without additional quantum evolution. Noisy high moments can violate positivity and make a nominal squared off-diagonal coefficient negative; the physically motivated mitigation rule stops at the last accepted Ritz value rather than taking a square root of that unphysical number.

Returns
-------
tuple (final_energy, accepted_order, trial_energies, beta_squared, alpha_values, stop_code), using native float/int plus float64 arrays in chronological order
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import math

from numbers import Integral, Real

import numpy as np

def run_moment_lanczos(
    moments: np.ndarray,
    max_order: int,
    energy_shift: float,
) -> tuple[float, int, np.ndarray, np.ndarray, np.ndarray, int]:
    """Run determinant-based moment-Lanczos energy mitigation.

    Parameters
    ----------
    moments : np.ndarray
        Real finite sequence ``[mu_0, ..., mu_{2 * max_order}]`` with
        ``mu_0 = 1``.
    max_order : int
        Positive maximum tridiagonal order.
    energy_shift : float
        Finite scalar added to every shifted-Hamiltonian Ritz energy.

    Returns
    -------
    final_energy : float
        Last accepted lowest Ritz energy after adding ``energy_shift``.
    accepted_order : int
        Order of the last accepted tridiagonal matrix.
    trial_energies : np.ndarray
        One-dimensional float64 original-energy trials in chronological order, including an
        energy-increase trial if that trial triggers stop code 2.
    beta_squared : np.ndarray
        One-dimensional float64 squared off-diagonal candidates evaluated after
        accepted orders below ``max_order``, before attempting another order.
    alpha_values : np.ndarray
        One-dimensional float64 diagonal recurrence values in chronological order.
    stop_code : int
        ``0`` completed, ``1`` negative beta square, ``2`` strict energy
        increase, or ``3`` exact zero beta square.

    Raises
    ------
    ValueError
        If inputs are invalid, moments are insufficient, or a determinant
        denominator becomes singular before a defined termination branch.
    """
    return (
        float("nan"),
        0,
        np.empty(0, dtype=np.float64),
        np.empty(0, dtype=np.float64),
        np.empty(0, dtype=np.float64),
        0,
    )

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_run_moment_lanczos(
    moments: np.ndarray,
    max_order: int,
    energy_shift: float,
) -> tuple[float, int, np.ndarray, np.ndarray, np.ndarray, int]:
    import math
    from numbers import Integral, Real

    import numpy as np

    if isinstance(max_order, bool) or not isinstance(max_order, Integral) or int(max_order) < 1:
        raise ValueError("max_order must be an integer >= 1")
    if isinstance(energy_shift, bool) or not isinstance(energy_shift, Real):
        raise ValueError("energy_shift must be finite")
    shift = float(energy_shift)
    if not math.isfinite(shift):
        raise ValueError("energy_shift must be finite")

    raw = np.asarray(moments)
    if raw.ndim != 1:
        raise ValueError("moments must be one-dimensional")
    if np.iscomplexobj(raw) and np.any(np.abs(np.imag(raw)) > 0.0):
        raise ValueError("moments must be real")
    try:
        mu = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("moments must be real numeric values") from exc
    order_limit = int(max_order)
    if mu.size < 2 * order_limit + 1:
        raise ValueError("moments must include mu_0 through mu_(2 * max_order)")
    if not np.all(np.isfinite(mu)):
        raise ValueError("moments must be finite")
    if abs(float(mu[0]) - 1.0) > 1.0e-12:
        raise ValueError("mu_0 must equal 1 within 1e-12")

    l_cache = {-1: 1.0, 0: 1.0}
    m_cache = {-1: 0.0, 0: float(mu[1])}

    def l_determinant(index: int) -> float:
        if index in l_cache:
            return l_cache[index]
        matrix = np.array(
            [[mu[row + column] for column in range(index + 1)] for row in range(index + 1)],
            dtype=np.float64,
        )
        value = float(np.linalg.det(matrix))
        l_cache[index] = value
        return value

    def m_determinant(index: int) -> float:
        if index in m_cache:
            return m_cache[index]
        exponents = list(range(index)) + [index + 1]
        matrix = np.array(
            [[mu[row + exponent] for exponent in exponents] for row in range(index + 1)],
            dtype=np.float64,
        )
        value = float(np.linalg.det(matrix))
        m_cache[index] = value
        return value

    alpha_values = []
    beta_values = []
    beta_squared = []
    trial_energies = []
    accepted_order = 0
    previous_shifted_energy = None

    for order in range(1, order_limit + 1):
        l_previous = l_determinant(order - 1)
        l_before_previous = l_determinant(order - 2)
        if l_previous == 0.0 or l_before_previous == 0.0:
            raise ValueError("a determinant denominator is zero before a defined stop")
        alpha = m_determinant(order - 1) / l_previous - m_determinant(order - 2) / l_before_previous
        if not math.isfinite(alpha):
            raise ValueError("a Lanczos diagonal coefficient is not finite")
        alpha_values.append(float(alpha))

        tridiagonal = np.diag(np.asarray(alpha_values, dtype=np.float64))
        for index, beta in enumerate(beta_values):
            tridiagonal[index, index + 1] = beta
            tridiagonal[index + 1, index] = beta
        shifted_energy = float(np.linalg.eigvalsh(tridiagonal)[0])
        trial_energy = shifted_energy + shift
        trial_energies.append(trial_energy)

        if previous_shifted_energy is not None and shifted_energy > previous_shifted_energy:
            return (
                float(previous_shifted_energy + shift),
                int(accepted_order),
                np.asarray(trial_energies, dtype=np.float64),
                np.asarray(beta_squared, dtype=np.float64),
                np.asarray(alpha_values, dtype=np.float64),
                2,
            )

        previous_shifted_energy = shifted_energy
        accepted_order = order
        if order < order_limit:
            denominator = l_previous * l_previous
            if denominator == 0.0 or not math.isfinite(denominator):
                raise ValueError("a squared determinant denominator is singular before a defined stop")
            beta_square = l_determinant(order) * l_before_previous / denominator
            if not math.isfinite(beta_square):
                raise ValueError("a squared Lanczos off-diagonal coefficient is not finite")
            beta_squared.append(float(beta_square))
            if beta_square < 0.0:
                return (
                    float(shifted_energy + shift),
                    int(accepted_order),
                    np.asarray(trial_energies, dtype=np.float64),
                    np.asarray(beta_squared, dtype=np.float64),
                    np.asarray(alpha_values, dtype=np.float64),
                    1,
                )
            if beta_square == 0.0:
                return (
                    float(shifted_energy + shift),
                    int(accepted_order),
                    np.asarray(trial_energies, dtype=np.float64),
                    np.asarray(beta_squared, dtype=np.float64),
                    np.asarray(alpha_values, dtype=np.float64),
                    3,
                )
            beta_values.append(math.sqrt(beta_square))

    return (
        float(previous_shifted_energy + shift),
        int(accepted_order),
        np.asarray(trial_energies, dtype=np.float64),
        np.asarray(beta_squared, dtype=np.float64),
        np.asarray(alpha_values, dtype=np.float64),
        0,
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
moments = np.array([1.0, -1.2156817677625402, 1.7607524182062582, -2.4630089113925990, 3.3990312749133818, -5.8075109027773860, -53.355048766002540])

def pack_result(result):
    final_energy, accepted_order, trial_energies, beta_squared, alpha_values, stop_code = result
    return np.concatenate([
        np.asarray([
            final_energy, accepted_order, stop_code,
            trial_energies.ndim, trial_energies.size,
            beta_squared.ndim, beta_squared.size, alpha_values.ndim, alpha_values.size,
        ], dtype=float),
        trial_energies.ravel(), beta_squared.ravel(), alpha_values.ravel(),
    ])
""",
            "call": "pack_result(run_moment_lanczos(moments, 3, 0.3))",
            "gold_call": "pack_result(_oracle_run_moment_lanczos(moments, 3, 0.3))",
        },
        {
            "setup": """import numpy as np
energies = np.array([-1.3, -0.1, 0.8, 1.6])
weights = np.array([0.2, 0.3, 0.1, 0.4])
moments = np.array([np.sum(weights * energies**q) for q in range(7)], dtype=float)

def pack_result(result):
    final_energy, accepted_order, trial_energies, beta_squared, alpha_values, stop_code = result
    return np.concatenate([
        np.asarray([
            final_energy, accepted_order, stop_code,
            trial_energies.ndim, trial_energies.size,
            beta_squared.ndim, beta_squared.size, alpha_values.ndim, alpha_values.size,
        ], dtype=float),
        trial_energies.ravel(), beta_squared.ravel(), alpha_values.ravel(),
    ])
""",
            "call": "pack_result(run_moment_lanczos(moments, 3, 0.0))",
            "gold_call": "pack_result(_oracle_run_moment_lanczos(moments, 3, 0.0))",
        },
        {
            "setup": """import numpy as np
moments = np.array([1.0, -0.4, 0.9], dtype=float)

def pack_result(result):
    final_energy, accepted_order, trial_energies, beta_squared, alpha_values, stop_code = result
    return np.concatenate([
        np.asarray([
            final_energy, accepted_order, stop_code,
            trial_energies.ndim, trial_energies.size,
            beta_squared.ndim, beta_squared.size, alpha_values.ndim, alpha_values.size,
        ], dtype=float),
        trial_energies.ravel(), beta_squared.ravel(), alpha_values.ravel(),
    ])
""",
            "call": "pack_result(run_moment_lanczos(moments, 1, 2.0))",
            "gold_call": "pack_result(_oracle_run_moment_lanczos(moments, 1, 2.0))",
        },
        {
            "setup": """import numpy as np
energy = 0.5
moments = np.array([energy**q for q in range(7)], dtype=float)

def pack_result(result):
    final_energy, accepted_order, trial_energies, beta_squared, alpha_values, stop_code = result
    return np.concatenate([
        np.asarray([
            final_energy, accepted_order, stop_code,
            trial_energies.ndim, trial_energies.size,
            beta_squared.ndim, beta_squared.size, alpha_values.ndim, alpha_values.size,
        ], dtype=float),
        trial_energies.ravel(), beta_squared.ravel(), alpha_values.ravel(),
    ])
""",
            "call": "pack_result(run_moment_lanczos(moments, 3, -0.1))",
            "gold_call": "pack_result(_oracle_run_moment_lanczos(moments, 3, -0.1))",
        },
        {
            "setup": """import numpy as np
moments = np.array([1.0, 0.0, 1.0, 0.0], dtype=float)
def run_model():
    try:
        run_moment_lanczos(moments, 2, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _oracle_run_moment_lanczos(moments, 2, 0.0)
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
moments = np.array([0.9, 0.0, 1.0], dtype=float)
def run_model():
    try:
        run_moment_lanczos(moments, 1, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _oracle_run_moment_lanczos(moments, 1, 0.0)
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
base = np.array([1.0, -1.2156817677625402, 1.7607524182062582, -2.4630089113925990, 3.3990312749133818, -5.8075109027773860, -53.355048766002540])
moments = np.concatenate([[1.0], base[1:] * 1.0e-300])
def run_model():
    try:
        run_moment_lanczos(moments, 3, 0.3)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _oracle_run_moment_lanczos(moments, 3, 0.3)
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
