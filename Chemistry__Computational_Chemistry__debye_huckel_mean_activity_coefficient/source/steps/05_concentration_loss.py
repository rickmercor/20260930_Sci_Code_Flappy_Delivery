"""
Score a fitted mechanism by integrating its ODE against concentration data.

The derivative fit is only a screen; SISR also checks whether the fitted ODE reproduces concentration trajectories.

Returns
-------
Float mean scaled concentration-trajectory loss.
"""

import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def concentration_loss(
    times: np.ndarray,
    initial_concentrations: np.ndarray,
    observed_concentrations: np.ndarray,
    reactions: np.ndarray,
    rates: np.ndarray,
    concentration_scales: np.ndarray,
) -> float:
    """Score a fitted mechanism by integrating its ODE against concentration data.

    Parameters
    ----------
    times : np.ndarray
        Strictly increasing sampling times.
    initial_concentrations : np.ndarray
        Initial species concentrations.
    observed_concentrations : np.ndarray
        Observed concentrations with shape (n_times, n_species).
    reactions : np.ndarray
        Integer rows in [reactants | products] form.
    rates : np.ndarray
        Nonnegative rate constants, one per reaction.
    concentration_scales : np.ndarray
        Positive scale for each species.

    Returns
    -------
    float
        Mean scaled concentration-trajectory loss.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_concentration_loss(
    times: np.ndarray,
    initial_concentrations: np.ndarray,
    observed_concentrations: np.ndarray,
    reactions: np.ndarray,
    rates: np.ndarray,
    concentration_scales: np.ndarray,
) -> float:
    """Reference ODE trajectory loss."""
    import numpy as np
    from scipy.integrate import solve_ivp

    t = np.asarray(times, dtype=float)
    y0 = np.asarray(initial_concentrations, dtype=float)
    yobs = np.asarray(observed_concentrations, dtype=float)
    r = np.asarray(reactions, dtype=float)
    k = np.asarray(rates, dtype=float)
    scales = np.asarray(concentration_scales, dtype=float)

    if t.ndim != 1 or t.size < 2 or np.any(np.diff(t) <= 0.0):
        raise ValueError("times must be strictly increasing")
    if yobs.ndim != 2 or yobs.shape[0] != t.size:
        raise ValueError("observed_concentrations must align with times")

    n_species = yobs.shape[1]
    if y0.shape != (n_species,) or scales.shape != (n_species,):
        raise ValueError("initial concentrations and scales must match species")
    if r.ndim != 2 or r.shape[1] != 2 * n_species or k.shape != (r.shape[0],):
        raise ValueError("reaction and rate dimensions do not match")
    if np.any(y0 < 0.0) or np.any(yobs < 0.0) or np.any(k < 0.0) or np.any(scales <= 0.0):
        raise ValueError("concentrations, rates, and scales must be valid")

    reactants = r[:, :n_species].astype(int)
    stoich = r[:, n_species:] - r[:, :n_species]

    def rhs(_time, y):
        y = np.maximum(y, 0.0)
        phi = np.prod(y[None, :] ** reactants, axis=1)
        return (k * phi) @ stoich

    sol = solve_ivp(
        rhs,
        (float(t[0]), float(t[-1])),
        y0,
        t_eval=t,
        rtol=1e-8,
        atol=1e-10,
    )
    if not sol.success:
        raise RuntimeError("ODE integration failed")

    residual = (sol.y.T - yobs) / scales[None, :]
    return float(np.mean(np.sum(residual * residual, axis=1)))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np
times = np.array([0.0, 0.5, 1.0])
initial_concentrations = np.array([1.0, 0.0])
observed_concentrations = np.array([[1.0, 0.0], [0.60653066, 0.39346934], [0.36787944, 0.63212056]])
reactions = np.array([[1, 0, 0, 1]])
rates = np.array([1.0])
concentration_scales = np.array([1.0, 1.0])""",
            "call": "concentration_loss(times, initial_concentrations, observed_concentrations, reactions, rates, concentration_scales)",
            "gold_call": "_oracle_concentration_loss(times, initial_concentrations, observed_concentrations, reactions, rates, concentration_scales)",
        },
        {
            "setup": """import numpy as np
times = np.array([0.0, 0.25, 0.75])
initial_concentrations = np.array([2.0, 0.0])
observed_concentrations = np.array([[2.0, 0.0], [1.5576, 0.4424], [0.9447, 1.0553]])
reactions = np.array([[1, 0, 0, 1]])
rates = np.array([1.0])
concentration_scales = np.array([2.0, 2.0])""",
            "call": "concentration_loss(times, initial_concentrations, observed_concentrations, reactions, rates, concentration_scales)",
            "gold_call": "_oracle_concentration_loss(times, initial_concentrations, observed_concentrations, reactions, rates, concentration_scales)",
        },
        {
            "setup": """import numpy as np
times = np.array([0.0, 0.2, 0.4, 0.8])
initial_concentrations = np.array([1.0, 1.0, 0.0])
observed_concentrations = np.array([[1.0, 1.0, 0.0], [0.84, 0.84, 0.16], [0.72, 0.72, 0.28], [0.56, 0.56, 0.44]])
reactions = np.array([[1, 1, 0, 0, 0, 1]])
rates = np.array([0.9])
concentration_scales = np.array([1.0, 1.0, 1.0])""",
            "call": "concentration_loss(times, initial_concentrations, observed_concentrations, reactions, rates, concentration_scales)",
            "gold_call": "_oracle_concentration_loss(times, initial_concentrations, observed_concentrations, reactions, rates, concentration_scales)",
        },
    ]
