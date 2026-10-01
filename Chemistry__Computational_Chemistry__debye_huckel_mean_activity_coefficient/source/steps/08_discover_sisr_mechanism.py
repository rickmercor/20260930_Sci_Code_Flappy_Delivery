"""
Run the SISR-style mechanism discovery pipeline and return the selected candidate.

Run the SISR-style mechanism discovery pipeline and return the selected candidate.

Returns
-------
One-based integer index of the selected candidate mechanism.
"""

import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def discover_sisr_mechanism(
    times: np.ndarray,
    concentrations: np.ndarray,
    candidate_reactions: list,
) -> int:
    """Run the SISR-style pipeline and select the final mechanism.

    Parameters
    ----------
    times : np.ndarray
        Strictly increasing sampling times.
    concentrations : np.ndarray
        Observed concentrations with shape (n_times, n_species).
    candidate_reactions : list
        Candidate mechanisms in [reactants | products] form.

    Returns
    -------
    int
        One-based index of the selected candidate.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_discover_sisr_mechanism(
    times,
    concentrations,
    candidate_reactions,
    tolerance=0.10,
):
    import numpy as np

    t = np.asarray(times, dtype=float)
    c = np.asarray(concentrations, dtype=float)

    if t.ndim != 1 or t.size < 3 or c.ndim != 2 or c.shape[0] != t.size:
        raise ValueError("times and concentrations have incompatible shapes")
    if np.any(np.diff(t) <= 0.0) or np.any(c < 0.0):
        raise ValueError("times must increase and concentrations must be nonnegative")
    if tolerance < 0.0:
        raise ValueError("tolerance must be nonnegative")

    dc = _oracle_estimate_derivatives(t, c)
    dscale = np.maximum(np.max(np.abs(dc), axis=0), 1e-12)
    cscale = np.maximum(np.max(np.abs(c), axis=0), 1e-12)

    derivative_losses = []
    concentration_losses = []
    complexities = []

    for mechanism in candidate_reactions:
        r = np.asarray(mechanism, dtype=float)
        n_species = c.shape[1]
        if r.ndim != 2 or r.shape[1] != 2 * n_species:
            raise ValueError("each mechanism must use [reactants | products] columns")
        try:
            features = _oracle_mass_action_features(c, r)
            design = _oracle_stoichiometric_design(features, r, dscale)
            fit = _oracle_fit_rate_constants(design, dc, dscale)
            rates = np.asarray(fit[:-1], dtype=float)
            derivative_losses.append(float(fit[-1]))
            concentration_losses.append(
                float(_oracle_concentration_loss(t, c[0], c, r, rates, cscale))
            )
            complexities.append(float(_oracle_expression_complexity(r)))
        except Exception:
            derivative_losses.append(np.inf)
            concentration_losses.append(np.inf)
            complexities.append(np.inf)

    dloss = np.asarray(derivative_losses)
    closs = np.asarray(concentration_losses)
    comp = np.asarray(complexities)

    best = float(np.min(closs))
    eligible = closs <= best * (1.0 + tolerance) + 1e-12
    pareto = np.asarray(_oracle_pareto_front(closs, comp), dtype=bool)
    candidates = np.flatnonzero(eligible & pareto)
    if candidates.size == 0:
        candidates = np.array([int(np.argmin(closs))])

    chosen = sorted(
        candidates.tolist(),
        key=lambda i: (comp[i], closs[i], dloss[i], i),
    )[0]
    return int(chosen + 1)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    base_setup = """import numpy as np
times = np.array([
    0.00, 0.07, 0.16, 0.31, 0.52, 0.80, 1.15, 1.58,
    2.10, 2.75, 3.55, 4.55, 5.80, 7.40, 9.40, 12.00,
])
concentrations = np.array([
    [1.20000000, 0.08000000, 0.03000000, 0.00000000],
    [1.19680873, 0.08504694, 0.03235152, 0.00034691],
    [1.17638285, 0.09311654, 0.03517167, 0.00083757],
    [1.16569019, 0.10653083, 0.04093369, 0.00174251],
    [1.12434706, 0.12663981, 0.05090539, 0.00329966],
    [1.08242928, 0.15862843, 0.06644897, 0.00592546], 
    [0.99813266, 0.20541817, 0.09329393, 0.01029542],
    [0.89449655, 0.26208953, 0.13680727, 0.01812513],
    [0.74625165, 0.32244038, 0.20655167, 0.03241652],
    [0.58036819, 0.35342079, 0.31732050, 0.05953419],
    [0.42378397, 0.31233118, 0.46467954, 0.10945823],
    [0.31666019, 0.20448149, 0.59276695, 0.19600106],
    [0.25529348, 0.09801058, 0.63728962, 0.32122890],
    [0.22857335, 0.03643595, 0.56993106, 0.47468379],
    [0.21591196, 0.01192926, 0.44379018, 0.63892659],
    [0.21277023, 0.00358169, 0.30327429, 0.78970170],
])
reaction_library = np.array([
    [1,1,0,0, 0,2,0,0],
    [0,1,0,0, 0,0,1,0],
    [0,1,1,0, 0,0,2,0],
    [0,0,1,0, 0,0,0,1],
    [1,0,0,0, 0,1,0,0],
    [1,1,0,0, 0,1,1,0],
    [0,1,0,0, 0,0,0,1],
    [1,0,0,0, 0,0,0,1],
    [0,0,2,0, 0,0,1,1],
    [0,1,0,1, 0,0,1,0],
])
candidate_reactions = [
    reaction_library[[0,1,3]],
    reaction_library[[0,1,2,3]],
    reaction_library[[0,1,2,3,9]],
    reaction_library[[4,1,2,3]],
    reaction_library[[5,1,3]],
    reaction_library[[0,2,3]],
    reaction_library[[0,1,2,8]],
    reaction_library[[0,1,2,3,6]],
]"""

    reordered_setup = base_setup + """
candidate_reactions = [
    candidate_reactions[index]
    for index in [2,5,1,0,6,7,3,4]
]"""
    fallback_setup = base_setup + """
candidate_reactions = [
    candidate_reactions[0],
    candidate_reactions[1],
]"""

    call = "discover_sisr_mechanism(times, concentrations, candidate_reactions)"
    gold_call = (
        "_oracle_discover_sisr_mechanism("
        "times, concentrations, candidate_reactions)"
    )
    return [
        {
            "setup": base_setup,
            "call": call,
            "gold_call": gold_call,
        },
        {
            "setup": reordered_setup,
            "call": call,
            "gold_call": gold_call,
        },
        {
            "setup": fallback_setup,
            "call": call,
            "gold_call": gold_call,
        },
    ]
