"""
Compute extraction and recombination yield responses from rate and population derivatives.

Extraction and molecular recombination compete for the injected population. Their fluxes divided by total generation define the charge and loss yields. Field derivatives include the responses of rates, populations and the generation normalization.

Returns
-------
ndarray (3,3): rows contain values, first derivatives and ordinary second derivatives; columns contain extraction, LE recombination and non-LE recombination yields.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def charge_yield_response(
    kinetics: "np.ndarray",
    population: "np.ndarray",
    labels: "np.ndarray",
) -> "np.ndarray":
    """Compute extraction and recombination yield responses.
    
    Parameters
    ----------
    kinetics : ndarray, shape (3, M, M+3)
        Finite value, first and second derivatives of outgoing transfers,
        recombination, extraction and generation, as in Step 4.
        Base entries are nonnegative and transfer diagonals are zero at
        every order. Sink and source derivatives may have either sign.
    population : ndarray, shape (3, M)
        Finite value, first and second derivatives of populations.
        Base populations are nonnegative. The input need not solve the
        microscopic balance; canonical-lifted populations are also valid.
    labels : ndarray, shape (M,)
        LE=0, CT=1, CS=2 labels. Individual pools may be empty here.
    
    Returns
    -------
    ndarray, shape (3, 3)
        Rows are value, first derivative and ordinary second derivative.
        Columns are extraction, LE recombination and non-LE recombination
        fluxes divided by total generation. Include derivatives of each
        flux product and of the generation denominator.
    
    Raises
    ------
    ValueError
        For invalid dimensions, finite values, signs or labels specified
        above, or nonpositive total base generation.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_charge_yield_response(
    kinetics: "np.ndarray",
    population: "np.ndarray",
    labels: "np.ndarray",
) -> "np.ndarray":
    import numpy as np

    data = _response_array(kinetics)
    m = data.shape[1]
    population, labels = _finite(population, "population"), _finite(
        labels, "labels"
    )
    if (
        population.shape != (3, m)
        or labels.shape != (m,)
        or np.any(population[0] < 0)
    ):
        raise ValueError(
            "invalid population shape or negative base population"
        )
    if not np.all(np.isin(labels, (0, 1, 2))):
        raise ValueError("labels must be 0/1/2")
    gen = data[:, :, -1].sum(axis=1)
    if gen[0] <= 0:
        raise ValueError("total generation at s=0 must be positive")
    flux = np.zeros((3, 3))
    for order in range(3):
        for k in range(order + 1):
            choose = 2 if order == 2 and k == 1 else 1
            rec = data[order - k, :, m] * population[k]
            flux[order] += choose * np.array(
                [
                    data[order - k, :, m + 1] @ population[k],
                    rec[labels == 0].sum(),
                    rec[labels != 0].sum(),
                ]
            )
    out = np.zeros((3, 3))
    out[0] = flux[0] / gen[0]
    out[1] = (flux[1] - gen[1] * out[0]) / gen[0]
    out[2] = (flux[2] - gen[2] * out[0] - 2 * gen[1] * out[1]) / gen[0]
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """
import numpy as np
from scipy.special import eval_genlaguerre, gammaln

positions = np.array(
    [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]]
)
homo = np.zeros(6)
lumo = np.full(6, 0.95)
binding = np.full(6, 0.24)
e, h = np.indices((6, 6))
disorder = (
    0.040 * np.cos(2 * np.pi * (e + 1) / 7)
    + 0.025 * np.sin(2 * np.pi * (h + 1) / 7)
    + 0.012 * np.sin((e + 1) * (h + 2))
)
field = np.array([0.045, 0.0])
direction = np.array([1.0, 0.0])
illumination = 1 + 0.2 * np.cos(np.arange(6) + 1)
parameters = np.array(
    [
        0.60,
        0.30,
        2.0,
        np.sqrt(2),
        1.0,
        0.40,
        0.35,
        0.004,
        0.003,
        0.002,
        0.025,
        0.105,
        0.032,
        0.090,
        0.060,
        310.0,
        12.0,
        40.0,
        0.020,
        0.001,
        10.0,
    ]
)
basis = _oracle_build_molecular_basis(
    positions.copy(),
    homo.copy(),
    lumo.copy(),
    binding.copy(),
    parameters[0],
    parameters[1],
    field.copy(),
    disorder.copy(),
    parameters[2],
    direction.copy(),
)
couplings = _oracle_build_couplings(positions.copy(), *parameters[3:10])
kinetics = _oracle_build_kinetics_response(
    basis.copy(),
    couplings.copy(),
    *parameters[10:16],
    int(parameters[16]),
    int(parameters[17]),
    parameters[18],
    parameters[19],
    parameters[3],
    parameters[20],
    illumination.copy()
)
population = _oracle_coupled_population_response(
    basis[:, [2, 5]].copy(), basis[:, 4].copy(),
    kinetics.copy(), parameters[15],
)[0]
""",
            "call": """
charge_yield_response(kinetics.copy(), population.copy(), basis[:, 4].copy())
""",
            "gold_call": """
_oracle_charge_yield_response(
    kinetics.copy(), population.copy(), basis[:, 4].copy()
)
""",
            "tol": 1e-06,
        },
        {
            "setup": """
import numpy as np
from scipy.special import eval_genlaguerre, gammaln

basis = np.zeros((4, 6))
basis[:, 2] = [0.1, 0.12, 0.15, 0.18]
basis[:, 5] = [-0.2, 0.7, 0.1, -0.3]
basis[:, 4] = [0, 0, 1, 2]
parameters = np.zeros(21)
parameters[15] = 315.0
kinetics = np.zeros((3, 4, 7))
kinetics[0, :, :4] = np.array(
    [
        [0.0, 2.0, 3.0, 1.0],
        [0.7, 0.0, 2.2, 4.0],
        [1.0, 0.2, 0.0, 3.0],
        [0.3, 2.0, 1.1, 0.0],
    ]
)
kinetics[0, :, 4:] = np.array(
    [[0.4, 0.2, 1.0], [0.7, 0.3, 2.0], [0.1, 0.9, 0.3], [0.2, 3.0, 0.7]]
)
kinetics[1] = kinetics[0] * np.array([0.1, -0.2, 0.3, -0.1])[:, None]
kinetics[2] = kinetics[0] * np.array([-0.2, 0.05, 0.1, 0.3])[:, None]
population = np.array(
    [[0.2, 0.3, 0.1, 0.4], [0.1, -0.2, 0.3, -0.4], [-0.5, 0.2, -0.1, 0.3]]
)
""",
            "call": """
charge_yield_response(kinetics.copy(), population.copy(), basis[:, 4].copy())
""",
            "gold_call": """
_oracle_charge_yield_response(
    kinetics.copy(), population.copy(), basis[:, 4].copy()
)
""",
            "tol": 1e-06,
        },
        {
            "setup": """
import numpy as np
from scipy.special import eval_genlaguerre, gammaln

basis = np.zeros((4, 6))
basis[:, 2] = [0.1, 0.12, 0.15, 0.18]
basis[:, 5] = [-0.2, 0.7, 0.1, -0.3]
basis[:, 4] = [0, 0, 1, 2]
parameters = np.zeros(21)
parameters[15] = 315.0
kinetics = np.zeros((3, 4, 7))
kinetics[0, :, :4] = np.array(
    [
        [0.0, 2.0, 3.0, 1.0],
        [0.7, 0.0, 2.2, 4.0],
        [1.0, 0.2, 0.0, 3.0],
        [0.3, 2.0, 1.1, 0.0],
    ]
)
kinetics[0, :, 4:] = np.array(
    [[0.4, 0.2, 1.0], [0.7, 0.3, 2.0], [0.1, 0.9, 0.3], [0.2, 3.0, 0.7]]
)
kinetics[1] = kinetics[0] * np.array([0.1, -0.2, 0.3, -0.1])[:, None]
kinetics[2] = kinetics[0] * np.array([-0.2, 0.05, 0.1, 0.3])[:, None]
population = np.array(
    [[0.2, 0.3, 0.1, 0.4], [0.1, -0.2, 0.3, -0.4], [-0.5, 0.2, -0.1, 0.3]]
)
population[1:] = 0.0
kinetics[1:] = 0.0
basis[:, 4] = 2.0
""",
            "call": """
charge_yield_response(kinetics.copy(), population.copy(), basis[:, 4].copy())
""",
            "gold_call": """
_oracle_charge_yield_response(
    kinetics.copy(), population.copy(), basis[:, 4].copy()
)
""",
            "tol": 1e-06,
        },
        {
            "setup": """
import numpy as np
from scipy.special import eval_genlaguerre, gammaln

basis = np.zeros((4, 6))
basis[:, 2] = [0.1, 0.12, 0.15, 0.18]
basis[:, 5] = [-0.2, 0.7, 0.1, -0.3]
basis[:, 4] = [0, 0, 1, 2]
parameters = np.zeros(21)
parameters[15] = 315.0
kinetics = np.zeros((3, 4, 7))
kinetics[0, :, :4] = np.array(
    [
        [0.0, 2.0, 3.0, 1.0],
        [0.7, 0.0, 2.2, 4.0],
        [1.0, 0.2, 0.0, 3.0],
        [0.3, 2.0, 1.1, 0.0],
    ]
)
kinetics[0, :, 4:] = np.array(
    [[0.4, 0.2, 1.0], [0.7, 0.3, 2.0], [0.1, 0.9, 0.3], [0.2, 3.0, 0.7]]
)
kinetics[1] = kinetics[0] * np.array([0.1, -0.2, 0.3, -0.1])[:, None]
kinetics[2] = kinetics[0] * np.array([-0.2, 0.05, 0.1, 0.3])[:, None]
population = np.array(
    [[0.2, 0.3, 0.1, 0.4], [0.1, -0.2, 0.3, -0.4], [-0.5, 0.2, -0.1, 0.3]]
)
population *= 0.5
""",
            "call": """
charge_yield_response(kinetics.copy(), population.copy(), basis[:, 4].copy())
""",
            "gold_call": """
_oracle_charge_yield_response(
    kinetics.copy(), population.copy(), basis[:, 4].copy()
)
""",
            "tol": 1e-06,
        },
        {
            "setup": """
import numpy as np
from scipy.special import eval_genlaguerre, gammaln

basis = np.zeros((4, 6))
basis[:, 2] = [0.1, 0.12, 0.15, 0.18]
basis[:, 5] = [-0.2, 0.7, 0.1, -0.3]
basis[:, 4] = [0, 0, 1, 2]
parameters = np.zeros(21)
parameters[15] = 315.0
kinetics = np.zeros((3, 4, 7))
kinetics[0, :, :4] = np.array(
    [
        [0.0, 2.0, 3.0, 1.0],
        [0.7, 0.0, 2.2, 4.0],
        [1.0, 0.2, 0.0, 3.0],
        [0.3, 2.0, 1.1, 0.0],
    ]
)
kinetics[0, :, 4:] = np.array(
    [[0.4, 0.2, 1.0], [0.7, 0.3, 2.0], [0.1, 0.9, 0.3], [0.2, 3.0, 0.7]]
)
kinetics[1] = kinetics[0] * np.array([0.1, -0.2, 0.3, -0.1])[:, None]
kinetics[2] = kinetics[0] * np.array([-0.2, 0.05, 0.1, 0.3])[:, None]
population = np.array(
    [[0.2, 0.3, 0.1, 0.4], [0.1, -0.2, 0.3, -0.4], [-0.5, 0.2, -0.1, 0.3]]
)
kinetics[0, :, -1] = 0.0


def check_domain(fn):
    try:
        fn(kinetics.copy(), population.copy(), basis[:, 4].copy())
    except ValueError:
        return 1
    return 0
""",
            "call": "check_domain(charge_yield_response)",
            "gold_call": "check_domain(_oracle_charge_yield_response)",
            "tol": 1e-06,
        },
    ]
