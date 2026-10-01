"""
Solve the coupled microscopic and canonical population responses.

Canonical thermalization changes both the reduced transition rates

and the distribution of a pool population among its molecular states.

The field response therefore couples driven balance, normalized

Boltzmann weights and microscopic population reconstruction.

Returns
-------
ndarray (2, 3, M): full and canonical-lifted microscopic populations at ordinary derivative orders 0, 1 and 2.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def coupled_population_response(
    energy: "np.ndarray",
    labels: "np.ndarray",
    kinetics: "np.ndarray",
    temperature: float,
) -> "np.ndarray":
    """Compute microscopic and canonical-lifted population responses.
    
    Parameters
    ----------
    energy : ndarray, shape (M, 2)
        Finite energies E_i(0) and slopes E_i'(0), with E_i(s) affine in s.
        Energies are in eV; s is in V/nm.
    labels : ndarray, shape (M,)
        Fixed LE=0, CT=1, CS=2 labels. Each pool must be nonempty.
    kinetics : ndarray, shape (3, M, M+3)
        Ordinary value, first and second derivatives. Row i contains
        outgoing transfers K[i,j], then recombination, extraction and
        generation. Transfer diagonals vanish at every order.
        Base rates and sources are finite and nonnegative; derivatives
        may have either sign, including derivatives of sinks and sources.
        Every base state must reach a positive recombination/extraction
        sink. Total base generation may be zero. No detailed balance or
        stationarity of these input rate derivatives is assumed.
    temperature : float
        Positive finite temperature in K; k_B=8.617333262145e-5 eV/K.
    
    Returns
    -------
    ndarray, shape (2, 3, M)
        Axis 0 contains full microscopic populations P and reconstructed
        canonical populations P_hat in the original microscopic order.
        Axis 1 contains values, first derivatives and ordinary second
        derivatives at s=0. These are driven populations, not normalized
        probabilities or factorial-scaled coefficients.
    
        The full model balances generation and incoming transfer against
        outgoing transfer, recombination and extraction at each state.
        The canonical model assigns weights proportional to
        exp[-E_i(s)/(k_B*T)], normalized within each origin pool. Average
        each interpool outgoing rate and each sink with these weights,
        sum destination states, omit intrapool transfers and sum actual
        generation into each pool without thermal weighting. Solve the
        three-pool driven balance for Q_a(s). Reconstruct microscopic
        canonical populations as P_hat_i(s)=w_i(s)*Q_label(i)(s).
        Include the field response of the weights, both driven solutions
        and the reconstruction. All three derivative orders are required.
        Equivalent stable numerical or analytic methods are accepted.
    
    Raises
    ------
    ValueError
        For the invalid dimensions, finite values, signs, temperature or
        labels described above; an empty pool; or a singular/sinkless base
        balance. A nonfinite computed response is invalid.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _response_array(kinetics):
    import numpy as np

    data = _finite(kinetics, "kinetics")
    if data.ndim != 3 or data.shape[0] != 3:
        raise ValueError("kinetics must have shape (3,M,M+3)")
    m = data.shape[1]
    if m < 1 or data.shape[2] != m + 3:
        raise ValueError("kinetics must have shape (3,M,M+3)")
    if np.any(data[0] < 0):
        raise ValueError(
            "zeroth-order rates and generation must be nonnegative"
        )
    if any(np.any(np.diag(c[:, :m]) != 0) for c in data):
        raise ValueError("all transfer diagonals must be zero")
    return data


def _solve_response(kinetics: "np.ndarray") -> "np.ndarray":
    import numpy as np

    data = _response_array(kinetics)
    m = data.shape[1]
    reaches = data[0, :, m] + data[0, :, m + 1] > 0
    for _ in range(m):
        new = reaches | np.any((data[0, :, :m] > 0) & reaches[None, :], axis=1)
        if np.array_equal(new, reaches):
            break
        reaches = new
    if not np.all(reaches):
        raise ValueError("every state must reach a positive sink at s=0")
    loss = np.array(
        [
            np.diag(c[:, :m].sum(axis=1) + c[:, m] + c[:, m + 1]) - c[:, :m].T
            for c in data
        ]
    )
    gen = data[:, :, -1]
    population = np.zeros((3, m))
    try:
        population[0] = np.linalg.solve(loss[0], gen[0])
        population[1] = np.linalg.solve(
            loss[0], gen[1] - loss[1] @ population[0]
        )
        population[2] = np.linalg.solve(
            loss[0],
            gen[2] - loss[2] @ population[0] - 2 * loss[1] @ population[1],
        )
    except np.linalg.LinAlgError as exc:
        raise ValueError("zeroth-order balance must be nonsingular") from exc
    if not np.all(np.isfinite(population)):
        raise ValueError("nonfinite response")
    return population


def _reduce_response(
    energy: "np.ndarray",
    labels: "np.ndarray",
    kinetics: "np.ndarray",
    temperature: float,
) -> "np.ndarray":
    import numpy as np

    data = _response_array(kinetics)
    m = data.shape[1]
    energy, labels = _finite(energy, "energy"), _finite(labels, "labels")
    temperature = _positive(temperature, "temperature")
    if (
        energy.shape != (m, 2)
        or labels.shape != (m,)
        or set(labels) != {0, 1, 2}
    ):
        raise ValueError(
            "energy must be (M,2) and labels cover all three pools"
        )
    member = np.eye(3)[labels.astype(int)]
    weight = np.zeros((3, m, 3))
    kt = 8.617333262145e-5 * temperature
    for a in range(3):
        mask = labels == a
        w = np.exp(-(energy[mask, 0] - energy[mask, 0].min()) / kt)
        w /= w.sum()
        score = -energy[mask, 1] / kt
        centered = score - w @ score
        weight[0, mask, a] = w
        weight[1, mask, a] = w * centered
        weight[2, mask, a] = w * (centered**2 - w @ (centered**2))
    out = np.zeros((3, 3, 6))
    for order in range(3):
        for k in range(order + 1):
            choose = 2 if order == 2 and k == 1 else 1
            out[order, :, :3] += (
                choose * weight[k].T @ data[order - k, :, :m] @ member
            )
            out[order, :, 3:5] += (
                choose * weight[k].T @ data[order - k, :, m: m + 2]
            )
        np.fill_diagonal(out[order, :, :3], 0.0)
        out[order, :, 5] = member.T @ data[order, :, -1]
    return out


def _oracle_coupled_population_response(
    energy: 'np.ndarray',
    labels: 'np.ndarray',
    kinetics: 'np.ndarray',
    temperature: float,
) -> "np.ndarray":
    reduced = _reduce_response(energy, labels, kinetics, temperature)
    full = _solve_response(kinetics)
    pools = _solve_response(reduced)
    e = np.asarray(energy, dtype=float)
    lab = np.asarray(labels, dtype=int)
    kt = 8.617333262145e-5 * temperature
    lifted = np.zeros_like(full)
    for a in range(3):
        mask = lab == a
        w = np.exp(-(e[mask, 0] - e[mask, 0].min()) / kt)
        w /= w.sum()
        score = -e[mask, 1] / kt
        centered = score - w @ score
        wp = w * centered
        wpp = w * (centered**2 - w @ (centered**2))
        lifted[0, mask] = w * pools[0, a]
        lifted[1, mask] = wp * pools[0, a] + w * pools[1, a]
        lifted[2, mask] = (
            wpp * pools[0, a] + 2 * wp * pools[1, a]
            + w * pools[2, a]
        )
    if not np.all(np.isfinite(lifted)):
        raise ValueError("nonfinite canonical population response")
    return np.stack((full, lifted))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return independent scientific cases."""
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
""",
            "call": """
coupled_population_response(
    basis[:, [2, 5]].copy(),
    basis[:, 4].copy(),
    kinetics.copy(),
    parameters[15],
)
""",
            "gold_call": """
_oracle_coupled_population_response(
    basis[:, [2, 5]].copy(),
    basis[:, 4].copy(),
    kinetics.copy(),
    parameters[15],
)
""",
            "tol": 1e-07,
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
""",
            "call": """
coupled_population_response(
    basis[:, [2, 5]].copy(),
    basis[:, 4].copy(),
    kinetics.copy(),
    parameters[15],
)
""",
            "gold_call": """
_oracle_coupled_population_response(
    basis[:, [2, 5]].copy(),
    basis[:, 4].copy(),
    kinetics.copy(),
    parameters[15],
)
""",
            "tol": 1e-07,
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
basis[:, 2] = -0.5
basis[:, 5] = 0.3
""",
            "call": """
coupled_population_response(
    basis[:, [2, 5]].copy(),
    basis[:, 4].copy(),
    kinetics.copy(),
    parameters[15],
)
""",
            "gold_call": """
_oracle_coupled_population_response(
    basis[:, [2, 5]].copy(),
    basis[:, 4].copy(),
    kinetics.copy(),
    parameters[15],
)
""",
            "tol": 1e-07,
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
basis[:, 2] += 1000.0
""",
            "call": """
coupled_population_response(
    basis[:, [2, 5]].copy(),
    basis[:, 4].copy(),
    kinetics.copy(),
    parameters[15],
)
""",
            "gold_call": """
_oracle_coupled_population_response(
    basis[:, [2, 5]].copy(),
    basis[:, 4].copy(),
    kinetics.copy(),
    parameters[15],
)
""",
            "tol": 1e-07,
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

kinetics[:, :, -1] *= 3.7
""",
            "call": """
coupled_population_response(
    basis[:, [2, 5]].copy(),
    basis[:, 4].copy(),
    kinetics.copy(),
    parameters[15],
)
""",
            "gold_call": """
_oracle_coupled_population_response(
    basis[:, [2, 5]].copy(),
    basis[:, 4].copy(),
    kinetics.copy(),
    parameters[15],
)
""",
            "tol": 1e-07,
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

kinetics[:, :, -1] = 0.0
""",
            "call": """
coupled_population_response(
    basis[:, [2, 5]].copy(),
    basis[:, 4].copy(),
    kinetics.copy(),
    parameters[15],
)
""",
            "gold_call": """
_oracle_coupled_population_response(
    basis[:, [2, 5]].copy(),
    basis[:, 4].copy(),
    kinetics.copy(),
    parameters[15],
)
""",
            "tol": 1e-07,
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

basis[:, 5] *= -2.0
""",
            "call": """
coupled_population_response(
    basis[:, [2, 5]].copy(),
    basis[:, 4].copy(),
    kinetics.copy(),
    parameters[15],
)
""",
            "gold_call": """
_oracle_coupled_population_response(
    basis[:, [2, 5]].copy(),
    basis[:, 4].copy(),
    kinetics.copy(),
    parameters[15],
)
""",
            "tol": 1e-07,
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

basis[:, 5] += 0.45
""",
            "call": """
coupled_population_response(
    basis[:, [2, 5]].copy(),
    basis[:, 4].copy(),
    kinetics.copy(),
    parameters[15],
)
""",
            "gold_call": """
_oracle_coupled_population_response(
    basis[:, [2, 5]].copy(),
    basis[:, 4].copy(),
    kinetics.copy(),
    parameters[15],
)
""",
            "tol": 1e-07,
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
basis[:, 4] = 0.0


def check_domain(fn):
    try:
        fn(
            basis[:, [2, 5]].copy(),
            basis[:, 4].copy(),
            kinetics.copy(),
            parameters[15],
        )
    except ValueError:
        return 1
    return 0
""",
            "call": """
check_domain(coupled_population_response)
""",
            "gold_call": """
check_domain(_oracle_coupled_population_response)
""",
            "tol": 0.0,
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
kinetics[0, :, 4:6] = 0.0


def check_domain(fn):
    try:
        fn(
            basis[:, [2, 5]].copy(),
            basis[:, 4].copy(),
            kinetics.copy(),
            parameters[15],
        )
    except ValueError:
        return 1
    return 0
""",
            "call": """
check_domain(coupled_population_response)
""",
            "gold_call": """
check_domain(_oracle_coupled_population_response)
""",
            "tol": 0.0,
        },
    ]
