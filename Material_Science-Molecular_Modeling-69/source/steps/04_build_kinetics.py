"""
Build localized molecular transfer, loss and generation kinetics.

The localized molecular prescription uses finite-temperature

Marcus-Levich-Jortner transitions with independent exciton and polaron

baths. Transfers and recombination have distinct reorganization energies.

Returns
-------
np.ndarray (3,M,M+3): value/first/second outgoing K, rec, ext, gen
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def build_kinetics_response(
    basis: "np.ndarray",
    couplings: "np.ndarray",
    low_x: float,
    high_x: float,
    low_p: float,
    high_p: float,
    quantum: float,
    temperature: float,
    nmax: int,
    mmax: int,
    vx: float,
    vct: float,
    cutoff: float,
    kext: float,
    illumination: "np.ndarray",
) -> "np.ndarray":
    """Build the field response of finite-temperature molecular rates.
    
    Parameters
    ----------
    basis : ndarray, shape (M, 6), M=N*N, N >= 2
        Finite rows [e,h,E(0),separation,label,E'(0)] in e*N+h order.
        E(s)=E(0)+s*E'(0); separations are nonnegative and labels are
        LE=0, CT=1, CS=2. LE labels occur exactly when e=h.
    couplings : ndarray, shape (M, M)
        Finite real transfer amplitudes, symmetric to absolute tolerance
        1e-12 with zero diagonal. Signed amplitudes are allowed.
    low_x, high_x, low_p, high_p : float
        Single-exciton and single-polaron low/high reorganization energies
        in eV. Low components are positive; high components nonnegative.
    quantum, temperature : float
        Positive oscillator energy in eV and temperature in K.
        Use k_B=8.617333262145e-5 eV/K, hbar=6.582119569e-7 eV ns.
    nmax, mmax : int
        Nonnegative inclusive initial/final cutoffs, not booleans.
    vx, vct : float
        Nonnegative LE and adjacent non-LE ground couplings in eV.
    cutoff : float
        Positive adjacency distance in nm. Non-LE ground decay is active
        only at separation <= cutoff+1e-12. Ground energy and slope are 0.
    kext : float
        Nonnegative extraction rate on every CS state; zero otherwise.
    illumination : ndarray, shape (N,)
        Finite nonnegative generation into each |e,e>; zero into non-LE.
    
    Returns
    -------
    ndarray, shape (3, M, M+3)
        Axis 0 is value, first derivative, ordinary second derivative in s.
        Row i contains outgoing rates K[i,j] for j=0..M-1, followed by
        recombination, extraction and generation. Transfer diagonals are 0.
        Derivatives are not factorial-scaled Taylor coefficients.
        Transfer uses twice the exciton low/high components for LE-LE,
        and twice the polaron components otherwise. LE ground decay uses
        one exciton bath; adjacent non-LE decay uses two polaron baths.
        The finite-temperature MLJ density sums the Step 3 weights with
        exp[-(Delta+low+(m-n)*quantum)^2/(4*low*k_B*T)] and normalization
        sqrt(4*pi*low*k_B*T). The rate prefactor is 2*pi*V^2/hbar.
        Delta is the final-minus-initial electronic gap; ground decay
        uses -E(s). Energies vary with s; every other input stays fixed.
    
    Raises
    ------
    ValueError
        For invalid dimensions, labels, finite values, signs or cutoffs
        specified above.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _density_response(delta, slope, low, weights, quantum, temperature):
    import numpy as np

    n, m = np.indices(weights.shape)
    kt = 8.617333262145e-5 * temperature
    gap = np.asarray(delta)[..., None, None] + low + (m - n) * quantum
    terms = (
        weights
        * np.exp(-(gap**2) / (4 * low * kt))
        / np.sqrt(4 * np.pi * low * kt)
    )
    log_derivative = -gap / (2 * low * kt)
    slope = np.asarray(slope)
    return np.array(
        [
            terms.sum(axis=(-2, -1)),
            slope * (terms * log_derivative).sum(axis=(-2, -1)),
            slope**2
            * (terms * (log_derivative**2 - 1 / (2 * low * kt))).sum(
                axis=(-2, -1)
            ),
        ]
    )


def _oracle_build_kinetics_response(
    basis: "np.ndarray",
    couplings: "np.ndarray",
    low_x: float,
    high_x: float,
    low_p: float,
    high_p: float,
    quantum: float,
    temperature: float,
    nmax: int,
    mmax: int,
    vx: float,
    vct: float,
    cutoff: float,
    kext: float,
    illumination: "np.ndarray",
) -> "np.ndarray":
    import numpy as np

    basis = _finite(basis, "basis")
    coupling = _finite(couplings, "couplings")
    if basis.ndim != 2 or basis.shape[1] != 6 or len(basis) < 4:
        raise ValueError("basis must have shape (N*N,6), N>=2")
    size = len(basis)
    n = int(np.sqrt(size))
    e, h = np.indices((n, n)).reshape(2, -1)
    if n * n != size or not np.array_equal(
        basis[:, :2], np.column_stack((e, h))
    ):
        raise ValueError("basis must follow e*N+h order")
    labels = basis[:, 4]
    if not np.all(np.isin(labels, (0, 1, 2))) or not np.array_equal(
        labels == 0, e == h
    ):
        raise ValueError("invalid labels")
    if np.any(basis[:, 3] < 0) or coupling.shape != (size, size):
        raise ValueError("invalid separations or coupling shape")
    if not np.allclose(coupling, coupling.T, rtol=0, atol=1e-12) or np.any(
        np.diag(coupling) != 0
    ):
        raise ValueError("couplings must be symmetric with zero diagonal")
    low_x, low_p = [_positive(x, "low") for x in (low_x, low_p)]
    high_x, high_p = [
        _positive(x, "high", zero=True) for x in (high_x, high_p)
    ]
    quantum, temperature, cutoff = [
        _positive(x, "scale") for x in (quantum, temperature, cutoff)
    ]
    vx, vct, kext = [_positive(x, "sink", zero=True) for x in (vx, vct, kext)]
    light = _finite(illumination, "illumination")
    if light.shape != (n,) or np.any(light < 0):
        raise ValueError("illumination must be a nonnegative (N,) vector")
    wx = _oracle_thermal_vibronic_weights(
        high_x, quantum, temperature, nmax, mmax
    )
    wxx = _oracle_thermal_vibronic_weights(
        2 * high_x, quantum, temperature, nmax, mmax
    )
    wpp = _oracle_thermal_vibronic_weights(
        2 * high_p, quantum, temperature, nmax, mmax
    )
    energy, slope = basis[:, 2], basis[:, 5]
    delta = energy[None, :] - energy[:, None]
    delta_slope = slope[None, :] - slope[:, None]
    le = labels == 0
    lele = le[:, None] & le[None, :]
    density = np.where(
        lele[None],
        _density_response(
            delta, delta_slope, 2 * low_x, wxx, quantum, temperature
        ),
        _density_response(
            delta, delta_slope, 2 * low_p, wpp, quantum, temperature
        ),
    )
    factor = 2 * np.pi / 6.582119569e-7
    rates = factor * coupling[None] ** 2 * density
    rec = factor * np.where(
        le[None],
        vx**2
        * _density_response(-energy, -slope, low_x, wx, quantum, temperature),
        vct**2
        * _density_response(
            -energy, -slope, 2 * low_p, wpp, quantum, temperature
        )
        * (basis[:, 3] <= cutoff + 1e-12)[None],
    )
    result = np.zeros((3, size, size + 3))
    result[:, :, :size] = rates
    result[:, :, size] = rec
    result[0, :, size + 1] = kext * (labels == 2)
    result[0, :, size + 2] = np.where(le, light[e], 0.0)
    return result

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
""",
            "call": """
build_kinetics_response(
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
            "gold_call": """
_oracle_build_kinetics_response(
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
            "tol": 1e-06,
        },
        {
            "setup": """
import numpy as np
from scipy.special import eval_genlaguerre, gammaln

positions = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
homo = np.array([0.01, -0.01, 0.02])
lumo = np.array([0.94, 0.98, 0.91])
binding = np.array([0.23, 0.25, 0.22])
disorder = np.array(
    [[0.02, -0.01, 0.03], [0.01, 0.0, -0.02], [-0.01, 0.015, 0.01]]
)
field = np.array([0.018, -0.01])
direction = np.array([0.6, 0.8])
illumination = np.array([1.0, 0.4, 1.7])
parameters = np.array(
    [
        0.60,
        0.30,
        1.8,
        1.01,
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
        330.0,
        10.0,
        35.0,
        0.020,
        0.001,
        6.0,
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
""",
            "call": """
build_kinetics_response(
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
            "gold_call": """
_oracle_build_kinetics_response(
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
            "tol": 1e-06,
        },
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
direction[:] = 0.0
parameters[11] = 0.0
parameters[13] = 0.0
parameters[20] = 0.0
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
""",
            "call": """
build_kinetics_response(
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
            "gold_call": """
_oracle_build_kinetics_response(
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
            "tol": 1e-06,
        },
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
direction = np.array([-0.4, 1.2])
parameters[17] = 2.0
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
couplings = -_oracle_build_couplings(positions.copy(), *parameters[3:10])
""",
            "call": """
build_kinetics_response(
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
            "gold_call": """
_oracle_build_kinetics_response(
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
            "tol": 1e-06,
        },
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
illumination[0] = -1.0


def check_domain(fn):
    try:
        fn(
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
    except ValueError:
        return 1
    return 0
""",
            "call": "check_domain(build_kinetics_response)",
            "gold_call": "check_domain(_oracle_build_kinetics_response)",
            "tol": 1e-06,
        },
    ]
