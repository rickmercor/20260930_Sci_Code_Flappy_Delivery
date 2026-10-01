"""
Orchestrate the pole search, coupled scattering, two regulator prescriptions, and halo branching comparison.

The final scientific observable combines the paper's regular-only multistate prescription with the stated halo distribution. Each preceding public subproblem contributes to the end-to-end calculation.

Returns
-------
return value
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def resonant_branching_shift(detuning: float = 1e-4, eta: float = 3e-4, cold_weight: float = 0.7, order: int = 64) -> float:
    """Orchestrate the pole search, coupled scattering, two regulator prescriptions, and halo branching comparison.

    detuning : float
        Fractional change of the potential depth relative to the pole in [.9,1.2];
        supported range [-0.001,0.001].
    eta : float
        Positive amplitude scale in GeV^-1, f=eta*(H+iG); supported range
        [1e-5,0.003]. H and G are the prompt matrices.
    cold_weight : float
        Mixture weight of M(p;0.0002), in [0,1]; M(p;0.006) has weight 1-cold_weight.
    order : int
        Gauss-Legendre order per stated interval, 16 through 256 inclusive.
    Returns
    -------
    float
        Dimensionless B_F/B_A-1 for the prompt's fixed three-state matrices,
        shell widths, thresholds, pole bracket, momentum interval and mixture
        scales. F uses the leading resonant regulator; A uses Z=0. Both retain
        the full complex f. The defaults specify the main task.
        Call and combine every preceding public function: propagate_shells,
        threshold_pole, match_channels, resonant_regulator, dress_channels,
        annihilation_loss, conversion_strength, and halo_branching. Valid parameters
        use nonsingular linear systems throughout this benchmark.
    Raises
    ------
    ValueError
        Invalid rank, shape, nonnumeric or nonfinite input. All dimensions are nonzero.
        An input outside its supported interval, a nonscalar parameter or an order that is not an integer (Boolean orders are invalid).
        Numerical linear systems must be nonsingular and outputs finite.
    """
    return value

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.linalg import expm
from scipy.optimize import brentq

def _finite_array(value, name, ndim, real=False):
    try:
        array = np.asarray(value)
        if array.dtype.kind not in 'iufc' or (real and np.iscomplexobj(array)):
            raise ValueError(name + ' must contain real numeric values' if real else name + ' must be numeric')
        array = np.asarray(array, dtype=float if real else complex)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(name + ' must be a finite numeric array') from exc
    if array.ndim != ndim or not np.all(np.isfinite(array)):
        raise ValueError(name + ' has invalid rank or nonfinite entries')
    return array


def _finite_scalar(value, name):
    return float(_finite_array(value, name, 0, real=True))


def _thresholds(gaps, strict=False):
    gaps = _finite_array(gaps, 'gaps', 1, real=True)
    if gaps.size == 0 or gaps[0] != 0 or np.any(gaps < 0):
        raise ValueError('gaps must be nonempty, nonnegative and start at zero')
    if np.any(np.diff(gaps) <= 0 if strict else np.diff(gaps) < 0):
        raise ValueError('gaps must be strictly increasing' if strict else 'gaps must be nondecreasing')
    return gaps


def _shell_data(gaps, wells, widths, strict=False):
    gaps = _thresholds(gaps, strict)
    wells = _finite_array(wells, 'wells', 3, real=True)
    widths = _finite_array(widths, 'widths', 1, real=True)
    n = len(gaps)
    if widths.size == 0 or wells.shape != (len(widths), n, n) or np.any(widths <= 0):
        raise ValueError('wells and positive widths must describe the same nonempty shell sequence')
    if not np.allclose(wells, wells.transpose(0, 2, 1), rtol=1e-10, atol=1e-12):
        raise ValueError('wells must be real symmetric')
    return gaps, wells, widths


def _open_momenta(p_open):
    p_open = _finite_array(p_open, 'p_open', 1, real=True)
    if p_open.size == 0 or np.any(p_open <= 0):
        raise ValueError('p_open must be nonempty and positive')
    return p_open


def _matched_data(p_open, matched):
    p_open = _open_momenta(p_open)
    matched = _finite_array(matched, 'matched', 2)
    m = len(p_open)
    if matched.shape[1] != m or matched.shape[0] < 2*m:
        raise ValueError('matched must have shape (N+M,M) with N>=M>=1')
    sigma, s0 = matched[:-m], matched[-m:]
    if not np.allclose(s0.conj().T @ s0, np.eye(m), rtol=1e-8, atol=1e-8):
        raise ValueError('S_0 must be unitary')
    return p_open, sigma, s0


def _hermitian_matrix(value, name, n):
    matrix = _finite_array(value, name, 2)
    if matrix.shape != (n, n):
        raise ValueError(name + ' has incompatible shape')
    if not np.allclose(matrix, matrix.conj().T, rtol=1e-10, atol=1e-12):
        raise ValueError(name + ' must be Hermitian')
    return matrix


def _short_matrix(value, n):
    matrix = _finite_array(value, 'short_amplitude', 2)
    if matrix.shape != (n, n):
        raise ValueError('short_amplitude has incompatible shape')
    gamma = (matrix - matrix.conj().T) / (2j)
    if np.linalg.eigvalsh(gamma)[0] < -1e-12 * max(1., np.linalg.norm(gamma, ord=2)):
        raise ValueError('short_amplitude must have positive semidefinite absorptive part')
    return matrix


def _dressed_data(p_open, dressed):
    p_open = _open_momenta(p_open)
    dressed = _finite_array(dressed, 'dressed', 2)
    m = len(p_open)
    if dressed.shape[1] != m or dressed.shape[0] < m:
        raise ValueError('dressed must have shape (N,M) with N>=M>=1')
    return p_open, dressed


def _checked_solve(matrix, rhs, name):
    try:
        result = np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError as exc:
        raise ValueError(name + ' must be nonsingular') from exc
    return _checked_result(result, name)


def _checked_result(result, name):
    if not np.all(np.isfinite(result)):
        raise ValueError(name + ' produced a nonfinite result')
    return result


def _oracle_resonant_branching_shift(detuning: float=0.0001, eta: float=0.0003, cold_weight: float=0.7, order: int=64) -> float:
    detuning = _finite_scalar(detuning, 'detuning')
    eta = _finite_scalar(eta, 'eta')
    cold_weight = _finite_scalar(cold_weight, 'cold_weight')
    if not -0.001 <= detuning <= 0.001 or not 1e-05 <= eta <= 0.003 or (not 0 <= cold_weight <= 1):
        raise ValueError('detuning, eta or cold_weight lies outside the supported interval')
    if isinstance(order, (bool, np.bool_)) or not isinstance(order, (int, np.integer)) or (not 16 <= order <= 256):
        raise ValueError('order must be an integer from 16 through 256')
    gaps = np.array([0.0, 0.004 ** 2, 0.011 ** 2])
    wells = np.array([[[2.9, -1.3, 0.55], [-1.3, 4.2, -0.8], [0.55, -0.8, 1.5]], [[1.4, 0.6, -0.45], [0.6, 2.7, 1.1], [-0.45, 1.1, 3.2]]])
    widths = np.array([0.43, 0.57])
    h = np.array([[0.2, -0.13, 0.09], [-0.13, -0.11, 0.05], [0.09, 0.05, 0.07]])
    gamma = np.array([[1.0, 0.45, -0.2], [0.45, 0.7, 0.15], [-0.2, 0.15, 0.6]])
    f = eta * (h + 1j * gamma)
    pole = _oracle_threshold_pole(gaps, wells, widths, (0.9, 1.2))
    depth = pole * (1 + detuning)
    (abscissae, quadrature) = np.polynomial.legendre.leggauss(order)
    breaks = [1e-05, 0.001, 0.004, 0.011, 0.03]
    nodes = np.concatenate([(lo + hi) / 2 + (hi - lo) / 2 * abscissae for (lo, hi) in zip(breaks[:-1], breaks[1:])])
    weights = np.concatenate([(hi - lo) / 2 * quadrature for (lo, hi) in zip(breaks[:-1], breaks[1:])])
    density = np.zeros_like(nodes)
    for (fraction, scale) in [(cold_weight, 0.0002), (1 - cold_weight, 0.006)]:
        density += fraction * 4 / np.sqrt(np.pi) * nodes ** 2 / scale ** 3 * np.exp(-(nodes / scale) ** 2)
    weights *= density
    losses = np.empty((2, len(nodes)))
    conversions = np.empty_like(losses)
    for (j, p) in enumerate(nodes):
        y = _oracle_propagate_shells(p, gaps, wells, widths, depth)
        matched = _oracle_match_channels(p, gaps, np.sum(widths), y)
        po = np.sqrt(p * p - gaps[gaps < p * p])
        z = _oracle_resonant_regulator(po, matched)
        for (mode, regulator) in enumerate([z, np.zeros_like(z)]):
            dressed = _oracle_dress_channels(po, matched, regulator, f)
            losses[mode, j] = _oracle_annihilation_loss(po, dressed, f)[0, 0].real
            conversions[mode, j] = _oracle_conversion_strength(po, matched, dressed, f)[:, 0].sum()
    full = _oracle_halo_branching(nodes, weights, losses[0], conversions[0], 0.001)
    absorptive = _oracle_halo_branching(nodes, weights, losses[1], conversions[1], 0.001)
    return float(_checked_result(full[2] / absorptive[2] - 1, 'branching shift'))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Deterministic scientific differential cases."""
    return [{'setup': 'import numpy as np\n',
      'call': 'resonant_branching_shift()',
      'gold_call': '_oracle_resonant_branching_shift()'},
     {'setup': 'import numpy as np\n',
      'call': 'resonant_branching_shift(-0.0001,0.0003,0.7,64)',
      'gold_call': '_oracle_resonant_branching_shift(-0.0001,0.0003,0.7,64)'},
     {'setup': 'import numpy as np\n',
      'call': 'resonant_branching_shift(0.0001,0.0003,0.0,64)',
      'gold_call': '_oracle_resonant_branching_shift(0.0001,0.0003,0.0,64)'},
     {'setup': 'import numpy as np\n',
      'call': 'resonant_branching_shift(0.0001,0.0003,1.0,64)',
      'gold_call': '_oracle_resonant_branching_shift(0.0001,0.0003,1.0,64)'},
     {'setup': 'import numpy as np\n',
      'call': 'resonant_branching_shift(0.0001,1e-05,0.7,64)',
      'gold_call': '_oracle_resonant_branching_shift(0.0001,1e-05,0.7,64)'},
     {'setup': 'import numpy as np\n',
      'call': 'resonant_branching_shift(0.0001,0.003,0.7,64)',
      'gold_call': '_oracle_resonant_branching_shift(0.0001,0.003,0.7,64)'},
     {'setup': 'import numpy as np\n',
      'call': 'resonant_branching_shift(0.0,0.0003,0.7,64)',
      'gold_call': '_oracle_resonant_branching_shift(0.0,0.0003,0.7,64)'},
     {'setup': 'import numpy as np\n',
      'call': 'resonant_branching_shift(0.0001,0.0003,0.7,128)',
      'gold_call': '_oracle_resonant_branching_shift(0.0001,0.0003,0.7,128)'},
     {'setup': 'import numpy as np\n'
               '\n'
               '\n'
               '\n'
               'def _exception_code(function, *args, **kwargs):\n'
               '    try:\n'
               '        function(*args, **kwargs)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n',
      'call': '_exception_code(resonant_branching_shift, detuning=0.01)',
      'gold_call': '_exception_code(_oracle_resonant_branching_shift, detuning=0.01)'},
     {'setup': 'import numpy as np\n'
               '\n'
               '\n'
               '\n'
               'def _exception_code(function, *args, **kwargs):\n'
               '    try:\n'
               '        function(*args, **kwargs)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n',
      'call': '_exception_code(resonant_branching_shift, eta=0.0)',
      'gold_call': '_exception_code(_oracle_resonant_branching_shift, eta=0.0)'},
     {'setup': 'import numpy as np\n'
               '\n'
               '\n'
               '\n'
               'def _exception_code(function, *args, **kwargs):\n'
               '    try:\n'
               '        function(*args, **kwargs)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n',
      'call': '_exception_code(resonant_branching_shift, cold_weight=1.1)',
      'gold_call': '_exception_code(_oracle_resonant_branching_shift, cold_weight=1.1)'},
     {'setup': 'import numpy as np\n'
               '\n'
               '\n'
               '\n'
               'def _exception_code(function, *args, **kwargs):\n'
               '    try:\n'
               '        function(*args, **kwargs)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n',
      'call': '_exception_code(resonant_branching_shift, detuning=np.nan)',
      'gold_call': '_exception_code(_oracle_resonant_branching_shift, detuning=np.nan)'},
     {'setup': 'import numpy as np\n'
               '\n'
               '\n'
               '\n'
               'def _exception_code(function, *args, **kwargs):\n'
               '    try:\n'
               '        function(*args, **kwargs)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n',
      'call': '_exception_code(resonant_branching_shift, cold_weight=np.array([0.7]))',
      'gold_call': '_exception_code(_oracle_resonant_branching_shift, cold_weight=np.array([0.7]))'},
     {'setup': 'import numpy as np\n'
               '\n'
               '\n'
               '\n'
               'def _exception_code(function, *args, **kwargs):\n'
               '    try:\n'
               '        function(*args, **kwargs)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n',
      'call': '_exception_code(resonant_branching_shift, order=64.5)',
      'gold_call': '_exception_code(_oracle_resonant_branching_shift, order=64.5)'},
     {'setup': 'import numpy as np\n'
               '\n'
               '\n'
               '\n'
               'def _exception_code(function, *args, **kwargs):\n'
               '    try:\n'
               '        function(*args, **kwargs)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n',
      'call': '_exception_code(resonant_branching_shift, order=8)',
      'gold_call': '_exception_code(_oracle_resonant_branching_shift, order=8)'}]
