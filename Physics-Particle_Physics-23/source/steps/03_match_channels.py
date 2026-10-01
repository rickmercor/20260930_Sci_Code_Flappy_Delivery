"""
Match a regular radial basis to incident and outgoing waves and closed-channel decay.

Sec. 4 defines distinct full and open momentum spaces, Eq. (4.2), and the regular scattering normalization, Eqs. (4.3), (4.17), (4.18). Closed components remain in Sigma_0 although S_0 has only open channels.

Returns
-------
return result
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def match_channels(p: float, gaps: np.ndarray, radius: float, fundamental: np.ndarray) -> np.ndarray:
    """Match a regular radial basis to incident and outgoing waves and closed-channel decay.

    p : float
        Positive reference momentum in GeV; p^2 differs from every threshold.
    gaps : ndarray, shape (N,)
        Nondecreasing nonnegative threshold offsets in GeV^2, gaps[0]=0.
    radius : float
        Positive outer radius R in GeV^-1.
    fundamental : ndarray, shape (2*N,N)
        Real U(R) above U'(R), normalized by U(0)=0, U'(0)=I, as in propagate_shells.
    Returns
    -------
    ndarray, complex, shape (N+M,M)
        Top N rows: dimensionless ordinary Sommerfeld matrix Sigma_0.
        Bottom M rows: dimensionless unit-flux elastic matrix S_0.
        M counts gaps[i]<p^2. Open channels occur in increasing physical order;
        columns are incident channels. For k_i=sqrt(p^2-gaps[i]), use positive
        real or positive imaginary roots. Open outgoing waves are exp(i*k_i*r).
        The regular solution has w~diag(k_i*r)Q at the origin and incident sine
        waves in open channels; Sigma_0=diag(k_i)Q P^-1. Closed waves decay.
    Raises
    ------
    ValueError
        Invalid rank, shape, nonnumeric or nonfinite input. All dimensions are nonzero.
        Nonpositive momentum/radius, exact channel threshold, unordered gaps, inconsistent fundamental shape or singular matching system.
        Numerical linear systems must be nonsingular and outputs finite.
    """
    return result

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


def _oracle_match_channels(p: float, gaps: np.ndarray, radius: float, fundamental: np.ndarray) -> np.ndarray:
    gaps = _thresholds(gaps)
    p = _finite_scalar(p, 'p')
    radius = _finite_scalar(radius, 'radius')
    if p <= 0 or p > np.sqrt(np.finfo(float).max) or radius <= 0 or np.any(p * p == gaps):
        raise ValueError('p and radius must be positive and p^2 must differ from each threshold')
    n = len(gaps)
    fundamental = _finite_array(fundamental, 'fundamental', 2, real=True)
    if fundamental.shape != (2 * n, n):
        raise ValueError('fundamental must have shape (2*N,N)')
    k = np.sqrt((p ** 2 - gaps).astype(complex))
    opened = np.flatnonzero(gaps < p ** 2)
    po = k[opened].real
    (u, du) = (np.asarray(fundamental)[:n], np.asarray(fundamental)[n:])
    jost = np.exp(1j * k * radius)[:, None] * (du - 1j * k[:, None] * u)
    sigma = _checked_solve(jost, np.eye(n)[:, opened], 'Jost matrix')
    outgoing = np.exp(-1j * k[opened] * radius)[:, None] * (du[opened] + 1j * k[opened, None] * u[opened])
    s0 = outgoing @ sigma * np.sqrt(po)[None, :] / np.sqrt(po)[:, None]
    return _checked_result(np.vstack((sigma, s0)), 'channel matching')

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Deterministic scientific differential cases."""
    return [{'setup': 'from copy import deepcopy as _copy_input\n'
               'import numpy as np\n'
               'p = 0.002\n'
               'gaps = np.array([0.0, 1.6e-05, 0.00012099999999999999], dtype=float)\n'
               'radius = 1.0\n'
               'fundamental = np.array([[0.6865311416057324, 0.026841852031857967, 0.0073970665459447546], '
               '[0.016843393296099404, 0.5170193142680312, -0.050950403414416263], [0.004864501554036709, '
               '-0.038505009969559925, 0.6124059926352794], [0.25811885261761386, -0.08717635246341333, '
               '0.10955586701049677], [-0.12925114453179762, -0.1604386802522381, -0.29110982991222556], '
               '[0.09929134348036098, -0.24022022690061734, -0.2074011139920885]], dtype=float)\n',
      'call': 'match_channels(*_copy_input((p, gaps, radius, fundamental)))',
      'gold_call': '_oracle_match_channels(*_copy_input((p, gaps, radius, fundamental)))'},
     {'setup': 'from copy import deepcopy as _copy_input\n'
               'import numpy as np\n'
               'p = 0.003999\n'
               'gaps = np.array([0.0, 1.6e-05, 0.00012099999999999999], dtype=float)\n'
               'radius = 1.0\n'
               'fundamental = np.array([[0.6947102772918583, 0.026065768415044765, 0.007234599884568339], '
               '[0.016584239074170223, 0.5286850322160744, -0.049733322474913036], [0.004829690832705381, '
               '-0.0379242706347132, 0.6222463259255734], [0.27607420538573724, -0.08568049409586533, '
               '0.10711824763664168], [-0.12566410454839275, -0.13659281557239136, -0.28524726233513364], '
               '[0.09733843154567583, -0.2368114643823953, -0.18015179540311138]], dtype=float)\n',
      'call': 'match_channels(*_copy_input((p, gaps, radius, fundamental)))',
      'gold_call': '_oracle_match_channels(*_copy_input((p, gaps, radius, fundamental)))'},
     {'setup': 'from copy import deepcopy as _copy_input\n'
               'import numpy as np\n'
               'p = 0.004001\n'
               'gaps = np.array([0.0, 1.6e-05, 0.00012099999999999999], dtype=float)\n'
               'radius = 1.0\n'
               'fundamental = np.array([[0.6947102751324667, 0.026065768367029535, 0.007234599875990089], '
               '[0.016584239036736302, 0.5286850303445059, -0.04973332239954434], [0.004829690826838732, '
               '-0.037924270572595, 0.6222463238760663], [0.27607419966436053, -0.0856804941015613, '
               '0.10711824747160086], [-0.1256641044864183, -0.13659281995813743, -0.28524726175530074], '
               '[0.09733843139768483, -0.23681146388625632, -0.18015180022354116]], dtype=float)\n',
      'call': 'match_channels(*_copy_input((p, gaps, radius, fundamental)))',
      'gold_call': '_oracle_match_channels(*_copy_input((p, gaps, radius, fundamental)))'},
     {'setup': 'from copy import deepcopy as _copy_input\n'
               'import numpy as np\n'
               'p = 0.010999\n'
               'gaps = np.array([0.0, 1.6e-05, 0.00012099999999999999], dtype=float)\n'
               'radius = 1.0\n'
               'fundamental = np.array([[0.6892368975984791, 0.026582880617388865, 0.007343150006828806], '
               '[0.01675799742355628, 0.5208810811190725, -0.05054576213739444], [0.004853418325076871, '
               '-0.038313971554378004, 0.615663360225916], [0.26404254109905984, -0.08668349429589971, '
               '0.1087464680772353], [-0.1280567597727491, -0.15257634987964672, -0.2891667973576825], '
               '[0.09864430613664327, -0.23909962607825028, -0.19839755868542977]], dtype=float)\n',
      'call': 'match_channels(*_copy_input((p, gaps, radius, fundamental)))',
      'gold_call': '_oracle_match_channels(*_copy_input((p, gaps, radius, fundamental)))'},
     {'setup': 'from copy import deepcopy as _copy_input\n'
               'import numpy as np\n'
               'p = 0.011001\n'
               'gaps = np.array([0.0, 1.6e-05, 0.00012099999999999999], dtype=float)\n'
               'radius = 1.0\n'
               'fundamental = np.array([[0.6892368916858868, 0.026582880482831146, 0.00734314998284849], '
               '[0.016757997319172966, 0.520881076010481, -0.05054576192655418], [0.004853418308823841, '
               '-0.03831397138131293, 0.6156633546204768], [0.26404252547780904, -0.08668349431343583, '
               '0.10874646761593933], [-0.12805675959745338, -0.15257636177589162, -0.2891667957378999], '
               '[0.09864430572388094, -0.23909962469682772, -0.19839757179074852]], dtype=float)\n',
      'call': 'match_channels(*_copy_input((p, gaps, radius, fundamental)))',
      'gold_call': '_oracle_match_channels(*_copy_input((p, gaps, radius, fundamental)))'},
     {'setup': 'from copy import deepcopy as _copy_input\n'
               'import numpy as np\n'
               'p = 0.02\n'
               'gaps = np.array([0.0, 1.6e-05, 0.00012099999999999999], dtype=float)\n'
               'radius = 1.0\n'
               'fundamental = np.array([[0.999933334666654, 0.0, 0.0], [0.0, 0.9999360012287887, 0.0], [0.0, '
               '0.0, 0.9999535006486706], [0.9998000066665778, 0.0, 0.0], [0.0, 0.9998080061439213, 0.0], [0.0, '
               '0.0, 0.9998605032433449]], dtype=float)\n',
      'call': 'match_channels(*_copy_input((p, gaps, radius, fundamental)))',
      'gold_call': '_oracle_match_channels(*_copy_input((p, gaps, radius, fundamental)))'},
     {'setup': 'from copy import deepcopy as _copy_input\n'
               'import numpy as np\n'
               'p = 0.006\n'
               'gaps = np.array([0.0, 0.0, 0.0], dtype=float)\n'
               'radius = 1.0\n'
               'fundamental = np.array([[0.7821508025802294, 0.018023078732681434, 0.005391457995386246], '
               '[0.013205791951166565, 0.6566608576304201, -0.036304101984103776], [0.004152743624379308, '
               '-0.03026809361590916, 0.7284085966858126], [0.4731179249261132, -0.06704920197563168, '
               '0.07957739854150207], [-0.08780192857368009, 0.14047091194853267, -0.2163610156676706], '
               '[0.07436809174518481, -0.19083069976139694, 0.12628165977890443]], dtype=float)\n',
      'call': 'match_channels(*_copy_input((p, gaps, radius, fundamental)))',
      'gold_call': '_oracle_match_channels(*_copy_input((p, gaps, radius, fundamental)))'},
     {'setup': 'from copy import deepcopy as _copy_input\n'
               'import numpy as np\n'
               'p = 0.014\n'
               'gaps = np.array([0.0, 1.6e-05, 0.00012099999999999999], dtype=float)\n'
               'radius = 1.0\n'
               'fundamental = np.array([[0.5185753004448638, 0.04365485499788197, 0.010333790408375948], '
               '[0.01999151604013738, 0.28979374965513827, -0.0742059133345657], [0.004519660003162601, '
               '-0.045149446423070054, 0.4141945209730985], [-0.0904474635761679, -0.10838333522045583, '
               '0.1539258638478144], [-0.20326278625944078, -0.5684288678456276, -0.3871877136152853], '
               '[0.13214186065900624, -0.27661645027162507, -0.7081531477358382]], dtype=float)\n',
      'call': 'match_channels(*_copy_input((p, gaps, radius, fundamental)))',
      'gold_call': '_oracle_match_channels(*_copy_input((p, gaps, radius, fundamental)))'},
     {'setup': 'from copy import deepcopy as _copy_input\n'
               'import numpy as np\n'
               'p = 0.002\n'
               'gaps = np.array([0.0, 1.6e-05, 0.00012099999999999999], dtype=float)\n'
               'radius = 1.0\n'
               'fundamental = np.array([[0.6865311416057324, 0.026841852031857967, 0.0073970665459447546], '
               '[0.016843393296099404, 0.5170193142680312, -0.050950403414416263], [0.004864501554036709, '
               '-0.038505009969559925, 0.6124059926352794], [0.25811885261761386, -0.08717635246341333, '
               '0.10955586701049677], [-0.12925114453179762, -0.1604386802522381, -0.29110982991222556], '
               '[0.09929134348036098, -0.24022022690061734, -0.2074011139920885]], dtype=float)\n'
               '\n'
               'p = 0.004\n'
               '\n'
               'def _exception_code(function, *args, **kwargs):\n'
               '    try:\n'
               '        function(*args, **kwargs)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n',
      'call': '_exception_code(match_channels, *_copy_input((p, gaps, radius, fundamental)))',
      'gold_call': '_exception_code(_oracle_match_channels, *_copy_input((p, gaps, radius, fundamental)))'},
     {'setup': 'from copy import deepcopy as _copy_input\n'
               'import numpy as np\n'
               'p = 0.002\n'
               'gaps = np.array([0.0, 1.6e-05, 0.00012099999999999999], dtype=float)\n'
               'radius = 1.0\n'
               'fundamental = np.array([[0.6865311416057324, 0.026841852031857967, 0.0073970665459447546], '
               '[0.016843393296099404, 0.5170193142680312, -0.050950403414416263], [0.004864501554036709, '
               '-0.038505009969559925, 0.6124059926352794], [0.25811885261761386, -0.08717635246341333, '
               '0.10955586701049677], [-0.12925114453179762, -0.1604386802522381, -0.29110982991222556], '
               '[0.09929134348036098, -0.24022022690061734, -0.2074011139920885]], dtype=float)\n'
               '\n'
               'radius = 0.0\n'
               '\n'
               'def _exception_code(function, *args, **kwargs):\n'
               '    try:\n'
               '        function(*args, **kwargs)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n',
      'call': '_exception_code(match_channels, *_copy_input((p, gaps, radius, fundamental)))',
      'gold_call': '_exception_code(_oracle_match_channels, *_copy_input((p, gaps, radius, fundamental)))'},
     {'setup': 'from copy import deepcopy as _copy_input\n'
               'import numpy as np\n'
               'p = 0.002\n'
               'gaps = np.array([0.0, 1.6e-05, 0.00012099999999999999], dtype=float)\n'
               'radius = 1.0\n'
               'fundamental = np.array([[0.6865311416057324, 0.026841852031857967, 0.0073970665459447546], '
               '[0.016843393296099404, 0.5170193142680312, -0.050950403414416263], [0.004864501554036709, '
               '-0.038505009969559925, 0.6124059926352794], [0.25811885261761386, -0.08717635246341333, '
               '0.10955586701049677], [-0.12925114453179762, -0.1604386802522381, -0.29110982991222556], '
               '[0.09929134348036098, -0.24022022690061734, -0.2074011139920885]], dtype=float)\n'
               '\n'
               'fundamental[0,0] = np.nan\n'
               '\n'
               'def _exception_code(function, *args, **kwargs):\n'
               '    try:\n'
               '        function(*args, **kwargs)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n',
      'call': '_exception_code(match_channels, *_copy_input((p, gaps, radius, fundamental)))',
      'gold_call': '_exception_code(_oracle_match_channels, *_copy_input((p, gaps, radius, fundamental)))'},
     {'setup': 'from copy import deepcopy as _copy_input\n'
               'import numpy as np\n'
               'p = 0.002\n'
               'gaps = np.array([0.0, 1.6e-05, 0.00012099999999999999], dtype=float)\n'
               'radius = 1.0\n'
               'fundamental = np.array([[0.6865311416057324, 0.026841852031857967, 0.0073970665459447546], '
               '[0.016843393296099404, 0.5170193142680312, -0.050950403414416263], [0.004864501554036709, '
               '-0.038505009969559925, 0.6124059926352794], [0.25811885261761386, -0.08717635246341333, '
               '0.10955586701049677], [-0.12925114453179762, -0.1604386802522381, -0.29110982991222556], '
               '[0.09929134348036098, -0.24022022690061734, -0.2074011139920885]], dtype=float)\n'
               '\n'
               'fundamental = fundamental[:-1]\n'
               '\n'
               'def _exception_code(function, *args, **kwargs):\n'
               '    try:\n'
               '        function(*args, **kwargs)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n',
      'call': '_exception_code(match_channels, *_copy_input((p, gaps, radius, fundamental)))',
      'gold_call': '_exception_code(_oracle_match_channels, *_copy_input((p, gaps, radius, fundamental)))'},
     {'setup': 'from copy import deepcopy as _copy_input\n'
               'import numpy as np\n'
               'p = 0.002\n'
               'gaps = np.array([0.0, 1.6e-05, 0.00012099999999999999], dtype=float)\n'
               'radius = 1.0\n'
               'fundamental = np.array([[0.6865311416057324, 0.026841852031857967, 0.0073970665459447546], '
               '[0.016843393296099404, 0.5170193142680312, -0.050950403414416263], [0.004864501554036709, '
               '-0.038505009969559925, 0.6124059926352794], [0.25811885261761386, -0.08717635246341333, '
               '0.10955586701049677], [-0.12925114453179762, -0.1604386802522381, -0.29110982991222556], '
               '[0.09929134348036098, -0.24022022690061734, -0.2074011139920885]], dtype=float)\n'
               '\n'
               'gaps = gaps[[0,2,1]]\n'
               '\n'
               'def _exception_code(function, *args, **kwargs):\n'
               '    try:\n'
               '        function(*args, **kwargs)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n',
      'call': '_exception_code(match_channels, *_copy_input((p, gaps, radius, fundamental)))',
      'gold_call': '_exception_code(_oracle_match_channels, *_copy_input((p, gaps, radius, fundamental)))'},
     {'setup': 'from copy import deepcopy as _copy_input\n'
               'import numpy as np\n'
               'p = 0.002\n'
               'gaps = np.array([0.0, 1.6e-05, 0.00012099999999999999], dtype=float)\n'
               'radius = 1.0\n'
               'fundamental = np.array([[0.6865311416057324, 0.026841852031857967, 0.0073970665459447546], '
               '[0.016843393296099404, 0.5170193142680312, -0.050950403414416263], [0.004864501554036709, '
               '-0.038505009969559925, 0.6124059926352794], [0.25811885261761386, -0.08717635246341333, '
               '0.10955586701049677], [-0.12925114453179762, -0.1604386802522381, -0.29110982991222556], '
               '[0.09929134348036098, -0.24022022690061734, -0.2074011139920885]], dtype=float)\n'
               '\n'
               'fundamental[:] = 0.0\n'
               '\n'
               'def _exception_code(function, *args, **kwargs):\n'
               '    try:\n'
               '        function(*args, **kwargs)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n',
      'call': '_exception_code(match_channels, *_copy_input((p, gaps, radius, fundamental)))',
      'gold_call': '_exception_code(_oracle_match_channels, *_copy_input((p, gaps, radius, fundamental)))'}]
