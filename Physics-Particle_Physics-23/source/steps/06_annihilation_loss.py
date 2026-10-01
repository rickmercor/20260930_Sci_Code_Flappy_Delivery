"""
Compute the inclusive annihilation loss operator in the incident unit-flux basis.

The annihilation contraction in Sec. 4 and the corrected regular wave in Eq. (5.1) are equivalent to missing flux in the full elastic scattering matrix. The loss is inclusive over annihilation products.

Returns
-------
return result
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def annihilation_loss(p_open: np.ndarray, dressed: np.ndarray, short_amplitude: np.ndarray) -> np.ndarray:
    """Compute the inclusive annihilation loss operator in the incident unit-flux basis.

    p_open : ndarray, shape (M,)
        Positive incident open momenta in GeV.
    dressed : ndarray, shape (N,M)
        Dimensionless dressed regular-wave enhancement from dress_channels.
    short_amplitude : ndarray, shape (N,N)
        Finite amplitude f in GeV^-1 with positive semidefinite absorptive part.
    Returns
    -------
    ndarray, complex, shape (M,M)
        Dimensionless Hermitian inclusive loss operator L=I-S^dagger S.
        L[i,i] gives sigma_ann*v_rel multiplied by mu*p_open[i]/pi for
        distinguishable particles in the s-wave. Off-diagonal entries retain
        the coherence of incident channels. Evaluate from the absorptive
        contraction of the dressed wave. Return its Hermitian part.
    Raises
    ------
    ValueError
        Invalid rank, shape, nonnumeric or nonfinite input. All dimensions are nonzero.
        Nonpositive momenta, inconsistent shapes with N>=M>=1, or a negative absorptive eigenvalue.
        Absorptive eigenvalues may be negative only within 1e-12*max(1,||Im_H(f)||_2).
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


def _oracle_annihilation_loss(p_open: np.ndarray, dressed: np.ndarray, short_amplitude: np.ndarray) -> np.ndarray:
    (p_open, dressed) = _dressed_data(p_open, dressed)
    f = _short_matrix(short_amplitude, len(dressed))
    absorptive = (f - f.conj().T) / 2j
    b = np.asarray(dressed) * np.sqrt(p_open)
    loss = 4 * b.conj().T @ absorptive @ b
    return _checked_result((loss + loss.conj().T) / 2, 'annihilation loss')

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Deterministic scientific differential cases."""
    return [{'setup': 'from copy import deepcopy as _copy_input\n'
               'import numpy as np\n'
               'p_open = np.array([0.07], dtype=float)\n'
               'dressed = np.array([[(4.336145153515082+2.850936414716249j)]], dtype=complex)\n'
               'short_amplitude = np.array([[0.04j]], dtype=complex)\n',
      'call': 'annihilation_loss(*_copy_input((p_open, dressed, short_amplitude)))',
      'gold_call': '_oracle_annihilation_loss(*_copy_input((p_open, dressed, short_amplitude)))'},
     {'setup': 'from copy import deepcopy as _copy_input\n'
               'import numpy as np\n'
               'p_open = np.array([0.03], dtype=float)\n'
               'dressed = np.array([[(8.019901366229949-4.117046632049065j)], '
               '[(0.8710908567242177-0.44717777864281383j)], [(-2.587363759052456+1.3282329499645986j)]], '
               'dtype=complex)\n'
               'short_amplitude = np.array([[0.014285714285714289j, 0.028571428571428577j, '
               '0.042857142857142864j], [0.028571428571428577j, 0.057142857142857155j, 0.08571428571428573j], '
               '[0.042857142857142864j, 0.08571428571428573j, 0.1285714285714286j]], dtype=complex)\n',
      'call': 'annihilation_loss(*_copy_input((p_open, dressed, short_amplitude)))',
      'gold_call': '_oracle_annihilation_loss(*_copy_input((p_open, dressed, short_amplitude)))'},
     {'setup': 'from copy import deepcopy as _copy_input\n'
               'import numpy as np\n'
               'p_open = np.array([0.012, 0.04], dtype=float)\n'
               'dressed = np.array([[(-12.125448384811843-2.5613775975944635j), '
               '(1.4681401145487056-0.2749529393465179j)], [(3.8206449362022057+0.8835490730781961j), '
               '(-0.8223718264748284+0.3435063656471525j)]], dtype=complex)\n'
               'short_amplitude = np.array([[(-0.0015000000000000013+0.075j), (0.0135+0.05999999999999999j)], '
               '[(0.0135+0.05999999999999999j), (0.043500000000000004+0.16499999999999998j)]], dtype=complex)\n',
      'call': 'annihilation_loss(*_copy_input((p_open, dressed, short_amplitude)))',
      'gold_call': '_oracle_annihilation_loss(*_copy_input((p_open, dressed, short_amplitude)))'},
     {'setup': 'from copy import deepcopy as _copy_input\n'
               'import numpy as np\n'
               'p_open = np.array([0.008, 0.019], dtype=float)\n'
               'dressed = np.array([[(-9.929449765651155+0.3713235314285907j), '
               '(3.0885519303144715+6.5179690702406745j)], [(3.1697165671858154-0.20104089641576836j), '
               '(-1.235127766864195-2.7182530454522915j)], [(5.7922871128387685-0.5834480894377526j), '
               '(-2.9096378280289485-6.636966631440567j)]], dtype=complex)\n'
               'short_amplitude = np.array([[(-0.010000000000000009+0j), (0.09+0j), (0.09+0j)], [(0.09+0j), '
               '(0.14+0j), (0.09+0j)], [(0.09+0j), (0.09+0j), (0.29000000000000004+0j)]], dtype=complex)\n',
      'call': 'annihilation_loss(*_copy_input((p_open, dressed, short_amplitude)))',
      'gold_call': '_oracle_annihilation_loss(*_copy_input((p_open, dressed, short_amplitude)))'},
     {'setup': 'from copy import deepcopy as _copy_input\n'
               'import numpy as np\n'
               'p_open = np.array([0.02, 0.02, 0.02], dtype=float)\n'
               'dressed = np.array([[(-7.091688286494627-3.167083722380627j), '
               '(3.092243077412013+1.3810318256553615j), (-7.351768405593116-3.2833245880574276j)], '
               '[(6.6670111913326915+2.9776271551331264j), (-1.8277276246926788-0.8161551461966244j), '
               '(-0.11171147203083225-0.050029608318965246j)], [(-8.133523803221307-3.632909305153038j), '
               '(-3.6759637523703694-1.6419271472742378j), (9.049239391226413+4.041868994436778j)]], '
               'dtype=complex)\n'
               'short_amplitude = np.array([[(-1.000000000000001e-07+3.7142857142857146e-06j), '
               '(9.000000000000001e-07+1.4285714285714288e-06j), '
               '(9.000000000000001e-07+2.142857142857143e-06j)], '
               '[(9.000000000000001e-07+1.4285714285714288e-06j), '
               '(1.4000000000000001e-06+5.857142857142858e-06j), '
               '(9.000000000000001e-07+4.285714285714286e-06j)], '
               '[(9.000000000000001e-07+2.142857142857143e-06j), (9.000000000000001e-07+4.285714285714286e-06j), '
               '(2.9000000000000006e-06+9.42857142857143e-06j)]], dtype=complex)\n',
      'call': 'annihilation_loss(*_copy_input((p_open, dressed, short_amplitude)))',
      'gold_call': '_oracle_annihilation_loss(*_copy_input((p_open, dressed, short_amplitude)))'},
     {'setup': 'from copy import deepcopy as _copy_input\n'
               'import numpy as np\n'
               'p_open = np.array([0.005, 0.017, 0.08], dtype=float)\n'
               'dressed = np.array([[(0.09793876098708774+0.6005679085608904j), '
               '(0.32782759757431945+0.21348973624039574j), (0.13636432162070863+0.009918173933337906j)], '
               '[(-0.23263584993430997-0.3240023708480974j), (-0.3722142306360201-0.9065234229608082j), '
               '(-0.11296197414349227-0.45622135192925517j)], [(0.1708520544455568-0.7804820576027014j), '
               '(-0.0021033375740113804-0.05804753863464435j), (-0.09145463056678879+0.3421840697547279j)], '
               '[(-0.04372418985910638+0.20813617394865247j), (0.04156951639321246-0.4026926858899284j), '
               '(0.03276453109568687-0.23976099677703525j)]], dtype=complex)\n'
               'short_amplitude = np.array([[(-0.10000000000000009+3.333333333333333j), '
               '(0.8999999999999999+0.6666666666666665j), (0.8999999999999999+0.9999999999999999j), '
               '(0.8999999999999999+1.333333333333333j)], [(0.8999999999999999+0.6666666666666665j), '
               '(0.9000000000000001+4.333333333333333j), (0.8999999999999999+1.9999999999999998j), '
               '(0.8999999999999999+2.666666666666666j)], [(0.8999999999999999+0.9999999999999999j), '
               '(0.8999999999999999+1.9999999999999998j), (1.9000000000000004+5.999999999999998j), '
               '(0.8999999999999999+3.9999999999999996j)], [(0.8999999999999999+1.333333333333333j), '
               '(0.8999999999999999+2.666666666666666j), (0.8999999999999999+3.9999999999999996j), '
               '(2.9000000000000004+8.333333333333332j)]], dtype=complex)\n',
      'call': 'annihilation_loss(*_copy_input((p_open, dressed, short_amplitude)))',
      'gold_call': '_oracle_annihilation_loss(*_copy_input((p_open, dressed, short_amplitude)))'},
     {'setup': 'from copy import deepcopy as _copy_input\n'
               'import numpy as np\n'
               'p_open = np.array([0.02, 0.03, 0.07], dtype=float)\n'
               'dressed = np.array([[(0.11315720304408267-7.949683489859146j), '
               '(-0.7880218397177108+2.2024441467798317j), (-0.6508856089753824-0.8932648920610126j)], '
               '[(0.7516846353021931+3.182006367753144j), (0.16583621350963856-1.711569279022107j), '
               '(-0.6967348146986292+1.9183566285054983j)], [(-3.4817845987519074+1.6712333545202767j), '
               '(-2.6573336742014955-0.5780859950825483j), (2.5652985572312783-3.0308225391412584j)]], '
               'dtype=complex)\n'
               'short_amplitude = np.array([[(-0.0010000000000000009+0.1j), (0.009+0j), (0.009+0j)], '
               '[(0.009+0j), (0.014000000000000002+0.1j), (0.009+0j)], [(0.009+0j), (0.009+0j), '
               '(0.029000000000000005+0.1j)]], dtype=complex)\n',
      'call': 'annihilation_loss(*_copy_input((p_open, dressed, short_amplitude)))',
      'gold_call': '_oracle_annihilation_loss(*_copy_input((p_open, dressed, short_amplitude)))'},
     {'setup': 'from copy import deepcopy as _copy_input\n'
               'import numpy as np\n'
               'p_open = np.array([0.004, 0.05], dtype=float)\n'
               'dressed = np.array([[(-0.1786384486366727-0.7672752912082988j), '
               '(-0.3794712028935936-1.5066340329391399j)], [(0.057338926349471825+0.08101253866560622j), '
               '(0.12228351565360907+0.12415423959148453j)], [(0.10560063861233467-0.2812399884891341j), '
               '(0.22646340409468585-0.7075241829136016j)]], dtype=complex)\n'
               'short_amplitude = np.array([[(-0.007000000000000005+0.26j), (0.063+0.1j), (0.063+0.15j)], '
               '[(0.063+0.1j), (0.098+0.41j), (0.063+0.3j)], [(0.063+0.15j), (0.063+0.3j), '
               '(0.203+0.6599999999999999j)]], dtype=complex)\n',
      'call': 'annihilation_loss(*_copy_input((p_open, dressed, short_amplitude)))',
      'gold_call': '_oracle_annihilation_loss(*_copy_input((p_open, dressed, short_amplitude)))'},
     {'setup': 'from copy import deepcopy as _copy_input\n'
               'import numpy as np\n'
               'p_open = np.array([0.07], dtype=float)\n'
               'dressed = np.array([[(4.336145153515082+2.850936414716249j)]], dtype=complex)\n'
               'short_amplitude = np.array([[0.04j]], dtype=complex)\n'
               '\n'
               'p_open[0] = 0.0\n'
               '\n'
               'def _exception_code(function, *args, **kwargs):\n'
               '    try:\n'
               '        function(*args, **kwargs)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n',
      'call': '_exception_code(annihilation_loss, *_copy_input((p_open, dressed, short_amplitude)))',
      'gold_call': '_exception_code(_oracle_annihilation_loss, *_copy_input((p_open, dressed, short_amplitude)))'},
     {'setup': 'from copy import deepcopy as _copy_input\n'
               'import numpy as np\n'
               'p_open = np.array([0.07], dtype=float)\n'
               'dressed = np.array([[(4.336145153515082+2.850936414716249j)]], dtype=complex)\n'
               'short_amplitude = np.array([[0.04j]], dtype=complex)\n'
               '\n'
               'dressed[0,0] = np.inf\n'
               '\n'
               'def _exception_code(function, *args, **kwargs):\n'
               '    try:\n'
               '        function(*args, **kwargs)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n',
      'call': '_exception_code(annihilation_loss, *_copy_input((p_open, dressed, short_amplitude)))',
      'gold_call': '_exception_code(_oracle_annihilation_loss, *_copy_input((p_open, dressed, short_amplitude)))'},
     {'setup': 'from copy import deepcopy as _copy_input\n'
               'import numpy as np\n'
               'p_open = np.array([0.07], dtype=float)\n'
               'dressed = np.array([[(4.336145153515082+2.850936414716249j)]], dtype=complex)\n'
               'short_amplitude = np.array([[0.04j]], dtype=complex)\n'
               '\n'
               'dressed = dressed[:,:0]\n'
               '\n'
               'def _exception_code(function, *args, **kwargs):\n'
               '    try:\n'
               '        function(*args, **kwargs)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n',
      'call': '_exception_code(annihilation_loss, *_copy_input((p_open, dressed, short_amplitude)))',
      'gold_call': '_exception_code(_oracle_annihilation_loss, *_copy_input((p_open, dressed, short_amplitude)))'},
     {'setup': 'from copy import deepcopy as _copy_input\n'
               'import numpy as np\n'
               'p_open = np.array([0.07], dtype=float)\n'
               'dressed = np.array([[(4.336145153515082+2.850936414716249j)]], dtype=complex)\n'
               'short_amplitude = np.array([[0.04j]], dtype=complex)\n'
               '\n'
               'short_amplitude = np.eye(2)*1j\n'
               '\n'
               'def _exception_code(function, *args, **kwargs):\n'
               '    try:\n'
               '        function(*args, **kwargs)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n',
      'call': '_exception_code(annihilation_loss, *_copy_input((p_open, dressed, short_amplitude)))',
      'gold_call': '_exception_code(_oracle_annihilation_loss, *_copy_input((p_open, dressed, short_amplitude)))'},
     {'setup': 'from copy import deepcopy as _copy_input\n'
               'import numpy as np\n'
               'p_open = np.array([0.07], dtype=float)\n'
               'dressed = np.array([[(4.336145153515082+2.850936414716249j)]], dtype=complex)\n'
               'short_amplitude = np.array([[0.04j]], dtype=complex)\n'
               '\n'
               'short_amplitude[0,0] = -0.1j\n'
               '\n'
               'def _exception_code(function, *args, **kwargs):\n'
               '    try:\n'
               '        function(*args, **kwargs)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n',
      'call': '_exception_code(annihilation_loss, *_copy_input((p_open, dressed, short_amplitude)))',
      'gold_call': '_exception_code(_oracle_annihilation_loss, *_copy_input((p_open, dressed, short_amplitude)))'}]
