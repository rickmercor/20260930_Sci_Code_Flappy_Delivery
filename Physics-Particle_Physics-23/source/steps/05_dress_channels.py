"""
Dress the ordinary Sommerfeld matrix with short-distance absorptive and dispersive feedback.

Eq. (5.1) combines the ordinary enhancement, dispersive regulator, open-channel spectral term, and the finite matched short-distance amplitude. These matrices can fail to commute.

Returns
-------
return result
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def dress_channels(p_open: np.ndarray, matched: np.ndarray, regulator: np.ndarray, short_amplitude: np.ndarray) -> np.ndarray:
    """Dress the ordinary Sommerfeld matrix with short-distance absorptive and dispersive feedback.

    p_open : ndarray, shape (M,)
        Positive open momenta in GeV.
    matched : ndarray, shape (N+M,M)
        Ordinary Sigma_0 above S_0 as in match_channels.
    regulator : ndarray, shape (N,N)
        Hermitian outgoing-wave regulator Z in GeV; zero is an allowed comparison.
    short_amplitude : ndarray, shape (N,N)
        Finite matched complex amplitude f in GeV^-1, with positive semidefinite
        (f-f^dagger)/(2i). Singular f is allowed; the dressing system is nonsingular.
    Returns
    -------
    ndarray, complex, shape (N,M)
        Dimensionless dressed regular-wave enhancement, in physical state rows
        and incident open-channel columns, to contract with the absorptive part
        of f. It is the continuous extension of the resonant prescription to
        singular f. Inputs retain their original values.
    Raises
    ------
    ValueError
        Invalid rank, shape, nonnumeric or nonfinite input. All dimensions are nonzero.
        Nonpositive momenta, inconsistent shapes, nonunitary S_0, non-Hermitian regulator, negative absorptive eigenvalue or singular dressing system.
        Symmetry/Hermiticity is checked with rtol=1e-10 and atol=1e-12.
        Elastic unitarity is checked with rtol=atol=1e-8.
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


def _oracle_dress_channels(p_open: np.ndarray, matched: np.ndarray, regulator: np.ndarray, short_amplitude: np.ndarray) -> np.ndarray:
    (p_open, sigma, _) = _matched_data(p_open, matched)
    regulator = _hermitian_matrix(regulator, 'regulator', len(sigma))
    short_amplitude = _short_matrix(short_amplitude, len(sigma))
    spectral = sigma * p_open @ sigma.conj().T
    kernel = np.asarray(regulator) + 1j * spectral
    return _checked_solve(np.eye(len(sigma)) - kernel @ short_amplitude, sigma, 'dressing system')

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Deterministic scientific differential cases."""
    return [{'setup': 'from copy import deepcopy as _copy_input\n\ndef _preservation_checked(function, *args):\n    before = _copy_input(args)\n    result = function(*args)\n    if any(not np.array_equal(after, old, equal_nan=True)\n           for after, old in zip(args, before)):\n        raise AssertionError("dress_channels must retain input values")\n    return result\n'
               'import numpy as np\n'
               'p_open = np.array([0.07], dtype=float)\n'
               'matched = np.array([[(5.632622160261771+1.8042838036432807j)], '
               '[(0.8138784566625339+0.5810351605373051j)]], dtype=complex)\n'
               'regulator = np.array([[(7.644460847998718+0j)]], dtype=complex)\n'
               'short_amplitude = np.array([[0.04j]], dtype=complex)\n',
      'call': '_preservation_checked(dress_channels, *_copy_input((p_open, matched, regulator, short_amplitude)))',
      'gold_call': '_preservation_checked(_oracle_dress_channels, *_copy_input((p_open, matched, regulator, short_amplitude)))'},
     {'setup': 'from copy import deepcopy as _copy_input\n\ndef _preservation_checked(function, *args):\n    before = _copy_input(args)\n    result = function(*args)\n    if any(not np.array_equal(after, old, equal_nan=True)\n           for after, old in zip(args, before)):\n        raise AssertionError("dress_channels must retain input values")\n    return result\n'
               'import numpy as np\n'
               'p_open = np.array([0.03], dtype=float)\n'
               'matched = np.array([[(8.054981205610996-4.091655754593365j)], '
               '[(0.874901094025746-0.4444199191398281j)], [(-2.5986811432624535+1.3200413983308035j)], '
               '[(0.5897880250310983-0.8075581004051142j)]], dtype=complex)\n'
               'regulator = np.array([[(-4.820660372810161+0j), (-0.5236015983699929-4.224360319583381e-18j), '
               '(1.5552313393565604-3.611478092986617e-17j)], [(-0.5236015983699929+4.224360319583381e-18j), '
               '(-0.05687159281370263+0j), (0.16892324954381688-1.2224396844774502e-17j)], '
               '[(1.5552313393565604+3.611478092986617e-17j), (0.16892324954381688+1.2224396844774502e-17j), '
               '(-0.5017454729976789+0j)]], dtype=complex)\n'
               'short_amplitude = np.array([[0.014285714285714289j, 0.028571428571428577j, '
               '0.042857142857142864j], [0.028571428571428577j, 0.057142857142857155j, 0.08571428571428573j], '
               '[0.042857142857142864j, 0.08571428571428573j, 0.1285714285714286j]], dtype=complex)\n',
      'call': '_preservation_checked(dress_channels, *_copy_input((p_open, matched, regulator, short_amplitude)))',
      'gold_call': '_preservation_checked(_oracle_dress_channels, *_copy_input((p_open, matched, regulator, short_amplitude)))'},
     {'setup': 'from copy import deepcopy as _copy_input\n\ndef _preservation_checked(function, *args):\n    before = _copy_input(args)\n    result = function(*args)\n    if any(not np.array_equal(after, old, equal_nan=True)\n           for after, old in zip(args, before)):\n        raise AssertionError("dress_channels must retain input values")\n    return result\n'
               'import numpy as np\n'
               'p_open = np.array([0.012, 0.04], dtype=float)\n'
               'matched = np.array([[(-13.695644900901067-3.0703777706363558j), '
               '(1.660825510602718-0.28772116850688967j)], [(4.335927681591876+1.0507345371816845j), '
               '(-0.8859652370776125+0.3481258512043014j)], [(0.9031335059433552+0.3854674696422496j), '
               '(-0.07711385209250922-0.17267933896537724j)], [(-0.0771138520925092-0.17267933896537724j), '
               '(0.3157723427472424-0.9297973581731952j)]], dtype=complex)\n'
               'regulator = np.array([[0j, 0j], [0j, 0j]], dtype=complex)\n'
               'short_amplitude = np.array([[(-0.0015000000000000013+0.075j), (0.0135+0.05999999999999999j)], '
               '[(0.0135+0.05999999999999999j), (0.043500000000000004+0.16499999999999998j)]], dtype=complex)\n',
      'call': '_preservation_checked(dress_channels, *_copy_input((p_open, matched, regulator, short_amplitude)))',
      'gold_call': '_preservation_checked(_oracle_dress_channels, *_copy_input((p_open, matched, regulator, short_amplitude)))'},
     {'setup': 'from copy import deepcopy as _copy_input\n\ndef _preservation_checked(function, *args):\n    before = _copy_input(args)\n    result = function(*args)\n    if any(not np.array_equal(after, old, equal_nan=True)\n           for after, old in zip(args, before)):\n        raise AssertionError("dress_channels must retain input values")\n    return result\n'
               'import numpy as np\n'
               'p_open = np.array([0.008, 0.019], dtype=float)\n'
               'matched = np.array([[(-17.18560802789641-1.169798283364947j), '
               '(2.0036213442647393+1.2060378570520867j)], [(5.487641583905056+0.26591104874096094j), '
               '(-0.9316073683418679-0.9494452484067624j)], [(10.032179867011925+0.20435270505190375j), '
               '(-2.4671162093364165-3.213205248404743j)], [(0.9613952676547358+0.1699606380060729j), '
               '(-0.20009234813143056+0.08243526599033932j)], [(-0.20009234813143054+0.08243526599033932j), '
               '(-0.5626691015344547+0.7978539724277632j)]], dtype=complex)\n'
               'regulator = np.array([[(30.558514508373467+0j), (-9.866283601449894-5.551115123125783e-17j), '
               '(-18.320903845952543+0j)], [(-9.866283601449894+5.551115123125783e-17j), '
               '(3.1893654393190722+0j), (5.9324572258035015+0j)], [(-18.320903845952543+0j), '
               '(5.9324572258035015+0j), (11.06082711385437+0j)]], dtype=complex)\n'
               'short_amplitude = np.array([[(-0.010000000000000009+0j), (0.09+0j), (0.09+0j)], [(0.09+0j), '
               '(0.14+0j), (0.09+0j)], [(0.09+0j), (0.09+0j), (0.29000000000000004+0j)]], dtype=complex)\n',
      'call': '_preservation_checked(dress_channels, *_copy_input((p_open, matched, regulator, short_amplitude)))',
      'gold_call': '_preservation_checked(_oracle_dress_channels, *_copy_input((p_open, matched, regulator, short_amplitude)))'},
     {'setup': 'from copy import deepcopy as _copy_input\n\ndef _preservation_checked(function, *args):\n    before = _copy_input(args)\n    result = function(*args)\n    if any(not np.array_equal(after, old, equal_nan=True)\n           for after, old in zip(args, before)):\n        raise AssertionError("dress_channels must retain input values")\n    return result\n'
               'import numpy as np\n'
               'p_open = np.array([0.02, 0.02, 0.02], dtype=float)\n'
               'matched = np.array([[(-7.0918268166115475-3.167015159303594j), '
               '(3.092349018101233+1.3809581750141355j), (-7.35197158060742-3.2831886689638345j)], '
               '[(6.66718818915394+2.9773832061890535j), (-1.827722494803457-0.8162106884060124j), '
               '(-0.11173117272311264-0.049896074302324456j)], [(-8.133810674802804-3.6323365440435085j), '
               '(-3.676136297924166-1.6416615470532192j), (9.049593336885362+4.041299939293001j)], '
               '[(0.6674628258413082+0.7446431199708593j), (2.630947422402707e-17+1.8746478584303784e-17j), '
               '(1.274990579227792e-17+3.168237764009003e-17j)], '
               '[(1.3175154249260747e-17-1.0443587305000175e-17j), (0.6674628258413081+0.7446431199708593j), '
               '(-8.170403639448021e-19-1.719945305593733e-17j)], '
               '[(4.213528024864062e-17+3.7150950137396716e-17j), '
               '(-4.925512509009907e-18+3.856689825679965e-18j), (0.6674628258413081+0.7446431199708593j)]], '
               'dtype=complex)\n'
               'regulator = np.array([[(6.1188031213169145+0j), (-2.7993573860678826-4.440892098500626e-16j), '
               '(-1.0859791646828556-2.7755575615628914e-17j)], [(-2.7993573860678826+4.440892098500626e-16j), '
               '(2.567912087843273+0j), (-2.6064464886898473-5.551115123125783e-17j)], '
               '[(-1.0859791646828556+2.7755575615628914e-17j), (-2.6064464886898473+5.551115123125783e-17j), '
               '(8.678949118271547+0j)]], dtype=complex)\n'
               'short_amplitude = np.array([[(-1.000000000000001e-07+3.7142857142857146e-06j), '
               '(9.000000000000001e-07+1.4285714285714288e-06j), '
               '(9.000000000000001e-07+2.142857142857143e-06j)], '
               '[(9.000000000000001e-07+1.4285714285714288e-06j), '
               '(1.4000000000000001e-06+5.857142857142858e-06j), '
               '(9.000000000000001e-07+4.285714285714286e-06j)], '
               '[(9.000000000000001e-07+2.142857142857143e-06j), (9.000000000000001e-07+4.285714285714286e-06j), '
               '(2.9000000000000006e-06+9.42857142857143e-06j)]], dtype=complex)\n',
      'call': '_preservation_checked(dress_channels, *_copy_input((p_open, matched, regulator, short_amplitude)))',
      'gold_call': '_preservation_checked(_oracle_dress_channels, *_copy_input((p_open, matched, regulator, short_amplitude)))'},
     {'setup': 'from copy import deepcopy as _copy_input\n\ndef _preservation_checked(function, *args):\n    before = _copy_input(args)\n    result = function(*args)\n    if any(not np.array_equal(after, old, equal_nan=True)\n           for after, old in zip(args, before)):\n        raise AssertionError("dress_channels must retain input values")\n    return result\n'
               'import numpy as np\n'
               'p_open = np.array([0.005, 0.017, 0.08], dtype=float)\n'
               'matched = np.array([[(-17.633550295131986+6.120289647265752j), '
               '(3.9567725574571666-1.877146483852491j), (-2.6350520236521464-1.16701528449509j)], '
               '[(11.620143781523776+1.4256658411297434j), (-2.234027846319984+1.0891781306808406j), '
               '(1.1757718290916068-1.753034316167282j)], [(-7.255669505575971-13.602908596772481j), '
               '(-3.477955974219149-3.161405813674027j), (0.35371721954242197+5.364012641743451j)], '
               '[(-8.493151745319924+4.4334129931187185j), (1.5133794437492019-1.2930295495018935j), '
               '(-1.44863264196349-1.4464090055190977j)], [(0.4313628051549057-0.03930062071284705j), '
               '(-0.23505723571015077+0.4694862869297617j), (0.7029694014947151-0.20626762816265226j)], '
               '[(-0.23505723571015083+0.4694862869297617j), (0.29832050362868473+0.7476850785760338j), '
               '(-0.16386225840270707+0.22237756853120497j)], [(0.7029694014947151-0.20626762816265226j), '
               '(-0.16386225840270705+0.22237756853120497j), (-0.5808775125491223+0.22263507997252807j)]], '
               'dtype=complex)\n'
               'regulator = np.array([[(-12.001712523082293+0j), (7.288741123096019-2.220446049250313e-16j), '
               '(-1.440020728173221-7.216449660063518e-16j), (-5.743664246713516+1.1102230246251565e-16j)], '
               '[(7.288741123096019+2.220446049250313e-16j), (-4.316008025277551+0j), '
               '(0.5890928416540157+5.551115123125783e-17j), (3.523297390041928+0j)], '
               '[(-1.440020728173221+7.216449660063518e-16j), (0.5890928416540157-5.551115123125783e-17j), '
               '(1.10125668197943+0j), (-0.713641719616124+1.942890293094024e-16j)], '
               '[(-5.743664246713516-1.1102230246251565e-16j), (3.523297390041928+0j), '
               '(-0.713641719616124-1.942890293094024e-16j), (-2.729412046848853+0j)]], dtype=complex)\n'
               'short_amplitude = np.array([[(-0.10000000000000009+3.333333333333333j), '
               '(0.8999999999999999+0.6666666666666665j), (0.8999999999999999+0.9999999999999999j), '
               '(0.8999999999999999+1.333333333333333j)], [(0.8999999999999999+0.6666666666666665j), '
               '(0.9000000000000001+4.333333333333333j), (0.8999999999999999+1.9999999999999998j), '
               '(0.8999999999999999+2.666666666666666j)], [(0.8999999999999999+0.9999999999999999j), '
               '(0.8999999999999999+1.9999999999999998j), (1.9000000000000004+5.999999999999998j), '
               '(0.8999999999999999+3.9999999999999996j)], [(0.8999999999999999+1.333333333333333j), '
               '(0.8999999999999999+2.666666666666666j), (0.8999999999999999+3.9999999999999996j), '
               '(2.9000000000000004+8.333333333333332j)]], dtype=complex)\n',
      'call': '_preservation_checked(dress_channels, *_copy_input((p_open, matched, regulator, short_amplitude)))',
      'gold_call': '_preservation_checked(_oracle_dress_channels, *_copy_input((p_open, matched, regulator, short_amplitude)))'},
     {'setup': 'from copy import deepcopy as _copy_input\n\ndef _preservation_checked(function, *args):\n    before = _copy_input(args)\n    result = function(*args)\n    if any(not np.array_equal(after, old, equal_nan=True)\n           for after, old in zip(args, before)):\n        raise AssertionError("dress_channels must retain input values")\n    return result\n'
               'import numpy as np\n'
               'p_open = np.array([0.02, 0.03, 0.07], dtype=float)\n'
               'matched = np.array([[(0.9161706126205966-9.964169330244452j), '
               '(-0.910687475906394+3.480815874329809j), (-1.5201869454615402-1.6030239286287995j)], '
               '[(1.6531362246725956+4.645219608680386j), (0.5911214822246484-1.9497587639751477j), '
               '(-1.3050188026461842+2.440128355223857j)], [(-6.764333470016928+1.3209583757012169j), '
               '(-3.5139471378087603-1.2092706640396005j), (4.63505530220009-3.793963633653709j)], '
               '[(-0.45899782772882175-0.08696984752085045j), (0.5865912116988481+0.24882487300829986j), '
               '(-0.4552364713549425+0.4105044796842104j)], [(0.586591211698848+0.24882487300829986j), '
               '(0.43833075267998306+0.5563458194233j), (0.2951454311346694+0.07232971640776249j)], '
               '[(-0.4552364713549425+0.41050447968421044j), (0.2951454311346694+0.0723297164077625j), '
               '(0.15847516318469979-0.7118911690460712j)]], dtype=complex)\n'
               'regulator = np.array([[(-0.31453181812880604+0j), (-0.4375722651610251+0j), '
               '(1.7267616868506965+1.1102230246251565e-16j)], [(-0.4375722651610251+0j), '
               '(-0.31885315518109997+0j), (1.0739725406721023-1.6653345369377348e-16j)], '
               '[(1.7267616868506965-1.1102230246251565e-16j), (1.0739725406721023+1.6653345369377348e-16j), '
               '(-3.393705422406922+0j)]], dtype=complex)\n'
               'short_amplitude = np.array([[(-0.0010000000000000009+0.1j), (0.009+0j), (0.009+0j)], '
               '[(0.009+0j), (0.014000000000000002+0.1j), (0.009+0j)], [(0.009+0j), (0.009+0j), '
               '(0.029000000000000005+0.1j)]], dtype=complex)\n',
      'call': '_preservation_checked(dress_channels, *_copy_input((p_open, matched, regulator, short_amplitude)))',
      'gold_call': '_preservation_checked(_oracle_dress_channels, *_copy_input((p_open, matched, regulator, short_amplitude)))'},
     {'setup': 'from copy import deepcopy as _copy_input\n\ndef _preservation_checked(function, *args):\n    before = _copy_input(args)\n    result = function(*args)\n    if any(not np.array_equal(after, old, equal_nan=True)\n           for after, old in zip(args, before)):\n        raise AssertionError("dress_channels must retain input values")\n    return result\n'
               'import numpy as np\n'
               'p_open = np.array([0.004, 0.05], dtype=float)\n'
               'matched = np.array([[(-24.534854975625983+0.2704551205689916j), '
               '(0.9041677395523329+0.7565040266865762j)], [(7.913473483044101-0.26415635927788156j), '
               '(-0.29588520904593124-0.6316330591784719j)], [(14.67402929377214-0.9483995468702763j), '
               '(-0.5596911060141848-2.175949005587194j)], [(0.9671987204698124+0.006255558339704907j), '
               '(-0.25393617990655903+0.0019798095564927537j)], [(-0.2539361799065591+0.0019798095564927537j), '
               '(-0.9669836081287407+0.021335381386455854j)]], dtype=complex)\n'
               'regulator = np.array([[(816.241553541258+0j), (-263.3342985075047+2.609024107869118e-15j), '
               '(-488.4684112117317+4.440892098500626e-16j)], [(-263.3342985075047-2.609024107869118e-15j), '
               '(84.9564932946915+0j), (157.58911694933443+9.992007221626409e-16j)], '
               '[(-488.4684112117317-4.440892098500626e-16j), (157.58911694933443-9.992007221626409e-16j), '
               '(292.3187604981176+0j)]], dtype=complex)\n'
               'short_amplitude = np.array([[(-0.007000000000000005+0.26j), (0.063+0.1j), (0.063+0.15j)], '
               '[(0.063+0.1j), (0.098+0.41j), (0.063+0.3j)], [(0.063+0.15j), (0.063+0.3j), '
               '(0.203+0.6599999999999999j)]], dtype=complex)\n',
      'call': '_preservation_checked(dress_channels, *_copy_input((p_open, matched, regulator, short_amplitude)))',
      'gold_call': '_preservation_checked(_oracle_dress_channels, *_copy_input((p_open, matched, regulator, short_amplitude)))'},
     {'setup': 'from copy import deepcopy as _copy_input\n'
               'import numpy as np\n'
               'p_open = np.array([0.07], dtype=float)\n'
               'matched = np.array([[(5.632622160261771+1.8042838036432807j)], '
               '[(0.8138784566625339+0.5810351605373051j)]], dtype=complex)\n'
               'regulator = np.array([[(7.644460847998718+0j)]], dtype=complex)\n'
               'short_amplitude = np.array([[0.04j]], dtype=complex)\n'
               '\n'
               'p_open[0] = -0.1\n'
               '\n'
               'def _exception_code(function, *args, **kwargs):\n'
               '    try:\n'
               '        function(*args, **kwargs)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n',
      'call': '_exception_code(dress_channels, *_copy_input((p_open, matched, regulator, short_amplitude)))',
      'gold_call': '_exception_code(_oracle_dress_channels, *_copy_input((p_open, matched, regulator, short_amplitude)))'},
     {'setup': 'from copy import deepcopy as _copy_input\n'
               'import numpy as np\n'
               'p_open = np.array([0.07], dtype=float)\n'
               'matched = np.array([[(5.632622160261771+1.8042838036432807j)], '
               '[(0.8138784566625339+0.5810351605373051j)]], dtype=complex)\n'
               'regulator = np.array([[(7.644460847998718+0j)]], dtype=complex)\n'
               'short_amplitude = np.array([[0.04j]], dtype=complex)\n'
               '\n'
               'short_amplitude[0,0] = np.nan\n'
               '\n'
               'def _exception_code(function, *args, **kwargs):\n'
               '    try:\n'
               '        function(*args, **kwargs)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n',
      'call': '_exception_code(dress_channels, *_copy_input((p_open, matched, regulator, short_amplitude)))',
      'gold_call': '_exception_code(_oracle_dress_channels, *_copy_input((p_open, matched, regulator, short_amplitude)))'},
     {'setup': 'from copy import deepcopy as _copy_input\n'
               'import numpy as np\n'
               'p_open = np.array([0.07], dtype=float)\n'
               'matched = np.array([[(5.632622160261771+1.8042838036432807j)], '
               '[(0.8138784566625339+0.5810351605373051j)]], dtype=complex)\n'
               'regulator = np.array([[(7.644460847998718+0j)]], dtype=complex)\n'
               'short_amplitude = np.array([[0.04j]], dtype=complex)\n'
               '\n'
               'regulator = np.zeros((2,2))\n'
               '\n'
               'def _exception_code(function, *args, **kwargs):\n'
               '    try:\n'
               '        function(*args, **kwargs)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n',
      'call': '_exception_code(dress_channels, *_copy_input((p_open, matched, regulator, short_amplitude)))',
      'gold_call': '_exception_code(_oracle_dress_channels, *_copy_input((p_open, matched, regulator, short_amplitude)))'},
     {'setup': 'from copy import deepcopy as _copy_input\n'
               'import numpy as np\n'
               'p_open = np.array([0.07], dtype=float)\n'
               'matched = np.array([[(5.632622160261771+1.8042838036432807j)], '
               '[(0.8138784566625339+0.5810351605373051j)]], dtype=complex)\n'
               'regulator = np.array([[(7.644460847998718+0j)]], dtype=complex)\n'
               'short_amplitude = np.array([[0.04j]], dtype=complex)\n'
               '\n'
               'regulator[0,0] += 1j\n'
               '\n'
               'def _exception_code(function, *args, **kwargs):\n'
               '    try:\n'
               '        function(*args, **kwargs)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n',
      'call': '_exception_code(dress_channels, *_copy_input((p_open, matched, regulator, short_amplitude)))',
      'gold_call': '_exception_code(_oracle_dress_channels, *_copy_input((p_open, matched, regulator, short_amplitude)))'},
     {'setup': 'from copy import deepcopy as _copy_input\n'
               'import numpy as np\n'
               'p_open = np.array([0.07], dtype=float)\n'
               'matched = np.array([[(5.632622160261771+1.8042838036432807j)], '
               '[(0.8138784566625339+0.5810351605373051j)]], dtype=complex)\n'
               'regulator = np.array([[(7.644460847998718+0j)]], dtype=complex)\n'
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
      'call': '_exception_code(dress_channels, *_copy_input((p_open, matched, regulator, short_amplitude)))',
      'gold_call': '_exception_code(_oracle_dress_channels, *_copy_input((p_open, matched, regulator, short_amplitude)))'},
     {'setup': 'from copy import deepcopy as _copy_input\n'
               'import numpy as np\n'
               'p_open = np.array([0.07], dtype=float)\n'
               'matched = np.array([[(5.632622160261771+1.8042838036432807j)], '
               '[(0.8138784566625339+0.5810351605373051j)]], dtype=complex)\n'
               'regulator = np.array([[(7.644460847998718+0j)]], dtype=complex)\n'
               'short_amplitude = np.array([[0.04j]], dtype=complex)\n'
               '\n'
               'matched = np.array([[0.0],[-1.0]],dtype=complex)\n'
               'regulator = np.array([[1.0]],dtype=complex)\n'
               'short_amplitude = np.array([[1.0]],dtype=complex)\n'
               '\n'
               'def _exception_code(function, *args, **kwargs):\n'
               '    try:\n'
               '        function(*args, **kwargs)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n',
      'call': '_exception_code(dress_channels, *_copy_input((p_open, matched, regulator, short_amplitude)))',
      'gold_call': '_exception_code(_oracle_dress_channels, *_copy_input((p_open, matched, regulator, short_amplitude)))'}]
