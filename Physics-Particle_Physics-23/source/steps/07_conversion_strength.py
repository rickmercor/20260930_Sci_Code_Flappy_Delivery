"""
Compute state-conversion probabilities from the full regulated scattering matrix.

Eq. (5.2) supplies the full regulated S matrix, and Eq. (4.30) fixes incident-momentum normalization for state-changing scattering. Its conversion sector contributes to the benchmark branching fraction.

Returns
-------
return result
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def conversion_strength(p_open: np.ndarray, matched: np.ndarray, dressed: np.ndarray, short_amplitude: np.ndarray) -> np.ndarray:
    """Compute state-conversion probabilities from the full regulated scattering matrix.

    p_open : ndarray, shape (M,)
        Positive open momenta in GeV, ordered by physical channel.
    matched : ndarray, shape (N+M,M)
        Ordinary Sigma_0 above S_0 as in match_channels.
    dressed : ndarray, shape (N,M)
        Dressed regular-wave enhancement for this same configuration.
    short_amplitude : ndarray, shape (N,N)
        Finite matched amplitude f in GeV^-1, allowing a singular matrix,
        with positive semidefinite absorptive part (f-f^dagger)/(2i).
    Returns
    -------
    ndarray, real, shape (M,M)
        Dimensionless transition strengths: entry [j,i] is |S[j,i]|^2 for j!=i,
        and every diagonal entry is zero. Rows are outgoing open channels;
        columns are incident ones. Thus column i summed and multiplied by
        pi/(mu*p_open[i]) is the conversion rate coefficient in GeV^-2.
    Raises
    ------
    ValueError
        Invalid rank, shape, nonnumeric or nonfinite input. All dimensions are nonzero.
        Nonpositive momenta, inconsistent shapes with N>=M>=1, nonunitary S_0 or a negative absorptive eigenvalue.
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


def _oracle_conversion_strength(p_open: np.ndarray, matched: np.ndarray, dressed: np.ndarray, short_amplitude: np.ndarray) -> np.ndarray:
    (p_open, sigma, s0) = _matched_data(p_open, matched)
    m = len(p_open)
    (_, dressed) = _dressed_data(p_open, dressed)
    if dressed.shape != sigma.shape:
        raise ValueError('dressed and Sigma_0 must have the same shape')
    short_amplitude = _short_matrix(short_amplitude, len(sigma))
    scattering = s0 @ (np.eye(m) + 2j * np.sqrt(p_open)[:, None] * (sigma.conj().T @ short_amplitude @ dressed) * np.sqrt(p_open)[None, :])
    result = np.abs(scattering) ** 2
    np.fill_diagonal(result, 0.0)
    return _checked_result(result, 'conversion strengths')

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Deterministic scientific differential cases."""
    return [{'setup': 'from copy import deepcopy as _copy_input\n'
               'import numpy as np\n'
               'p_open = np.array([0.07], dtype=float)\n'
               'matched = np.array([[(5.632622160261771+1.8042838036432807j)], '
               '[(0.8138784566625339+0.5810351605373051j)]], dtype=complex)\n'
               'dressed = np.array([[(4.336145153515082+2.850936414716249j)]], dtype=complex)\n'
               'short_amplitude = np.array([[0.04j]], dtype=complex)\n',
      'call': 'conversion_strength(*_copy_input((p_open, matched, dressed, short_amplitude)))',
      'gold_call': '_oracle_conversion_strength(*_copy_input((p_open, matched, dressed, short_amplitude)))'},
     {'setup': 'from copy import deepcopy as _copy_input\n'
               'import numpy as np\n'
               'p_open = np.array([0.03], dtype=float)\n'
               'matched = np.array([[(8.054981205610996-4.091655754593365j)], '
               '[(0.874901094025746-0.4444199191398281j)], [(-2.5986811432624535+1.3200413983308035j)], '
               '[(0.5897880250310983-0.8075581004051142j)]], dtype=complex)\n'
               'dressed = np.array([[(8.019901366229949-4.117046632049065j)], '
               '[(0.8710908567242177-0.44717777864281383j)], [(-2.587363759052456+1.3282329499645986j)]], '
               'dtype=complex)\n'
               'short_amplitude = np.array([[0.014285714285714289j, 0.028571428571428577j, '
               '0.042857142857142864j], [0.028571428571428577j, 0.057142857142857155j, 0.08571428571428573j], '
               '[0.042857142857142864j, 0.08571428571428573j, 0.1285714285714286j]], dtype=complex)\n',
      'call': 'conversion_strength(*_copy_input((p_open, matched, dressed, short_amplitude)))',
      'gold_call': '_oracle_conversion_strength(*_copy_input((p_open, matched, dressed, short_amplitude)))'},
     {'setup': 'from copy import deepcopy as _copy_input\n'
               'import numpy as np\n'
               'p_open = np.array([0.012, 0.04], dtype=float)\n'
               'matched = np.array([[(-13.695644900901067-3.0703777706363558j), '
               '(1.660825510602718-0.28772116850688967j)], [(4.335927681591876+1.0507345371816845j), '
               '(-0.8859652370776125+0.3481258512043014j)], [(0.9031335059433552+0.3854674696422496j), '
               '(-0.07711385209250922-0.17267933896537724j)], [(-0.0771138520925092-0.17267933896537724j), '
               '(0.3157723427472424-0.9297973581731952j)]], dtype=complex)\n'
               'dressed = np.array([[(-12.125448384811843-2.5613775975944635j), '
               '(1.4681401145487056-0.2749529393465179j)], [(3.8206449362022057+0.8835490730781961j), '
               '(-0.8223718264748284+0.3435063656471525j)]], dtype=complex)\n'
               'short_amplitude = np.array([[(-0.0015000000000000013+0.075j), (0.0135+0.05999999999999999j)], '
               '[(0.0135+0.05999999999999999j), (0.043500000000000004+0.16499999999999998j)]], dtype=complex)\n',
      'call': 'conversion_strength(*_copy_input((p_open, matched, dressed, short_amplitude)))',
      'gold_call': '_oracle_conversion_strength(*_copy_input((p_open, matched, dressed, short_amplitude)))'},
     {'setup': 'from copy import deepcopy as _copy_input\n'
               'import numpy as np\n'
               'p_open = np.array([0.008, 0.019], dtype=float)\n'
               'matched = np.array([[(-17.18560802789641-1.169798283364947j), '
               '(2.0036213442647393+1.2060378570520867j)], [(5.487641583905056+0.26591104874096094j), '
               '(-0.9316073683418679-0.9494452484067624j)], [(10.032179867011925+0.20435270505190375j), '
               '(-2.4671162093364165-3.213205248404743j)], [(0.9613952676547358+0.1699606380060729j), '
               '(-0.20009234813143056+0.08243526599033932j)], [(-0.20009234813143054+0.08243526599033932j), '
               '(-0.5626691015344547+0.7978539724277632j)]], dtype=complex)\n'
               'dressed = np.array([[(-9.929449765651155+0.3713235314285907j), '
               '(3.0885519303144715+6.5179690702406745j)], [(3.1697165671858154-0.20104089641576836j), '
               '(-1.235127766864195-2.7182530454522915j)], [(5.7922871128387685-0.5834480894377526j), '
               '(-2.9096378280289485-6.636966631440567j)]], dtype=complex)\n'
               'short_amplitude = np.array([[(-0.010000000000000009+0j), (0.09+0j), (0.09+0j)], [(0.09+0j), '
               '(0.14+0j), (0.09+0j)], [(0.09+0j), (0.09+0j), (0.29000000000000004+0j)]], dtype=complex)\n',
      'call': 'conversion_strength(*_copy_input((p_open, matched, dressed, short_amplitude)))',
      'gold_call': '_oracle_conversion_strength(*_copy_input((p_open, matched, dressed, short_amplitude)))'},
     {'setup': 'from copy import deepcopy as _copy_input\n'
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
      'call': 'conversion_strength(*_copy_input((p_open, matched, dressed, short_amplitude)))',
      'gold_call': '_oracle_conversion_strength(*_copy_input((p_open, matched, dressed, short_amplitude)))'},
     {'setup': 'from copy import deepcopy as _copy_input\n'
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
      'call': 'conversion_strength(*_copy_input((p_open, matched, dressed, short_amplitude)))',
      'gold_call': '_oracle_conversion_strength(*_copy_input((p_open, matched, dressed, short_amplitude)))'},
     {'setup': 'from copy import deepcopy as _copy_input\n'
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
               'dressed = np.array([[(0.11315720304408267-7.949683489859146j), '
               '(-0.7880218397177108+2.2024441467798317j), (-0.6508856089753824-0.8932648920610126j)], '
               '[(0.7516846353021931+3.182006367753144j), (0.16583621350963856-1.711569279022107j), '
               '(-0.6967348146986292+1.9183566285054983j)], [(-3.4817845987519074+1.6712333545202767j), '
               '(-2.6573336742014955-0.5780859950825483j), (2.5652985572312783-3.0308225391412584j)]], '
               'dtype=complex)\n'
               'short_amplitude = np.array([[(-0.0010000000000000009+0.1j), (0.009+0j), (0.009+0j)], '
               '[(0.009+0j), (0.014000000000000002+0.1j), (0.009+0j)], [(0.009+0j), (0.009+0j), '
               '(0.029000000000000005+0.1j)]], dtype=complex)\n',
      'call': 'conversion_strength(*_copy_input((p_open, matched, dressed, short_amplitude)))',
      'gold_call': '_oracle_conversion_strength(*_copy_input((p_open, matched, dressed, short_amplitude)))'},
     {'setup': 'from copy import deepcopy as _copy_input\n'
               'import numpy as np\n'
               'p_open = np.array([0.004, 0.05], dtype=float)\n'
               'matched = np.array([[(-24.534854975625983+0.2704551205689916j), '
               '(0.9041677395523329+0.7565040266865762j)], [(7.913473483044101-0.26415635927788156j), '
               '(-0.29588520904593124-0.6316330591784719j)], [(14.67402929377214-0.9483995468702763j), '
               '(-0.5596911060141848-2.175949005587194j)], [(0.9671987204698124+0.006255558339704907j), '
               '(-0.25393617990655903+0.0019798095564927537j)], [(-0.2539361799065591+0.0019798095564927537j), '
               '(-0.9669836081287407+0.021335381386455854j)]], dtype=complex)\n'
               'dressed = np.array([[(-0.1786384486366727-0.7672752912082988j), '
               '(-0.3794712028935936-1.5066340329391399j)], [(0.057338926349471825+0.08101253866560622j), '
               '(0.12228351565360907+0.12415423959148453j)], [(0.10560063861233467-0.2812399884891341j), '
               '(0.22646340409468585-0.7075241829136016j)]], dtype=complex)\n'
               'short_amplitude = np.array([[(-0.007000000000000005+0.26j), (0.063+0.1j), (0.063+0.15j)], '
               '[(0.063+0.1j), (0.098+0.41j), (0.063+0.3j)], [(0.063+0.15j), (0.063+0.3j), '
               '(0.203+0.6599999999999999j)]], dtype=complex)\n',
      'call': 'conversion_strength(*_copy_input((p_open, matched, dressed, short_amplitude)))',
      'gold_call': '_oracle_conversion_strength(*_copy_input((p_open, matched, dressed, short_amplitude)))'},
     {'setup': 'from copy import deepcopy as _copy_input\n'
               'import numpy as np\n'
               'p_open = np.array([0.07], dtype=float)\n'
               'matched = np.array([[(5.632622160261771+1.8042838036432807j)], '
               '[(0.8138784566625339+0.5810351605373051j)]], dtype=complex)\n'
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
      'call': '_exception_code(conversion_strength, *_copy_input((p_open, matched, dressed, short_amplitude)))',
      'gold_call': '_exception_code(_oracle_conversion_strength, *_copy_input((p_open, matched, dressed, short_amplitude)))'},
     {'setup': 'from copy import deepcopy as _copy_input\n'
               'import numpy as np\n'
               'p_open = np.array([0.07], dtype=float)\n'
               'matched = np.array([[(5.632622160261771+1.8042838036432807j)], '
               '[(0.8138784566625339+0.5810351605373051j)]], dtype=complex)\n'
               'dressed = np.array([[(4.336145153515082+2.850936414716249j)]], dtype=complex)\n'
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
      'call': '_exception_code(conversion_strength, *_copy_input((p_open, matched, dressed, short_amplitude)))',
      'gold_call': '_exception_code(_oracle_conversion_strength, *_copy_input((p_open, matched, dressed, short_amplitude)))'},
     {'setup': 'from copy import deepcopy as _copy_input\n'
               'import numpy as np\n'
               'p_open = np.array([0.07], dtype=float)\n'
               'matched = np.array([[(5.632622160261771+1.8042838036432807j)], '
               '[(0.8138784566625339+0.5810351605373051j)]], dtype=complex)\n'
               'dressed = np.array([[(4.336145153515082+2.850936414716249j)]], dtype=complex)\n'
               'short_amplitude = np.array([[0.04j]], dtype=complex)\n'
               '\n'
               'dressed = np.vstack([dressed,dressed])\n'
               '\n'
               'def _exception_code(function, *args, **kwargs):\n'
               '    try:\n'
               '        function(*args, **kwargs)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n',
      'call': '_exception_code(conversion_strength, *_copy_input((p_open, matched, dressed, short_amplitude)))',
      'gold_call': '_exception_code(_oracle_conversion_strength, *_copy_input((p_open, matched, dressed, short_amplitude)))'},
     {'setup': 'from copy import deepcopy as _copy_input\n'
               'import numpy as np\n'
               'p_open = np.array([0.07], dtype=float)\n'
               'matched = np.array([[(5.632622160261771+1.8042838036432807j)], '
               '[(0.8138784566625339+0.5810351605373051j)]], dtype=complex)\n'
               'dressed = np.array([[(4.336145153515082+2.850936414716249j)]], dtype=complex)\n'
               'short_amplitude = np.array([[0.04j]], dtype=complex)\n'
               '\n'
               'matched[-1,0] = 0.5\n'
               '\n'
               'def _exception_code(function, *args, **kwargs):\n'
               '    try:\n'
               '        function(*args, **kwargs)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n',
      'call': '_exception_code(conversion_strength, *_copy_input((p_open, matched, dressed, short_amplitude)))',
      'gold_call': '_exception_code(_oracle_conversion_strength, *_copy_input((p_open, matched, dressed, short_amplitude)))'},
     {'setup': 'from copy import deepcopy as _copy_input\n'
               'import numpy as np\n'
               'p_open = np.array([0.07], dtype=float)\n'
               'matched = np.array([[(5.632622160261771+1.8042838036432807j)], '
               '[(0.8138784566625339+0.5810351605373051j)]], dtype=complex)\n'
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
      'call': '_exception_code(conversion_strength, *_copy_input((p_open, matched, dressed, short_amplitude)))',
      'gold_call': '_exception_code(_oracle_conversion_strength, *_copy_input((p_open, matched, dressed, short_amplitude)))'}]
