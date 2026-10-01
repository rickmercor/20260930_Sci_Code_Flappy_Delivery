"""
Assemble the strike-independent coefficient matrix of the expansion.

The matrix collects correlation powers by row and moneyness powers by column. Its entries depend on the signed physical moment family.

Returns
-------
numpy.ndarray Float array of shape (M_max + 1, 2 * N_max + 2).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def coefficient_matrix(exps: dict, M_max: int, N_max: int, rho: float,
                       T: float, v: float) -> np.ndarray:
    """Return the coefficient matrix whose row ``m`` multiplies ``rho**m`` and
    whose column ``j`` multiplies the ``j``-th power of the moneyness.

    Let ``q = 1 - rho**2``, ``g[j] = log(j!)``, ``n = 0, 1, ..., N_max``, and let
    ``base_even``, ``base_odd``, ``high_even`` and ``high_odd`` be the entries of
    ``exps``. Build an all-zero array of shape ``(M_max + 1, 2 * N_max + 2)`` and
    fill only these entries. Algebraically equivalent stable evaluations of
    the stated factorial ratios and powers are allowed:

    ``A[0, 2 * n] = -sqrt(T) / sqrt(2 * pi) * (-1)**n
      * (base_even[n] - v**(1 - 2 * n))
      / ((2 * n - 1) * exp(g[n] + n * log(2 * T) + (n - 0.5) * log(q)))``

    ``A[1, 2 * n + 1] = 1 / (2 * sqrt(2 * pi)) * (-1)**n * base_odd[n]
      * (4 / (4 * n + 2))
      / exp(g[n] + n * log(2) + (n + 0.5) * log(T) + (n + 0.5) * log(q))``

    ``A[2 * p + 2, 2 * n] = 1 / sqrt(2 * pi) * (-1)**(n + p) * high_even[p][n]
      * exp(g[2 * n + 2 * p] - g[2 * n] - g[n + p] - (n + p) * log(2)
            - (n + p + 0.5) * log(q * T)) / (2 * p + 2)!``

    ``A[2 * p + 3, 2 * n + 1] = -1 / sqrt(2 * pi) * (-1)**(n + p)
      * high_odd[p][n]
      * exp(g[2 * n + 2 * p + 2] - g[2 * n + 1] - g[n + p + 1]
            - (n + p + 1) * log(2) - (n + p + 1.5) * log(q * T)) / (2 * p + 3)!``

    Every other entry stays zero.

    Parameters
    ----------
    exps : dict
        Mixed-moment family with keys ``"base_even"``, ``"base_odd"``,
        ``"high_even"`` and ``"high_odd"``. The two arrays have length
        ``N_max + 1``; ``"high_even"`` holds ``M_max // 2`` such arrays and
        ``"high_odd"`` holds ``(M_max - 1) // 2``.
    M_max : int
        Outer truncation order, an integer ``>= 1``.
    N_max : int
        Inner truncation order, an integer ``>= 0``.
    rho : float
        Correlation with ``-1 < rho < 1``.
    T : float
        Maturity, strictly positive and finite.
    v : float
        Root-mean-square volatility level, strictly positive and finite.

    Returns
    -------
    numpy.ndarray
        Float array of shape ``(M_max + 1, 2 * N_max + 2)``.

    Raises
    ------
    ValueError
        If ``M_max`` is not an integer ``>= 1``, if ``N_max`` is not an integer
        ``>= 0``, if ``exps`` lacks a required key or carries arrays of the
        wrong length or lists of the wrong count, if ``rho`` is not finite with
        ``-1 < rho < 1``, or if ``T`` or ``v`` is not strictly positive and
        finite.
    """
    return np.zeros((1, 1))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_coefficient_matrix(exps: dict, M_max: int, N_max: int, rho: float,
                               T: float, v: float) -> np.ndarray:
    """Reference implementation of the coefficient matrix."""
    np = __import__("numpy")
    gammaln = __import__("scipy.special", fromlist=["gammaln"]).gammaln

    if isinstance(M_max, bool) or not isinstance(M_max, (int, np.integer)) or int(M_max) < 1:
        raise ValueError("M_max must be an integer >= 1")
    if isinstance(N_max, bool) or not isinstance(N_max, (int, np.integer)) or int(N_max) < 0:
        raise ValueError("N_max must be an integer >= 0")
    M_max = int(M_max)
    N_max = int(N_max)
    if not isinstance(exps, dict) or any(
            key not in exps for key in ("base_even", "base_odd", "high_even", "high_odd")):
        raise ValueError("exps must carry base_even, base_odd, high_even and high_odd")
    base_even = np.asarray(exps["base_even"], dtype=float)
    base_odd = np.asarray(exps["base_odd"], dtype=float)
    if base_even.shape != (N_max + 1,) or base_odd.shape != (N_max + 1,):
        raise ValueError("base_even and base_odd must both have length N_max + 1")
    high_even = [np.asarray(a, dtype=float) for a in exps["high_even"]]
    high_odd = [np.asarray(a, dtype=float) for a in exps["high_odd"]]
    if len(high_even) != M_max // 2 or len(high_odd) != (M_max - 1) // 2:
        raise ValueError("high_even and high_odd hold the wrong number of arrays")
    if any(a.shape != (N_max + 1,) for a in high_even + high_odd):
        raise ValueError("every high-order array must have length N_max + 1")
    rho = float(rho)
    if not np.isfinite(rho) or not -1.0 < rho < 1.0:
        raise ValueError("rho must be finite with -1 < rho < 1")
    for name, value in (("T", T), ("v", v)):
        val = float(value)
        if not np.isfinite(val) or val <= 0.0:
            raise ValueError(f"{name} must be finite and strictly positive")
    T = float(T)
    v = float(v)

    q = 1.0 - rho * rho
    root_two_pi = np.sqrt(2.0 * np.pi)
    n = np.arange(N_max + 1)
    g = gammaln(np.arange(max(4 * N_max + 12, 2 * N_max + M_max + 3)) + 1.0)
    A = np.zeros((M_max + 1, 2 * N_max + 2))

    delta = base_even - v ** (1.0 - 2.0 * n)
    log_den = g[n] + n * np.log(2.0 * T) + (n - 0.5) * np.log(q)
    A[0, 0::2] = (-np.sqrt(T) / root_two_pi * ((-1.0) ** n) * delta
                  / ((2.0 * n - 1.0) * np.exp(log_den)))

    log_den = g[n] + n * np.log(2.0) + (n + 0.5) * np.log(T) + (n + 0.5) * np.log(q)
    A[1, 1::2] = ((1.0 / (2.0 * root_two_pi)) * ((-1.0) ** n) * base_odd
                  / np.exp(log_den) * (4.0 / (4.0 * n + 2.0)))

    for p, moments in enumerate(high_even):
        log_ratio = (g[2 * n + 2 * p] - g[2 * n] - g[n + p] - (n + p) * np.log(2.0)
                     - (n + p + 0.5) * np.log(q * T))
        A[2 * p + 2, 0::2] = ((1.0 / root_two_pi) * ((-1.0) ** (n + p))
                              * np.exp(log_ratio) * moments / np.exp(g[2 * p + 2]))

    for p, moments in enumerate(high_odd):
        log_ratio = (g[2 * n + 2 * p + 2] - g[2 * n + 1] - g[n + p + 1]
                     - (n + p + 1) * np.log(2.0) - (n + p + 1.5) * np.log(q * T))
        A[2 * p + 3, 1::2] = (-(1.0 / root_two_pi) * ((-1.0) ** (n + p))
                              * np.exp(log_ratio) * moments / np.exp(g[2 * p + 3]))

    return A

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np

def test_cases():
    """Normal, boundary, edge and invalid-input cases with portable fixtures."""
    return [{'setup': 'import numpy as np\n'
           '# Fixed numerical inputs; no reference execution is needed to construct them.\n'
           "e = {'base_even': [1.0050811017968906,\n"
           '               1.0151979195928866,\n'
           '               1.1093878590592978,\n'
           '               1.3068131977718953,\n'
           '               1.6536850904119902,\n'
           '               2.2408141982273477,\n'
           '               3.2416409476012054,\n'
           '               4.992348213248177,\n'
           '               8.163530444465644,\n'
           '               14.138683846134683,\n'
           '               25.875564735743847,\n'
           '               49.93158930932141,\n'
           '               101.38617838757484],\n'
           " 'base_odd': [-0.04331231909561211,\n"
           '              -0.1352053822472552,\n'
           '              -0.25317824183427035,\n'
           '              -0.4284901591555836,\n'
           '              -0.7142463326441342,\n'
           '              -1.2100378952145394,\n'
           '              -2.113166571403283,\n'
           '              -3.8306520501457024,\n'
           '              -7.233117597927456,\n'
           '              -14.249354739956868,\n'
           '              -29.303900739578154,\n'
           '              -62.90531197812034,\n'
           '              -140.89196947017004],\n'
           " 'high_even': [[0.12374948313032026,\n"
           '                0.1325016122367931,\n'
           '                0.1686724757111233,\n'
           '                0.24414769925845914,\n'
           '                0.3870187171426814,\n'
           '                0.6562333814002171,\n'
           '                1.1751573540361404,\n'
           '                2.208110770089141,\n'
           '                4.33898939086138,\n'
           '                8.900403460359467,\n'
           '                19.036518845149374,\n'
           '                42.417592193995496,\n'
           '                98.38994903776144],\n'
           '               [0.0457295696953653,\n'
           '                0.05173919539135861,\n'
           '                0.07486666519855432,\n'
           '                0.12557125352774037,\n'
           '                0.22931829757148303,\n'
           '                0.44321429695370795,\n'
           '                0.8963816213210469,\n'
           '                1.888582009773773,\n'
           '                4.137281800480468,\n'
           '                9.414604146375918,\n'
           '                22.238355606465422,\n'
           '                54.495604513282835,\n'
           '                138.46115729467252]],\n'
           " 'high_odd': [[-0.016005349393377864,\n"
           '               -0.05191006394878933,\n'
           '               -0.10425673490278496,\n'
           '               -0.19379832589383517,\n'
           '               -0.3604154005610063,\n'
           '               -0.6873161612275093,\n'
           '               -1.3561565527039972,\n'
           '               -2.7779248702540547,\n'
           '               -5.913959031552621,\n'
           '               -13.087732558472425,\n'
           '               -30.100658237712594,\n'
           '               -71.91518274392513,\n'
           '               -178.38472336952304]]}\n'
           'v = 1.0155097011894159\n',
  'call': 'coefficient_matrix(e,4,12,-0.6,0.125,v)',
  'gold_call': '_oracle_coefficient_matrix(e,4,12,-0.6,0.125,v)'},
 {'setup': 'import numpy as np\n'
           '# Fixed numerical inputs; no reference execution is needed to construct them.\n'
           "e = {'base_even': [35.17783856289117],\n"
           " 'base_odd': [-0.04331231909561211],\n"
           " 'high_even': [],\n"
           " 'high_odd': []}\n"
           'v = 35.542839541629554\n',
  'call': 'coefficient_matrix(e,1,0,0.0,0.125,v)',
  'gold_call': '_oracle_coefficient_matrix(e,1,0,0.0,0.125,v)'},
 {'setup': 'import numpy as np\n'
           '# Fixed numerical inputs; no reference execution is needed to construct them.\n'
           "e = {'base_even': [2.0165449388989147,\n"
           '               0.5123494056900023,\n'
           '               0.1476027431404012,\n'
           '               0.04777482181246777,\n'
           '               0.01722979798176267],\n'
           " 'base_odd': [-0.04919334627807177,\n"
           '              -0.03932726367245014,\n'
           '              -0.019706855390545945,\n'
           '              -0.009278920313856907,\n'
           '              -0.004453523183008216],\n'
           " 'high_even': [[0.19677338511228595,\n"
           '                0.05483917355180021,\n'
           '                0.01978632430602322,\n'
           '                0.008450788167946949,\n'
           '                0.004030254346622753]],\n'
           " 'high_odd': [[-0.014416931931200118,\n"
           '               -0.012238747233442301,\n'
           '               -0.006810481790582288,\n'
           '               -0.0036617460054718153,\n'
           '               -0.0020345406943626317]]}\n'
           'v = 2.0510574645840385\n',
  'call': 'coefficient_matrix(e,3,4,0.4,0.1,v)',
  'gold_call': '_oracle_coefficient_matrix(e,3,4,0.4,0.1,v)'},
 {'setup': 'def run_model_invalid():\n'
           '    try:\n'
           '        coefficient_matrix({},0,0,0.,1.,1.)\n'
           '        return 0\n'
           '    except ValueError:\n'
           '        return 1\n'
           '    except Exception:\n'
           '        return 2\n'
           'def run_oracle_invalid():\n'
           '    try:\n'
           '        _oracle_coefficient_matrix({},0,0,0.,1.,1.)\n'
           '        return 0\n'
           '    except ValueError:\n'
           '        return 1\n'
           '    except Exception:\n'
           '        return 2\n',
  'call': 'run_model_invalid()',
  'gold_call': 'run_oracle_invalid()'}]
