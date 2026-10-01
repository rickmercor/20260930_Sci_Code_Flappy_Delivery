"""
Apply closed-form image-charge form factors and free-carrier screening.

The ground-state profile \(p(z)=2\cos^2(\pi z/a)/a\) on \(|z|\le a/2\) sits between identical dielectrics, with \(\gamma=(\epsilon_s-\epsilon_e)/(\epsilon_s+\epsilon_e)\), \(x=qa\), and \(r=\gamma e^{-x}\). The image-charge form factors are



\[

F_I=\int p(z)\left[e^{-q|z|}+\frac{2r\cosh(qz)}{1-r}\right]dz=A(x)+\frac{2r\,C(x)}{1-r},

\]



\[

F_{ee}=\iint p(z)p(z')\left[e^{-q|z-z'|}+\frac{2r\cosh(q(z-z'))}{1-r}\right]dz\,dz'=\Phi(x)+\frac{2r\,C(x)^2}{1-r},

\]



evaluated with the exact closed forms



\[

A=\frac{2(1-e^{-x/2})}{x}+\frac{2x(1+e^{-x/2})}{x^2+4\pi^2},\qquad

C=\frac{8\pi^2\sinh(x/2)}{x(x^2+4\pi^2)},

\]



\[

\Phi=\frac{1}{x^2+4\pi^2}\left[3x+\frac{8\pi^2x+32\pi^4g(x)}{x^2+4\pi^2}\right],\qquad

g(x)=\frac{x-1+e^{-x}}{x^2}.

\]



Evaluate \(g\) without cancellation at small \(x\) (for example by its Taylor series \(1/2-x/6+x^2/24-\dots\)). Any \(x>700\), where the hyperbolic factors overflow double precision, raises `ValueError`. Then return \(\varepsilon_{2D}=1+e^2\Pi F_{ee}/(2\epsilon_0\epsilon_s q)\) and \(U=e^2F_I/(2\epsilon_0\epsilon_s q\varepsilon_{2D})\).

Returns
-------
A NumPy array whose first row is \(U(q)\) in J m\(^2\) and whose second row is the dimensionless \(\varepsilon_{2D}(q)\), preserving the input grid shape.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def screen_dielectric_kernel(q_m_inv: 'np.ndarray', polarizability_j_inv_m2: 'np.ndarray', thickness_angstrom: float, epsilon_layer: float, epsilon_environment: float) -> 'np.ndarray':
    """Return the screened impurity potential and dielectric response.

    With x = q a, the ground-state profile p(z) = 2 cos^2(pi z / a) / a on
    |z| <= a/2 gives the closed forms
    A(x) = 2 (1 - exp(-x/2)) / x + 2 x (1 + exp(-x/2)) / (x^2 + 4 pi^2),
    C(x) = 8 pi^2 sinh(x/2) / (x (x^2 + 4 pi^2)), and
    Phi(x) = [3 x + (8 pi^2 x + 32 pi^4 g(x)) / (x^2 + 4 pi^2)] / (x^2 + 4 pi^2)
    with g(x) = (x - 1 + exp(-x)) / x^2. With gamma = (eps_layer - eps_env) /
    (eps_layer + eps_env) and r = gamma exp(-x), the impurity and carrier form
    factors are F_I = A + 2 r C / (1 - r) and F_ee = Phi + 2 r C^2 / (1 - r).

    Parameters
    ----------
    q_m_inv : numpy.ndarray
        Positive two-dimensional wave-vector transfers in m^-1.
    polarizability_j_inv_m2 : numpy.ndarray
        Static polarizability with the same shape as q_m_inv.
    thickness_angstrom : float
        Layer thickness a in angstrom.
    epsilon_layer, epsilon_environment : float
        Relative permittivities of the layer and symmetric environment.

    Returns
    -------
    numpy.ndarray
        First row: screened potential U = e^2 F_I / (2 eps_0 eps_layer q
        eps_2D) in J m^2; second row: dimensionless dielectric response
        eps_2D = 1 + e^2 Pi F_ee / (2 eps_0 eps_layer q). Each row retains
        the input array shape.

    Raises
    ------
    ValueError
        For empty or mismatched q and polarizability arrays; nonfinite or
        nonpositive q; nonfinite or negative polarizability; nonfinite or
        nonpositive thickness or permittivities; or any q with q * a above
        700, where the hyperbolic image-charge factors overflow double
        precision.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

"""Apply closed-form image-charge form factors and free-carrier screening."""
import numpy as np
_E_CHARGE = 1.602176634e-19
_EPSILON_0 = 8.8541878128e-12

def _oracle_screen_dielectric_kernel(q_m_inv: 'np.ndarray', polarizability_j_inv_m2: 'np.ndarray', thickness_angstrom: float, epsilon_layer: float, epsilon_environment: float) -> 'np.ndarray':
    """Reference infinite-image dielectric kernel with closed-form profile integrals."""
    _E_CHARGE = 1.602176634e-19
    _EPSILON_0 = 8.8541878128e-12
    q_input = np.asarray(q_m_inv, dtype=float)
    polarizability = np.asarray(polarizability_j_inv_m2, dtype=float)
    if q_input.shape != polarizability.shape or q_input.size == 0:
        raise ValueError('q and polarizability must be nonempty arrays with matching shapes')
    if not np.all(np.isfinite(q_input)) or np.any(q_input <= 0.0):
        raise ValueError('q_m_inv must contain positive finite values')
    if not np.all(np.isfinite(polarizability)) or np.any(polarizability < 0.0):
        raise ValueError('polarizability must contain finite nonnegative values')
    values = (thickness_angstrom, epsilon_layer, epsilon_environment)
    if not all(np.isfinite(values)) or min(values) <= 0.0:
        raise ValueError('positive finite dielectric inputs are required')
    thickness_m = thickness_angstrom * 1e-10
    x = q_input * thickness_m
    if np.any(x > 700.0):
        raise ValueError('q times the layer thickness must not exceed 700')
    four_pi_sq = 4.0 * np.pi ** 2
    denominator = x ** 2 + four_pi_sq
    direct_impurity = -2.0 * np.expm1(-0.5 * x) / x + 2.0 * x * (1.0 + np.exp(-0.5 * x)) / denominator
    projected_cosh = 2.0 * np.pi ** 2 * (2.0 * np.sinh(0.5 * x) / x) * 2.0 / denominator
    small = x < 0.01
    g = np.empty_like(x)
    xs = x[small]
    g[small] = 0.5 - xs / 6.0 + xs ** 2 / 24.0 - xs ** 3 / 120.0 + xs ** 4 / 720.0
    xl = x[~small]
    g[~small] = (xl + np.expm1(-xl)) / xl ** 2
    direct_carrier = (3.0 * x + (8.0 * np.pi ** 2 * x + 32.0 * np.pi ** 4 * g) / denominator) / denominator
    contrast = (epsilon_layer - epsilon_environment) / (epsilon_layer + epsilon_environment)
    image_ratio = contrast * np.exp(-x)
    image_sum = 2.0 * image_ratio / (1.0 - image_ratio)
    impurity_form = direct_impurity + image_sum * projected_cosh
    carrier_form = direct_carrier + image_sum * projected_cosh ** 2
    coulomb_prefactor = _E_CHARGE ** 2 / (2.0 * _EPSILON_0 * epsilon_layer * q_input)
    dielectric = 1.0 + coulomb_prefactor * polarizability * carrier_form
    screened_potential = coulomb_prefactor * impurity_form / dielectric
    return np.stack((screened_potential, dielectric), axis=0)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, boundary, edge, and rejection differential tests."""
    return [{'setup': 'q = np.array([1e7, 3e8, 2e9, 8e9])\n'
               'pi = np.array([2e37, 1.8e37, 9e36, 1e36])\n'
               'params = (q, pi, 6.6, 7.6, 3.9)\n'
               'def scaled(value):\n'
               '    value = np.asarray(value)\n'
               '    return np.stack((value[0] / 1e-37, value[1]), axis=0)',
      'call': 'scaled(screen_dielectric_kernel(*params))',
      'gold_call': 'scaled(_oracle_screen_dielectric_kernel(*params))',
      'tol': 1e-09},
     {'setup': 'q = np.array([[1e8, 5e8], [1e9, 6e9]])\n'
               'params = (q, np.zeros_like(q), 6.6, 7.6, 7.6)\n'
               'def scaled(value):\n'
               '    value = np.asarray(value)\n'
               '    return np.stack((value[0] / 1e-37, value[1]), axis=0)',
      'call': 'scaled(screen_dielectric_kernel(*params))',
      'gold_call': 'scaled(_oracle_screen_dielectric_kernel(*params))',
      'tol': 1e-09},
     {'setup': 'q = np.array([1e-8, 1e-4, 0.0099, 0.0101]) / 1e-9\n'
               'params = (q, np.full(q.shape, 2e37), 10.0, 7.6, 23.0)\n'
               'def scaled(value):\n'
               '    value = np.asarray(value)\n'
               '    return np.stack((value[0] / 1e-37, value[1]), axis=0)',
      'call': 'scaled(screen_dielectric_kernel(*params))',
      'gold_call': 'scaled(_oracle_screen_dielectric_kernel(*params))',
      'tol': 1e-09},
     {'setup': 'q = np.array([1e10, 7e11])\n'
               'params = (q, np.array([1e37, 0.0]), 10.0, 7.6, 3.9)\n'
               'def scaled(value):\n'
               '    value = np.asarray(value)\n'
               '    return np.stack((value[0] / 1e-37, value[1]), axis=0)',
      'call': 'scaled(screen_dielectric_kernel(*params))',
      'gold_call': 'scaled(_oracle_screen_dielectric_kernel(*params))',
      'tol': 1e-09},
     {'setup': 'good = (np.array([1e8, 5e8]), np.array([1e37, 2e37]), 6.6, 7.6, 3.9)\n'
               'def bad_argument_sets():\n'
               '    bad = [list(good) for _ in range(13)]\n'
               '    bad[0][0] = np.array([])\n'
               '    bad[0][1] = np.array([])\n'
               '    bad[1][1] = np.array([1e37])\n'
               '    bad[2][0] = np.array([0.0, 5e8])\n'
               '    bad[3][0] = np.array([np.nan, 5e8])\n'
               '    bad[4][1] = np.array([-1e37, 2e37])\n'
               '    bad[5][1] = np.array([np.inf, 2e37])\n'
               '    bad[6][2] = 0.0\n'
               '    bad[7][2] = np.nan\n'
               '    bad[8][3] = -7.6\n'
               '    bad[9][3] = np.inf\n'
               '    bad[10][4] = 0.0\n'
               '    bad[11][4] = np.nan\n'
               '    bad[12][0] = np.array([1e8, 701.0 / (6.6e-10)])\n'
               '    return bad\n'
               'def rejected_public():\n'
               '    result = []\n'
               '    for args in bad_argument_sets():\n'
               '        try:\n'
               '            screen_dielectric_kernel(*args)\n'
               '        except ValueError:\n'
               '            result.append(1.0)\n'
               '        else:\n'
               '            result.append(0.0)\n'
               '    return np.asarray(result)\n'
               'def rejected_oracle():\n'
               '    result = []\n'
               '    for args in bad_argument_sets():\n'
               '        try:\n'
               '            _oracle_screen_dielectric_kernel(*args)\n'
               '        except ValueError:\n'
               '            result.append(1.0)\n'
               '        else:\n'
               '            result.append(0.0)\n'
               '    return np.asarray(result)\n',
      'call': 'rejected_public()',
      'gold_call': 'rejected_oracle()',
      'tol': 1e-09}]
