"""
Angular construction of two spherical quadratic responses.

The table stores $\widehat\omega_{220}$, $\widehat\omega_{320}$, and the supplied synthetic radial transfer coefficients $\rho_4,\rho_5$. The transfer from unequal parents is $\kappa_l=-0.20\rho_l$; this scale, the conversions below, the retained angular components, and the finite truncation are synthetic instance conventions.



For spin weight $s$, azimuthal number $m$, and $L=L_{\min},\ldots,13$, where $L_{\min}=\max(|s|,|m|)$, form the real symmetric cosine matrix



$$

X_{LL}=-\frac{sm}{L(L+1)},\qquad X_{L,L+1}=\sqrt{\frac{((L+1)^2-m^2)((L+1)^2-s^2)}{(L+1)^2(4(L+1)^2-1)}}.

$$



Set $X_{L+1,L}=X_{L,L+1}$, with all other undisplayed entries zero. Square this extended matrix first, then retain the rows and columns through $L=12$ in both $X$ and $X^2$. At complex spheroidicity $c$, diagonalize



$$

H^{(s,m)}(c)=\operatorname{diag}[L(L+1)-s(s+1)]+2csX-c^2X^2.

$$



For target $l$, choose the eigenvalue closest in complex modulus to $l(l+1)-s(s+1)$, normalize its right eigenvector $b$ by $b^\dagger b=1$, and rotate its phase so $b_l$ is real and positive. Construct parent vectors $b^{22}$ and $b^{32}$ with $(s,m,l,c)=(-2,2,2,\chi_f\widehat\omega_{220})$ and $(-2,2,3,\chi_f\widehat\omega_{320})$. For $ab\in\{22,23\}$ construct each child $b^{ab,l}$ with $(s,m,l,c)=(-2,4,l,\chi_f\Omega_{ab})$, where $\Omega_{22}=2\widehat\omega_{220}$ and $\Omega_{23}=\widehat\omega_{220}+\widehat\omega_{320}$. Define $u^a_L=-\sqrt{(L+2)(L-1)}b^a_L$, $p_l^{ab}=b^{ab,l}_6$, and



$$

A_l^{ab}=\sum_{L_1,L_2,J}u^a_{L_1}u^b_{L_2}\overline{b^{ab,l}_J}\sqrt{\frac{(2L_1+1)(2L_2+1)(2J+1)}{4\pi}}

\begin{pmatrix}L_1\&L_2\&J\\1&1&-2\end{pmatrix}

\begin{pmatrix}L_1\&L_2\&J\\2&2&-4\end{pmatrix}.

$$



For each integer Wigner symbol require $m_1+m_2+m_3=0$, $|m_i|\le j_i$, and $|j_1-j_2|\le j_3\le j_1+j_2$, returning zero otherwise. With $\Delta=(j_1+j_2-j_3)!(j_1-j_2+j_3)!(-j_1+j_2+j_3)!/(j_1+j_2+j_3+1)!$ and $N=\Delta\prod_i(j_i+m_i)!(j_i-m_i)!$, use



$$

\begin{pmatrix}j_1\&j_2\&j_3\\m_1\&m_2\&m_3\end{pmatrix}=(-1)^{j_1-j_2-m_3}\sqrt N\sum_{z=z_-}^{z_+}\frac{(-1)^z}{z!(j_1+j_2-j_3-z)!(j_1-m_1-z)!(j_2+m_2-z)!(j_3-j_2+m_1+z)!(j_3-j_1-m_2+z)!},

$$



where $z_-=\max(0,j_2-j_3-m_1,j_1-j_3+m_2)$ and $z_+=\min(j_1+j_2-j_3,j_1-m_1,j_2+m_2)$.



Use Python's standard `math.factorial`, not the removed `numpy.math`

namespace. Cast parity exponents to the standard Python `int` before applying $(-1)$ or

evaluate parity directly, so negative NumPy integer exponents cannot raise.



Finally compute



$$

\mathcal R_{64}^{22}=\sum_{l=4,5}p_l^{22}A_l^{22}\left[-\frac{i\widehat\omega_{220}}{48}\sqrt{\frac{(l+2)!}{(l-2)!}}\right]\rho_l,

\qquad

\mathcal R_{64}^{23}=\sum_{l=4,5}p_l^{23}A_l^{23}\left[-\frac{i\sqrt{\widehat\omega_{220}\widehat\omega_{320}}}{2\sqrt{24\cdot120}}\sqrt{\frac{(l+2)!}{(l-2)!}}\right]\kappa_l,

\quad \kappa_l=-0.20\rho_l.

$$



Use the principal complex square root. Both conversions use intrinsic frequencies and are independent of mass at fixed spin. Reject nonfinite or malformed input, spin outside $[0.70,0.86]$, nonpositive mass, a 220 or 320 frequency without positive real and negative imaginary parts, a nonfinite eigensystem, a branch tie within $10^{-10}$, or a zero normalization or target phase component.

Returns
-------
`numpy.ndarray` of shape `(n, 4)` containing $(\Re\mathcal R_{64}^{22},\Im\mathcal R_{64}^{22},\Re\mathcal R_{64}^{23},\Im\mathcal R_{64}^{23})$. No cells are unused. Every returned component is compared at numerical tolerance `1e-9`.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
def synthesize_quadratic_response(table: np.ndarray) -> np.ndarray:
    """Construct the self-coupled and mixed-parent responses in spherical 64.

    Parameters
    ----------
    table : numpy.ndarray, shape (n, 16)
        Interpolated table in the exact column order declared by step 02.

    Returns
    -------
    numpy.ndarray, shape (n, 4)
        Real and imaginary parts of the intrinsic spherical-64 220-by-220
        response followed by the 220-by-320 response, compared at tolerance
        1e-9. No cells are unused.

    Raises
    ------
    ValueError
        If the table is empty, nonfinite, has a shape other than `(n, 16)`,
        has spin outside `[0.70, 0.86]`, has nonpositive mass, or has a 220
        or 320 frequency whose real part is not positive or imaginary part
        is not negative. Also raised if an angular eigensystem is nonfinite,
        ambiguous at the selected branch, or cannot be normalized with the
        required phase convention.

    Notes
    -----
    Use Python's standard `math.factorial`, not the removed `numpy.math`
    namespace. Cast parity exponents to built-in `int` before applying `(-1)`
    or evaluate parity directly, so negative NumPy-integer exponents cannot
    trigger an environment-dependent exception.
    """
    return None
import math
import numpy as np

# =============================================================================
# GOLD SOLUTION
# =============================================================================

r"""
Angular construction of two spherical quadratic responses.

The table contains the intrinsic parent frequency and task-defined synthetic
radial transfer coefficients $\rho_4,\rho_5$. Construct normalized
spin-weighted spheroidal harmonics in a spin-weighted spherical basis through
$L=12$. Project both the squared raised 220 parent and the bilinear raised
220-by-320 source onto their respective driven children, project each child
into spherical 64, and apply the stated strain conversions. The unequal-parent
transfer is $\kappa_l=-0.20\rho_l$, where `0.20` is a supplied synthetic scale
for the unequal-parent channel. The
exact matrix, phase, Gaunt, and response
conventions are in the ordered specification.

Returns
-------
`numpy.ndarray` of shape `(n, 4)` containing real and imaginary parts of
$(\mathcal R_{64}^{22},\mathcal R_{64}^{23})$. Every component is compared at
numerical tolerance `1e-9`.
"""
"""Angular construction of the spherical quadratic responses."""
import numpy as np
"""Angular construction of the spherical quadratic responses."""
import math
import numpy as np
_ANGULAR_LMAX = 12
def _wigner_3j_integer(j1, j2, j3, m1, m2, m3):
    if (
        m1 + m2 + m3 != 0
        or abs(m1) > j1 or abs(m2) > j2 or abs(m3) > j3
        or j3 < abs(j1 - j2) or j3 > j1 + j2
    ):
        return 0.0
    triangle = (
        math.factorial(j1 + j2 - j3)
        * math.factorial(j1 - j2 + j3)
        * math.factorial(-j1 + j2 + j3)
        / math.factorial(j1 + j2 + j3 + 1)
    )
    factorial_product = triangle
    for j, m in ((j1, m1), (j2, m2), (j3, m3)):
        factorial_product *= math.factorial(j + m) * math.factorial(j - m)
    lower = max(0, j2 - j3 - m1, j1 - j3 + m2)
    upper = min(j1 + j2 - j3, j1 - m1, j2 + m2)
    series = 0.0
    for z in range(lower, upper + 1):
        denominator = (
            math.factorial(z)
            * math.factorial(j1 + j2 - j3 - z)
            * math.factorial(j1 - m1 - z)
            * math.factorial(j2 + m2 - z)
            * math.factorial(j3 - j2 + m1 + z)
            * math.factorial(j3 - j1 - m2 + z)
        )
        series += (-1) ** z / denominator
    return (-1) ** (j1 - j2 - m3) * math.sqrt(factorial_product) * series
def _angular_gaunt_tensor(parent_l, child_l):
    tensor = np.zeros((parent_l.size, parent_l.size, child_l.size), dtype=float)
    for a, l1 in enumerate(parent_l):
        for b, l2 in enumerate(parent_l):
            for d, j in enumerate(child_l):
                prefactor = math.sqrt(
                    (2 * l1 + 1) * (2 * l2 + 1) * (2 * j + 1) / (4.0 * math.pi)
                )
                tensor[a, b, d] = prefactor * _wigner_3j_integer(
                    int(l1), int(l2), int(j), 1, 1, -2
                ) * _wigner_3j_integer(int(l1), int(l2), int(j), 2, 2, -4)
    return tensor
def _spheroidal_vector(spin_weight, azimuthal, spheroidicity, target_l):
    lmin = max(abs(spin_weight), abs(azimuthal))
    extended_l = np.arange(lmin, _ANGULAR_LMAX + 2, dtype=int)
    cosine = np.zeros((extended_l.size, extended_l.size), dtype=float)
    for index, ell in enumerate(extended_l):
        cosine[index, index] = -spin_weight * azimuthal / (ell * (ell + 1))
        if index + 1 < extended_l.size:
            upper = ell + 1
            coupling = math.sqrt(
                (upper ** 2 - azimuthal ** 2)
                * (upper ** 2 - spin_weight ** 2)
                / (upper ** 2 * (4 * upper ** 2 - 1))
            )
            cosine[index, index + 1] = coupling
            cosine[index + 1, index] = coupling
    cosine_squared = (cosine @ cosine)[:-1, :-1]
    cosine = cosine[:-1, :-1]
    ell = extended_l[:-1]
    spherical_eigenvalues = ell * (ell + 1) - spin_weight * (spin_weight + 1)
    angular_matrix = (
        np.diag(spherical_eigenvalues.astype(complex))
        + 2.0 * spheroidicity * spin_weight * cosine
        - spheroidicity ** 2 * cosine_squared
    )
    eigenvalues, eigenvectors = np.linalg.eig(angular_matrix)
    if not np.all(np.isfinite(eigenvalues)) or not np.all(np.isfinite(eigenvectors)):
        raise ValueError("angular eigensystem is nonfinite")
    target = target_l * (target_l + 1) - spin_weight * (spin_weight + 1)
    distances = np.abs(eigenvalues - target)
    order = np.argsort(distances)
    if order.size < 2 or distances[order[1]] - distances[order[0]] <= 1e-10:
        raise ValueError("angular eigenbranch is ambiguous")
    vector = eigenvectors[:, order[0]].astype(complex)
    norm = math.sqrt(float(np.vdot(vector, vector).real))
    target_component = vector[target_l - lmin]
    if not np.isfinite(norm) or norm <= 0.0 or abs(target_component) <= 1e-12:
        raise ValueError("angular eigenvector cannot be normalized")
    vector /= norm
    vector *= np.exp(-1j * np.angle(vector[target_l - lmin]))
    return ell, vector
def _angular_response_at_spin(spin, omega220, radial4, radial5):
    parent_l, parent = _spheroidal_vector(-2, 2, spin * omega220, 2)
    raised_parent = -np.sqrt((parent_l + 2) * (parent_l - 1)) * parent
    response = 0.0j
    for target_l, radial in ((4, radial4), (5, radial5)):
        child_l, child = _spheroidal_vector(-2, 4, spin * (2.0 * omega220), target_l)
        gaunt = _angular_gaunt_tensor(parent_l, child_l)
        angular_source = np.einsum(
            "a,b,c,abc->", raised_parent, raised_parent, np.conjugate(child), gaunt
        )
        spherical64_overlap = child[6 - 4]
        conversion = (
            -1j * omega220 / 48.0
            * math.sqrt(math.factorial(target_l + 2) / math.factorial(target_l - 2))
        )
        response += spherical64_overlap * angular_source * conversion * radial
    return response
def _mixed_angular_response_at_spin(spin, omega220, omega320, radial4, radial5):
    parent_l, parent220 = _spheroidal_vector(-2, 2, spin * omega220, 2)
    other_l, parent320 = _spheroidal_vector(-2, 2, spin * omega320, 3)
    if not np.array_equal(parent_l, other_l):
        raise ValueError("mixed-parent angular bases do not agree")
    raised220 = -np.sqrt((parent_l + 2) * (parent_l - 1)) * parent220
    raised320 = -np.sqrt((parent_l + 2) * (parent_l - 1)) * parent320
    response = 0.0j
    parent_factor = math.sqrt(
        math.factorial(2 + 2) / math.factorial(2 - 2)
        * math.factorial(3 + 2) / math.factorial(3 - 2)
    )
    for target_l, radial in ((4, radial4), (5, radial5)):
        child_l, child = _spheroidal_vector(
            -2, 4, spin * (omega220 + omega320), target_l
        )
        gaunt = _angular_gaunt_tensor(parent_l, child_l)
        angular_source = np.einsum(
            "a,b,c,abc->", raised220, raised320, np.conjugate(child), gaunt
        )
        spherical64_overlap = child[6 - 4]
        conversion = (
            -1j * np.sqrt(omega220 * omega320) / (2.0 * parent_factor)
            * math.sqrt(math.factorial(target_l + 2) / math.factorial(target_l - 2))
        )
        unequal_parent_scale = 0.20
        response += (
            spherical64_overlap * angular_source * conversion
            * (-unequal_parent_scale * radial)
        )
    return response
def _oracle_synthesize_quadratic_response(table: np.ndarray) -> np.ndarray:
    values = np.asarray(table, dtype=float)
    if values.ndim != 2 or values.shape[1] != 16 or values.shape[0] == 0 or not np.all(np.isfinite(values)):
        raise ValueError("table must be a nonempty finite array with shape (n, 16)")
    if np.any(values[:, 0] < 0.70) or np.any(values[:, 0] > 0.86):
        raise ValueError("spin must lie in [0.70, 0.86]")
    if np.any(values[:, 1] <= 0.0):
        raise ValueError("mass ratio must be positive")
    omega220 = values[:, 2] + 1j * values[:, 3]
    omega320 = values[:, 4] + 1j * values[:, 5]
    if np.any(omega220.real <= 0.0) or np.any(omega220.imag >= 0.0):
        raise ValueError("220 frequencies require positive real and negative imaginary parts")
    if np.any(omega320.real <= 0.0) or np.any(omega320.imag >= 0.0):
        raise ValueError("320 frequencies require positive real and negative imaginary parts")
    radial4 = values[:, 12] + 1j * values[:, 13]
    radial5 = values[:, 14] + 1j * values[:, 15]
    cache = {}
    result = np.empty((values.shape[0], 2), dtype=complex)
    for row, (spin, frequency220, frequency320, coefficient4, coefficient5) in enumerate(
        zip(values[:, 0], omega220, omega320, radial4, radial5)
    ):
        key = (
            float(spin), complex(frequency220), complex(frequency320),
            complex(coefficient4), complex(coefficient5),
        )
        if key not in cache:
            cache[key] = (
                _angular_response_at_spin(
                    spin, frequency220, coefficient4, coefficient5
                ),
                _mixed_angular_response_at_spin(
                    spin, frequency220, frequency320, coefficient4, coefficient5
                ),
            )
        result[row] = cache[key]
    if not np.all(np.isfinite(result)):
        raise ValueError("quadratic response is nonfinite")
    return np.column_stack((
        result[:, 0].real, result[:, 0].imag,
        result[:, 1].real, result[:, 1].imag,
    ))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            'setup': 'def run_case(fn_under_test, fixture_interpolate_spin_tables):\n    import numpy as np\n    tab = fixture_interpolate_spin_tables(np.array([0.7, 0.78, 0.86]))\n    return fn_under_test(tab)',
            'call': 'run_case(synthesize_quadratic_response, interpolate_spin_tables)',
            'gold_call': 'run_case(_oracle_synthesize_quadratic_response, _oracle_interpolate_spin_tables)',
        },
        {
            'setup': 'def run_case(fn_under_test, fixture_interpolate_spin_tables):\n    import numpy as np\n    tab = fixture_interpolate_spin_tables(np.array([0.731, 0.768, 0.843]))\n    return fn_under_test(tab)',
            'call': 'run_case(synthesize_quadratic_response, interpolate_spin_tables)',
            'gold_call': 'run_case(_oracle_synthesize_quadratic_response, _oracle_interpolate_spin_tables)',
        },
        {
            'setup': 'def run_case(fn_under_test, fixture_interpolate_spin_tables):\n    import numpy as np\n\n    def catches_value_error(fn):\n        try:\n            fn()\n        except ValueError:\n            return True\n        return False\n    good = fixture_interpolate_spin_tables(np.array([0.75]))\n    outside = good.copy()\n    outside[0, 0] = 0.69\n    badmass = good.copy()\n    badmass[0, 1] = 0.0\n    bad320imag = good.copy()\n    bad320imag[0, 5] = 0.0\n    return catches_value_error(lambda: fn_under_test(np.zeros((0, 16)))) and catches_value_error(lambda: fn_under_test(np.zeros((1, 15)))) and catches_value_error(lambda: fn_under_test(outside)) and catches_value_error(lambda: fn_under_test(badmass)) and catches_value_error(lambda: fn_under_test(np.where(np.arange(16) == 2, 0.0, good))) and catches_value_error(lambda: fn_under_test(np.where(np.arange(16) == 3, np.nan, good))) and catches_value_error(lambda: fn_under_test(np.where(np.arange(16) == 4, 0.0, good))) and catches_value_error(lambda: fn_under_test(bad320imag))',
            'call': 'run_case(synthesize_quadratic_response, interpolate_spin_tables)',
            'gold_call': 'run_case(_oracle_synthesize_quadratic_response, _oracle_interpolate_spin_tables)',
        },
    ]
