"""
Extract the real roots of the balanced residual polynomial, sort them on the real axis, and evaluate the residual-polynomial derivative at each real root.

Let the balanced residual polynomial be



$$

\pi(\alpha) = \sum_{k=0}^{m} c_k \alpha^k,

$$



with coefficients stored in ascending order $[c_0, c_1, \ldots, c_m]$. Its derivative is



$$

\pi'(\alpha) = \sum_{k=1}^{m} k\, c_k\, \alpha^{k-1}.

$$



Use apply_balance_method2 to obtain the complete balanced root multiset. Compute the derivative of the residual polynomial normalized by $\pi(0)=1$ at every distinct real root. Multiplicity is part of the polynomial: at a repeated root the derivative is zero. All retained complex roots contribute to the derivative even though they do not appear in the output table.



The input domain includes polynomials of degree up to 64 with clustered roots and root magnitudes spanning $10^{-12}$ to $2\times10^{12}$. Only finite reference derivatives are tested; intermediate products need not be representable in binary64. Return values must satisfy $|y-y_{\text{ref}}| \le 10^{-9}\max(1,|y_{\text{ref}}|)$. The choice of numerical representation and differentiation method is part of the task.



For the later real-axis spline screening, retain only roots whose imaginary part is zero to numerical tolerance, that is $|\operatorname{Im} r| \le \tau \max(1, |r|)$ with $\tau = 10^{-12}$, and use their real parts. Real roots that coincide to the same tolerance, $|r - r'| \le \tau \max(1, |r|, |r'|)$, are a repeated root and produce exactly one row, at which $\pi' = 0$ up to roundoff. Sort the distinct real roots in strictly increasing order,



$$

r_1 < r_2 < \cdots < r_s,

$$



and evaluate $\pi'(r_j)$ for every sorted real root.



Return one row per real balanced root, $[r_j,\ \pi'(r_j)]$.



Complex roots remain factors of the residual polynomial. They are excluded only from the real-root table used by the spline screening.



Invalid input (empty or non-one-dimensional array, non-finite entries, or a balanced root set with fewer than two distinct real roots) must raise ValueError.

Returns
-------
A real NumPy array of shape (n_real, 2) containing each sorted real balanced root and the corresponding residual-polynomial derivative.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def evaluate_real_root_derivatives(
    harmonic_ritz_roots: np.ndarray,
) -> np.ndarray:
    """
    Evaluate the balanced residual-polynomial derivative at its real roots.

    Parameters
    ----------
    harmonic_ritz_roots : np.ndarray
        One-dimensional array of nonzero harmonic Ritz roots with
        complex roots occurring in conjugate pairs.

    Returns
    -------
    np.ndarray
        Real array of shape (n_real, 2) with columns
        [sorted_real_root, derivative_value].

    Raises
    ------
    ValueError
        If the array is empty or not one-dimensional, contains non-finite
        entries, or fewer than two distinct real balanced roots exist.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_evaluate_real_root_derivatives(
    harmonic_ritz_roots,
):
    roots = np.asarray(
        harmonic_ritz_roots,
        dtype=complex,
    )

    if roots.ndim != 1 or roots.size == 0:
        raise ValueError(
            "harmonic_ritz_roots must be a non-empty one-dimensional array."
        )

    if not np.all(
        np.isfinite(roots.real)
        & np.isfinite(roots.imag)
    ):
        raise ValueError(
            "harmonic_ritz_roots must contain only finite values."
        )

    balanced_roots = _oracle_apply_balance_method2(
        roots
    )

    tol = 1.0e-12

    real_roots = []

    for root in balanced_roots:
        scale = max(
            1.0,
            abs(root),
        )

        if abs(root.imag) > tol * scale:
            continue

        value = float(root.real)

        duplicate = any(
            abs(value - previous)
            <= tol * max(1.0, abs(value), abs(previous))
            for previous in real_roots
        )

        if not duplicate:
            real_roots.append(value)

    if len(real_roots) < 2:
        raise ValueError(
            "At least two distinct real balanced roots are required for spline screening."
        )

    real_roots = np.array(
        sorted(real_roots),
        dtype=float,
    )

    def _product_derivative(value):
        from math import fsum, log, exp
        matching = (np.abs(balanced_roots - value)
                    <= tol * np.maximum(1.0, np.maximum(abs(value), np.abs(balanced_roots))))
        if np.count_nonzero(matching) > 1:
            return 0.0
        others = balanced_roots[~matching]
        # Subtract before dividing to retain nearby-root differences. Accumulate
        # magnitudes in logarithms so intermediate products cannot overflow.
        factors = (others - value) / others
        magnitudes = np.abs(factors)
        phase = np.prod(factors / magnitudes).real
        logarithm = fsum([*(log(float(a)) for a in magnitudes), -log(abs(value))])
        return -np.sign(value) * float(phase) * exp(logarithm)

    derivatives = np.array(
        [
            _product_derivative(root)
            for root in real_roots
        ],
        dtype=float,
    )

    if not np.all(np.isfinite(derivatives)):
        raise ValueError(
            "Derivative evaluations must be finite."
        )

    return np.column_stack((
        real_roots,
        derivatives,
    ))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\n'
               '\n'
               'harmonic_ritz_roots = np.array([\n'
               '    -5.0,\n'
               '    -1.3,\n'
               '    0.9,\n'
               '    2.0 + 1.5j,\n'
               '    2.0 - 1.5j\n'
               '], dtype=complex)\n',
      'call': 'evaluate_real_root_derivatives(\n    harmonic_ritz_roots\n)',
      'gold_call': '_oracle_evaluate_real_root_derivatives(\n    harmonic_ritz_roots\n)'},
     {'setup': 'import numpy as np\n'
               '\n'
               'harmonic_ritz_roots = np.array([\n'
               '    -6.0,\n'
               '    -2.0,\n'
               '    0.8,\n'
               '    3.5\n'
               '], dtype=complex)\n',
      'call': 'evaluate_real_root_derivatives(\n    harmonic_ritz_roots\n)',
      'gold_call': '_oracle_evaluate_real_root_derivatives(\n    harmonic_ritz_roots\n)'},
     {'setup': 'import numpy as np\n'
               '\n'
               'harmonic_ritz_roots = np.array([\n'
               '    1.0,\n'
               '    -1.1111111111111112,\n'
               '    -20.0 + 20.0j,\n'
               '    -20.0 - 20.0j\n'
               '], dtype=complex)\n',
      'call': 'evaluate_real_root_derivatives(\n    harmonic_ritz_roots\n)',
      'gold_call': '_oracle_evaluate_real_root_derivatives(\n    harmonic_ritz_roots\n)'},
     {'setup': 'import numpy as np\n'
               '\n'
               'harmonic_ritz_roots = np.array([\n'
               '    2.0,\n'
               '    2.0,\n'
               '    -0.8\n'
               '], dtype=complex)\n',
      'call': 'evaluate_real_root_derivatives(\n    harmonic_ritz_roots\n)',
      'gold_call': '_oracle_evaluate_real_root_derivatives(\n    harmonic_ritz_roots\n)'},
     {'setup': 'import numpy as np\n'
               '\n'
               'harmonic_ritz_roots = np.array([\n'
               '    1.25,\n'
               '    2.0 + 1.0j,\n'
               '    -3.0 + 1.0e-14j,\n'
               '    2.0 - 1.0j,\n'
               '    -0.75\n'
               '], dtype=complex)\n',
      'call': 'evaluate_real_root_derivatives(\n    harmonic_ritz_roots\n)',
      'gold_call': '_oracle_evaluate_real_root_derivatives(\n    harmonic_ritz_roots\n)'},
     {'setup': 'import numpy as np\n'
               '\n'
               'harmonic_ritz_roots = np.array([\n'
               '    -3.0,\n'
               '    -1.0,\n'
               '    1.0678380296485983,\n'
               '    1.6094271405406886,\n'
               '    2.6866715986911203,\n'
               '    4.287768888388513,\n'
               '    6.395177052675791,\n'
               '    8.985806886701518,\n'
               '    12.03127490787994,\n'
               '    15.498214331265899,\n'
               '    19.348640643033054,\n'
               '    23.540367766756177,\n'
               '    28.027470262892436,\n'
               '    32.76078649750764,\n'
               '    37.68845726742523,\n'
               '    42.75649398050857,\n'
               '    47.90937016597429,\n'
               '    53.09062983402571,\n'
               '    58.243506019491434,\n'
               '    63.31154273257478,\n'
               '    68.23921350249236,\n'
               '    72.97252973710758,\n'
               '    77.45963223324384,\n'
               '    81.65135935696695,\n'
               '    85.5017856687341,\n'
               '    88.96872509212005,\n'
               '    92.01419311329849,\n'
               '    94.60482294732421,\n'
               '    96.71223111161149,\n'
               '    98.31332840130888,\n'
               '    99.39057285945933,\n'
               '    99.9321619703514\n'
               '], dtype=complex)\n',
      'call': 'evaluate_real_root_derivatives(\n    harmonic_ritz_roots\n)',
      'gold_call': '_oracle_evaluate_real_root_derivatives(\n    harmonic_ritz_roots\n)'},
     {'setup': 'import numpy as np\n'
               'harmonic_ritz_roots = np.r_[2e-12, np.linspace(1e-6, 2e-6, 20), '
               '1e12*(1+np.arange(40)*1e-8)].astype(complex)\n',
      'call': 'evaluate_real_root_derivatives(harmonic_ritz_roots)',
      'gold_call': '_oracle_evaluate_real_root_derivatives(harmonic_ritz_roots)'},
     {'setup': 'import numpy as np\n'
               'harmonic_ritz_roots = np.r_[2e-12, np.linspace(1e-6, 2e-6, 20), '
               '1e12*(1+np.arange(40)*1e-8)].astype(complex)\n'
               'harmonic_ritz_roots = harmonic_ritz_roots[np.r_[0, np.arange(60, 0, -1)]]\n',
      'call': 'evaluate_real_root_derivatives(harmonic_ritz_roots)',
      'gold_call': '_oracle_evaluate_real_root_derivatives(harmonic_ritz_roots)'},
     {'setup': 'import numpy as np\n'
               'centers = 0.9*np.cos((2*np.arange(30)+1)*np.pi/60)+0.05\n'
               'harmonic_ritz_roots = np.r_[1/1024, 1., centers+1e-5j, centers-1e-5j]\n',
      'call': 'evaluate_real_root_derivatives(harmonic_ritz_roots)',
      'gold_call': '_oracle_evaluate_real_root_derivatives(harmonic_ritz_roots)'},
     {'setup': 'import numpy as np\n'
               '\n'
               '\n'
               'def _run_invalid(fn, *args):\n'
               '    try:\n'
               '        fn(*args)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               '\n'
               'import numpy as np\n'
               '\n'
               'harmonic_ritz_roots = np.array([\n'
               '    -5.0\n'
               '], dtype=complex)\n',
      'call': '_run_invalid(\n    evaluate_real_root_derivatives, harmonic_ritz_roots\n)',
      'gold_call': '_run_invalid(\n    _oracle_evaluate_real_root_derivatives, harmonic_ritz_roots\n)'}]
