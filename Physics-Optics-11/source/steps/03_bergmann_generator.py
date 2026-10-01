"""
Build the dimensionless first-order TE/TM layer generator.

With chi=alpha^(-1)*partial_x(psi)/(i*k), both components of the state are continuous through a plane interface. The constitutive Bergmann equation then becomes a first-order evolution in k*x. Transverse differentiation acts on both sides of the reciprocal convolution. Alpha and beta have already been assigned for the polarization; use the earlier public reciprocal_fourier function in candidate implementations.

Returns
-------
return generator: finite complex ndarray (2*J,2*J) satisfying partial_(k*x)(psi,chi)=1j*generator@(psi,chi), in the stated component and diffraction-order ordering.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def bergmann_generator(sines, alpha_coefficients, beta_coefficients):
    """Return the dimensionless state generator with shape (2*J, 2*J).

    Use time dependence exp(-1j*omega*t), state (psi, chi), and
    chi=alpha**(-1)*partial_x(psi)/(1j*k). For the equation
    div(alpha**(-1)*grad(psi))+k**2*beta*psi=0, derive the matrix G
    satisfying partial_(k*x)(state)=1j*G@state. TE has (alpha,beta)
    =(mu,epsilon); TM has (epsilon,mu). The input coefficients already
    have this mapping. A positive harmonic h couples order j to j+h.
    sines is a nonempty finite real vector of length J. Both profiles
    are nonempty finite numeric coefficient vectors, and alpha's constant
    strictly dominates its harmonic tail. Use reciprocal_fourier for the
    reciprocal alpha coefficients, including harmonics through J-1.
    Return complex values, preserve inputs, and raise ValueError for
    invalid inputs. Evanescent transverse sines are allowed.

    Raise ValueError if the computation produces nonfinite coefficients
    or efficiencies; do not return nonfinite results.
    """
    return np.zeros((2 * len(sines), 2 * len(sines)), dtype=complex)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_bergmann_generator(sines, alpha_coefficients, beta_coefficients):
    import numpy as np

    sines = np.asarray(sines)
    if (
        sines.ndim != 1
        or sines.size == 0
        or sines.dtype.kind not in "iuf"
        or not np.all(np.isfinite(sines))
    ):
        raise ValueError("sines must be a nonempty finite real vector.")
    count = len(sines)
    alpha_inverse = _oracle_reciprocal_fourier(alpha_coefficients, count)

    def convolution(values):
        values = np.asarray(values)
        if (
            values.ndim != 1
            or values.size == 0
            or values.dtype.kind not in "iufc"
            or not np.all(np.isfinite(values))
        ):
            raise ValueError(
                "Material coefficients must be finite numeric vectors."
            )
        matrix = np.zeros((count, count), dtype=complex)
        for harmonic, value in enumerate(values[:count]):
            columns = np.arange(count - harmonic)
            matrix[columns + harmonic, columns] = value
        return matrix

    alpha = convolution(alpha_coefficients)
    beta = convolution(beta_coefficients)
    inverse = convolution(alpha_inverse)
    lower = beta - sines[:, None] * inverse * sines[None, :]
    zero = np.zeros_like(alpha)
    result = np.block([[zero, alpha], [lower, zero]])
    if not np.all(np.isfinite(result)):
        raise ValueError("The generator is not finite.")
    return result

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np


def test_cases():
    return [{'setup': '\nsines = np.array([-0.6, 0.1, 0.8])\n',
      'call': 'bergmann_generator(sines, [1.0], [1.0])',
      'gold_call': '_oracle_bergmann_generator(sines, [1.0], [1.0])'},
     {'setup': '\nsines = np.array([-0.6, -0.21, 0.18, 0.57, 0.96])\n',
      'call': 'bergmann_generator(\n'
              '    sines, [2.4 + 0.45j, 0.35, 0.04j], [1.2 + 0.14j, 0.10, 0.015]\n'
              ')',
      'gold_call': '_oracle_bergmann_generator(\n'
                   '    sines, [2.4 + 0.45j, 0.35, 0.04j], [1.2 + 0.14j, 0.10, 0.015]\n'
                   ')'},
     {'setup': '\nsines = np.array([0.0, 0.7, 1.4, 2.1])\n',
      'call': 'bergmann_generator(sines, [1.4, -0.2j], [2.1 + 0.3j, 0.1, 0.02j])',
      'gold_call': '_oracle_bergmann_generator(sines, [1.4, -0.2j], [2.1 + 0.3j, 0.1, 0.02j])'},
     {'setup': '',
      'call': 'bergmann_generator([0.3], [2.0], [1.0])',
      'gold_call': '_oracle_bergmann_generator([0.3], [2.0], [1.0])'},
     {'setup': 'def rejects_nonfinite(fn):\n'
               '    try:\n'
               '        fn([1e+200], [1.0], [1.0])\n'
               '    except ValueError:\n'
               '        return 1.0\n'
               '    return 0.0\n',
      'call': 'rejects_nonfinite(bergmann_generator)',
      'gold_call': 'rejects_nonfinite(_oracle_bergmann_generator)'}]
