"""
Construct the real, zero-mean massless Green function on a periodic cubic lattice of size L^3.



Use the nearest-neighbor lattice Laplacian

Delta_L f(x)=sum_mu[f(x+mu)+f(x-mu)-2*f(x)].



The Green function must satisfy

Delta_L G(x)=-delta_x0+1/L^3.



Follow the finite-volume prescription in Appendix A, Eqs. (A6)–(A8): use the inverse lattice-Laplacian eigenvalue for every nonzero momentum mode and set the joint zero-mode coefficient to zero. Do not substitute the continuum 1/r Green function.



Require an integer L>=2 and raise ValueError for invalid input.

A periodic massless lattice Laplacian has a constant zero mode and therefore cannot be inverted on all modes. Appendix A excludes the joint Fourier zero mode, fixing the Green function to have zero spatial mean.



For momentum indices n_mu=0,...,L-1, the positive eigenvalue of minus the lattice Laplacian is 4*sum_mu sin^2(pi*n_mu/L). Its reciprocal defines the nonzero Fourier coefficients.



Removing the zero mode gives Delta_L G=-delta_x0+1/L^3. This finite-volume term is responsible for the constant correction in the subsequent Hodge decomposition.

Returns
-------
A real NumPy array of shape (L,L,L), containing the zero-mean periodic massless lattice Green function with the source at index (0,0,0).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def periodic_green(L: int) -> np.ndarray:
    """Construct the zero-mean periodic massless Green function.

    Parameters
    ----------
    L : int
        Cubic side length at least 2; booleans are excluded.
        Odd lengths are allowed in this Green-function step.

    Returns
    -------
    numpy.ndarray
        Real array (L,L,L), with source at (0,0,0) and zero mean.
        Nonzero Fourier modes invert minus the lattice Laplacian;
        the joint zero mode is zero, giving Delta_L G=-delta+1/L**3.

    Raises
    ------
    ValueError
        If L is not an integer (excluding booleans) at least 2.
    """
    return np.empty(0, dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_periodic_green(L: int) -> np.ndarray:
    """Construct the zero-mean periodic massless lattice Green function."""
    if (
        isinstance(L, (bool, np.bool_))
        or not isinstance(L, (int, np.integer))
        or L < 2
    ):
        raise ValueError("L must be an integer >= 2")

    momenta = np.arange(L, dtype=float)

    one_dimensional_eigenvalues = (
        4.0 * np.sin(np.pi * momenta / L) ** 2
    )

    eigenvalues = (
        one_dimensional_eigenvalues[:, None, None]
        + one_dimensional_eigenvalues[None, :, None]
        + one_dimensional_eigenvalues[None, None, :]
    )

    fourier_kernel = np.zeros_like(eigenvalues)

    np.divide(
        1.0,
        eigenvalues,
        out=fourier_kernel,
        where=eigenvalues > 0.0,
    )

    fourier_kernel[0, 0, 0] = 0.0

    return np.fft.ifftn(fourier_kernel).real

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': '', 'call': 'periodic_green(4)', 'gold_call': '_oracle_periodic_green(4)'},
     {'setup': '', 'call': 'periodic_green(6)', 'gold_call': '_oracle_periodic_green(6)'},
     {'setup': '', 'call': 'periodic_green(8)', 'gold_call': '_oracle_periodic_green(8)'},
     {'setup': 'def raises_value_error(fn, *args, **kwargs):\n'
               '    try:\n'
               '        fn(*args, **kwargs)\n'
               '    except ValueError:\n'
               '        return 1.0\n'
               '    return 0.0\n',
      'call': 'raises_value_error(periodic_green, 1)',
      'gold_call': '1.0',
      'tol': 0.0}]
