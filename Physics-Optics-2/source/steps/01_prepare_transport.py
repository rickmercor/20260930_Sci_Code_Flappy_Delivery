"""
Compute the discrete target masses and the importance observable for one pixel.

The input fluxes are densities with respect to the supplied cell-volume measure. Cell order is preserved.

Returns
-------
np.ndarray of shape (N, 2): column 0 holds the unnormalized target masses and column 1 the selected pixel's importance observable, in cell order.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def prepare_transport(flux: "np.ndarray", volumes: "np.ndarray", pixel: int) -> "np.ndarray":
    """Parameters
    ----------
    flux : np.ndarray
        Nonnegative (N, P) pixel contribution densities, with positive row sums.
    volumes : np.ndarray
        Positive (N,) reference-measure cell volumes, summing to one.
    pixel : int
        Selected pixel index, between $0$ and $P-1$.

    Returns
    -------
    result : np.ndarray
        Shape (N, 2), with unnormalized target probability masses in column zero
        and the selected pixel's importance-sampling observable in column one.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_prepare_transport(flux: "np.ndarray", volumes: "np.ndarray", pixel: int) -> "np.ndarray":
    density = np.sum(flux, axis=1)
    return np.column_stack((density * volumes, flux[:, pixel] / density))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return independent numerical test configurations."""
    return [{'setup': 'import numpy as np\n'
               'flux = np.array([[.2,0],[.8,.1],[4,.5],[.1,2],[.3,7],[1.2,.2],[2.5,1],[.05,.9]])\n'
               'volumes = np.array([.08,.12,.09,.16,.14,.11,.17,.13])\n'
               'local = np.zeros((8,8))\n'
               'for i in range(8):\n'
               '    local[i,i] = .1\n'
               '    local[i,(i+1)%8] = .55\n'
               '    local[i,(i-1)%8] = .35\n'
               'global_prob = volumes.copy()\n'
               'large_step = .3\n'
               'pixel = 1\n',
      'call': 'prepare_transport(flux.copy(), volumes.copy(), pixel)',
      'gold_call': '_oracle_prepare_transport(flux.copy(), volumes.copy(), pixel)',
      'tol': 1e-10},
     {'setup': 'import numpy as np\n'
               'flux = np.array([[2.,3.]])\n'
               'volumes = np.array([1.])\n'
               'local = np.ones((1,1))\n'
               'global_prob = np.ones(1)\n'
               'large_step = 0.\n'
               'pixel = 1\n',
      'call': 'prepare_transport(flux.copy(), volumes.copy(), pixel)',
      'gold_call': '_oracle_prepare_transport(flux.copy(), volumes.copy(), pixel)',
      'tol': 1e-10},
     {'setup': 'import numpy as np\n'
               'flux = np.array([[.01,0.],[0.,9.],[.2,.1]])\n'
               'volumes = np.array([.2,.5,.3])\n'
               'local = np.array([[.1,.9,0.],[0.,.1,.9],[.9,0.,.1]])\n'
               'global_prob = np.array([.2,.5,.3])\n'
               'large_step = .15\n'
               'pixel = 1\n',
      'call': 'prepare_transport(flux.copy(), volumes.copy(), pixel)',
      'gold_call': '_oracle_prepare_transport(flux.copy(), volumes.copy(), pixel)',
      'tol': 1e-10}]
