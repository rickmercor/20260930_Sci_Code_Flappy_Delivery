"""
Compute the signed long-run percentage radiance distortion (final orchestrator).

This end-to-end calculation uses the acceptance-terminated vanilla MH estimator and the finite path-family conventions of the main problem.

Returns
-------
float: the unrounded signed percentage distortion, 100 times the ratio of limiting vanilla radiance to exact radiance minus 100.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_radiance_distortion(flux: "np.ndarray", volumes: "np.ndarray", local: "np.ndarray", global_prob: "np.ndarray", large_step: float, pixel: int) -> float:
    """Parameters
    ----------
    flux : np.ndarray
        Nonnegative (N, P) pixel contribution densities, with positive row sums.
    volumes : np.ndarray
        Positive (N,) cell volumes summing to one.
    local : np.ndarray
        Row-stochastic (N, N) small-step proposal matrix.
    global_prob : np.ndarray
        Nonnegative (N,) large-step destination probabilities summing to one.
    large_step : float
        Large-step probability in $[0,1]$. The resulting accepted chain is irreducible.
    pixel : int
        Selected pixel index. Its exact radiance must be positive.

    Returns
    -------
    result : float
        Signed percentage distortion, 100 times the ratio of limiting vanilla
        radiance to exact radiance minus 100. This is an unrounded scalar.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_compute_radiance_distortion(flux: "np.ndarray", volumes: "np.ndarray", local: "np.ndarray", global_prob: "np.ndarray", large_step: float, pixel: int) -> float:
    quantities = _oracle_prepare_transport(flux, volumes, pixel)
    proposal = _oracle_compose_proposal(local, global_prob, large_step)
    acceptance = _oracle_compute_acceptance(quantities[:, 0], proposal)
    moments = _oracle_compute_rejection_moments(proposal, acceptance)
    mean_weights = _oracle_compute_stopped_weights(moments)
    stationary = _oracle_compute_tour_law(proposal, acceptance, moments)
    radiances = _oracle_reconstruct_radiances(quantities, mean_weights, stationary)
    return float(100.0 * (radiances[1] / radiances[0] - 1.0))

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
      'call': 'compute_radiance_distortion(flux.copy(), volumes.copy(), local.copy(), global_prob.copy(), '
              'large_step, pixel)',
      'gold_call': '_oracle_compute_radiance_distortion(flux.copy(), volumes.copy(), local.copy(), '
                   'global_prob.copy(), large_step, pixel)',
      'tol': 1e-10},
     {'setup': 'import numpy as np\n'
               'flux = np.array([[2.,3.]])\n'
               'volumes = np.array([1.])\n'
               'local = np.ones((1,1))\n'
               'global_prob = np.ones(1)\n'
               'large_step = 0.\n'
               'pixel = 1\n',
      'call': 'compute_radiance_distortion(flux.copy(), volumes.copy(), local.copy(), global_prob.copy(), '
              'large_step, pixel)',
      'gold_call': '_oracle_compute_radiance_distortion(flux.copy(), volumes.copy(), local.copy(), '
                   'global_prob.copy(), large_step, pixel)',
      'tol': 1e-10},
     {'setup': 'import numpy as np\n'
               'flux = np.array([[.01,0.],[0.,9.],[.2,.1]])\n'
               'volumes = np.array([.2,.5,.3])\n'
               'local = np.array([[.1,.9,0.],[0.,.1,.9],[.9,0.,.1]])\n'
               'global_prob = np.array([.2,.5,.3])\n'
               'large_step = .15\n'
               'pixel = 1\n',
      'call': 'compute_radiance_distortion(flux.copy(), volumes.copy(), local.copy(), global_prob.copy(), '
              'large_step, pixel)',
      'gold_call': '_oracle_compute_radiance_distortion(flux.copy(), volumes.copy(), local.copy(), '
                   'global_prob.copy(), large_step, pixel)',
      'tol': 1e-10}]
