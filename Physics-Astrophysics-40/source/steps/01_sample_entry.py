"""
Step 1: draw one entry state, physical realization, and nuisance point.

Use one supplied or default `Generator(PCG64)` without reseeding. Draw six standard normals first and form `state6=mu+chol(C)@g`. Draw density scale next, then strength, dust fraction, largest-child fraction, `beta`, wind scale, drag scale, and ablation coefficient in the prompt's order. Derive the four nuisance coordinates from strength, dust, beta, and wind without additional random draws.

Returns
-------
Return one finite float vector of length 18 ordered `(x,y,z,vx,vy,vz,mass,strength,dust,largest,beta,wind_scale,cd_scale,sigma_abl,uS,uD,uBeta,uWind)`.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

"""Step 1: draw one entry state, physical realization, and nuisance point."""
import numpy as np

def sample_entry(rng=None):
    """Draw one entry realization in the stated random-stream order.

    Parameters
    ----------
    rng : numpy.random.Generator or None
        Advancing stream with standard_normal and uniform methods, or a new
        PCG64(761903) stream when None. Draws and ranges are in the statement.

    Returns
    -------
    ndarray, shape (18,)
        (x,y,z,vx,vy,vz,mass,strength,dust,largest,beta,wind_scale,
        cd_scale,sigma_abl,u_strength,u_dust,u_beta,u_wind).

    Raises
    ------
    ValueError
        If rng lacks either required random-number method.
    Notes
    -----
    Return the stated deterministic result to rtol=1e-10 and atol=1e-12.
    The supplied numerical controls define the discrete target.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

"""Step 1: draw one entry state, physical realization, and nuisance point."""
import numpy as np
_MEAN = np.array([0.0, 0.0, 100000.0, 12400.0, 180.0, -5250.0])
_COVARIANCE = np.array([[420.0 ** 2, 0.35 * 420.0 * 260.0, 0.0, 0.0, 0.0, 0.0], [0.35 * 420.0 * 260.0, 260.0 ** 2, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 95.0 ** 2, 0.0, 0.0, -0.25 * 95.0 * 18.0], [0.0, 0.0, 0.0, 26.0 ** 2, 0.2 * 26.0 * 11.0, 0.0], [0.0, 0.0, 0.0, 0.2 * 26.0 * 11.0, 11.0 ** 2, 0.0], [0.0, 0.0, -0.25 * 95.0 * 18.0, 0.0, 0.0, 18.0 ** 2]])

def _oracle_sample_entry(rng=None):
    if rng is None:
        rng = np.random.Generator(np.random.PCG64(761903))
    if not hasattr(rng, 'standard_normal') or not hasattr(rng, 'uniform'):
        raise ValueError('rng must provide standard_normal and uniform')
    state6 = _MEAN + np.linalg.cholesky(_COVARIANCE) @ rng.standard_normal(6)
    density_scale = rng.uniform(0.92, 1.08)
    state7 = np.concatenate((state6, [185000.0 * density_scale]))
    strength = rng.uniform(420000.0, 720000.0)
    dust = rng.uniform(0.18, 0.34)
    largest = rng.uniform(0.25, 0.38)
    beta = rng.uniform(0.48, 0.66)
    wind_scale = rng.uniform(0.88, 1.12)
    cd_scale = rng.uniform(0.9, 1.1)
    sigma_abl = rng.uniform(7.2e-08, 8.8e-08)
    nuisance = np.array([(strength - 420000.0) / 300000.0, (dust - 0.18) / 0.16, (beta - 0.48) / 0.18, (wind_scale - 0.88) / 0.24])
    physical = np.array([strength, dust, largest, beta, wind_scale, cd_scale, sigma_abl])
    return np.concatenate((state7, physical, nuisance))

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np

def test_cases():
    """Exercise the default stream, a second seed, and stream advancement."""
    return [{'setup': 'marker = 1.0', 'call': 'sample_entry()', 'gold_call': '_oracle_sample_entry()'}, {'setup': 'rng_candidate=np.random.Generator(np.random.PCG64(77)); rng_oracle=np.random.Generator(np.random.PCG64(77))', 'call': 'sample_entry(rng_candidate)', 'gold_call': '_oracle_sample_entry(rng_oracle)'}, {'setup': 'rng_candidate=np.random.Generator(np.random.PCG64(91)); rng_oracle=np.random.Generator(np.random.PCG64(91)); rng_candidate.random(13); rng_oracle.random(13)', 'call': 'sample_entry(rng_candidate)', 'gold_call': '_oracle_sample_entry(rng_oracle)'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\n', 'call': 'rejects(lambda: sample_entry(rng=object()))', 'gold_call': 'rejects(lambda: _oracle_sample_entry(rng=object()))'}]
