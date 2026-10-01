"""
Step 4: partition a disrupted parent and assign transverse kicks.

After dust loss set $available=m_parent*(1-dust)$ and $largest_mass=largest_fraction*available$. Starting from that child, propose ranks `k=2,...,7` with normalized fraction $(largest_fraction^(-beta)+beta*largest_fraction^(1-beta)*(k-1)/(1-beta))^(-1/beta)$, multiply by available mass, stop below $minimum_mass$, cap by the remaining mass, and append a residual only when it is at least $minimum_mass$. At the breakup state set `e=v/|v|`, choose axis `(0,0,1)` when $|e_z|<.9$ and `(0,1,0)` otherwise, set $b1=normalize(cross(e,axis))$ and $b2=cross(e,b1)$, and consume one uniform angle on `[0,2*pi)` per child. Kick magnitude is $sqrt(3*rho/(2*3050)*(r_parent/r_child)^kick_power)*|v-wind|$. When requested subtract $sum(m_i*kick_i)/sum(m_i)$ from each kick. Set child strength to $min(S_parent*(m_parent/m_child)^alpha,strength_cap)$ and increment depth.

Returns
-------
Return a finite `(n_children,9)` array in generated rank order, each row `(state7,child_strength,parent_depth+1)`.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

"""Step 4: partition a disrupted parent and assign transverse kicks."""
import math
import numpy as np

def partition_fragment_cloud(fragment, dust_fraction, largest_fraction, beta, wind_scale, rng=None, alpha=0.16, strength_cap=15000000.0, minimum_mass=0.2, kick_power=2.0, momentum_correct=True):
    """Partition one disrupted parent and assign transverse velocities.

    Parameters
    ----------
    fragment : array_like, shape (9,)
        Finite (state7,strength,depth), positive mass and strength,
        nonzero velocity, and nonnegative integer depth.
    dust_fraction, largest_fraction, beta : float
        Finite fractions in [0,1), (0,1), and (0,1), respectively.
    wind_scale : float
        Finite nonnegative wind multiplier.
    rng : numpy.random.Generator or None
        Advancing uniform stream, default PCG64(741).
    alpha, strength_cap, minimum_mass, kick_power : float
        Finite nonnegative strength exponent, positive cap in Pa, positive
        proposal/residual floor in kg, and nonnegative radius-ratio exponent
        inside the square root of the kick magnitude.
        The first largest child is retained even below minimum_mass.
    momentum_correct : bool
        Whether to remove the resolved-mass-weighted mean kick.

    Returns
    -------
    ndarray, shape (n_children,9)
        Generated-rank-order (state7,child_strength,parent_depth+1) rows.

    Raises
    ------
    ValueError
        For a shape or domain violation above, any nonfinite numeric input,
        zero parent velocity, nonboolean momentum_correct, or rng without
        a uniform method.
    Notes
    -----
    Return the stated deterministic result to rtol=1e-10 and atol=1e-12.
    The supplied numerical controls define the discrete target.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

"""Step 4: partition a disrupted parent and assign transverse kicks."""
import math
import numpy as np
_RHO_BODY = 3050.0

def _oracle_partition_fragment_cloud(fragment, dust_fraction, largest_fraction, beta, wind_scale, rng=None, alpha=0.16, strength_cap=15000000.0, minimum_mass=0.2, kick_power=2.0, momentum_correct=True):
    fragment = np.asarray(fragment, dtype=float)
    if fragment.shape != (9,) or not np.all(np.isfinite(fragment)):
        raise ValueError('fragment must contain state, strength, and depth')
    if not np.all(np.isfinite([dust_fraction, largest_fraction, beta, wind_scale, alpha, strength_cap, minimum_mass, kick_power])):
        raise ValueError('fragment controls must be finite')
    if not (0.0 <= dust_fraction < 1.0 and 0.0 < largest_fraction < 1.0 and (0.0 < beta < 1.0)):
        raise ValueError('fragment-distribution parameters are invalid')
    if fragment[6] <= 0.0 or fragment[7] <= 0.0 or minimum_mass <= 0.0:
        raise ValueError('masses and strength must be positive')
    if fragment[8] < 0 or fragment[8] != np.floor(fragment[8]) or np.linalg.norm(fragment[3:6]) == 0.0:
        raise ValueError('depth must be a nonnegative integer and velocity must be nonzero')
    if wind_scale < 0 or alpha < 0 or strength_cap <= 0 or (kick_power < 0):
        raise ValueError('wind, alpha, and kick power must be nonnegative and strength cap positive')
    if not isinstance(momentum_correct, (bool, np.bool_)):
        raise ValueError('momentum_correct must be boolean')
    if rng is None:
        rng = np.random.Generator(np.random.PCG64(741))
    if not hasattr(rng, 'uniform'):
        raise ValueError('rng must provide uniform')
    available = fragment[6] * (1.0 - dust_fraction)
    largest = largest_fraction * available
    masses = [largest]
    remaining = available - largest
    rank = 2
    while remaining >= minimum_mass and rank < 8:
        normalized = (largest_fraction ** (-beta) + beta / (1.0 - beta) * largest_fraction ** (1.0 - beta) * (rank - 1)) ** (-1.0 / beta)
        proposed = normalized * available
        if proposed < minimum_mass:
            break
        value = min(proposed, remaining)
        masses.append(value)
        remaining -= value
        if remaining < minimum_mass:
            break
        rank += 1
    if remaining >= minimum_mass:
        masses.append(remaining)
    masses = np.asarray(masses, dtype=float)
    parent_radius = (3.0 * fragment[6] / (4.0 * math.pi * _RHO_BODY)) ** (1.0 / 3.0)
    diagnostics = _oracle_atmospheric_rhs(fragment[:7], wind_scale, 1.0, 0.0)
    rho = diagnostics[7]
    relative_speed = diagnostics[12]
    velocity = fragment[3:6]
    unit = velocity / np.linalg.norm(velocity)
    seed_axis = np.array([0.0, 0.0, 1.0]) if abs(unit[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    basis1 = np.cross(unit, seed_axis)
    basis1 /= np.linalg.norm(basis1)
    basis2 = np.cross(unit, basis1)
    angles = rng.uniform(0.0, 2.0 * math.pi, len(masses))
    kicks = []
    for mass, angle in zip(masses, angles):
        radius = (3.0 * mass / (4.0 * math.pi * _RHO_BODY)) ** (1.0 / 3.0)
        magnitude = math.sqrt(3.0 * rho / (2.0 * _RHO_BODY)) * (parent_radius / radius) ** (0.5 * kick_power) * relative_speed
        kicks.append(magnitude * (math.cos(angle) * basis1 + math.sin(angle) * basis2))
    kicks = np.asarray(kicks)
    if momentum_correct:
        kicks -= np.sum(masses[:, None] * kicks, axis=0) / np.sum(masses)
    children = []
    for mass, kick in zip(masses, kicks):
        child = fragment.copy()
        child[3:6] += kick
        child[6] = mass
        child[7] = min(fragment[7] * (fragment[6] / mass) ** alpha, strength_cap)
        child[8] = fragment[8] + 1.0
        children.append(child)
    return np.asarray(children)

# =============================================================================
# TEST CASES
# =============================================================================

import math
import numpy as np

def test_cases():
    """Cover corrected kicks, a different partition, and the legacy exponent."""
    return [{'setup': 'fragment=np.array([0.,0.,34000.,11800.,120.,-4700.,120000.,6e5,0.]); rng_candidate=np.random.Generator(np.random.PCG64(12)); rng_oracle=np.random.Generator(np.random.PCG64(12))', 'call': 'partition_fragment_cloud(fragment,.24,.31,.57,1.0,rng_candidate)', 'gold_call': '_oracle_partition_fragment_cloud(fragment,.24,.31,.57,1.0,rng_oracle)'}, {'setup': 'fragment=np.array([2.,3.,22000.,8400.,-50.,-2600.,9000.,1.2e6,1.]); rng_candidate=np.random.Generator(np.random.PCG64(31)); rng_oracle=np.random.Generator(np.random.PCG64(31))', 'call': 'partition_fragment_cloud(fragment, 0.18, 0.38, 0.49, 0.92, rng_candidate, minimum_mass=0.5)', 'gold_call': '_oracle_partition_fragment_cloud(fragment, 0.18, 0.38, 0.49, 0.92, rng_oracle, minimum_mass=0.5)'}, {'setup': 'fragment=np.array([0.,0.,30000.,10000.,0.,-4000.,40000.,8e5,0.]); rng_candidate=np.random.Generator(np.random.PCG64(7)); rng_oracle=np.random.Generator(np.random.PCG64(7))', 'call': 'partition_fragment_cloud(fragment,.3,.28,.63,1.1,rng_candidate,kick_power=1.,momentum_correct=False)', 'gold_call': '_oracle_partition_fragment_cloud(fragment,.3,.28,.63,1.1,rng_oracle,kick_power=1.,momentum_correct=False)'}, {'setup': 'f=np.array([2.,-3.,9000.,10.,20.,-5000.,1.4,7e5,2.]); rng_candidate=np.random.default_rng(47); rng_oracle=np.random.default_rng(47)\n', 'call': 'partition_fragment_cloud(f,.13,.43,.68,.8,rng_candidate,alpha=.3,strength_cap=8e5,minimum_mass=.09,kick_power=1.5)', 'gold_call': '_oracle_partition_fragment_cloud(f,.13,.43,.68,.8,rng_oracle,alpha=.3,strength_cap=8e5,minimum_mass=.09,kick_power=1.5)'}, {'setup': 'f=np.array([11.,-7.,18000.,5300.,70.,-2100.,400.,7e5,1.])\n', 'call': 'partition_fragment_cloud(f,0.,.31,.55,0.,alpha=0.,strength_cap=1e9,minimum_mass=200.,kick_power=0.,momentum_correct=False)', 'gold_call': '_oracle_partition_fragment_cloud(f,0.,.31,.55,0.,alpha=0.,strength_cap=1e9,minimum_mass=200.,kick_power=0.,momentum_correct=False)'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\n', 'call': 'rejects(lambda: partition_fragment_cloud(np.ones(8),.2,.3,.6,1.))', 'gold_call': 'rejects(lambda: _oracle_partition_fragment_cloud(np.ones(8),.2,.3,.6,1.))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\nf=np.array([11.,-7.,18000.,5300.,70.,-2100.,400.,7e5,1.])\nf[6]=0.\n', 'call': 'rejects(lambda: partition_fragment_cloud(f,.2,.3,.6,1.))', 'gold_call': 'rejects(lambda: _oracle_partition_fragment_cloud(f,.2,.3,.6,1.))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\nf=np.array([11.,-7.,18000.,5300.,70.,-2100.,400.,7e5,1.])\nf[7]=0.\n', 'call': 'rejects(lambda: partition_fragment_cloud(f,.2,.3,.6,1.))', 'gold_call': 'rejects(lambda: _oracle_partition_fragment_cloud(f,.2,.3,.6,1.))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\nf=np.array([11.,-7.,18000.,5300.,70.,-2100.,400.,7e5,1.])\nf[8]=-1.\n', 'call': 'rejects(lambda: partition_fragment_cloud(f,.2,.3,.6,1.))', 'gold_call': 'rejects(lambda: _oracle_partition_fragment_cloud(f,.2,.3,.6,1.))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\nf=np.array([11.,-7.,18000.,5300.,70.,-2100.,400.,7e5,1.])\nf[8]=.5\n', 'call': 'rejects(lambda: partition_fragment_cloud(f,.2,.3,.6,1.))', 'gold_call': 'rejects(lambda: _oracle_partition_fragment_cloud(f,.2,.3,.6,1.))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\nf=np.array([11.,-7.,18000.,5300.,70.,-2100.,400.,7e5,1.])\nf[3:6]=0.\n', 'call': 'rejects(lambda: partition_fragment_cloud(f,.2,.3,.6,1.))', 'gold_call': 'rejects(lambda: _oracle_partition_fragment_cloud(f,.2,.3,.6,1.))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\nf=np.array([11.,-7.,18000.,5300.,70.,-2100.,400.,7e5,1.])\nf[0]=np.nan\n', 'call': 'rejects(lambda: partition_fragment_cloud(f,.2,.3,.6,1.))', 'gold_call': 'rejects(lambda: _oracle_partition_fragment_cloud(f,.2,.3,.6,1.))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\nf=np.array([11.,-7.,18000.,5300.,70.,-2100.,400.,7e5,1.])\n', 'call': 'rejects(lambda: partition_fragment_cloud(f,dust_fraction=-1.,largest_fraction=.3,beta=.6,wind_scale=1.))', 'gold_call': 'rejects(lambda: _oracle_partition_fragment_cloud(f,dust_fraction=-1.,largest_fraction=.3,beta=.6,wind_scale=1.))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\nf=np.array([11.,-7.,18000.,5300.,70.,-2100.,400.,7e5,1.])\n', 'call': 'rejects(lambda: partition_fragment_cloud(f,dust_fraction=1.,largest_fraction=.3,beta=.6,wind_scale=1.))', 'gold_call': 'rejects(lambda: _oracle_partition_fragment_cloud(f,dust_fraction=1.,largest_fraction=.3,beta=.6,wind_scale=1.))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\nf=np.array([11.,-7.,18000.,5300.,70.,-2100.,400.,7e5,1.])\n', 'call': 'rejects(lambda: partition_fragment_cloud(f,dust_fraction=np.nan,largest_fraction=.3,beta=.6,wind_scale=1.))', 'gold_call': 'rejects(lambda: _oracle_partition_fragment_cloud(f,dust_fraction=np.nan,largest_fraction=.3,beta=.6,wind_scale=1.))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\nf=np.array([11.,-7.,18000.,5300.,70.,-2100.,400.,7e5,1.])\n', 'call': 'rejects(lambda: partition_fragment_cloud(f,dust_fraction=.2,largest_fraction=0.,beta=.6,wind_scale=1.))', 'gold_call': 'rejects(lambda: _oracle_partition_fragment_cloud(f,dust_fraction=.2,largest_fraction=0.,beta=.6,wind_scale=1.))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\nf=np.array([11.,-7.,18000.,5300.,70.,-2100.,400.,7e5,1.])\n', 'call': 'rejects(lambda: partition_fragment_cloud(f,dust_fraction=.2,largest_fraction=1.,beta=.6,wind_scale=1.))', 'gold_call': 'rejects(lambda: _oracle_partition_fragment_cloud(f,dust_fraction=.2,largest_fraction=1.,beta=.6,wind_scale=1.))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\nf=np.array([11.,-7.,18000.,5300.,70.,-2100.,400.,7e5,1.])\n', 'call': 'rejects(lambda: partition_fragment_cloud(f,dust_fraction=.2,largest_fraction=np.inf,beta=.6,wind_scale=1.))', 'gold_call': 'rejects(lambda: _oracle_partition_fragment_cloud(f,dust_fraction=.2,largest_fraction=np.inf,beta=.6,wind_scale=1.))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\nf=np.array([11.,-7.,18000.,5300.,70.,-2100.,400.,7e5,1.])\n', 'call': 'rejects(lambda: partition_fragment_cloud(f,dust_fraction=.2,largest_fraction=.3,beta=0.,wind_scale=1.))', 'gold_call': 'rejects(lambda: _oracle_partition_fragment_cloud(f,dust_fraction=.2,largest_fraction=.3,beta=0.,wind_scale=1.))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\nf=np.array([11.,-7.,18000.,5300.,70.,-2100.,400.,7e5,1.])\n', 'call': 'rejects(lambda: partition_fragment_cloud(f,dust_fraction=.2,largest_fraction=.3,beta=1.,wind_scale=1.))', 'gold_call': 'rejects(lambda: _oracle_partition_fragment_cloud(f,dust_fraction=.2,largest_fraction=.3,beta=1.,wind_scale=1.))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\nf=np.array([11.,-7.,18000.,5300.,70.,-2100.,400.,7e5,1.])\n', 'call': 'rejects(lambda: partition_fragment_cloud(f,dust_fraction=.2,largest_fraction=.3,beta=-np.inf,wind_scale=1.))', 'gold_call': 'rejects(lambda: _oracle_partition_fragment_cloud(f,dust_fraction=.2,largest_fraction=.3,beta=-np.inf,wind_scale=1.))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\nf=np.array([11.,-7.,18000.,5300.,70.,-2100.,400.,7e5,1.])\n', 'call': 'rejects(lambda: partition_fragment_cloud(f,dust_fraction=.2,largest_fraction=.3,beta=.6,wind_scale=-1.))', 'gold_call': 'rejects(lambda: _oracle_partition_fragment_cloud(f,dust_fraction=.2,largest_fraction=.3,beta=.6,wind_scale=-1.))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\nf=np.array([11.,-7.,18000.,5300.,70.,-2100.,400.,7e5,1.])\n', 'call': 'rejects(lambda: partition_fragment_cloud(f,dust_fraction=.2,largest_fraction=.3,beta=.6,wind_scale=np.nan))', 'gold_call': 'rejects(lambda: _oracle_partition_fragment_cloud(f,dust_fraction=.2,largest_fraction=.3,beta=.6,wind_scale=np.nan))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\nf=np.array([11.,-7.,18000.,5300.,70.,-2100.,400.,7e5,1.])\n', 'call': 'rejects(lambda: partition_fragment_cloud(f,dust_fraction=.2,largest_fraction=.3,beta=.6,wind_scale=1.,alpha=-1.))', 'gold_call': 'rejects(lambda: _oracle_partition_fragment_cloud(f,dust_fraction=.2,largest_fraction=.3,beta=.6,wind_scale=1.,alpha=-1.))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\nf=np.array([11.,-7.,18000.,5300.,70.,-2100.,400.,7e5,1.])\n', 'call': 'rejects(lambda: partition_fragment_cloud(f,dust_fraction=.2,largest_fraction=.3,beta=.6,wind_scale=1.,alpha=np.inf))', 'gold_call': 'rejects(lambda: _oracle_partition_fragment_cloud(f,dust_fraction=.2,largest_fraction=.3,beta=.6,wind_scale=1.,alpha=np.inf))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\nf=np.array([11.,-7.,18000.,5300.,70.,-2100.,400.,7e5,1.])\n', 'call': 'rejects(lambda: partition_fragment_cloud(f,dust_fraction=.2,largest_fraction=.3,beta=.6,wind_scale=1.,strength_cap=0.))', 'gold_call': 'rejects(lambda: _oracle_partition_fragment_cloud(f,dust_fraction=.2,largest_fraction=.3,beta=.6,wind_scale=1.,strength_cap=0.))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\nf=np.array([11.,-7.,18000.,5300.,70.,-2100.,400.,7e5,1.])\n', 'call': 'rejects(lambda: partition_fragment_cloud(f,dust_fraction=.2,largest_fraction=.3,beta=.6,wind_scale=1.,strength_cap=np.nan))', 'gold_call': 'rejects(lambda: _oracle_partition_fragment_cloud(f,dust_fraction=.2,largest_fraction=.3,beta=.6,wind_scale=1.,strength_cap=np.nan))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\nf=np.array([11.,-7.,18000.,5300.,70.,-2100.,400.,7e5,1.])\n', 'call': 'rejects(lambda: partition_fragment_cloud(f,dust_fraction=.2,largest_fraction=.3,beta=.6,wind_scale=1.,minimum_mass=0.))', 'gold_call': 'rejects(lambda: _oracle_partition_fragment_cloud(f,dust_fraction=.2,largest_fraction=.3,beta=.6,wind_scale=1.,minimum_mass=0.))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\nf=np.array([11.,-7.,18000.,5300.,70.,-2100.,400.,7e5,1.])\n', 'call': 'rejects(lambda: partition_fragment_cloud(f,dust_fraction=.2,largest_fraction=.3,beta=.6,wind_scale=1.,minimum_mass=np.inf))', 'gold_call': 'rejects(lambda: _oracle_partition_fragment_cloud(f,dust_fraction=.2,largest_fraction=.3,beta=.6,wind_scale=1.,minimum_mass=np.inf))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\nf=np.array([11.,-7.,18000.,5300.,70.,-2100.,400.,7e5,1.])\n', 'call': 'rejects(lambda: partition_fragment_cloud(f,dust_fraction=.2,largest_fraction=.3,beta=.6,wind_scale=1.,kick_power=-1.))', 'gold_call': 'rejects(lambda: _oracle_partition_fragment_cloud(f,dust_fraction=.2,largest_fraction=.3,beta=.6,wind_scale=1.,kick_power=-1.))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\nf=np.array([11.,-7.,18000.,5300.,70.,-2100.,400.,7e5,1.])\n', 'call': 'rejects(lambda: partition_fragment_cloud(f,dust_fraction=.2,largest_fraction=.3,beta=.6,wind_scale=1.,kick_power=np.nan))', 'gold_call': 'rejects(lambda: _oracle_partition_fragment_cloud(f,dust_fraction=.2,largest_fraction=.3,beta=.6,wind_scale=1.,kick_power=np.nan))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\nf=np.array([11.,-7.,18000.,5300.,70.,-2100.,400.,7e5,1.])\n', 'call': 'rejects(lambda: partition_fragment_cloud(f,dust_fraction=.2,largest_fraction=.3,beta=.6,wind_scale=1.,momentum_correct=1))', 'gold_call': 'rejects(lambda: _oracle_partition_fragment_cloud(f,dust_fraction=.2,largest_fraction=.3,beta=.6,wind_scale=1.,momentum_correct=1))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\nf=np.array([11.,-7.,18000.,5300.,70.,-2100.,400.,7e5,1.])\n', 'call': 'rejects(lambda: partition_fragment_cloud(f,dust_fraction=.2,largest_fraction=.3,beta=.6,wind_scale=1.,rng=object()))', 'gold_call': 'rejects(lambda: _oracle_partition_fragment_cloud(f,dust_fraction=.2,largest_fraction=.3,beta=.6,wind_scale=1.,rng=object()))'}]
