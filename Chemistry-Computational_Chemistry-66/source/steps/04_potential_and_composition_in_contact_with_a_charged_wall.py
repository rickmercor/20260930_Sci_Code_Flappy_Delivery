"""
Return the reduced electrostatic potential at a positively charged planar wall, together with the volume fractions of every ionic species and of the solvent in contact with it.

A planar wall carrying a uniform positive surface charge faces the size-asymmetric lattice electrolyte of step 03, which is homogeneous parallel to the wall, so every quantity depends only on the distance x from it. Far from the wall the field vanishes and the composition is the reservoir composition. At the wall Gauss's law fixes the field, and in reduced units the combination a**3 / (8 pi l_B) (dPsi/dx)**2, with Psi = e psi / (k_B T) and l_B the Bjerrum length, equals the reduced surface parameter zeta of step 01 there. Between the wall and the reservoir the electrolyte is in mechanical equilibrium: its thermodynamic pressure, made of the electrostatic stress of the field and the osmotic pressure of the lattice gas of ions and solvent, has the same value at every distance from the wall.

Return the potential at the wall, measured from the reservoir, and the composition there. The wall potential must be accurate to ten significant figures from an uncharged wall, where it is zero, up to reduced surface parameters of order one hundred.

Returns
-------
np.ndarray of length N + 2: the reduced wall potential, the N contact volume fractions, then the contact solvent fraction
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def wall_state(z: "np.ndarray", v: "np.ndarray", phi_bulk: "np.ndarray", zeta: float) -> "np.ndarray":
    '''Reduced wall potential and contact composition of a size-asymmetric electrolyte.

    Parameters
    ----------
    z : np.ndarray
        Real array of length N holding the signed valency of each species.
    v : np.ndarray
        Real array of length N holding each species' molecular volume divided by the cell
        volume; every entry strictly positive.
    phi_bulk : np.ndarray
        Real array of length N holding the reservoir volume fractions of an electroneutral
        reservoir; every entry strictly positive, with a sum strictly less than one, and at least
        one species of negative valency.
    zeta : float
        Reduced surface parameter of the positively charged wall; non-negative.

    Returns
    -------
    result : np.ndarray
        Real array of length N + 2 holding the reduced wall potential Psi_s = e psi(0) / (k_B T),
        which is non-negative, then the N volume fractions in contact with the wall in the input
        species order, then the solvent fraction in contact with the wall.

    Raises
    ------
    ValueError
        If zeta is negative.
    '''
    return result  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.optimize import brentq


def _squared_field(z, v, phi_bulk, Psi):
    """a**3 / (8 pi l_B) (dPsi/dx)**2 at local potential Psi, from the uniform pressure."""
    st = _oracle_local_composition(z, v, phi_bulk, Psi)
    d = float(st[-1])
    w = 1.0 - 1.0 / v
    return -d - float((w * phi_bulk * np.expm1(-z * float(Psi) + v * d)).sum())


def _oracle_wall_state(z: "np.ndarray", v: "np.ndarray", phi_bulk: "np.ndarray", zeta: float) -> "np.ndarray":
    zz = np.atleast_1d(np.asarray(z, dtype=float))
    vv = np.atleast_1d(np.asarray(v, dtype=float))
    pb = np.atleast_1d(np.asarray(phi_bulk, dtype=float))
    zeta = float(zeta)
    if zeta < 0.0:
        raise ValueError("zeta must be non-negative")
    eb = 1.0 - float(pb.sum())
    if zeta == 0.0:
        return np.concatenate([[0.0], pb, [eb]])
    hi = 1.0
    while _squared_field(zz, vv, pb, hi) < zeta:
        hi *= 1.6
    Ps = brentq(lambda P: _squared_field(zz, vv, pb, P) - zeta, 0.0, hi,
                xtol=1e-300, rtol=4.0 * np.finfo(float).eps, maxiter=1000)
    st = _oracle_local_composition(zz, vv, pb, Ps)
    return np.concatenate([[Ps], st[:-1], [eb * np.exp(st[-1])]])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    task = "import numpy as np\nZ = np.array([-1.0, -2.0, -3.0, 1.0])\nV = np.array([0.5, 2.0, 8.0, 1.0])\nPB = np.array([2.0e-3, 1.0e-3, 5.0e-4, 5.1875e-3])"
    return [
        # normal composition at the task's strongly charged wall
        {
            "setup": task,
            "call": "wall_state(Z, V, PB, 5.9)",
            "gold_call": "_oracle_wall_state(Z, V, PB, 5.9)",
            "tol": 1e-8,
        },
        # normal composition at a weakly charged wall
        {
            "setup": task,
            "call": "wall_state(Z, V, PB, 0.28)",
            "gold_call": "_oracle_wall_state(Z, V, PB, 0.28)",
            "tol": 1e-8,
        },
        # boundary, an uncharged wall has zero potential and the reservoir composition
        {
            "setup": task,
            "call": "wall_state(Z, V, PB, 0.0)",
            "gold_call": "_oracle_wall_state(Z, V, PB, 0.0)",
            "tol": 1e-12,
        },
        # edge composition at zeta=100; trace excluded fractions retain a sensible absolute floor
        {
            "setup": task,
            "call": "wall_state(Z, V, PB, 100.0)",
            "gold_call": "_oracle_wall_state(Z, V, PB, 100.0)",
            "tol": 1e-8,
        },
        # edge, nearly salt-free single large counterion
        {
            "setup": "import numpy as np\nZ = np.array([-1.0, 1.0])\nV = np.array([2.0, 1.0])\nPB = np.array([2.0e-9, 1.0e-9])",
            "call": "wall_state(Z, V, PB, 2.0)",
            "gold_call": "_oracle_wall_state(Z, V, PB, 2.0)",
            "tol": 1e-8,
        },
        # normal, three-species composition in a less polar hotter solvent
        {
            "setup": "import numpy as np\nZ = np.array([-2.0, 1.0, -1.0])\nV = np.array([6.0, 0.3, 0.8])\nPB = np.array([4.0e-3, 1.6e-3, 3.2e-3])",
            "call": "wall_state(Z, V, PB, 15.0)",
            "gold_call": "_oracle_wall_state(Z, V, PB, 15.0)",
            "tol": 1e-8,
        },
        # projected normal-potential check: ten significant figures without overconstraining trace fractions
        {
            "setup": task,
            "call": "float(wall_state(Z, V, PB, 5.9)[0])",
            "gold_call": "float(_oracle_wall_state(Z, V, PB, 5.9)[0])",
            "tol": 2e-10,
        },
        # projected weak-wall potential check
        {
            "setup": task,
            "call": "float(wall_state(Z, V, PB, 0.28)[0])",
            "gold_call": "float(_oracle_wall_state(Z, V, PB, 0.28)[0])",
            "tol": 2e-10,
        },
        # projected high-wall potential check spans the upper advertised range
        {
            "setup": task,
            "call": "float(wall_state(Z, V, PB, 100.0)[0])",
            "gold_call": "float(_oracle_wall_state(Z, V, PB, 100.0)[0])",
            "tol": 2e-10,
        },
        # contract, a negative surface parameter must raise ValueError
        {
            "setup": "import numpy as np\ndef _raises(fn):\n    try:\n        fn()\n    except ValueError:\n        return 1.0\n    return 0.0\nZ = np.array([-1.0, 1.0])\nV = np.array([1.0, 1.0])\nPB = np.array([0.005, 0.005])",
            "call": "_raises(lambda: wall_state(Z, V, PB, -1.0))",
            "gold_call": "_raises(lambda: _oracle_wall_state(Z, V, PB, -1.0))",
        },
    ]
