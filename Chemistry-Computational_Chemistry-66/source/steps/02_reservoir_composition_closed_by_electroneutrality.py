"""
Return the reservoir volume fraction of every ionic species, with the last one fixed by electroneutrality, followed by the reservoir solvent fraction.

Far from any wall the electrolyte is a homogeneous reservoir. The solvent and the ions share a cubic lattice of cell volume a**3 and fill it completely. Species i carries the signed valency z_i and occupies v_i lattice cells per ion, where v_i is its molecular volume divided by a**3, so its volume fraction phi_i, the fraction of cells its ions occupy, corresponds to the number density c_i = phi_i / (a**3 v_i). The solvent takes every cell the ions leave.

The reservoir volume fractions of all species except the last are specified. The last species is present at whatever volume fraction makes the reservoir carry no net electric charge, and a reservoir that cannot be neutralised that way, or that would need more than the whole lattice, is not a valid input.

Returns
-------
np.ndarray of length N + 1: the N reservoir volume fractions, the last fixed by electroneutrality, then the reservoir solvent fraction
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def reservoir_composition(z: "np.ndarray", v: "np.ndarray", phi_free: "np.ndarray") -> "np.ndarray":
    '''Reservoir volume fractions of an electroneutral size-asymmetric electrolyte.

    Parameters
    ----------
    z : np.ndarray
        Real array of length N >= 2 holding the signed valency of each species.
    v : np.ndarray
        Real array of length N holding each species' molecular volume divided by the lattice
        cell volume; every entry strictly positive.
    phi_free : np.ndarray
        Real array of length N - 1 holding the reservoir volume fractions of the first N - 1
        species; every entry strictly positive.

    Returns
    -------
    result : np.ndarray
        Real array of length N + 1 holding the N reservoir volume fractions in the input species
        order, the last one fixed by electroneutrality, followed by the reservoir solvent fraction.

    Raises
    ------
    ValueError
        If the lengths are inconsistent, if any relative volume or specified fraction is not
        strictly positive, if the last species has zero valency, if no strictly positive fraction
        of the last species neutralises the reservoir, or if the ion fractions sum to one or more.
    '''
    return result  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_reservoir_composition(z: "np.ndarray", v: "np.ndarray", phi_free: "np.ndarray") -> "np.ndarray":
    zz = np.atleast_1d(np.asarray(z, dtype=float))
    vv = np.atleast_1d(np.asarray(v, dtype=float))
    free = np.atleast_1d(np.asarray(phi_free, dtype=float))
    n = zz.size
    if n < 2 or vv.size != n or free.size != n - 1:
        raise ValueError("z and v need N >= 2 entries and phi_free N - 1 entries")
    if np.any(vv <= 0.0) or np.any(free <= 0.0):
        raise ValueError("relative volumes and specified fractions must be strictly positive")
    if zz[-1] == 0.0:
        raise ValueError("the closing species must carry charge")
    # neutrality holds for number densities phi_i / (a**3 v_i), not for volume fractions
    charge = float((zz[:-1] * free / vv[:-1]).sum())
    last = -charge * vv[-1] / zz[-1]
    if last <= 0.0:
        raise ValueError("no positive fraction of the last species neutralises the reservoir")
    phi = np.concatenate([free, [last]])
    solvent = 1.0 - float(phi.sum())
    if solvent <= 0.0:
        raise ValueError("the ion volume fractions fill the whole lattice")
    return np.concatenate([phi, [solvent]])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # normal, the four-species electrolyte of the task closed by its monovalent co-ion
        {
            "setup": "import numpy as np\nZ = np.array([-1.0, -2.0, -3.0, 1.0])\nV = np.array([0.5, 2.0, 8.0, 1.0])\nF = np.array([2.0e-3, 1.0e-3, 5.0e-4])",
            "call": "reservoir_composition(Z, V, F)",
            "gold_call": "_oracle_reservoir_composition(Z, V, F)",
        },
        # normal, two cations of different size closed by a large anion
        {
            "setup": "import numpy as np\nZ = np.array([2.0, 1.0, -1.0])\nV = np.array([3.0, 0.4, 1.5])\nF = np.array([1.0e-3, 2.0e-3])",
            "call": "reservoir_composition(Z, V, F)",
            "gold_call": "_oracle_reservoir_composition(Z, V, F)",
        },
        # boundary, a symmetric solvent-sized salt where volume and number neutrality coincide
        {
            "setup": "import numpy as np\nZ = np.array([-1.0, 1.0])\nV = np.array([1.0, 1.0])\nF = np.array([0.01])",
            "call": "reservoir_composition(Z, V, F)",
            "gold_call": "_oracle_reservoir_composition(Z, V, F)",
        },
        # edge, a bulky tetravalent ion with a small counter-charge, where the size contrast is two hundredfold
        {
            "setup": "import numpy as np\nZ = np.array([-4.0, 1.0])\nV = np.array([40.0, 0.2])\nF = np.array([0.02])",
            "call": "reservoir_composition(Z, V, F)",
            "gold_call": "_oracle_reservoir_composition(Z, V, F)",
        },
        # contract, two anions cannot form a neutral reservoir and must raise ValueError
        {
            "setup": "import numpy as np\ndef _raises(fn):\n    try:\n        fn()\n    except ValueError:\n        return 1.0\n    return 0.0",
            "call": "_raises(lambda: reservoir_composition(np.array([-1.0, -2.0]), np.array([1.0, 1.0]), np.array([0.01])))",
            "gold_call": "_raises(lambda: _oracle_reservoir_composition(np.array([-1.0, -2.0]), np.array([1.0, 1.0]), np.array([0.01])))",
        },
        # contract, a reservoir that would need more than the whole lattice must raise ValueError
        {
            "setup": "import numpy as np\ndef _raises(fn):\n    try:\n        fn()\n    except ValueError:\n        return 1.0\n    return 0.0",
            "call": "_raises(lambda: reservoir_composition(np.array([-1.0, 1.0]), np.array([1.0, 1.0]), np.array([0.6])))",
            "gold_call": "_raises(lambda: _oracle_reservoir_composition(np.array([-1.0, 1.0]), np.array([1.0, 1.0]), np.array([0.6])))",
        },
    ]
