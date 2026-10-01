"""
Compose the detector response of the scintillator array from its redistribution and resolution operators.

A photon emitted into $\gamma$-energy bin $j$ is not simply registered in bin $j$. Part of its energy escapes the crystal through Compton scattering and pair-production escape, so the deposited energy is redistributed over lower bins, and the deposited energy is then smeared by the finite resolution of the detector. Both effects are linear on the emitted spectrum and are encoded as square operators on the $\gamma$-energy grid. The redistribution operator $\mathbf{D}$ carries in entry $(i, j)$ the probability that a photon emitted in bin $j$ deposits its energy in bin $i$. A photon that traverses the crystal without depositing energy is never registered, so column $j$ of $\mathbf{D}$ sums to the detection efficiency $\varepsilon_j \le 1$ of that bin. The resolution operator $\mathbf{G}_\gamma$ carries in entry $(i, j)$ the probability that an energy deposited in bin $j$ is registered in bin $i$, and it preserves the expected number of counts, so every one of its columns sums to one.

The two operators do not commute. The composite response $\mathbf{R}_\gamma$ that maps the emitted spectrum $\boldsymbol{\mu}$ to the expected detected signal $\boldsymbol{\nu} = \mathbf{R}_\gamma \boldsymbol{\mu}$ has to apply them in the order in which the physical processes happen, and its column sums are the efficiencies $\varepsilon_j$. The same resolution operator applied on its own maps the emitted spectrum to the resolution-limited spectrum $\boldsymbol{\eta} = \mathbf{G}_\gamma \boldsymbol{\mu}$ in which unfolded results are reported.

Returns
-------
np.ndarray, shape (J, J). The composite response applied to an emitted spectrum gives the expected detected signal, and its column sums are the detection efficiencies. The product is returned as it is and is not renormalised.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compose_detector_response(
    redistribution: np.ndarray,
    resolution: np.ndarray,
) -> np.ndarray:
    r"""Compose the redistribution and resolution operators into the detector response.

    Parameters
    ----------
    redistribution : np.ndarray
        Energy-redistribution operator, shape (J, J), non-negative and finite,
        entry (i, j) being the probability that a photon emitted in bin j
        deposits its energy in bin i. Every column sums to the detection
        efficiency of that bin, a value above zero and at most one to within
        1e-6.
    resolution : np.ndarray
        Resolution-broadening operator, shape (J, J), non-negative and
        finite, entry (i, j) being the probability that an energy deposited
        in bin j is registered in bin i. Every column sums to one to within
        1e-6.

    Returns
    -------
    response : np.ndarray
        Shape (J, J). The composite response applied to an emitted spectrum
        gives the expected detected signal, and its column sums are the
        detection efficiencies. The product is returned as it is and is not
        renormalised.

    Raises
    ------
    ValueError
        If either operator is not a square two-dimensional array, if the two
        differ in shape, if the grid is empty, if an entry is negative or not
        finite, if a column of the redistribution sums to zero or exceeds one
        by more than 1e-6, or if a column of the resolution does not sum to
        one to within 1e-6.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_compose_detector_response(
    redistribution: np.ndarray,
    resolution: np.ndarray,
) -> np.ndarray:
    d = np.asarray(redistribution, dtype=float)
    g = np.asarray(resolution, dtype=float)

    for name, m in (("redistribution", d), ("resolution", g)):
        if m.ndim != 2 or m.shape[0] != m.shape[1]:
            raise ValueError(f"{name} must be a square 2D array")
        if m.shape[0] == 0:
            raise ValueError(f"{name} must act on at least one bin")
        if not np.all(np.isfinite(m)):
            raise ValueError(f"{name} must be finite")
        if np.any(m < 0.0):
            raise ValueError(f"{name} must be non-negative")
    if np.any(d.sum(axis=0) <= 0.0) or np.any(d.sum(axis=0) > 1.0 + 1e-6):
        raise ValueError("every column of redistribution must sum to an efficiency in (0, 1]")
    if not np.allclose(g.sum(axis=0), 1.0, rtol=0.0, atol=1e-6):
        raise ValueError("every column of resolution must sum to one")
    if d.shape != g.shape:
        raise ValueError("redistribution and resolution must share one grid")

    # Deposition happens before the deposited energy is broadened, so the
    # redistribution acts first and the resolution acts on its output.
    return g @ d

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    err = ""
    for name, function in (("run_model", "compose_detector_response"),
                           ("run_gold", "_oracle_compose_detector_response")):
        err += (f"def {name}():\n    try:\n        {function}(*args)\n"
                "        return 0\n    except ValueError:\n        return 1\n")
    cases = [
        # the benchmark operators on an eight-bin grid, the redistribution columns summing to the efficiencies
        {
            "setup": """import numpy as np
redistribution = np.array([
    [0.9600, 0.0920, 0.0634, 0.0528, 0.0466, 0.0421, 0.0386, 0.0356],
    [0.0000, 0.8280, 0.0686, 0.0506, 0.0412, 0.0352, 0.0309, 0.0275],
    [0.0000, 0.0000, 0.7480, 0.0646, 0.0511, 0.0427, 0.0368, 0.0323],
    [0.0000, 0.0000, 0.0000, 0.6720, 0.0611, 0.0502, 0.0427, 0.0370],
    [0.0000, 0.0000, 0.0000, 0.0000, 0.6000, 0.0577, 0.0486, 0.0418],
    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.5321, 0.0545, 0.0466],
    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.4679, 0.0513],
    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.4079],
])
resolution = np.array([
    [0.9220, 0.0724, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
    [0.0780, 0.8552, 0.0724, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
    [0.0000, 0.0724, 0.8552, 0.0724, 0.0000, 0.0000, 0.0000, 0.0000],
    [0.0000, 0.0000, 0.0724, 0.8552, 0.0724, 0.0000, 0.0000, 0.0000],
    [0.0000, 0.0000, 0.0000, 0.0724, 0.8552, 0.0724, 0.0000, 0.0000],
    [0.0000, 0.0000, 0.0000, 0.0000, 0.0724, 0.8552, 0.0724, 0.0000],
    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0724, 0.8552, 0.0780],
    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0724, 0.9220],
])
""",
            "call": "compose_detector_response(redistribution, resolution)",
            "gold_call": "_oracle_compose_detector_response(redistribution, resolution)",
        },
        # a three-bin grid on which the two operators visibly fail to commute
        {
            "setup": """import numpy as np
redistribution = np.array([
    [1.0, 0.3, 0.2],
    [0.0, 0.7, 0.3],
    [0.0, 0.0, 0.5],
])
resolution = np.array([
    [0.8, 0.1, 0.0],
    [0.2, 0.8, 0.1],
    [0.0, 0.1, 0.9],
])
""",
            "call": "compose_detector_response(redistribution, resolution)",
            "gold_call": "_oracle_compose_detector_response(redistribution, resolution)",
        },
        # boundary: an ideal detector on both counts leaves the emitted spectrum untouched
        {
            "setup": """import numpy as np
redistribution = np.eye(4)
resolution = np.eye(4)
""",
            "call": "compose_detector_response(redistribution, resolution)",
            "gold_call": "_oracle_compose_detector_response(redistribution, resolution)",
        },
        # boundary: a single bin, where both operators are the number one
        {
            "setup": """import numpy as np
redistribution = np.array([[1.0]])
resolution = np.array([[1.0]])
""",
            "call": "compose_detector_response(redistribution, resolution)",
            "gold_call": "_oracle_compose_detector_response(redistribution, resolution)",
        },
        # edge: perfect resolution, so the response is the redistribution alone
        {
            "setup": """import numpy as np
redistribution = np.array([
    [1.0, 0.25, 0.10, 0.05],
    [0.0, 0.75, 0.15, 0.10],
    [0.0, 0.00, 0.75, 0.20],
    [0.0, 0.00, 0.00, 0.65],
])
resolution = np.eye(4)
""",
            "call": "compose_detector_response(redistribution, resolution)",
            "gold_call": "_oracle_compose_detector_response(redistribution, resolution)",
        },
        # edge: no redistribution at all, so the response is the resolution alone
        {
            "setup": """import numpy as np
redistribution = np.eye(5)
resolution = np.array([
    [0.90, 0.05, 0.00, 0.00, 0.00],
    [0.10, 0.90, 0.05, 0.00, 0.00],
    [0.00, 0.05, 0.90, 0.05, 0.00],
    [0.00, 0.00, 0.05, 0.90, 0.10],
    [0.00, 0.00, 0.00, 0.05, 0.90],
])
""",
            "call": "compose_detector_response(redistribution, resolution)",
            "gold_call": "_oracle_compose_detector_response(redistribution, resolution)",
        },
        # edge: a broad resolution that reaches two bins either side, with rounding at the
        # fourth decimal that still sums to one in every column
        {
            "setup": """import numpy as np
redistribution = np.array([
    [1.0000, 0.1500, 0.1000, 0.0800, 0.0700, 0.0600],
    [0.0000, 0.8500, 0.1200, 0.0900, 0.0800, 0.0700],
    [0.0000, 0.0000, 0.7800, 0.1300, 0.1000, 0.0900],
    [0.0000, 0.0000, 0.0000, 0.7000, 0.1300, 0.1100],
    [0.0000, 0.0000, 0.0000, 0.0000, 0.6200, 0.1400],
    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.5300],
])
resolution = np.array([
    [0.7000, 0.1500, 0.0200, 0.0000, 0.0000, 0.0000],
    [0.2500, 0.6600, 0.1500, 0.0200, 0.0000, 0.0000],
    [0.0500, 0.1500, 0.6600, 0.1500, 0.0200, 0.0000],
    [0.0000, 0.0200, 0.1500, 0.6600, 0.1500, 0.0500],
    [0.0000, 0.0200, 0.0200, 0.1500, 0.6600, 0.2500],
    [0.0000, 0.0000, 0.0000, 0.0200, 0.1700, 0.7000],
])
""",
            "call": "compose_detector_response(redistribution, resolution)",
            "gold_call": "_oracle_compose_detector_response(redistribution, resolution)",
        },
        # edge: operators rounded so that their columns sum to one only to within the 1e-6
        # tolerance, where the response is the plain product and is not renormalised
        {
            "setup": """import numpy as np
redistribution = np.array([
    [1.0000000, 0.3000000, 0.2000000],
    [0.0000000, 0.6999995, 0.2999996],
    [0.0000000, 0.0000000, 0.4999998],
])
resolution = np.array([
    [0.8000003, 0.1000000, 0.0000000],
    [0.1999995, 0.7999996, 0.1000004],
    [0.0000000, 0.1000000, 0.8999992],
])
""",
            "call": "compose_detector_response(redistribution, resolution)",
            "gold_call": "_oracle_compose_detector_response(redistribution, resolution)",
        },
        # edge: a two-bin grid in which all redistributed energy lands in the lower bin
        {
            "setup": """import numpy as np
redistribution = np.array([
    [1.0, 0.4],
    [0.0, 0.6],
])
resolution = np.array([
    [0.85, 0.15],
    [0.15, 0.85],
])
""",
            "call": "compose_detector_response(redistribution, resolution)",
            "gold_call": "_oracle_compose_detector_response(redistribution, resolution)",
        },
        {
            "setup": """import numpy as np
# invalid: a redistribution column that sums to more than one by well over the 1e-6 tolerance
redistribution = np.array([
    [0.900, 0.100, 0.050],
    [0.000, 0.801, 0.050],
    [0.000, 0.100, 0.700],
])
resolution = np.array([
    [0.9, 0.1, 0.0],
    [0.1, 0.8, 0.1],
    [0.0, 0.1, 0.9],
])
args = (redistribution, resolution)
""" + err,
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
# invalid: a resolution column that falls short of one by more than the 1e-6 tolerance
redistribution = np.array([
    [0.90, 0.10, 0.05],
    [0.00, 0.80, 0.05],
    [0.00, 0.10, 0.70],
])
resolution = np.array([
    [0.9, 0.1, 0.0],
    [0.1, 0.8, 0.1],
    [0.0, 0.1, 0.899],
])
args = (redistribution, resolution)
""" + err,
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
# invalid: an emitted bin that is never detected, so its redistribution column sums to zero
redistribution = np.array([
    [0.90, 0.10, 0.0],
    [0.00, 0.80, 0.0],
    [0.00, 0.10, 0.0],
])
resolution = np.eye(3)
args = (redistribution, resolution)
""" + err,
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
# invalid: a negative entry in the resolution operator, which is not a probability
redistribution = np.eye(3)
resolution = np.array([
    [1.1, 0.1, 0.0],
    [-0.1, 0.8, 0.1],
    [0.0, 0.1, 0.9],
])
args = (redistribution, resolution)
""" + err,
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # boundary: a redistribution column 5e-7 above one, inside the stated tolerance, so the product is returned
        {
            "setup": """import numpy as np
redistribution = np.array([
    [0.90, 0.10, 0.05],
    [0.00, 0.80, 0.05],
    [0.00, 0.1000005, 0.70],
])
resolution = np.array([
    [0.85, 0.10, 0.00],
    [0.15, 0.80, 0.15],
    [0.00, 0.10, 0.85],
])
""",
            "call": "compose_detector_response(redistribution, resolution)",
            "gold_call": "_oracle_compose_detector_response(redistribution, resolution)",
        },
        {
            "setup": """import numpy as np
# invalid: a resolution column 5e-6 above one, outside the stated 1e-6 tolerance
redistribution = np.eye(3)
resolution = np.array([
    [0.85, 0.10, 0.00],
    [0.15, 0.800005, 0.15],
    [0.00, 0.10, 0.85],
])
args = (redistribution, resolution)
""" + err,
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # boundary: a resolution column 5e-7 above one, inside the stated tolerance
        {
            "setup": """import numpy as np
redistribution = np.eye(3)
resolution = np.array([
    [0.85, 0.10, 0.00],
    [0.15, 0.8000005, 0.15],
    [0.00, 0.10, 0.85],
])
""",
            "call": "compose_detector_response(redistribution, resolution)",
            "gold_call": "_oracle_compose_detector_response(redistribution, resolution)",
        },
    ]
    # Both sides receive their own deep copy of every argument, so an in-place
    # implementation on either side cannot contaminate the other's inputs.
    isolate = (
        "\nimport copy as _test_copy\n\n"
        "def _independent_inputs(_function):\n"
        "    def _invoke(*_args, **_kwargs):\n"
        "        _args, _kwargs = _test_copy.deepcopy((_args, _kwargs))\n"
        "        return _function(*_args, **_kwargs)\n"
        "    return _invoke\n\n"
        "compose_detector_response = _independent_inputs(compose_detector_response)\n"
        "_oracle_compose_detector_response = _independent_inputs(_oracle_compose_detector_response)\n"
    )
    for case in cases:
        case["setup"] += isolate
    return cases
