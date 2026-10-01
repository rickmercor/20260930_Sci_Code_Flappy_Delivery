"""
Run the background-aware Richardson-Lucy iteration on the signal run and keep every iterate.

The reference spectrum that seeds the empirical prior is a Poisson maximum-likelihood iterate. Each update multiplies the current emitted spectrum $\boldsymbol{\mu}^{(t)}$, bin by bin, by the back-projection through the response of the ratio between the observed signal-run counts $\mathbf{n}$ and the counts the current spectrum predicts. The prediction folds the emitted spectrum through the composite response $\mathbf{R}_\gamma$ and adds the fixed background reference $\mathbf{b}_{\mathrm{ref}}$, so the observed counts enter as they are and no background is ever subtracted from them. The back-projected ratio is divided by the sensitivity of the bin, $s_j = \sum_i R_{ij}$, the fraction of photons emitted in bin $j$ that is registered at all, which is the general form of the update and reduces to the plain multiplicative form only when the response preserves counts and every $s_j$ equals one. A small guard constant $\epsilon$ is added to every predicted count before the division so that an empty prediction cannot divide by zero.

The iteration starts from a flat emitted spectrum $\boldsymbol{\mu}^{(0)}$. Its level is the total of the signal run less the total of the background reference, spread evenly over the $J$ bins, which must be positive for the start to be defined. The iterates are kept from the start onwards, because the reference is selected among them afterwards by a rule that watches how they change.

Returns
-------
np.ndarray, shape (n_iterations + 1, J). Row t is the emitted spectrum after t updates, row 0 the flat start whose bins all equal the net count of the signal run divided by the number of bins.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def run_richardson_lucy_iterates(
    on_counts: np.ndarray,
    response: np.ndarray,
    background_reference: np.ndarray,
    n_iterations: int,
    guard: float,
) -> np.ndarray:
    r"""Iterate the background-aware update from the flat start and keep every iterate.

    Parameters
    ----------
    on_counts : np.ndarray
        Counts of the signal run, shape (J,), non-negative integer values.
    response : np.ndarray
        Composite detector response, shape (J, J), non-negative and finite,
        entry (i, j) mapping emitted bin j to registered bin i, every column
        summing to the detection efficiency of that bin, above zero and at
        most one to within 1e-6. The column sums are the sensitivities that
        divide the update.
    background_reference : np.ndarray
        Fixed background expectation of every bin, shape (J,), non-negative
        and finite, in counts.
    n_iterations : int
        Number of updates to apply, zero or more. Zero returns the flat
        start alone.
    guard : float
        Non-negative constant added to every predicted count before the
        observed counts are divided by it.

    Returns
    -------
    iterates : np.ndarray
        Shape (n_iterations + 1, J). Row t is the emitted spectrum after t
        updates, row 0 the flat start whose bins all equal the net count of
        the signal run divided by the number of bins.

    Raises
    ------
    ValueError
        If the arrays are not of the shapes above, are empty or disagree on
        the number of bins, if an entry is negative or not finite, if a count
        is not an integer value, if a column of the response sums to zero or
        exceeds one by more than 1e-6, if ``n_iterations`` is negative or not
        an integer, if ``guard`` is negative or not finite, or if the net count
        of the signal run, its total less the total background reference, is
        not positive.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_run_richardson_lucy_iterates(
    on_counts: np.ndarray,
    response: np.ndarray,
    background_reference: np.ndarray,
    n_iterations: int,
    guard: float,
) -> np.ndarray:
    n_on = np.asarray(on_counts, dtype=float)
    r = np.asarray(response, dtype=float)
    b_ref = np.asarray(background_reference, dtype=float)

    if n_on.ndim != 1 or n_on.size == 0:
        raise ValueError("on_counts must be a non-empty 1D array")
    n_bins = n_on.size
    if r.ndim != 2 or r.shape != (n_bins, n_bins):
        raise ValueError("response must have shape (J, J) matching on_counts")
    if b_ref.ndim != 1 or b_ref.size != n_bins:
        raise ValueError("background_reference must have shape (J,) matching on_counts")
    for name, arr in (("on_counts", n_on), ("response", r), ("background_reference", b_ref)):
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} must be finite")
        if np.any(arr < 0.0):
            raise ValueError(f"{name} must be non-negative")
    if np.any(n_on != np.round(n_on)):
        raise ValueError("on_counts must hold integer values")
    sensitivity = r.sum(axis=0)
    if np.any(sensitivity <= 0.0) or np.any(sensitivity > 1.0 + 1e-6):
        raise ValueError("every column of response must sum to an efficiency in (0, 1]")
    if int(n_iterations) != n_iterations or n_iterations < 0:
        raise ValueError("n_iterations must be a non-negative integer")
    eps = float(guard)
    if not np.isfinite(eps) or eps < 0.0:
        raise ValueError("guard must be a finite non-negative number")

    net = float(n_on.sum() - b_ref.sum())
    if net <= 0.0:
        raise ValueError("the signal run must hold more counts than the background reference")

    n_it = int(n_iterations)
    iterates = np.empty((n_it + 1, n_bins), dtype=float)
    estimate = np.full(n_bins, net / n_bins, dtype=float)
    iterates[0] = estimate
    for t in range(1, n_it + 1):
        predicted = r @ estimate + b_ref
        estimate = estimate / sensitivity * (r.T @ (n_on / (predicted + eps)))
        iterates[t] = estimate
    return iterates

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    def _benchmark_setup():
        return """import numpy as np
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
response = resolution @ redistribution
on_counts = np.array([381, 1541, 460, 157, 293, 39, 0, 0])
off_counts = np.array([48, 154, 51, 22, 51, 15, 1, 0])
background_reference = (1.0 + off_counts) / (1.0 / off_counts.mean() + 1.0)
"""

    err = ""
    for name, function in (("run_model", "run_richardson_lucy_iterates"),
                           ("run_gold", "_oracle_run_richardson_lucy_iterates")):
        err += (f"def {name}():\n    try:\n        {function}(*args)\n"
                "        return 0\n    except ValueError:\n        return 1\n")
    cases = [
        {
            "setup": _benchmark_setup() + """n_iterations = 15
guard = 1e-12
""",
            "call": "run_richardson_lucy_iterates(on_counts, response, background_reference, n_iterations, guard)",
            "gold_call": "_oracle_run_richardson_lucy_iterates(on_counts, response, background_reference, n_iterations, guard)",
        },
        # many updates, where the empty tail bins decay through many orders of magnitude
        {
            "setup": _benchmark_setup() + """n_iterations = 500
guard = 1e-12
""",
            "call": "run_richardson_lucy_iterates(on_counts, response, background_reference, n_iterations, guard)",
            "gold_call": "_oracle_run_richardson_lucy_iterates(on_counts, response, background_reference, n_iterations, guard)",
        },
        # a small grid with a non-diagonal response and a background that varies across bins
        {
            "setup": """import numpy as np
response = np.array([
    [0.90, 0.20, 0.10],
    [0.10, 0.70, 0.30],
    [0.00, 0.10, 0.60],
])
on_counts = np.array([120, 340, 60])
background_reference = np.array([15.0, 22.5, 8.0])
n_iterations = 25
guard = 1e-12
""",
            "call": "run_richardson_lucy_iterates(on_counts, response, background_reference, n_iterations, guard)",
            "gold_call": "_oracle_run_richardson_lucy_iterates(on_counts, response, background_reference, n_iterations, guard)",
        },
        # boundary: zero updates return the flat start alone
        {
            "setup": """import numpy as np
response = np.array([
    [0.90, 0.20, 0.10],
    [0.10, 0.70, 0.30],
    [0.00, 0.10, 0.60],
])
on_counts = np.array([120, 340, 60])
background_reference = np.array([15.0, 22.5, 8.0])
n_iterations = 0
guard = 1e-12
""",
            "call": "run_richardson_lucy_iterates(on_counts, response, background_reference, n_iterations, guard)",
            "gold_call": "_oracle_run_richardson_lucy_iterates(on_counts, response, background_reference, n_iterations, guard)",
        },
        # boundary: a single update, which exposes the flat start and the first back-projection
        {
            "setup": """import numpy as np
response = np.array([
    [0.90, 0.20, 0.10],
    [0.10, 0.70, 0.30],
    [0.00, 0.10, 0.60],
])
on_counts = np.array([120, 340, 60])
background_reference = np.array([15.0, 22.5, 8.0])
n_iterations = 1
guard = 1e-12
""",
            "call": "run_richardson_lucy_iterates(on_counts, response, background_reference, n_iterations, guard)",
            "gold_call": "_oracle_run_richardson_lucy_iterates(on_counts, response, background_reference, n_iterations, guard)",
        },
        # boundary: no background at all, so the update is the plain Richardson-Lucy step
        {
            "setup": """import numpy as np
response = np.array([
    [0.80, 0.15, 0.05],
    [0.20, 0.70, 0.25],
    [0.00, 0.15, 0.70],
])
on_counts = np.array([50, 200, 90])
background_reference = np.zeros(3)
n_iterations = 40
guard = 1e-12
""",
            "call": "run_richardson_lucy_iterates(on_counts, response, background_reference, n_iterations, guard)",
            "gold_call": "_oracle_run_richardson_lucy_iterates(on_counts, response, background_reference, n_iterations, guard)",
        },
        # boundary: an identity response, where each bin evolves on its own toward its net count
        {
            "setup": """import numpy as np
response = np.eye(4)
on_counts = np.array([30, 0, 75, 12])
background_reference = np.array([4.0, 3.0, 5.0, 2.0])
n_iterations = 30
guard = 1e-12
""",
            "call": "run_richardson_lucy_iterates(on_counts, response, background_reference, n_iterations, guard)",
            "gold_call": "_oracle_run_richardson_lucy_iterates(on_counts, response, background_reference, n_iterations, guard)",
        },
        # edge: an empty signal bin with no background and no response overlap, where the bin
        # is driven to zero in one update and the guard keeps every later division defined
        {
            "setup": """import numpy as np
response = np.eye(3)
on_counts = np.array([80, 0, 40])
background_reference = np.array([2.0, 0.0, 1.0])
n_iterations = 3
guard = 1e-12
""",
            "call": "run_richardson_lucy_iterates(on_counts, response, background_reference, n_iterations, guard)",
            "gold_call": "_oracle_run_richardson_lucy_iterates(on_counts, response, background_reference, n_iterations, guard)",
        },
        # edge: a guard large enough to matter, which damps every ratio measurably
        {
            "setup": """import numpy as np
response = np.array([
    [0.90, 0.20, 0.10],
    [0.10, 0.70, 0.30],
    [0.00, 0.10, 0.60],
])
on_counts = np.array([12, 34, 6])
background_reference = np.array([1.5, 2.5, 0.8])
n_iterations = 10
guard = 0.5
""",
            "call": "run_richardson_lucy_iterates(on_counts, response, background_reference, n_iterations, guard)",
            "gold_call": "_oracle_run_richardson_lucy_iterates(on_counts, response, background_reference, n_iterations, guard)",
        },
        # edge: a single bin, where the flat start already equals the net count and every update leaves it unchanged
        {
            "setup": """import numpy as np
response = np.array([[1.0]])
on_counts = np.array([57])
background_reference = np.array([7.0])
n_iterations = 4
guard = 1e-12
""",
            "call": "run_richardson_lucy_iterates(on_counts, response, background_reference, n_iterations, guard)",
            "gold_call": "_oracle_run_richardson_lucy_iterates(on_counts, response, background_reference, n_iterations, guard)",
        },
        # edge: a response whose columns sum to one only to within the 1e-6 tolerance, where the
        # sensitivities still divide the update even though they differ from one in the seventh decimal
        {
            "setup": """import numpy as np
response = np.array([
    [0.8999996, 0.2000000, 0.1000000],
    [0.1000000, 0.6999995, 0.3000000],
    [0.0000000, 0.1000000, 0.5999996],
])
on_counts = np.array([120, 340, 60])
background_reference = np.array([15.0, 22.5, 8.0])
n_iterations = 25
guard = 1e-12
""",
            "call": "run_richardson_lucy_iterates(on_counts, response, background_reference, n_iterations, guard)",
            "gold_call": "_oracle_run_richardson_lucy_iterates(on_counts, response, background_reference, n_iterations, guard)",
        },
        # a response with detection efficiencies below one, where the sensitivities that divide
        # the update differ from bin to bin
        {
            "setup": """import numpy as np
response = np.array([
    [0.81, 0.16, 0.07],
    [0.09, 0.56, 0.21],
    [0.00, 0.08, 0.42],
])
on_counts = np.array([120, 340, 60])
background_reference = np.array([15.0, 22.5, 8.0])
n_iterations = 25
guard = 1e-12
""",
            "call": "run_richardson_lucy_iterates(on_counts, response, background_reference, n_iterations, guard)",
            "gold_call": "_oracle_run_richardson_lucy_iterates(on_counts, response, background_reference, n_iterations, guard)",
        },
        # boundary: a uniform efficiency, where the sensitivity is one common factor in every bin
        {
            "setup": """import numpy as np
response = 0.5 * np.array([
    [0.90, 0.20, 0.10],
    [0.10, 0.70, 0.30],
    [0.00, 0.10, 0.60],
])
on_counts = np.array([120, 340, 60])
background_reference = np.array([15.0, 22.5, 8.0])
n_iterations = 25
guard = 1e-12
""",
            "call": "run_richardson_lucy_iterates(on_counts, response, background_reference, n_iterations, guard)",
            "gold_call": "_oracle_run_richardson_lucy_iterates(on_counts, response, background_reference, n_iterations, guard)",
        },
        # edge: a background reference that exceeds the signal run in one bin, so that bin decays
        {
            "setup": """import numpy as np
response = np.array([
    [0.95, 0.05, 0.00],
    [0.05, 0.90, 0.05],
    [0.00, 0.05, 0.95],
])
on_counts = np.array([300, 4, 150])
background_reference = np.array([20.0, 9.0, 12.0])
n_iterations = 60
guard = 1e-12
""",
            "call": "run_richardson_lucy_iterates(on_counts, response, background_reference, n_iterations, guard)",
            "gold_call": "_oracle_run_richardson_lucy_iterates(on_counts, response, background_reference, n_iterations, guard)",
        },
        {
            "setup": """import numpy as np
# invalid: a background reference whose total is not below the signal run, so the flat start is not positive
response = np.array([
    [0.90, 0.20, 0.10],
    [0.10, 0.70, 0.30],
    [0.00, 0.10, 0.60],
])
on_counts = np.array([12, 34, 6])
background_reference = np.array([15.0, 30.0, 7.0])
args = (on_counts, response, background_reference, 10, 1e-12)
""" + err,
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
# invalid: a number of updates that is not an integer
response = np.array([
    [0.90, 0.20, 0.10],
    [0.10, 0.70, 0.30],
    [0.00, 0.10, 0.60],
])
on_counts = np.array([120, 340, 60])
background_reference = np.array([15.0, 22.5, 8.0])
args = (on_counts, response, background_reference, 2.5, 1e-12)
""" + err,
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
# invalid: a response column that exceeds one by well over the 1e-6 tolerance
response = np.array([
    [0.90, 0.20, 0.10],
    [0.10, 0.70, 0.30],
    [0.00, 0.11, 0.60],
])
on_counts = np.array([120, 340, 60])
background_reference = np.array([15.0, 22.5, 8.0])
args = (on_counts, response, background_reference, 25, 1e-12)
""" + err,
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
# invalid: a signal count that is not an integer value
response = np.array([
    [0.90, 0.20, 0.10],
    [0.10, 0.70, 0.30],
    [0.00, 0.10, 0.60],
])
on_counts = np.array([120.0, 340.4, 60.0])
background_reference = np.array([15.0, 22.5, 8.0])
args = (on_counts, response, background_reference, 25, 1e-12)
""" + err,
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # boundary: a response column 5e-7 above one, inside the stated tolerance, whose sensitivity still divides the update
        {
            "setup": """import numpy as np
response = np.array([
    [0.90, 0.20, 0.10],
    [0.10, 0.70, 0.30],
    [0.00, 0.1000005, 0.60],
])
on_counts = np.array([120, 340, 60])
background_reference = np.array([15.0, 22.5, 8.0])
n_iterations = 25
guard = 1e-12
""",
            "call": "run_richardson_lucy_iterates(on_counts, response, background_reference, n_iterations, guard)",
            "gold_call": "_oracle_run_richardson_lucy_iterates(on_counts, response, background_reference, n_iterations, guard)",
        },
        {
            "setup": """import numpy as np
# invalid: a response column 5e-6 above one, outside the stated 1e-6 tolerance
response = np.array([
    [0.90, 0.20, 0.10],
    [0.10, 0.70, 0.30],
    [0.00, 0.100005, 0.60],
])
on_counts = np.array([120, 340, 60])
background_reference = np.array([15.0, 22.5, 8.0])
args = (on_counts, response, background_reference, 25, 1e-12)
""" + err,
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # signal counts supplied as a float array of whole numbers, which holds integer values
        {
            "setup": """import numpy as np
response = np.array([
    [0.90, 0.20, 0.10],
    [0.10, 0.70, 0.30],
    [0.00, 0.10, 0.60],
])
on_counts = np.array([120.0, 340.0, 60.0])
background_reference = np.array([15.0, 22.5, 8.0])
n_iterations = 25
guard = 1e-12
""",
            "call": "run_richardson_lucy_iterates(on_counts, response, background_reference, n_iterations, guard)",
            "gold_call": "_oracle_run_richardson_lucy_iterates(on_counts, response, background_reference, n_iterations, guard)",
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
        "run_richardson_lucy_iterates = _independent_inputs(run_richardson_lucy_iterates)\n"
        "_oracle_run_richardson_lucy_iterates = _independent_inputs(_oracle_run_richardson_lucy_iterates)\n"
    )
    for case in cases:
        case["setup"] += isolate
    return cases
