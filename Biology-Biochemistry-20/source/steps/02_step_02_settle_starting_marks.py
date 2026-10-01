"""
Sort the starting chromatin marks of a gene class into the basins of its slow flow and return the constant regulatory input each occupied attractor carries together with the fraction of the class that settles there.

A mark relaxes monotonically towards the terminal equilibrium on its own side of the repelling watersheds. At a saddle-node fold the marginal equilibrium remains attracting from its outer side. Once settled, a gene contributes only the constant part of its regulatory input, the settled mark plus the external input, so a bistable or marginal class can enter the collective dynamics as more than one population.

Returns
-------
np.ndarray: float array of shape (J, 2), rows [constant regulatory input, population share].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def settle_starting_marks(starting: "np.ndarray", equilibria: "np.ndarray", c: float) -> "np.ndarray":
    """Return the occupied constant inputs of a class and their population shares.

    ``equilibria`` lists rows ``[theta, rate]`` ordered by increasing ``theta``.
    Away from a saddle-node fold, attractors (rate below ``-1e-12``) and
    repelling watersheds (rate above ``1e-12``) alternate, beginning and ending
    with an attractor. The basin of an attractor is the open interval between
    its flanking watersheds, unbounded beyond the outermost watersheds.

    A rate whose magnitude is at most ``1e-12`` is marginal. The two additional
    admissible tables produced at bifurcations are a single marginal equilibrium
    (the critical monostable state), or two equilibria consisting of one
    attractor and one marginal fold. At a two-equilibrium fold, the marginal
    point attracts from the unbounded side: marks at or left of a left marginal
    point settle there, and marks at or right of a right marginal point settle
    there. Marks on the side between the fold and the ordinary attractor settle
    at that attractor. A mark exactly on a positive-rate watershed remains
    invalid.

    Row ``k`` of the result holds ``[input_k, share_k]``, where ``input_k`` is a
    terminal equilibrium's mark plus ``c`` and ``share_k`` is the fraction of
    starting marks that settle there. Unoccupied terminal equilibria are
    omitted, rows are ordered by increasing input, and shares sum to one.

    Parameters
    ----------
    starting : np.ndarray
        Non-empty one-dimensional array of finite starting marks.
    equilibria : np.ndarray
        Finite ``(M, 2)`` array in one of the equilibrium patterns described
        above, with rows ordered by strictly increasing mark.
    c : float
        Finite external input shared by the whole class.

    Returns
    -------
    np.ndarray
        Float array of shape ``(J, 2)`` with one row per occupied terminal
        equilibrium.

    Raises
    ------
    ValueError
        If ``starting`` is not a non-empty one-dimensional array of finite
        numbers, if ``c`` is not finite, if ``equilibria`` is not a finite
        ``(M, 2)`` array with strictly increasing marks and one of the stated
        sign patterns, or if a starting mark coincides with a positive-rate
        watershed.
    """
    return table

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_settle_starting_marks(starting: "np.ndarray", equilibria: "np.ndarray",
                                  c: float) -> "np.ndarray":
    """Reference implementation (watershed search and basin counting)."""
    import numpy as np

    marks = _epi_require_vector(starting, "starting")
    c = _epi_require_scalar(c, "c", -np.inf, False)
    try:
        table = np.asarray(equilibria, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("equilibria must be an array of real numbers") from error
    if (table.ndim != 2 or table.shape[1] != 2 or table.shape[0] == 0
            or not bool(np.all(np.isfinite(table)))):
        raise ValueError("equilibria must be a non-empty finite (M,2) array")
    if table.shape[0] > 1 and not bool(np.all(np.diff(table[:, 0]) > 0.0)):
        raise ValueError("equilibria marks must be strictly increasing")
    rates = table[:, 1]
    tolerance = 1e-12
    stable = rates < -tolerance
    repelling = rates > tolerance
    marginal = ~(stable | repelling)

    if table.shape[0] == 1 and (stable[0] or marginal[0]):
        terminal = table[:, 0]
        basin = np.zeros(marks.size, dtype=int)
    elif (table.shape[0] == 2 and int(np.sum(stable)) == 1
          and int(np.sum(marginal)) == 1 and not bool(np.any(repelling))):
        terminal = table[:, 0]
        marginal_index = int(np.nonzero(marginal)[0][0])
        fold = float(terminal[marginal_index])
        if marginal_index == 0:
            basin = np.where(marks <= fold, 0, 1)
        else:
            basin = np.where(marks >= fold, 1, 0)
    else:
        standard = (table.shape[0] % 2 == 1
                    and bool(np.all(stable[0::2]))
                    and bool(np.all(repelling[1::2])))
        if not standard:
            raise ValueError("equilibrium rates do not form an admissible attractor/watershed pattern")
        terminal = table[0::2, 0]
        watersheds = table[1::2, 0]
        if bool(np.any(np.isin(marks, watersheds))):
            raise ValueError("a starting mark sits exactly on a watershed")
        basin = np.searchsorted(watersheds, marks)

    counts = np.bincount(basin, minlength=terminal.size)
    rows = [[float(terminal[k]) + c, counts[k] / float(marks.size)]
            for k in range(terminal.size) if counts[k] > 0]
    return np.array(sorted(rows), dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
    digest = (
        "import numpy as np\n"
        "def _digest(values, rows, cols):\n"
        "    array = np.asarray(values, dtype=float)\n"
        "    if cols == 0:\n"
        "        if array.ndim != 1 or array.shape[0] != rows:\n"
        "            return -1.0\n"
        "    elif array.shape != (rows, cols):\n"
        "        return -1.0\n"
        "    flat = array.ravel()\n"
        "    phase = np.cos(np.arange(flat.size, dtype=float) + 1.0)\n"
        "    return float(flat.size) + float(np.sum(np.abs(flat)) + flat @ phase)\n"
    )
    bistable = "table = np.array([[-0.42618017, -0.64], [-0.2421404, 0.71], [0.49196363, -0.93]])\n"
    status = (
        "import numpy as np\n"
        "def _status(action):\n"
        "    try:\n"
        "        action()\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2\n"
    )
    return [
        # --- Normal: an unequal split of a uniform starting grid across one watershed ---
        {
            "setup": digest + bistable,
            "call": "_digest(settle_starting_marks(np.linspace(-2.5, 2.5, 1000), table, 0.11), 2, 2)",
            "gold_call": "_digest(_oracle_settle_starting_marks(np.linspace(-2.5, 2.5, 1000), table, 0.11), 2, 2)",
        },
        # --- Boundary: two marks straddling the watershed at a hair's distance ---
        {
            "setup": digest + bistable,
            "call": "_digest(settle_starting_marks(np.array([-0.24214041, -0.24214039]), table, 0.11), 2, 2)",
            "gold_call": "_digest(_oracle_settle_starting_marks(np.array([-0.24214041, -0.24214039]), table, 0.11), 2, 2)",
        },
        # --- Edge: every mark inside the upper basin, so one row with share one ---
        {
            "setup": digest + bistable,
            "call": "_digest(settle_starting_marks(np.linspace(0.0, 2.0, 9), table, 0.11), 1, 2)",
            "gold_call": "_digest(_oracle_settle_starting_marks(np.linspace(0.0, 2.0, 9), table, 0.11), 1, 2)",
        },
        # --- Edge: monostable class with a single equilibrium ---
        {
            "setup": digest + "table = np.array([[-0.49856987, -0.99]])\n",
            "call": "_digest(settle_starting_marks(np.linspace(-2.5, 2.5, 40), table, -0.32), 1, 2)",
            "gold_call": "_digest(_oracle_settle_starting_marks(np.linspace(-2.5, 2.5, 40), table, -0.32), 1, 2)",
        },
        # --- Normal: five equilibria, three basins, one of them left empty ---
        {
            "setup": digest + "table = np.array([[-2.0, -0.5], [-1.0, 0.4], [0.0, -0.3], [1.0, 0.6], [2.0, -0.8]])\n",
            "call": "_digest(settle_starting_marks(np.array([-3.0, -1.5, 1.2, 2.4, 1.01]), table, 0.25), 2, 2)",
            "gold_call": "_digest(_oracle_settle_starting_marks(np.array([-3.0, -1.5, 1.2, 2.4, 1.01]), table, 0.25), 2, 2)",
        },
        # --- Boundary: a left marginal fold attracts from its unbounded side ---
        {
            "setup": digest + "table = np.array([[-0.35, -2e-16], [0.48, -0.72]])\n",
            "call": "_digest(settle_starting_marks(np.array([-1.2, -0.35, -0.2, 0.9]), table, 0.13), 2, 2)",
            "gold_call": "_digest(_oracle_settle_starting_marks(np.array([-1.2, -0.35, -0.2, 0.9]), table, 0.13), 2, 2)",
        },
        # --- Boundary: a right marginal fold attracts from its unbounded side ---
        {
            "setup": digest + "table = np.array([[-0.48, -0.72], [0.35, 3e-16]])\n",
            "call": "_digest(settle_starting_marks(np.array([-0.9, 0.2, 0.35, 1.1]), table, -0.13), 2, 2)",
            "gold_call": "_digest(_oracle_settle_starting_marks(np.array([-0.9, 0.2, 0.35, 1.1]), table, -0.13), 2, 2)",
        },
        # --- Critical: a single marginal equilibrium receives the whole class ---
        {
            "setup": digest + "table = np.array([[0.0, 0.0]])\n",
            "call": "_digest(settle_starting_marks(np.array([1.0, -2.0, 0.0]), table, 0.0), 1, 2)",
            "gold_call": "_digest(_oracle_settle_starting_marks(np.array([1.0, -2.0, 0.0]), table, 0.0), 1, 2)",
        },
        # --- Invalid: a starting mark exactly on the watershed never settles ---
        {
            "setup": status + bistable,
            "call": "_status(lambda: settle_starting_marks(np.array([0.3, -0.2421404]), table, 0.11))",
            "gold_call": "_status(lambda: _oracle_settle_starting_marks(np.array([0.3, -0.2421404]), table, 0.11))",
        },
        # --- Invalid: two attractors without a watershed between them ---
        {
            "setup": status + "table = np.array([[-0.4, -0.6], [0.1, -0.2], [0.5, -0.9]])\n",
            "call": "_status(lambda: settle_starting_marks(np.array([0.0]), table, 0.0))",
            "gold_call": "_status(lambda: _oracle_settle_starting_marks(np.array([0.0]), table, 0.0))",
        },
        # --- Invalid: an even table that is neither a stable/neutral fold nor alternating ---
        {
            "setup": status + "table = np.array([[-0.4, -0.6], [0.1, 0.2]])\n",
            "call": "_status(lambda: settle_starting_marks(np.array([0.0]), table, 0.0))",
            "gold_call": "_status(lambda: _oracle_settle_starting_marks(np.array([0.0]), table, 0.0))",
        },
    ]
