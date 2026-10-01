"""
Select the first moiety radius that resolves a nonzero transformation.

For some transferase reactions, the standard radius-one moiety representation produces a zero overall change despite a real chemical transformation. The source method method addresses this by increasing the representation radius until the product-minus-reactant moiety-change vector becomes nonzero. All rule and target representations must then use that selected radius consistently.

Returns
-------
Integer vector [selected_radius, *selected_change].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def select_informative_radius(
    changes_by_radius: 'np.ndarray',
    radii: 'np.ndarray',
) -> 'np.ndarray':
    """Select the first radius whose overall moiety-change vector is nonzero.

    Parameters
    ----------
    changes_by_radius
        Integer array with shape ``(n_radii, n_moieties)`` in increasing-radius
        order. Each row is the overall product-minus-reactant moiety change at
        that radius.
    radii
        Strictly increasing positive integer radii with shape ``(n_radii,)``.

    Returns
    -------
    np.ndarray
        Integer vector ``[selected_radius, *selected_change]``.

    Raises
    ------
    ValueError
        If the inputs are malformed or every supplied change vector is zero.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_select_informative_radius(
    changes_by_radius: 'np.ndarray',
    radii: 'np.ndarray',
) -> 'np.ndarray':
    import numpy as np

    changes = np.asarray(changes_by_radius)
    radius_values = np.asarray(radii)

    if changes.ndim != 2 or changes.shape[0] == 0 or changes.shape[1] == 0:
        raise ValueError(
            "changes_by_radius must be a nonempty two-dimensional array"
        )
    if radius_values.ndim != 1 or radius_values.size != changes.shape[0]:
        raise ValueError("radii must have one entry per change vector")
    if (
        not np.issubdtype(changes.dtype, np.number)
        or not np.isrealobj(changes)
        or np.any(~np.isfinite(changes))
    ):
        raise ValueError("changes_by_radius must be finite and numeric")
    if (
        not np.issubdtype(radius_values.dtype, np.number)
        or not np.isrealobj(radius_values)
        or np.any(~np.isfinite(radius_values))
    ):
        raise ValueError("radii must be finite and numeric")
    if np.any(changes != np.rint(changes)):
        raise ValueError("moiety changes must be integers")
    if (
        np.any(radius_values != np.rint(radius_values))
        or np.any(radius_values <= 0)
    ):
        raise ValueError("radii must be positive integers")

    radius_values = radius_values.astype(int, copy=False)
    if np.any(np.diff(radius_values) <= 0):
        raise ValueError("radii must be strictly increasing")

    changes = changes.astype(int, copy=False)

    for radius, row in zip(radius_values, changes):
        if np.any(row != 0):
            return np.concatenate(
                (np.array([radius], dtype=int), row.copy())
            )

    raise ValueError(
        "no supplied radius resolves a nonzero transformation"
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np\nx=np.array([[0,0,0],[-1,0,1],[-2,0,2]])\nr=np.array([1,2,3])",
            "call": "select_informative_radius(x.copy(),r.copy())",
            "gold_call": "_oracle_select_informative_radius(x.copy(),r.copy())",
        },
        {
            "setup": "import numpy as np\nx=np.array([[2,-1],[-3,4]])\nr=np.array([1,4])",
            "call": "select_informative_radius(x.copy(),r.copy())",
            "gold_call": "_oracle_select_informative_radius(x.copy(),r.copy())",
        },
        {
            "setup": "import numpy as np\nx=np.array([[0,0],[0,0],[-3,3]])\nr=np.array([1,2,7])",
            "call": "select_informative_radius(x.copy(),r.copy())",
            "gold_call": "_oracle_select_informative_radius(x.copy(),r.copy())",
        },
        {
            "setup": "import numpy as np\nx=np.zeros((3,4),dtype=int)\nr=np.array([1,2,5])\ndef check(fn):\n try: fn(x.copy(),r.copy())\n except ValueError: return 1\n except Exception: return 2\n return 0",
            "call": "check(select_informative_radius)",
            "gold_call": "check(_oracle_select_informative_radius)",
        },
    ]
