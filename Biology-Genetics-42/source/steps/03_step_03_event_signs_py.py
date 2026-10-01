"""
Return the sequence of event types read off the lineage counts, using +1 for a bifurcation and -1 for a hybridization, in order from the root.

The type of every internal event is already determined by the lineage counts: the count rises by one when a lineage splits and falls by one when two lineages merge. Recovering the sequence this way rather than from the graph keeps the encoding self-contained, and the sequence is what has to be put into correspondence when two networks of different complexity are compared. A network with n leaves and m hybridizations has n + 2m - 1 events, of which m are hybridizations.

Returns
-------
list of int - one entry per event, +1 for a bifurcation and -1 for a hybridization
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def event_signs(counts):
    """Classify ranked network events from consecutive lineage counts.

    Parameters
    ----------
    counts : list of int
        Lineage counts ordered from the rootward interval to the most recent
        interval.

    Returns
    -------
    list of int
        Event signs between consecutive intervals: ``+1`` for a bifurcation
        and ``-1`` for a hybridization.

    Raises
    ------
    ValueError
        If ``counts`` is not a sequence of at least two positive integers; if
        the first count is not one; or if any consecutive change is not
        exactly ``+1`` or ``-1``.
    """
    return []

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_event_signs(counts):
    """Return +1 bifurcation and -1 hybridization events from lineage counts."""
    import numpy as np

    if not isinstance(counts, (list, tuple, np.ndarray)) or len(counts) < 2:
        raise ValueError("counts must contain at least two intervals")

    values = list(counts)

    if any(
        not isinstance(value, (int, np.integer))
        or isinstance(value, bool)
        or int(value) < 1
        for value in values
    ):
        raise ValueError("lineage counts must be positive integers")

    values = [int(value) for value in values]

    if values[0] != 1:
        raise ValueError("the root interval must contain exactly one lineage")

    differences = [
        values[index] - values[index - 1]
        for index in range(1, len(values))
    ]

    if any(difference not in (-1, 1) for difference in differences):
        raise ValueError(
            "each internal event must change the lineage count by exactly one"
        )

    return differences

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return event-sign test specifications."""
    invalid_counts = """counts = [1, 3, 2]

def run_model():
    try:
        event_signs(counts)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _oracle_event_signs(counts)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
"""

    return [
        {
            # Normal: five leaves and one hybridization.
            "setup": "counts = [1, 2, 3, 2, 3, 4, 5]",
            "call": "event_signs(counts)",
            "gold_call": "_oracle_event_signs(counts)",
        },
        {
            # Normal: five leaves and two hybridizations.
            "setup": "counts = [1, 2, 3, 2, 3, 2, 3, 4, 5]",
            "call": "event_signs(counts)",
            "gold_call": "_oracle_event_signs(counts)",
        },
        {
            # Boundary: a three-leaf tree, with two consecutive bifurcations.
            "setup": "counts = [1, 2, 3]",
            "call": "event_signs(counts)",
            "gold_call": "_oracle_event_signs(counts)",
        },
        {
            # Invalid: a single event cannot change the count by two.
            "setup": invalid_counts,
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
