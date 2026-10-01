"""
Construct the joint refined F-matrices and boundary times for two ranked networks whose event histories need not have the same length.

Event correspondence, persistence counts, and elapsed time belong to one common refinement. A real event retains its identity and time, whereas an artificial event refines an interval without changing its lineage set. The source matches event types by recency, not by numerical clock time. The same matched event coordinates must determine both refined matrices and both refined clocks, even when artificial events are required on both sides.

Returns
-------
tuple (refined_matrix_a, refined_times_a, refined_matrix_b, refined_times_b): two M-by-M nested integer lists and two float lists of length M+1.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def reconcile_timed_matrices(signs_a, matrix_a, times_a,
                             signs_b, matrix_b, times_b):
    """Return the joint event refinement of two timed F-matrix encodings.
 
    Parameters
    ----------
    signs_a, signs_b : list of int
        Real event signs in root-to-tip order: +1 for a bifurcation and -1
        for a hybridization. Each sequence begins with a bifurcation.
    matrix_a, matrix_b : list of list of int
        Original lower-triangular F-matrices. A sequence with K real events
        has a (K+1)-by-(K+1) matrix. Diagonal differences equal the supplied
        event signs and the first diagonal entry is one.
    times_a, times_b : list of float
        Strictly decreasing boundary times [root, real-event times..., tip].
        A sequence with K events has K+2 boundary times. The two networks
        retain their own clocks; equal event coordinates do not imply equal
        times.
 
    Returns
    -------
    tuple
        (refined_matrix_a, refined_times_a, refined_matrix_b, refined_times_b).
        The matrices are M-by-M nested integer lists and each boundary vector
        is a list of M+1 floats, for the common refined interval count M.
 
        Event correspondence is the source's recent-first correspondence:
        bifurcations with the same rank counted from the most recent end
        correspond. Hybridizations correspond by their recency rank within
        each resulting branching span, including the terminal spans. Missing
        event coordinates are artificial events, and all results are returned
        in root-to-tip order. No real event may be moved or discarded.
 
        Artificial events leave the lineage set unchanged. Each refined
        F-entry counts persistence through its refined interval span. The
        original times are retained, and a run of r artificial events divides
        its enclosing original interval into r+1 equal durations. Either or
        both networks may be refined; M need not equal the larger input size.
        Inputs are not modified.
 
    Raises
    ------
    ValueError
        If signs are empty, are not integer +1/-1 values, or do not begin with
        +1; if a matrix is not square with nonnegative integer entries and
        zeros above the diagonal; if matrix size, positive diagonal, or
        diagonal differences disagree with the signs; or if a boundary vector
        has the wrong length, non-finite/non-numeric entries, or times that
        are not strictly decreasing. Booleans are not integers or times here.
        Lists and tuples are accepted for the input sequences and matrix rows.
    """
    return ([], [], [], [])

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_reconcile_timed_matrices(signs_a, matrix_a, times_a,
                                     signs_b, matrix_b, times_b):
    """Joint source-defined alignment, lineage refinement, and time refinement."""
    import math
    import numbers
 
    def _validate(signs, matrix, times):
        if not isinstance(signs, (list, tuple)) or not signs:
            raise ValueError("signs must be a nonempty list or tuple")
        if any(isinstance(s, bool) or not isinstance(s, numbers.Integral)
               or s not in (-1, 1) for s in signs):
            raise ValueError("real signs must be integer +1 or -1")
        if signs[0] != 1:
            raise ValueError("the first real event must be a bifurcation")
        n = len(signs) + 1
        if not isinstance(matrix, (list, tuple)) or len(matrix) != n:
            raise ValueError("matrix dimension must be one more than real events")
        if any(not isinstance(row, (list, tuple)) or len(row) != n
               for row in matrix):
            raise ValueError("matrix must be square")
        for i, row in enumerate(matrix):
            for j, x in enumerate(row):
                if isinstance(x, bool) or not isinstance(x, numbers.Integral) or x < 0:
                    raise ValueError("matrix entries must be nonnegative integers")
                if j > i and x != 0:
                    raise ValueError("matrix must be lower triangular")
        diagonal = [matrix[i][i] for i in range(n)]
        if diagonal[0] != 1 or any(x < 1 for x in diagonal):
            raise ValueError("diagonal must be positive and begin with one")
        if any(diagonal[i+1] - diagonal[i] != signs[i] for i in range(n-1)):
            raise ValueError("diagonal changes must equal real event signs")
        if not isinstance(times, (list, tuple)) or len(times) != n + 1:
            raise ValueError("provide root, all real-event times, and tip")
        if any(isinstance(t, bool) or not isinstance(t, numbers.Real)
               or not math.isfinite(float(t)) for t in times):
            raise ValueError("times must be finite real numbers")
        if any(times[i] <= times[i+1] for i in range(n)):
            raise ValueError("boundary times must decrease strictly")
        return ([int(s) for s in signs],
                [[int(x) for x in row] for row in matrix],
                [float(t) for t in times])
 
    signs_a, matrix_a, times_a = _validate(signs_a, matrix_a, times_a)
    signs_b, matrix_b, times_b = _validate(signs_b, matrix_b, times_b)
 
    def _align(signs_a, signs_b):
        """Align two root-to-tip event sequences with artificial-event zeros."""
 
        def validate(signs, name):
            if not isinstance(signs, (list, tuple)) or len(signs) == 0:
                raise ValueError(f"{name} must be a non-empty list or tuple")
            if any(
                not isinstance(sign, int)
                or isinstance(sign, bool)
                or sign not in (-1, 1)
                for sign in signs
            ):
                raise ValueError(f"{name} must contain only +1 and -1")
            if signs[0] != 1:
                raise ValueError(f"{name} must begin with a bifurcation")
 
        validate(signs_a, "signs_a")
        validate(signs_b, "signs_b")
 
        def buckets(signs):
            output = []
            current = []
 
            for sign in signs:
                if sign == 1:
                    output.append(current)
                    current = []
                else:
                    current.append(sign)
 
            output.append(current)
            return output
 
        # The source alignment is recent-first. Reverse the public root-to-tip
        # sequences, perform the source alignment, then reverse the result back.
        reversed_a = list(reversed(signs_a))
        reversed_b = list(reversed(signs_b))
 
        n_bifurcations = max(
            reversed_a.count(1),
            reversed_b.count(1),
        )
 
        def layout(signs):
            grouped = buckets(signs)
            missing = n_bifurcations - (len(grouped) - 1)
            real_bifurcations = [True] * (len(grouped) - 1)
 
            return (
                grouped[:-1] + [[]] * missing + [grouped[-1]],
                real_bifurcations + [False] * missing,
            )
 
        buckets_a, real_a = layout(reversed_a)
        buckets_b, real_b = layout(reversed_b)
 
        output_a = []
        output_b = []
 
        for index in range(n_bifurcations + 1):
            hybrids_a = list(buckets_a[index])
            hybrids_b = list(buckets_b[index])
 
            width = max(len(hybrids_a), len(hybrids_b))
 
            # In the source's recent-first orientation, real hybridizations are
            # matched left-to-right and unmatched positions are padded afterward.
            hybrids_a = hybrids_a + [0] * (width - len(hybrids_a))
            hybrids_b = hybrids_b + [0] * (width - len(hybrids_b))
 
            output_a.extend(hybrids_a)
            output_b.extend(hybrids_b)
 
            if index < n_bifurcations:
                output_a.append(1 if real_a[index] else 0)
                output_b.append(1 if real_b[index] else 0)
 
        return [list(reversed(output_a)), list(reversed(output_b))]
 
    def _expand_counts(aug_signs, matrix):
        """Expand an F-matrix according to a reconciled event sequence."""
 
        if not isinstance(aug_signs, (list, tuple)) or len(aug_signs) == 0:
            raise ValueError("aug_signs must be a non-empty list or tuple")
        if any(
            not isinstance(sign, int)
            or isinstance(sign, bool)
            or sign not in (-1, 0, 1)
            for sign in aug_signs
        ):
            raise ValueError("aug_signs must contain only -1, 0, and +1")
 
        if not isinstance(matrix, (list, tuple)) or len(matrix) == 0:
            raise ValueError("matrix must be a non-empty square matrix")
 
        original_size = len(matrix)
 
        if any(
            not isinstance(row, (list, tuple)) or len(row) != original_size
            for row in matrix
        ):
            raise ValueError("matrix must be square")
 
        if original_size != sum(sign != 0 for sign in aug_signs) + 1:
            raise ValueError(
                "matrix size must equal the number of non-artificial events plus one"
            )
 
        for row_index, row in enumerate(matrix):
            for column_index, value in enumerate(row):
                if (
                    not isinstance(value, int)
                    or isinstance(value, bool)
                    or value < 0
                ):
                    raise ValueError("matrix entries must be non-negative integers")
                if column_index > row_index and value != 0:
                    raise ValueError("matrix must be lower triangular")
 
        n_events = len(aug_signs)
        augmented_size = n_events + 1
 
        source_index = []
        seen_original_events = 0
 
        for interval in range(augmented_size):
            source_index.append(seen_original_events)
            if interval < n_events and aug_signs[interval] != 0:
                seen_original_events += 1
 
        return [
            [
                matrix[source_index[row]][source_index[column]]
                if column <= row
                else 0
                for column in range(augmented_size)
            ]
            for row in range(augmented_size)
        ]
 
    def _expand_times(aug_signs, times):
        """Insert equally spaced times for artificial events."""
        import math
        import numpy as np
 
        if not isinstance(aug_signs, (list, tuple)) or len(aug_signs) == 0:
            raise ValueError("aug_signs must be a non-empty list or tuple")
        if any(
            not isinstance(sign, int)
            or isinstance(sign, bool)
            or sign not in (-1, 0, 1)
            for sign in aug_signs
        ):
            raise ValueError("aug_signs must contain only -1, 0, and +1")
 
        if not isinstance(times, (list, tuple, np.ndarray)):
            raise ValueError("times must be a sequence")
 
        expected_length = sum(sign != 0 for sign in aug_signs) + 2
 
        if len(times) != expected_length:
            raise ValueError(
                "times must contain the root, one time per real event, and the tips"
            )
 
        if any(
            not isinstance(time, (int, float, np.integer, np.floating))
            or isinstance(time, bool)
            or not math.isfinite(float(time))
            for time in times
        ):
            raise ValueError("times must be finite real numbers")
 
        times = [float(time) for time in times]
 
        if any(
            times[index] <= times[index + 1]
            for index in range(len(times) - 1)
        ):
            raise ValueError("times must be strictly decreasing from root to tips")
 
        n_events = len(aug_signs)
        real_event_times = times[1:-1]
        real_slots = [
            index
            for index, sign in enumerate(aug_signs)
            if sign != 0
        ]
 
        expanded_event_times = [None] * n_events
 
        for slot, event_time in zip(real_slots, real_event_times):
            expanded_event_times[slot] = event_time
 
        index = 0
 
        while index < n_events:
            if expanded_event_times[index] is not None:
                index += 1
                continue
 
            end = index
 
            while end < n_events and expanded_event_times[end] is None:
                end += 1
 
            upper_time = (
                expanded_event_times[index - 1]
                if index > 0
                else times[0]
            )
            lower_time = (
                expanded_event_times[end]
                if end < n_events
                else times[-1]
            )
 
            step = (upper_time - lower_time) / (end - index + 1)
 
            for offset in range(end - index):
                expanded_event_times[index + offset] = (
                    upper_time - step * (offset + 1)
                )
 
            index = end
 
        return [times[0]] + expanded_event_times + [times[-1]]
 
    aligned_a, aligned_b = _align(signs_a, signs_b)
    refined_a = _expand_counts(aligned_a, matrix_a)
    refined_b = _expand_counts(aligned_b, matrix_b)
    clock_a = _expand_times(aligned_a, times_a)
    clock_b = _expand_times(aligned_b, times_b)
    return (refined_a, clock_a, refined_b, clock_b)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return deterministic, independently pinned scientific controls."""
    return [{'setup': 'sa = [1, 1, -1, 1, 1, 1]\n'
               'fa = [[1, 0, 0, 0, 0, 0, 0], [0, 2, 0, 0, 0, 0, 0], [0, 1, 3, 0, 0, 0, 0], [0, 0, 1, '
               '2, 0, 0, 0], [0, 0, 1, 1, 3, 0, 0], [0, 0, 1, 1, 2, 4, 0], [0, 0, 0, 0, 1, 3, 5]]\n'
               'ta = [10.0, 8.5, 6.5, 6.0, 5.5, 3.0, 1.5, 1.0]\n'
               'sb = [1, 1, -1, 1, -1, 1, 1, 1]\n'
               'fb = [[1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 2, 0, 0, 0, 0, 0, 0, 0], [0, 1, 3, 0, 0, 0, 0, '
               '0, 0], [0, 0, 1, 2, 0, 0, 0, 0, 0], [0, 0, 1, 1, 3, 0, 0, 0, 0], [0, 0, 0, 0, 1, 2, 0, '
               '0, 0], [0, 0, 0, 0, 1, 1, 3, 0, 0], [0, 0, 0, 0, 0, 0, 2, 4, 0], [0, 0, 0, 0, 0, 0, 1, '
               '3, 5]]\n'
               'tb = [10.0, 9.0, 8.5, 8.0, 7.5, 6.0, 5.0, 4.5, 3.0, 1.0]',
      'call': 'reconcile_timed_matrices(sa, fa, ta, sb, fb, tb)',
      'gold_call': '([[1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 2, 0, 0, 0, 0, '
                   '0, 0], [0, 0, 2, 2, 0, 0, 0, 0, 0], [0, 0, 1, 1, 3, 0, 0, 0, 0], [0, 0, 0, 0, 1, '
                   '2, 0, 0, 0], [0, 0, 0, 0, 1, 1, 3, 0, 0], [0, 0, 0, 0, 1, 1, 2, 4, 0], [0, 0, 0, '
                   '0, 0, 0, 1, 3, 5]], [10.0, 9.25, 8.5, 7.5, 6.5, 6.0, 5.5, 3.0, 1.5, 1.0], [[1, 0, '
                   '0, 0, 0, 0, 0, 0, 0], [0, 2, 0, 0, 0, 0, 0, 0, 0], [0, 1, 3, 0, 0, 0, 0, 0, 0], '
                   '[0, 0, 1, 2, 0, 0, 0, 0, 0], [0, 0, 1, 1, 3, 0, 0, 0, 0], [0, 0, 0, 0, 1, 2, 0, 0, '
                   '0], [0, 0, 0, 0, 1, 1, 3, 0, 0], [0, 0, 0, 0, 0, 0, 2, 4, 0], [0, 0, 0, 0, 0, 0, '
                   '1, 3, 5]], [10.0, 9.0, 8.5, 8.0, 7.5, 6.0, 5.0, 4.5, 3.0, 1.0])'},
     {'setup': 'sa = [1, 1]\n'
               'fa = [[1, 0, 0], [0, 2, 0], [0, 1, 3]]\n'
               'ta = [6.0, 4.0, 2.0, 1.0]\n'
               'sb = [1, 1, -1, 1]\n'
               'fb = [[1, 0, 0, 0, 0], [0, 2, 0, 0, 0], [0, 1, 3, 0, 0], [0, 0, 1, 2, 0], [0, 0, 0, 1, '
               '3]]\n'
               'tb = [8.0, 7.0, 5.0, 4.0, 2.0, 1.0]',
      'call': 'reconcile_timed_matrices(sa, fa, ta, sb, fb, tb)',
      'gold_call': '([[1, 0, 0, 0, 0], [1, 1, 0, 0, 0], [0, 0, 2, 0, 0], [0, 0, 2, 2, 0], [0, 0, 1, 1, '
                   '3]], [6.0, 5.0, 4.0, 3.0, 2.0, 1.0], [[1, 0, 0, 0, 0], [0, 2, 0, 0, 0], [0, 1, 3, '
                   '0, 0], [0, 0, 1, 2, 0], [0, 0, 0, 1, 3]], [8.0, 7.0, 5.0, 4.0, 2.0, 1.0])'},
     {'setup': 'sa = [1, 1]\n'
               'fa = [[1, 0, 0], [0, 2, 0], [0, 1, 3]]\n'
               'ta = [6.0, 4.0, 2.0, 1.0]\n'
               'sb = [1, 1]\n'
               'fb = [[1, 0, 0], [0, 2, 0], [0, 1, 3]]\n'
               'tb = [9.0, 5.0, 3.0, 1.0]',
      'call': 'reconcile_timed_matrices(sa, fa, ta, sb, fb, tb)',
      'gold_call': '([[1, 0, 0], [0, 2, 0], [0, 1, 3]], [6.0, 4.0, 2.0, 1.0], [[1, 0, 0], [0, 2, 0], '
                   '[0, 1, 3]], [9.0, 5.0, 3.0, 1.0])'},
     {'setup': 'sa = [1, 1, 1, -1]\n'
               'fa = [[1, 0, 0, 0, 0], [0, 2, 0, 0, 0], [0, 1, 3, 0, 0], [0, 0, 2, 4, 0], [0, 0, 1, 2, '
               '3]]\n'
               'ta = [8.0, 7.0, 6.0, 5.0, 4.0, 1.0]\n'
               'sb = [1, 1, 1, -1, -1, 1]\n'
               'fb = [[1, 0, 0, 0, 0, 0, 0], [0, 2, 0, 0, 0, 0, 0], [0, 1, 3, 0, 0, 0, 0], [0, 0, 2, '
               '4, 0, 0, 0], [0, 0, 1, 2, 3, 0, 0], [0, 0, 0, 0, 1, 2, 0], [0, 0, 0, 0, 0, 1, 3]]\n'
               'tb = [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]',
      'call': 'reconcile_timed_matrices(sa, fa, ta, sb, fb, tb)',
      'gold_call': '([[1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0], [0, 0, 2, 0, 0, 0, 0, 0], '
                   '[0, 0, 1, 3, 0, 0, 0, 0], [0, 0, 1, 3, 3, 0, 0, 0], [0, 0, 1, 3, 3, 3, 0, 0], [0, '
                   '0, 0, 2, 2, 2, 4, 0], [0, 0, 0, 1, 1, 1, 2, 3]], [8.0, 7.5, 7.0, 6.0, '
                   '5.666666666666667, 5.333333333333333, 5.0, 4.0, 1.0], [[1, 0, 0, 0, 0, 0, 0, 0], '
                   '[0, 2, 0, 0, 0, 0, 0, 0], [0, 1, 3, 0, 0, 0, 0, 0], [0, 0, 2, 4, 0, 0, 0, 0], [0, '
                   '0, 1, 2, 3, 0, 0, 0], [0, 0, 0, 0, 1, 2, 0, 0], [0, 0, 0, 0, 0, 1, 3, 0], [0, 0, '
                   '0, 0, 0, 1, 3, 3]], [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.5, 1.0])'},
     {'setup': 'sa = [1, 1, -1, 1, 1, 1]\n'
               'fa = [[1, 0, 0, 0, 0, 0, 0], [0, 2, 0, 0, 0, 0, 0], [0, 1, 3, 0, 0, 0, 0], [0, 0, 1, '
               '2, 0, 0, 0], [0, 0, 1, 1, 3, 0, 0], [0, 0, 1, 1, 2, 4, 0], [0, 0, 0, 0, 1, 3, 5]]\n'
               'ta = [10.0, 8.5, 6.5, 6.0, 5.5, 3.0, 1.5, 1.0]\n'
               'sb = [1, 1, -1, 1, 1, 1]\n'
               'fb = [[1, 0, 0, 0, 0, 0, 0], [0, 2, 0, 0, 0, 0, 0], [0, 1, 3, 0, 0, 0, 0], [0, 0, 1, '
               '2, 0, 0, 0], [0, 0, 1, 1, 3, 0, 0], [0, 0, 1, 1, 2, 4, 0], [0, 0, 0, 0, 1, 3, 5]]\n'
               'tb = [10.0, 8.5, 6.5, 6.0, 5.5, 3.0, 1.5, 1.0]',
      'call': 'reconcile_timed_matrices(sa, fa, ta, sb, fb, tb)',
      'gold_call': '([[1, 0, 0, 0, 0, 0, 0], [0, 2, 0, 0, 0, 0, 0], [0, 1, 3, 0, 0, 0, 0], [0, 0, 1, '
                   '2, 0, 0, 0], [0, 0, 1, 1, 3, 0, 0], [0, 0, 1, 1, 2, 4, 0], [0, 0, 0, 0, 1, 3, 5]], '
                   '[10.0, 8.5, 6.5, 6.0, 5.5, 3.0, 1.5, 1.0], [[1, 0, 0, 0, 0, 0, 0], [0, 2, 0, 0, 0, '
                   '0, 0], [0, 1, 3, 0, 0, 0, 0], [0, 0, 1, 2, 0, 0, 0], [0, 0, 1, 1, 3, 0, 0], [0, 0, '
                   '1, 1, 2, 4, 0], [0, 0, 0, 0, 1, 3, 5]], [10.0, 8.5, 6.5, 6.0, 5.5, 3.0, 1.5, '
                   '1.0])'},
     {'setup': 'sa = [1, 1, 1, -1]\n'
               'fa = [[1, 0, 0, 0, 0], [0, 2, 0, 0, 0], [0, 1, 3, 0, 0], [0, 0, 2, 4, 0], [0, 0, 1, 2, '
               '3]]\n'
               'ta = [30.0, 29.25, 27.75, 25.5, 22.5, 21.75]\n'
               'sb = [1, 1, 1, 1, -1, -1]\n'
               'fb = [[1, 0, 0, 0, 0, 0, 0], [0, 2, 0, 0, 0, 0, 0], [0, 1, 3, 0, 0, 0, 0], [0, 1, 2, '
               '4, 0, 0, 0], [0, 1, 1, 3, 5, 0, 0], [0, 0, 0, 2, 3, 4, 0], [0, 0, 0, 1, 1, 2, 3]]\n'
               'tb = [30.0, 29.375, 28.125, 26.25, 25.625, 24.375, 22.5, 21.875]',
      'call': 'reconcile_timed_matrices(sa, fa, ta, sb, fb, tb)',
      'gold_call': '([[1, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0], [0, 0, 2, 0, 0, 0, 0], [0, 0, 1, '
                   '3, 0, 0, 0], [0, 0, 0, 2, 4, 0, 0], [0, 0, 0, 2, 4, 4, 0], [0, 0, 0, 1, 2, 2, 3]], '
                   '[30.0, 29.625, 29.25, 27.75, 25.5, 24.0, 22.5, 21.75], [[1, 0, 0, 0, 0, 0, 0], [0, '
                   '2, 0, 0, 0, 0, 0], [0, 1, 3, 0, 0, 0, 0], [0, 1, 2, 4, 0, 0, 0], [0, 1, 1, 3, 5, '
                   '0, 0], [0, 0, 0, 2, 3, 4, 0], [0, 0, 0, 1, 1, 2, 3]], [30.0, 29.375, 28.125, '
                   '26.25, 25.625, 24.375, 22.5, 21.875])'},
     {'setup': 'sa = [1, 1, 1, 1, -1, -1]\n'
               'fa = [[1, 0, 0, 0, 0, 0, 0], [0, 2, 0, 0, 0, 0, 0], [0, 1, 3, 0, 0, 0, 0], [0, 0, 2, '
               '4, 0, 0, 0], [0, 0, 2, 3, 5, 0, 0], [0, 0, 1, 1, 3, 4, 0], [0, 0, 1, 1, 2, 2, 3]]\n'
               'ta = [30.0, 29.25, 27.75, 25.5, 22.5, 21.75, 20.25, 18.0]\n'
               'sb = [1, 1, -1, -1, 1, 1]\n'
               'fb = [[1, 0, 0, 0, 0, 0, 0], [0, 2, 0, 0, 0, 0, 0], [0, 1, 3, 0, 0, 0, 0], [0, 0, 1, '
               '2, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 2, 0], [0, 0, 0, 0, 0, 1, 3]]\n'
               'tb = [30.0, 29.375, 28.125, 26.25, 25.625, 24.375, 22.5, 21.875]',
      'call': 'reconcile_timed_matrices(sa, fa, ta, sb, fb, tb)',
      'gold_call': '([[1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 2, 0, 0, 0, 0, 0, 0, 0], [0, 1, 3, 0, 0, 0, 0, '
                   '0, 0], [0, 1, 3, 3, 0, 0, 0, 0, 0], [0, 1, 3, 3, 3, 0, 0, 0, 0], [0, 0, 2, 2, 2, '
                   '4, 0, 0, 0], [0, 0, 2, 2, 2, 3, 5, 0, 0], [0, 0, 1, 1, 1, 1, 3, 4, 0], [0, 0, 1, '
                   '1, 1, 1, 2, 2, 3]], [30.0, 29.25, 27.75, 27.0, 26.25, 25.5, 22.5, 21.75, 20.25, '
                   '18.0], [[1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 2, 0, 0, 0, 0, 0, 0, 0], [0, 1, 3, 0, 0, '
                   '0, 0, 0, 0], [0, 0, 1, 2, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, '
                   '0, 0, 2, 0, 0, 0], [0, 0, 0, 0, 0, 1, 3, 0, 0], [0, 0, 0, 0, 0, 1, 3, 3, 0], [0, '
                   '0, 0, 0, 0, 1, 3, 3, 3]], [30.0, 29.375, 28.125, 26.25, 25.625, 24.375, 22.5, '
                   '22.291666666666668, 22.083333333333332, 21.875])'},
     {'setup': 'sa = [1, 1, -1, -1, 1, 1, 1]\n'
               'fa = [[1, 0, 0, 0, 0, 0, 0, 0], [0, 2, 0, 0, 0, 0, 0, 0], [0, 1, 3, 0, 0, 0, 0, 0], '
               '[0, 0, 1, 2, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 2, 0, 0], [0, 0, '
               '0, 0, 0, 1, 3, 0], [0, 0, 0, 0, 0, 1, 2, 4]]\n'
               'ta = [30.0, 29.25, 27.75, 25.5, 22.5, 21.75, 20.25, 18.0, 15.0]\n'
               'sb = [1, 1, 1, -1, 1, -1, 1]\n'
               'fb = [[1, 0, 0, 0, 0, 0, 0, 0], [0, 2, 0, 0, 0, 0, 0, 0], [0, 1, 3, 0, 0, 0, 0, 0], '
               '[0, 1, 2, 4, 0, 0, 0, 0], [0, 1, 1, 2, 3, 0, 0, 0], [0, 1, 1, 2, 2, 4, 0, 0], [0, 0, '
               '0, 1, 1, 2, 3, 0], [0, 0, 0, 1, 1, 1, 2, 4]]\n'
               'tb = [30.0, 29.375, 28.125, 26.25, 25.625, 24.375, 22.5, 21.875, 20.625]',
      'call': 'reconcile_timed_matrices(sa, fa, ta, sb, fb, tb)',
      'gold_call': '([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 2, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 3, 0, 0, '
                   '0, 0, 0, 0, 0], [0, 0, 1, 2, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0, 0], '
                   '[0, 0, 0, 0, 0, 2, 0, 0, 0, 0], [0, 0, 0, 0, 0, 2, 2, 0, 0, 0], [0, 0, 0, 0, 0, 1, '
                   '1, 3, 0, 0], [0, 0, 0, 0, 0, 1, 1, 3, 3, 0], [0, 0, 0, 0, 0, 1, 1, 2, 2, 4]], '
                   '[30.0, 29.25, 27.75, 25.5, 22.5, 21.75, 21.0, 20.25, 19.125, 18.0, 15.0], [[1, 0, '
                   '0, 0, 0, 0, 0, 0, 0, 0], [0, 2, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 3, 0, 0, 0, 0, 0, '
                   '0, 0], [0, 1, 3, 3, 0, 0, 0, 0, 0, 0], [0, 1, 3, 3, 3, 0, 0, 0, 0, 0], [0, 1, 2, '
                   '2, 2, 4, 0, 0, 0, 0], [0, 1, 1, 1, 1, 2, 3, 0, 0, 0], [0, 1, 1, 1, 1, 2, 2, 4, 0, '
                   '0], [0, 0, 0, 0, 0, 1, 1, 2, 3, 0], [0, 0, 0, 0, 0, 1, 1, 1, 2, 4]], [30.0, '
                   '29.375, 28.125, 27.5, 26.875, 26.25, 25.625, 24.375, 22.5, 21.875, 20.625])'},
     {'setup': 'sa = [1, 1, 1, -1, 1]\n'
               'fa = [[1, 0, 0, 0, 0, 0], [0, 2, 0, 0, 0, 0], [0, 1, 3, 0, 0, 0], [0, 0, 2, 4, 0, 0], '
               '[0, 0, 1, 2, 3, 0], [0, 0, 1, 1, 2, 4]]\n'
               'ta = [30.0, 29.25, 27.75, 25.5, 22.5, 21.75, 20.25]\n'
               'sb = [1, 1, 1, 1, -1, -1, -1, 1, 1]\n'
               'fb = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 2, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 3, 0, 0, '
               '0, 0, 0, 0, 0], [0, 0, 2, 4, 0, 0, 0, 0, 0, 0], [0, 0, 2, 3, 5, 0, 0, 0, 0, 0], [0, 0, '
               '1, 2, 3, 4, 0, 0, 0, 0], [0, 0, 0, 1, 1, 2, 3, 0, 0, 0], [0, 0, 0, 0, 0, 1, 1, 2, 0, '
               '0], [0, 0, 0, 0, 0, 0, 0, 1, 3, 0], [0, 0, 0, 0, 0, 0, 0, 1, 2, 4]]\n'
               'tb = [30.0, 29.375, 28.125, 26.25, 25.625, 24.375, 22.5, 21.875, 20.625, 18.75, '
               '18.125]',
      'call': 'reconcile_timed_matrices(sa, fa, ta, sb, fb, tb)',
      'gold_call': '([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, '
                   '0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 3, 0, 0, '
                   '0, 0, 0, 0], [0, 0, 0, 1, 3, 3, 0, 0, 0, 0, 0], [0, 0, 0, 1, 3, 3, 3, 0, 0, 0, 0], '
                   '[0, 0, 0, 1, 3, 3, 3, 3, 0, 0, 0], [0, 0, 0, 0, 2, 2, 2, 2, 4, 0, 0], [0, 0, 0, 0, '
                   '1, 1, 1, 1, 2, 3, 0], [0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 4]], [30.0, 29.75, 29.5, '
                   '29.25, 27.75, 27.1875, 26.625, 26.0625, 25.5, 22.5, 21.75, 20.25], [[1, 0, 0, 0, '
                   '0, 0, 0, 0, 0, 0, 0], [0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 3, 0, 0, 0, 0, 0, '
                   '0, 0, 0], [0, 0, 2, 4, 0, 0, 0, 0, 0, 0, 0], [0, 0, 2, 3, 5, 0, 0, 0, 0, 0, 0], '
                   '[0, 0, 1, 2, 3, 4, 0, 0, 0, 0, 0], [0, 0, 0, 1, 1, 2, 3, 0, 0, 0, 0], [0, 0, 0, 0, '
                   '0, 1, 1, 2, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 3, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, '
                   '3, 3, 0], [0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 4]], [30.0, 29.375, 28.125, 26.25, '
                   '25.625, 24.375, 22.5, 21.875, 20.625, 19.6875, 18.75, 18.125])'},
     {'setup': 'sa = [1, 1, -1, 1, 1, 1]\n'
               'fa = [[1, 0, 0, 0, 0, 0, 0], [0, 2, 0, 0, 0, 0, 0], [0, 1, 3, 0, 0, 0, 0], [0, 0, 1, '
               '2, 0, 0, 0], [0, 0, 1, 1, 3, 0, 0], [0, 0, 1, 1, 2, 4, 0], [0, 0, 0, 0, 1, 3, 5]]\n'
               'ta = [10.0, 8.5, 6.5, 6.0, 5.5, 3.0, 1.5, 1.0]\n'
               'sb = [1, 1, -1, 1, -1, 1, 1, 1]\n'
               'fb = [[1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 2, 0, 0, 0, 0, 0, 0, 0], [0, 1, 3, 0, 0, 0, 0, '
               '0, 0], [0, 0, 1, 2, 0, 0, 0, 0, 0], [0, 0, 1, 1, 3, 0, 0, 0, 0], [0, 0, 0, 0, 1, 2, 0, '
               '0, 0], [0, 0, 0, 0, 1, 1, 3, 0, 0], [0, 0, 0, 0, 0, 0, 2, 4, 0], [0, 0, 0, 0, 0, 0, 1, '
               '3, 5]]\n'
               'tb = [10.0, 9.0, 8.5, 8.0, 7.5, 6.0, 5.0, 4.5, 3.0, 1.0]\n'
               'sa[0] = 0\n'
               'def run_model():\n'
               '    try:\n'
               '        reconcile_timed_matrices(sa, fa, ta, sb, fb, tb)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n',
      'call': 'run_model()',
      'gold_call': '1'},
     {'setup': 'sa = [1, 1, -1, 1, 1, 1]\n'
               'fa = [[1, 0, 0, 0, 0, 0, 0], [0, 2, 0, 0, 0, 0, 0], [0, 1, 3, 0, 0, 0, 0], [0, 0, 1, '
               '2, 0, 0, 0], [0, 0, 1, 1, 3, 0, 0], [0, 0, 1, 1, 2, 4, 0], [0, 0, 0, 0, 1, 3, 5]]\n'
               'ta = [10.0, 8.5, 6.5, 6.0, 5.5, 3.0, 1.5, 1.0]\n'
               'sb = [1, 1, -1, 1, -1, 1, 1, 1]\n'
               'fb = [[1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 2, 0, 0, 0, 0, 0, 0, 0], [0, 1, 3, 0, 0, 0, 0, '
               '0, 0], [0, 0, 1, 2, 0, 0, 0, 0, 0], [0, 0, 1, 1, 3, 0, 0, 0, 0], [0, 0, 0, 0, 1, 2, 0, '
               '0, 0], [0, 0, 0, 0, 1, 1, 3, 0, 0], [0, 0, 0, 0, 0, 0, 2, 4, 0], [0, 0, 0, 0, 0, 0, 1, '
               '3, 5]]\n'
               'tb = [10.0, 9.0, 8.5, 8.0, 7.5, 6.0, 5.0, 4.5, 3.0, 1.0]\n'
               'fa[0] = fa[0][:-1]\n'
               'def run_model():\n'
               '    try:\n'
               '        reconcile_timed_matrices(sa, fa, ta, sb, fb, tb)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n',
      'call': 'run_model()',
      'gold_call': '1'},
     {'setup': 'sa = [1, 1, -1, 1, 1, 1]\n'
               'fa = [[1, 0, 0, 0, 0, 0, 0], [0, 2, 0, 0, 0, 0, 0], [0, 1, 3, 0, 0, 0, 0], [0, 0, 1, '
               '2, 0, 0, 0], [0, 0, 1, 1, 3, 0, 0], [0, 0, 1, 1, 2, 4, 0], [0, 0, 0, 0, 1, 3, 5]]\n'
               'ta = [10.0, 8.5, 6.5, 6.0, 5.5, 3.0, 1.5, 1.0]\n'
               'sb = [1, 1, -1, 1, -1, 1, 1, 1]\n'
               'fb = [[1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 2, 0, 0, 0, 0, 0, 0, 0], [0, 1, 3, 0, 0, 0, 0, '
               '0, 0], [0, 0, 1, 2, 0, 0, 0, 0, 0], [0, 0, 1, 1, 3, 0, 0, 0, 0], [0, 0, 0, 0, 1, 2, 0, '
               '0, 0], [0, 0, 0, 0, 1, 1, 3, 0, 0], [0, 0, 0, 0, 0, 0, 2, 4, 0], [0, 0, 0, 0, 0, 0, 1, '
               '3, 5]]\n'
               'tb = [10.0, 9.0, 8.5, 8.0, 7.5, 6.0, 5.0, 4.5, 3.0, 1.0]\n'
               'ta = ta[:-1]\n'
               'def run_model():\n'
               '    try:\n'
               '        reconcile_timed_matrices(sa, fa, ta, sb, fb, tb)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n',
      'call': 'run_model()',
      'gold_call': '1'},
     {'setup': 'sa = [1, 1, -1, 1, 1, 1]\n'
               'fa = [[1, 0, 0, 0, 0, 0, 0], [0, 2, 0, 0, 0, 0, 0], [0, 1, 3, 0, 0, 0, 0], [0, 0, 1, '
               '2, 0, 0, 0], [0, 0, 1, 1, 3, 0, 0], [0, 0, 1, 1, 2, 4, 0], [0, 0, 0, 0, 1, 3, 5]]\n'
               'ta = [10.0, 8.5, 6.5, 6.0, 5.5, 3.0, 1.5, 1.0]\n'
               'sb = [1, 1, -1, 1, -1, 1, 1, 1]\n'
               'fb = [[1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 2, 0, 0, 0, 0, 0, 0, 0], [0, 1, 3, 0, 0, 0, 0, '
               '0, 0], [0, 0, 1, 2, 0, 0, 0, 0, 0], [0, 0, 1, 1, 3, 0, 0, 0, 0], [0, 0, 0, 0, 1, 2, 0, '
               '0, 0], [0, 0, 0, 0, 1, 1, 3, 0, 0], [0, 0, 0, 0, 0, 0, 2, 4, 0], [0, 0, 0, 0, 0, 0, 1, '
               '3, 5]]\n'
               'tb = [10.0, 9.0, 8.5, 8.0, 7.5, 6.0, 5.0, 4.5, 3.0, 1.0]\n'
               'ta[2] = ta[1]\n'
               'def run_model():\n'
               '    try:\n'
               '        reconcile_timed_matrices(sa, fa, ta, sb, fb, tb)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n',
      'call': 'run_model()',
      'gold_call': '1'}]
