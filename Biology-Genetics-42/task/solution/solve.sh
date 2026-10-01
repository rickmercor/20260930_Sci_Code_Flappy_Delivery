#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

def lineage_counts(edges, size):
    """Reference implementation for ranked-interval lineage counts."""
    if isinstance(size, bool) or not isinstance(size, int) or size < 2:
        raise ValueError("size must be an integer of at least 2")

    if not isinstance(edges, (list, tuple)) or len(edges) == 0:
        raise ValueError("edges must be a non-empty list or tuple")

    def rank(vertex):
        if not isinstance(vertex, str):
            raise ValueError("vertex labels must be strings")

        if vertex == "*":
            return 0

        if vertex.startswith("v") and vertex[1:].isdigit():
            value = int(vertex[1:])
            if not 1 <= value < size:
                raise ValueError("internal vertex rank is outside 1..size-1")
            return value

        if vertex.startswith("L") and vertex[1:].isdigit():
            return size + 1

        raise ValueError("invalid vertex label")

    ranked_edges = []
    for edge in edges:
        if not isinstance(edge, (tuple, list)) or len(edge) != 2:
            raise ValueError("each edge must be an ancestor-descendant pair")

        ancestor, descendant = edge
        ancestor_rank = rank(ancestor)
        descendant_rank = rank(descendant)

        if descendant == "*":
            raise ValueError("the root cannot be an edge descendant")
        if isinstance(ancestor, str) and ancestor.startswith("L"):
            raise ValueError("a leaf cannot be an edge ancestor")
        if ancestor_rank >= descendant_rank:
            raise ValueError("each edge must point from an older to a younger vertex")

        ranked_edges.append((ancestor_rank, descendant_rank))

    return [
        sum(1 for ancestor_rank, descendant_rank in ranked_edges
            if ancestor_rank < interval <= descendant_rank)
        for interval in range(1, size + 1)
    ]

def count_matrix(edges, size):
    """Return the F-matrix of persisting lineages."""
    import numpy as np

    if not (
        isinstance(size, (int, np.integer))
        and not isinstance(size, bool)
        and int(size) >= 2
    ):
        raise ValueError("size must be an integer of at least 2")

    size = int(size)

    if not isinstance(edges, (list, tuple)) or len(edges) == 0:
        raise ValueError("edges must be a non-empty list or tuple")

    def rank(vertex):
        if not isinstance(vertex, str):
            raise ValueError("each endpoint must be a string")
        if vertex == "*":
            return 0
        if vertex.startswith("v") and vertex[1:].isdigit():
            value = int(vertex[1:])
            if not 1 <= value < size:
                raise ValueError("internal vertex rank must lie in 1..size-1")
            return value
        if vertex.startswith("L") and vertex[1:].isdigit():
            return size
        raise ValueError("vertex labels must be '*', v<number>, or L<number>")

    checked_edges = []
    for edge in edges:
        if not isinstance(edge, (tuple, list)) or len(edge) != 2:
            raise ValueError("each edge must be an ancestor-descendant pair")

        ancestor, descendant = edge
        ancestor_rank = rank(ancestor)
        descendant_rank = rank(descendant)

        if descendant == "*":
            raise ValueError("the root cannot be a descendant")
        if ancestor.startswith("L"):
            raise ValueError("a leaf cannot be an ancestor")
        if ancestor_rank >= descendant_rank:
            raise ValueError("edges must point from older to younger ranks")

        checked_edges.append((ancestor_rank, descendant_rank, tuple(edge)))

    alive = [
        {
            edge
            for ancestor_rank, descendant_rank, edge in checked_edges
            if ancestor_rank < interval <= descendant_rank
        }
        for interval in range(1, size + 1)
    ]

    return [
        [
            len(alive[i] & alive[j]) if j <= i else 0
            for j in range(size)
        ]
        for i in range(size)
    ]

def event_signs(counts):
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

def reconcile_timed_matrices(signs_a, matrix_a, times_a,
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

def weight_matrix(times):

    """Return the elapsed-time matrix for a reconciled time sequence."""

    import math

    import numpy as np



    if not isinstance(times, (list, tuple, np.ndarray)) or len(times) < 2:

        raise ValueError("times must contain at least a root and a tip time")



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



    n_intervals = len(times) - 1



    return [

        [

            times[column - 1] - times[row]

            if column < row

            else 0.0

            for column in range(1, n_intervals + 1)

        ]

        for row in range(1, n_intervals + 1)

    ]

def weighted_l1(matrix_a, weights_a, matrix_b, weights_b):

    """Return the L1 distance between two weighted count matrices."""

    import math

    import numpy as np



    matrices = [matrix_a, weights_a, matrix_b, weights_b]



    if any(not isinstance(matrix, (list, tuple)) for matrix in matrices):

        raise ValueError("all inputs must be list or tuple matrices")



    sizes = [len(matrix) for matrix in matrices]



    if any(size == 0 for size in sizes) or len(set(sizes)) != 1:

        raise ValueError("all matrices must be non-empty and have the same size")



    size = sizes[0]



    if any(

        not isinstance(row, (list, tuple)) or len(row) != size

        for matrix in matrices

        for row in matrix

    ):

        raise ValueError("all matrices must be square")



    for matrix in (matrix_a, matrix_b):

        for row_index, row in enumerate(matrix):

            for column_index, value in enumerate(row):

                if (

                    not isinstance(value, (int, np.integer))

                    or isinstance(value, bool)

                    or int(value) < 0

                ):

                    raise ValueError(

                        "count-matrix entries must be non-negative integers"

                    )

                if column_index > row_index and value != 0:

                    raise ValueError("count matrices must be lower triangular")



    for matrix in (weights_a, weights_b):

        for row_index, row in enumerate(matrix):

            for column_index, value in enumerate(row):

                if (

                    not isinstance(value, (int, float, np.integer, np.floating))

                    or isinstance(value, bool)

                    or not math.isfinite(float(value))

                    or float(value) < 0.0

                ):

                    raise ValueError(

                        "weight-matrix entries must be finite and non-negative"

                    )

                if column_index >= row_index and value != 0.0:

                    raise ValueError(

                        "weight matrices must be zero on and above the diagonal"

                    )



    return sum(

        abs(

            matrix_a[row][column] * weights_a[row][column]

            - matrix_b[row][column] * weights_b[row][column]

        )

        for row in range(size)

        for column in range(size)

    )

def network_distance(edges_a, size_a, times_a, edges_b, size_b, times_b):
    counts_a = lineage_counts(edges_a, size_a)
    counts_b = lineage_counts(edges_b, size_b)
    signs_a = event_signs(counts_a)
    signs_b = event_signs(counts_b)
    matrix_a = count_matrix(edges_a, size_a)
    matrix_b = count_matrix(edges_b, size_b)
    if ([matrix_a[i][i] for i in range(size_a)] != counts_a
            or [matrix_b[i][i] for i in range(size_b)] != counts_b):
        raise ValueError("F-matrix diagonals and lineage counts disagree")
    refined_a, clock_a, refined_b, clock_b = reconcile_timed_matrices(
        signs_a, matrix_a, times_a, signs_b, matrix_b, times_b)
    weights_a = weight_matrix(clock_a)
    weights_b = weight_matrix(clock_b)
    return weighted_l1(refined_a, weights_a, refined_b, weights_b)
SCICODE_GOLD_EOF
