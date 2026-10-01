"""
Build the source-filtered library graph from pair records and return canonical weighted edges.

Each pair record contains two zero-based labels, spectral evidence, and two structural scores. The pinned source governs threshold application and conversion to an edge weight; preserve identity and canonicalize retained pairs.

Returns
-------
A zero-based canonical edge array together with its aligned source-defined edge-weight vector.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def build_hsqc_network(
    n_nodes: int,
    pair_records: "np.ndarray",
    distance_limit: float,
    hybrid_floor: float,
) -> "tuple[np.ndarray, np.ndarray]":
    """Return source-filtered canonical library edges and aligned weights.

    Each record contains two zero-based node labels, spectral evidence, and
    two structural scores. Preserve candidate identity and canonicalize every
    retained pair with the smaller node index first.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_build_hsqc_network(
    n_nodes: int,
    pair_records: "np.ndarray",
    distance_limit: float,
    hybrid_floor: float,
) -> "tuple[np.ndarray, np.ndarray]":
    import numpy as np

    records = np.asarray(pair_records, dtype=float)

    if (
        n_nodes < 1
        or records.ndim != 2
        or records.shape[1:] != (5,)
    ):
        raise ValueError("records must have five columns")

    edges = []
    weights = []
    seen = set()

    for raw_i, raw_j, distance, tanimoto, mcs in records:
        i = int(raw_i)
        j = int(raw_j)

        if (
            raw_i != i
            or raw_j != j
            or not (
                0 <= i < n_nodes
                and 0 <= j < n_nodes
            )
        ):
            raise ValueError("invalid node")

        edge = (
            min(i, j),
            max(i, j),
        )

        if i == j or edge in seen:
            raise ValueError(
                "pairs must be unique non-self edges"
            )

        seen.add(edge)
        hybrid = 0.5 * (
            float(tanimoto)
            + float(mcs)
        )

        if (
            distance < distance_limit
            and hybrid > hybrid_floor
        ):
            edges.append(edge)
            weights.append(hybrid)

    if not edges:
        return (
            np.empty((0, 2), dtype=int),
            np.empty(0, dtype=float),
        )

    return (
        np.asarray(edges, dtype=int),
        np.asarray(weights, dtype=float),
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import copy\n'
               'import numpy as np\n'
               'def pack(x):\n'
               ' edges,weights=x; return '
               'np.concatenate((np.asarray(edges.shape,dtype=float),edges.ravel().astype(float),weights))\n'
               'r=np.array([[0,1,30.0,.70,.74],[1,2,29.999999,.61,.63],[0,2,30.1,.9,.9]])',
      'call': 'pack(build_hsqc_network(copy.deepcopy(3), copy.deepcopy(r), copy.deepcopy(30.0), '
              'copy.deepcopy(0.6)))',
      'gold_call': 'pack(_oracle_build_hsqc_network(copy.deepcopy(3), copy.deepcopy(r), '
                   'copy.deepcopy(30.0), copy.deepcopy(0.6)))'},
     {'setup': 'import copy\n'
               'import numpy as np\n'
               'def pack(x):\n'
               ' edges,weights=x; return '
               'np.concatenate((np.asarray(edges.shape,dtype=float),edges.ravel().astype(float),weights))\n'
               'r=np.array([[2,0,12,.7,.8]])',
      'call': 'pack(build_hsqc_network(copy.deepcopy(3), copy.deepcopy(r), copy.deepcopy(30.0), '
              'copy.deepcopy(0.6)))',
      'gold_call': 'pack(_oracle_build_hsqc_network(copy.deepcopy(3), copy.deepcopy(r), '
                   'copy.deepcopy(30.0), copy.deepcopy(0.6)))'},
     {'setup': 'import copy\n'
               'import numpy as np\n'
               'def pack(x):\n'
               ' edges,weights=x; return '
               'np.concatenate((np.asarray(edges.shape,dtype=float),edges.ravel().astype(float),weights))\n'
               'r=np.array([[0,1,31,.9,.9]])',
      'call': 'pack(build_hsqc_network(copy.deepcopy(2), copy.deepcopy(r), copy.deepcopy(30.0), '
              'copy.deepcopy(0.6)))',
      'gold_call': 'pack(_oracle_build_hsqc_network(copy.deepcopy(2), copy.deepcopy(r), '
                   'copy.deepcopy(30.0), copy.deepcopy(0.6)))'},
     {'setup': 'import copy\n'
               'import numpy as np\n'
               'def pack(x):\n'
               ' edges,weights=x; return '
               'np.concatenate((np.asarray(edges.shape,dtype=float),edges.ravel().astype(float),weights))\n'
               'r=np.array([[0,1,20.,.6,.6],[0,2,20.,.5,.5],[1,2,20.,.8,.8]])',
      'call': 'pack(build_hsqc_network(copy.deepcopy(3), copy.deepcopy(r), copy.deepcopy(30.0), '
              'copy.deepcopy(0.6)))',
      'gold_call': 'pack(_oracle_build_hsqc_network(copy.deepcopy(3), copy.deepcopy(r), '
                   'copy.deepcopy(30.0), copy.deepcopy(0.6)))'}]
