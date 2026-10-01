"""
Build a rooted maximal-tree reduction atlas for the source paper's gauge hierarchy.

Figure 1 and Equation (2) of the source paper fix a rooted tree on the open cubic lattice: A_3=0 throughout the volume, A_2=0 on the n3=0 plane, and A_1=0 on the n2=n3=0 line. Starting at the origin, paths first move in positive n1 along the n2=n3=0 line, then in positive n2 on the n3=0 plane, and finally in positive n3 through the volume. The gauge-fixed tree links are nevertheless returned in the displayed kernel order: all n3 links, then boundary-plane n2 links, then boundary-line n1 links. Within a family, sort by increasing n1-fastest tail-site index. Return one numerical atlas with one row per oriented tree link and `6+2*V` columns, where `V=L**3`. Columns 0:6 are `(family, tail, head, parent_link, head_depth, subtree_size).` The family codes are 3, 2, 1. parent_link is the atlas row of the unique link entering the tail, or -1 when the tail is the origin. head_depth is the graph distance from the origin to that link's head, so the origin has depth 0 and a link leaving the origin has head_depth 1; on this tree it equals n1+n2+n3 of the head. subtree_size is the number of sites in the descendant subtree rooted at that head, the head itself counted, so it equals the sum of the row's descendant block. The next V columns are the oriented incidence row (-1 at the tail and +1 at the head). The final V columns are the 0/1 indicator of the descendant subtree rooted at the head, including the head itself. The descendant sets follow directly from the hierarchy. For an n3 link ending at (x,y,z+1), they are (x,y,z') with z'>=z+1. For an n2 link ending at (x,y+1,0), they are (x,y',z') with y'>=y+1. For an n1 link ending at (x+1,0,0), they are (x',y',z') with x'>=x+1. Flatten every site as `flat(x,y,z)=x+L*(y+L*z)`. This complete chain-complex atlas is the shared source of the kernel incidence and rooted charge-flow constructions later in the task.

Returns
-------
atlas : np.ndarray, shape (L**3 - 1, 6 + 2*L**3), float Link metadata, oriented incidence rows, and descendant-subtree indicators in the source hierarchy's ordered n3/n2/n1 families.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def build_maximal_tree_reduction_atlas(side_length) -> np.ndarray:
    '''Rooted incidence-and-subtree atlas for the paper's maximal tree.

    Parameters
    ----------
    side_length : int
        Number L of sites per open-lattice direction, 2 <= L <= 5.

    Returns
    -------
    atlas : np.ndarray, shape (L**3 - 1, 6 + 2*L**3), float
        Link metadata, oriented incidence rows, and descendant-subtree indicators
        in the source hierarchy's ordered n3/n2/n1 families.

    Raises
    ------
    ValueError
        If side_length is not a Python int or NumPy integer in [2, 5]
        (bool excluded).
    '''
    return np.zeros((int(side_length) ** 3 - 1, 6 + 2 * int(side_length) ** 3))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_build_maximal_tree_reduction_atlas(side_length):
    import numpy as np

    if isinstance(side_length, (bool, np.bool_)) or not isinstance(
        side_length, (int, np.integer)
    ):
        raise ValueError("side_length must be an ordinary integer")
    L = int(side_length)
    if L < 2 or L > 5:
        raise ValueError("invalid side length")

    V = L ** 3
    atlas = np.zeros((V - 1, 6 + 2 * V), dtype=float)
    z_count = L * L * (L - 1)
    y_start = z_count
    x_start = z_count + L * (L - 1)

    def flat(x, y, z):
        return x + L * (y + L * z)

    def z_row(x, y, z):
        return z * L * L + y * L + x

    def y_row(x, y):
        return y_start + y * L + x

    def x_row(x):
        return x_start + x

    def incoming(x, y, z):
        if z > 0:
            return z_row(x, y, z - 1)
        if y > 0:
            return y_row(x, y - 1)
        if x > 0:
            return x_row(x - 1)
        return -1

    def fill(row, family, tail_xyz, head_xyz, descendants):
        tail = flat(*tail_xyz)
        head = flat(*head_xyz)
        parent = incoming(*tail_xyz)
        atlas[row, :6] = (
            family,
            tail,
            head,
            parent,
            sum(head_xyz),
            len(descendants),
        )
        atlas[row, 6 + tail] = -1.0
        atlas[row, 6 + head] = 1.0
        atlas[row, 6 + V + np.asarray(descendants, dtype=int)] = 1.0

    for z in range(L - 1):
        for y in range(L):
            for x in range(L):
                descendants = [flat(x, y, zz) for zz in range(z + 1, L)]
                fill(z_row(x, y, z), 3, (x, y, z), (x, y, z + 1), descendants)

    for y in range(L - 1):
        for x in range(L):
            descendants = [
                flat(x, yy, zz)
                for zz in range(L)
                for yy in range(y + 1, L)
            ]
            fill(y_row(x, y), 2, (x, y, 0), (x, y + 1, 0), descendants)

    for x in range(L - 1):
        descendants = [
            flat(xx, yy, zz)
            for zz in range(L)
            for yy in range(L)
            for xx in range(x + 1, L)
        ]
        fill(x_row(x), 1, (x, 0, 0), (x + 1, 0, 0), descendants)

    return atlas

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': '# case: normal\nL=3\n',
      'call': 'build_maximal_tree_reduction_atlas(L)',
      'gold_call': '_oracle_build_maximal_tree_reduction_atlas(L)'},
     {'setup': '# case: boundary\nL=2\n',
      'call': 'build_maximal_tree_reduction_atlas(L)',
      'gold_call': '_oracle_build_maximal_tree_reduction_atlas(L)'},
     {'setup': '# case: edge\nimport numpy as np\nL=np.int64(5)\n',
      'call': 'build_maximal_tree_reduction_atlas(L)',
      'gold_call': '_oracle_build_maximal_tree_reduction_atlas(L)'},
     {'setup': '# case: normal\nL=4\n',
      'call': 'build_maximal_tree_reduction_atlas(L)',
      'gold_call': '_oracle_build_maximal_tree_reduction_atlas(L)'},
     {'setup': '# case: edge\n'
               'import numpy as np\n'
               'def status(fn):\n'
               '    try: fn(); return 0\n'
               '    except ValueError: return 1\n'
               '    except Exception: return 2\n',
      'call': 'tuple(status(fn) for fn in (lambda: build_maximal_tree_reduction_atlas(True),lambda: '
              'build_maximal_tree_reduction_atlas(1),lambda: build_maximal_tree_reduction_atlas(6),lambda: '
              'build_maximal_tree_reduction_atlas(3.0),lambda: build_maximal_tree_reduction_atlas(np.nan)))',
      'gold_call': 'tuple(status(fn) for fn in (lambda: '
                   '_oracle_build_maximal_tree_reduction_atlas(True),lambda: '
                   '_oracle_build_maximal_tree_reduction_atlas(1),lambda: '
                   '_oracle_build_maximal_tree_reduction_atlas(6),lambda: '
                   '_oracle_build_maximal_tree_reduction_atlas(3.0),lambda: '
                   '_oracle_build_maximal_tree_reduction_atlas(np.nan)))'}]
