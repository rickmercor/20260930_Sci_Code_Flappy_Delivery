"""
Assemble the global stiffness and mass matrices of a rectangular box, apply the restraint, and return them reordered into the block structure used by every later step.

The box spans Lx by Ly by Lz and is meshed with nx by ny by nz eight-node hexahedral elements of uniform size, each contributing the element matrices of the preceding step. All three degrees of freedom of every node on the face at the lower end of the first coordinate are removed.

The remaining equations are reordered into four consecutive blocks: the interior of substructure one, the interior of substructure two, the interior of substructure three, and the interface. The two interface node layers are those at one third and two thirds of nx. The three interiors are, in order, the node layers strictly between the restrained face and the first interface layer, strictly between the two interface layers, and strictly beyond the second interface layer. Within every block, degrees of freedom appear in ascending node index and, within a node, in ascending coordinate direction, where the node at grid position (i, j, k) has index i * (ny+1) * (nz+1) + j * (nz+1) + k, so the first coordinate varies slowest and the third fastest. Later steps index against this ordering, so it is part of the contract.

The function returns one float array of shape (2, n, n); plane 0 is stiffness and plane 1 is mass.

Raises ValueError if nx is not divisible by three; if nx is below six or ny or nz is below one, since each interior must retain at least one node layer; or if any edge length is not strictly positive.

Assembling a global finite element system is a bookkeeping operation: each element contributes its matrices to the rows and columns of the global arrays belonging to its own nodes, and overlapping contributions add. Restraints are imposed by deleting the equations of the restrained degrees of freedom, which leaves a symmetric positive definite pair for a properly supported structure.

Substructuring methods add a second layer of bookkeeping. The structure is divided into parts, and the equations are grouped so that degrees of freedom interior to each part are separated from those shared between parts. The shared set is called the interface. This grouping is what allows each part to be treated independently, because in the reordered system the interior blocks of different parts are coupled to one another only through the interface.

The grouping is a permutation and changes no eigenvalue of the system, but every subsequent operation indexes against it, so the convention has to be fixed once and adhered to. A frequent source of error is the treatment of the dividing surface itself: the nodes lying exactly on it belong to the interface, not to either interior, and an off-by-one choice here silently changes the sizes of all four blocks while leaving their total intact.

Returns
-------
np.ndarray of shape (2, n, n), dtype float, with n the number of free equations
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def assemble_partitioned_system(nx: int, ny: int, nz: int, Lx: float, Ly: float,
                                Lz: float, E: float, nu: float,
                                rho: float) -> 'np.ndarray':
    '''Assemble the restrained system in partitioned order.

    Parameters
    ----------
    nx, ny, nz : int
        Element counts along the three coordinate directions.
    Lx, Ly, Lz : float
        Edge lengths of the box.
    E, nu, rho : float
        Young's modulus, Poisson's ratio and density.

    Returns
    -------
    np.ndarray
        (2, n, n) float array; plane 0 is stiffness, plane 1 is mass, both in the
        four-block partitioned ordering.
    '''
    return partitioned_system  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_assemble_partitioned_system(nx: int, ny: int, nz: int, Lx: float,
                                        Ly: float, Lz: float, E: float, nu: float,
                                        rho: float) -> 'np.ndarray':
    import numpy as np
    CORNERS = [(0,0,0),(1,0,0),(1,1,0),(0,1,0),(0,0,1),(1,0,1),(1,1,1),(0,1,1)]
    if nx % 3 != 0:
        raise ValueError("nx must be divisible by three")
    if nx < 6 or ny < 1 or nz < 1:
        raise ValueError("element counts are too small; each interior needs at least one node layer")
    if min(Lx, Ly, Lz) <= 0:
        raise ValueError("edge lengths must be strictly positive")

    def _nid(i, j, k):
        return i*((ny+1)*(nz+1)) + j*(nz+1) + k

    nn = (nx+1)*(ny+1)*(nz+1)
    X = np.zeros((nn, 3))
    for i in range(nx+1):
        for j in range(ny+1):
            for k in range(nz+1):
                X[_nid(i, j, k)] = [i*Lx/nx, j*Ly/ny, k*Lz/nz]
    K = np.zeros((3*nn, 3*nn))
    M = np.zeros((3*nn, 3*nn))
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                conn = [_nid(i+c[0], j+c[1], k+c[2]) for c in CORNERS]
                EM = _oracle_hex8_element_matrices(X[conn], E, nu, rho)
                d = np.array([3*n+q for n in conn for q in range(3)])
                K[np.ix_(d, d)] += EM[0]
                M[np.ix_(d, d)] += EM[1]
    cl = [_nid(0, j, k) for j in range(ny+1) for k in range(nz+1)]
    cdof = np.array([3*n+q for n in cl for q in range(3)])
    free = np.setdiff1d(np.arange(3*nn), cdof)
    pos = -np.ones(3*nn, int)
    pos[free] = np.arange(free.size)

    def _blk(ii):
        return np.array([pos[3*_nid(i, j, k)+d] for i in ii
                         for j in range(ny+1) for k in range(nz+1) for d in range(3)])

    c = nx//3
    order = np.concatenate([_blk(range(1, c)), _blk(range(c+1, 2*c)),
                            _blk(range(2*c+1, nx+1)), _blk([c, 2*c])])
    Kf = K[np.ix_(free, free)]
    Mf = M[np.ix_(free, free)]
    return np.stack([Kf[np.ix_(order, order)], Mf[np.ix_(order, order)]])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        # --- normal: the six-element surrogate, stiffness plane ---
        # (stiffness is divided by E and rounded to 12 decimals so that rounding
        #  noise in entries that are exactly zero is not graded against 1e-9)
        {
            "setup": """import numpy as np
""",
            "call": "np.round(assemble_partitioned_system(6, 3, 1, 1.0, 0.6, 0.01, 210e9, 0.3, 7850.0)[0] / 210e9, 12)",
            "gold_call": "np.round(_oracle_assemble_partitioned_system(6, 3, 1, 1.0, 0.6, 0.01, 210e9, 0.3, 7850.0)[0] / 210e9, 12)",
        },
        # --- normal: the six-element surrogate, mass plane ---
        {
            "setup": """import numpy as np
""",
            "call": "assemble_partitioned_system(6, 3, 1, 1.0, 0.6, 0.01, 210e9, 0.3, 7850.0)[1]",
            "gold_call": "_oracle_assemble_partitioned_system(6, 3, 1, 1.0, 0.6, 0.01, 210e9, 0.3, 7850.0)[1]",
        },
        # --- boundary: smallest admissible mesh (stiffness plane scaled by E) ---
        {
            "setup": """import numpy as np
""",
            "call": "np.round(assemble_partitioned_system(6, 1, 1, 1.0, 0.6, 0.01, 210e9, 0.3, 7850.0) / np.array([210e9, 1.0]).reshape(2, 1, 1), 12)",
            "gold_call": "np.round(_oracle_assemble_partitioned_system(6, 1, 1, 1.0, 0.6, 0.01, 210e9, 0.3, 7850.0) / np.array([210e9, 1.0]).reshape(2, 1, 1), 12)",
        },
        # --- invalid: mesh too coarse to leave a node layer in each interior ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        assemble_partitioned_system(3, 1, 1, 1.0, 0.6, 0.01, 210e9, 0.3, 7850.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_assemble_partitioned_system(3, 1, 1, 1.0, 0.6, 0.01, 210e9, 0.3, 7850.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- edge: elongated, one element across width and thickness (stiffness plane scaled by E) ---
        {
            "setup": """import numpy as np
""",
            "call": "np.round(assemble_partitioned_system(9, 1, 1, 2.0, 0.1, 0.005, 70e9, 0.33, 2700.0) / np.array([70e9, 1.0]).reshape(2, 1, 1), 12)",
            "gold_call": "np.round(_oracle_assemble_partitioned_system(9, 1, 1, 2.0, 0.1, 0.005, 70e9, 0.33, 2700.0) / np.array([70e9, 1.0]).reshape(2, 1, 1), 12)",
        },
        # --- invalid: element count not divisible by three ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        assemble_partitioned_system(7, 3, 1, 1.0, 0.6, 0.01, 210e9, 0.3, 7850.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_assemble_partitioned_system(7, 3, 1, 1.0, 0.6, 0.01, 210e9, 0.3, 7850.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- invalid: zero edge length ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        assemble_partitioned_system(6, 3, 1, 0.0, 0.6, 0.01, 210e9, 0.3, 7850.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_assemble_partitioned_system(6, 3, 1, 0.0, 0.6, 0.01, 210e9, 0.3, 7850.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
