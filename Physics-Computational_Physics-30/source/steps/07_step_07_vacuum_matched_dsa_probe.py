"""
Polyhedral vacuum-correction comparison.

Use the supplied geometry and angles, or the frozen definitions in the scientific background when each optional pair is None. Call all six public predecessors and numerically compose their returned objects: assemble A_local, stack directional data for every ordinate, construct L, solve for Y, assemble A_exact from those data, L and the returned Y, then compute the discrepancy from the returned exact and local forms. Return `round(delta,6)` exactly once. The default uses four polyhedra, total degree 2, epsilon=1/768, unit unscaled coefficients, C_tr=96, theta=1/2 and the fourteen directions specified in the scientific background. Optional geometry and angular arguments must each be supplied as a pair.

Returns
-------
delta : float Relative correction-form discrepancy rounded once to six decimal places.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def vacuum_matched_dsa_probe(vertices: np.ndarray | None = None, cells: list[list[list[int]]] | None = None, degree: int = 2, epsilon: float = 1/768, sigma_t: float = 1.0, sigma_a: float = 1.0, C_tr: float = 96.0, theta: float = 0.5, omegas: np.ndarray | None = None, weights: np.ndarray | None = None) -> float:
    '''Polyhedral vacuum-correction comparison.

    Geometry and angular pairs default together to the frozen definitions when None.

    Parameters
    ----------
    vertices : np.ndarray or None, shape (N_v, 3)
        Finite vertex coordinates of the guaranteed conforming convex polyhedral mesh.
    cells : list[list[list[int]]] or None, nonempty
        Each cell is a list of its planar polygonal faces. A face is an outward
        counterclockwise vertex-index cycle with no repeated endpoint or index.
        Shared faces use identical indices in reverse cyclic order. Each cell
        has at least four faces and positive volume. Cells and faces are convex.
    degree : int
        Total polynomial degree, one of 1, 2, 3, common to every cell.
    epsilon : float
        Positive finite diffusive scale, less than sqrt(sigma_t/sigma_a).
    sigma_t, sigma_a : float
        Positive finite unscaled total and absorption coefficients.
    C_tr : float
        Positive finite penalty prefactor. The resulting local form must be SPD.
    theta : float
        Finite SIP split in the open interval (0, 1).
    omegas : np.ndarray or None, shape (N_omega, 3)
        Finite unit directions with the shared moments and equal-weight opposite pairs.
    weights : np.ndarray or None, shape (N_omega,)
        Positive finite angular weights, summing to one.

    Returns
    -------
    delta : float
        Relative correction-form discrepancy rounded once to six decimal places.

    Raises
    ------
    ValueError
        If optional pairs are incomplete or a predecessor input-domain condition
        is violated, including the required positive-definite correction forms.
    '''
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_vacuum_matched_dsa_probe(vertices: np.ndarray | None = None, cells: list[list[list[int]]] | None = None, degree: int = 2, epsilon: float = 1/768, sigma_t: float = 1.0, sigma_a: float = 1.0, C_tr: float = 96.0, theta: float = 0.5, omegas: np.ndarray | None = None, weights: np.ndarray | None = None) -> float:
    import numpy as np
    if vertices is None and cells is None:
        vertices = np.asarray([[-0.0, 0.14790909090909096, 1.0], [-0.0, -0.0, 1.0], [0.36532786885245905, -0.0, 1.0], [0.36382520053475936, 0.09166276737967909, 1.0], [0.3480709342560553, 0.10993771626297587, 1.0], [-0.0, -0.0, -0.0], [-0.0, 0.8388181818181818, -0.0], [0.545655737704918, -0.0, -0.0], [0.5328575694237085, 0.7806882651537773, -0.0], [0.3754696582190189, 0.15557790177994812, 0.929615722441809], [1.0, 0.13061224489795908, 1.0], [1.0, -0.0, 1.0], [1.0, -0.0, -0.0], [1.0, 1.0, -0.0], [1.0, 1.0, 0.03181818181818184], [0.7481818181818182, 1.0, -0.0], [0.7556220607097051, 1.0, 0.015156049593843621], [-0.0, 1.0, -0.0], [-0.0, 1.0, 1.0], [0.4336538461538462, 1.0, 1.0], [1.0, 1.0, 1.0]], dtype=float)
        cells = [[[0, 1, 2, 3, 4], [5, 1, 0, 6], [1, 5, 7, 2], [5, 6, 8, 7], [9, 3, 2, 7, 8], [6, 0, 4, 9, 8], [4, 3, 9]], [[10, 11, 12, 13, 14], [15, 16, 14, 13], [3, 2, 11, 10], [2, 7, 12, 11], [7, 8, 15, 13, 12], [7, 2, 3, 9, 8], [16, 15, 8, 9], [9, 3, 10, 14, 16]], [[17, 18, 19, 16, 15], [18, 0, 4, 19], [0, 18, 17, 6], [6, 17, 15, 8], [0, 6, 8, 9, 4], [8, 15, 16, 9], [4, 9, 16, 19]], [[10, 14, 20], [16, 19, 20, 14], [19, 4, 3, 10, 20], [3, 4, 9], [3, 9, 16, 14, 10], [9, 4, 19, 16]]]
    elif vertices is None or cells is None:
        raise ValueError("supply both geometry arguments")
    if omegas is None and weights is None:
        omegas = np.concatenate([np.eye(3), -np.eye(3),
            np.array([[x,y,z] for x in (-1.,1.) for y in (-1.,1.) for z in (-1.,1.)]) / np.sqrt(3)])
        weights = np.r_[np.full(6, 1/12), np.full(8, 1/16)]
    elif omegas is None or weights is None:
        raise ValueError("supply both angular arguments")
    local = _oracle_assemble_transport_matched_correction(
        vertices, cells, degree, epsilon, sigma_t, sigma_a, C_tr, theta, omegas, weights)
    data = np.stack([_oracle_assemble_directional_transport_data(
        vertices, cells, degree, direction, sigma_t, epsilon) for direction in omegas])
    lifting = _oracle_assemble_jump_lifting(vertices, cells, degree, sigma_t, omegas, weights)
    micro_action = _oracle_solve_mean_zero_micro_action(data, lifting, epsilon, sigma_t, omegas, weights)
    exact = _oracle_assemble_exact_scalar_correction(
        data, lifting, micro_action, epsilon, sigma_t, sigma_a, omegas, weights)
    delta = _oracle_relative_correction_discrepancy(exact, local)
    return round(float(delta), 6)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    imports = 'import numpy as np\n'
    mesh = 'vertices=np.array([[-0.0, 0.14790909090909096, 1.0], [-0.0, -0.0, 1.0], [0.36532786885245905, -0.0, 1.0], [0.36382520053475936, 0.09166276737967909, 1.0], [0.3480709342560553, 0.10993771626297587, 1.0], [-0.0, -0.0, -0.0], [-0.0, 0.8388181818181818, -0.0], [0.545655737704918, -0.0, -0.0], [0.5328575694237085, 0.7806882651537773, -0.0], [0.3754696582190189, 0.15557790177994812, 0.929615722441809], [1.0, 0.13061224489795908, 1.0], [1.0, -0.0, 1.0], [1.0, -0.0, -0.0], [1.0, 1.0, -0.0], [1.0, 1.0, 0.03181818181818184], [0.7481818181818182, 1.0, -0.0], [0.7556220607097051, 1.0, 0.015156049593843621], [-0.0, 1.0, -0.0], [-0.0, 1.0, 1.0], [0.4336538461538462, 1.0, 1.0], [1.0, 1.0, 1.0]]);cells=[[[0, 1, 2, 3, 4], [5, 1, 0, 6], [1, 5, 7, 2], [5, 6, 8, 7], [9, 3, 2, 7, 8], [6, 0, 4, 9, 8], [4, 3, 9]], [[10, 11, 12, 13, 14], [15, 16, 14, 13], [3, 2, 11, 10], [2, 7, 12, 11], [7, 8, 15, 13, 12], [7, 2, 3, 9, 8], [16, 15, 8, 9], [9, 3, 10, 14, 16]], [[17, 18, 19, 16, 15], [18, 0, 4, 19], [0, 18, 17, 6], [6, 17, 15, 8], [0, 6, 8, 9, 4], [8, 15, 16, 9], [4, 9, 16, 19]], [[10, 14, 20], [16, 19, 20, 14], [19, 4, 3, 10, 20], [3, 4, 9], [3, 9, 16, 14, 10], [9, 4, 19, 16]]]\n'
    angles = 'om=np.concatenate([np.eye(3),-np.eye(3),np.array([[x,y,z] for x in (-1.,1.) for y in (-1.,1.) for z in (-1.,1.)])/np.sqrt(3)]);wt=np.r_[np.full(6,1/12),np.full(8,1/16)]\n'
    parameters = 'p=2;eps=1/768;st=1.;sa=1.;ctr=96.;theta=.5\n'
    cube = 'vertices=np.array([[x,y,z] for x in (0.,1.) for y in (0.,1.) for z in (0.,1.)]);cells=[[[0,1,3,2],[4,6,7,5],[0,4,5,1],[2,3,7,6],[0,2,6,4],[1,5,7,3]]]\n'
    domain_wrapper = 'def check(f):\n try:\n  f();return 0\n except ValueError:return 1\n except Exception:return 2\n'
    step_call = 'vacuum_matched_dsa_probe(vertices,cells,p,eps,st,sa,ctr,theta,om,wt)'
    call_3 = '#case:normal\n' + step_call
    call_4 = '_oracle_' + step_call
    call_7 = '#case:edge\ncheck(lambda:' + step_call + ')'
    call_8 = 'check(lambda:_oracle_' + step_call + ')'
    return [
        {'setup': 'import numpy as np', 'call': 'vacuum_matched_dsa_probe()', 'gold_call': '_oracle_vacuum_matched_dsa_probe()'},
        {'setup': imports + mesh + angles + parameters, 'call': call_3, 'gold_call': call_4},
        {'setup': imports + mesh + angles + parameters + 'eps=1/8192\n', 'call': '#case:boundary\n' + step_call, 'gold_call': call_4},
        {'setup': imports + cube + angles + parameters + 'p=1;eps=1/128\n', 'call': '#case:edge\n' + step_call, 'gold_call': call_4},
        {'setup': imports + mesh + angles + parameters + 'p=3;eps=1/2048\n', 'call': call_3, 'gold_call': call_4},
        {'setup': imports + mesh + angles + parameters + 'eps=1/32;st=2.5;sa=.7\n', 'call': call_3, 'gold_call': call_4},
        {'setup': imports + mesh + angles + parameters + 'a=.23;b=.37;om=om@np.array([[np.cos(a),-np.sin(a),0.],[np.sin(a),np.cos(a),0.],[0.,0.,1.]])@np.array([[np.cos(b),0.,np.sin(b)],[0.,1.,0.],[-np.sin(b),0.,np.cos(b)]]);wt=np.full(14,1/14)\n', 'call': call_3, 'gold_call': call_4},
        {'setup': imports + cube + angles + parameters + 'theta=1\n' + domain_wrapper, 'call': call_7, 'gold_call': call_8},
        {'setup': imports + cube + angles + parameters + 'st=0\n' + domain_wrapper, 'call': call_7, 'gold_call': call_8},
    ]
