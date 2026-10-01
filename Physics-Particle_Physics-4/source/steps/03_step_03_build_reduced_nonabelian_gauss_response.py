"""
Build the reduced non-Abelian Gauss response and interaction pullback.

Equation (6), Table I, and Equations (34)-(35) of the source paper leave the first gauge-field family away from the n2=n3=0 line and the second family away from the n3=0 plane. The underlying hierarchy is A_3=0 throughout the volume, A_2=0 on the n3=0 plane, and A_1=0 on the n2=n3=0 line. Use the preceding tree atlas to identify the retained positive n1 and n2 links. Put the n1 family first, looping n3 and n2 over 0,...,L-1 except n2=n3=0, then tail n1 over 0,...,L-2. Next put the n2 family, looping n3 over 1,...,L-1, tail n2 over 0,...,L-2, and n1 over 0,...,L-1. These are direction-first blocks with n3, n2, and n1 as the nested outer-to-inner coordinates. These are links, with longitudinal tails ending at L-2 and transverse site indices ending at L-1. Within each link, colour r=0,...,7 varies fastest. Resolve the paper's covariant left divergence into an ordinary one-sided difference M0 and a connection response M1, so the unprojected source is `(M0+g*M1)@p`. Pull Equation (8) back to physical-momentum space through the singlet Green matrix. Return the three symmetric coefficient matrices C0, C1, C2 for p.T@(C0+g*C1+`g**2`*C2)@p. This dense response and coefficient packing is a task-side representation of the paper's reduced source and nonlocal interaction. Let lambda[0],...,lambda[7] be the conventional Gell-Mann matrices lambda_1,...,lambda_8 in that order, with Tr(lambda[r] lambda[s])=2 delta_rs. For frame_index j in {0,1,2}, start from `theta=(0.29,-0.17,0.23,0.11,-0.19,0.13,-0.07,0.31),` roll it right by `2*j` entries, and multiply its odd 0-based entries by `(-1)**j`. Set H=sum_r theta_r lambda[r]/2, U=exp(i H), and `R[a,b] = Re Tr(lambda[a] U lambda[b] U^dagger)/2.` Evaluate the exponential spectrally from the Hermitian H. Derive the structure constants from [lambda[r],lambda[s]]=2i f[r,s,t] lambda[t], equivalently f[r,s,t]=Tr([lambda[r],lambda[s]] lambda[t])/(4i). At a retained direction-I forward-link tail set X=(n1+0.5)/L, Y=(n2+0.5)/L, and Z=(n3+0.5)/L. Use I=0 for n1 and I=1 for n2. For colour r=0,...,7 define `A_I^r = 0.41*sin(pi*((r+1)*X + 0.17*(I+1)*Z)) + 0.23*cos(pi*(((r mod 3)+1)*Y - 0.13*(I+1)*X)),` `P_I^r = cos(pi*(((r mod 4)+1)*Z + 0.11*(I+1)*Y)) + 0.37*sin(pi*((r+2)*Y - 0.19*(I+1)*X)).` Replace A and P by R@A and R@P. The unprojected colour charge is the paper's reduced Gauss operator of Equations (34)-(35): the left covariant divergence of the physical momenta. Use the endnote's one-sided left difference on Pi and the site-local structure-constant coupling to A. Split that operator into an Abelian piece M0 (ordinary left divergence, independent of A) and a connection piece M1 (linear in A, no extra factor of g), so the unprojected source is `(M0+g*M1)@p`. Do not project or normalize. This is not a right difference on Pi, not a centred stencil, and not a divergence that places the connection on both endpoints of a link. If B is the atlas incidence block, form the site Green matrix of `K=B.T@B/a**2` on the site-singlet subspace. Equation (8) is quadratic in the unprojected source; pull that quadratic back to physical-momentum space as three symmetric coefficient matrices in powers of g, each already containing the Hamiltonian's factor one half.

Returns
-------
response : np.ndarray, shape (2, 8*L**3, 8*Nphys), float Abelian and connection pieces M0, M1 of the reduced Gauss response. coefficients : np.ndarray, shape (3, 8*Nphys, 8*Nphys), float Symmetric g**0, g**1, and g**2 nonlocal interaction coefficients. momenta : np.ndarray, shape (8*Nphys,), float Rotated physical-link momenta, where Nphys=(L-1)**2*(2*L+1).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def build_reduced_nonabelian_gauss_response(
    side_length, lattice_spacing, frame_index, tree_atlas
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''Reduced Gauss response, nonlocal coefficients, and physical momenta.

    Parameters
    ----------
    side_length : int
        Number L of sites per cubic direction, 2 <= L <= 5.
    lattice_spacing : float
        Positive lattice spacing a.
    frame_index : int
        Residual global SU(3) frame label in {0, 1, 2}; bool is excluded.
    tree_atlas : array-like, shape (L**3 - 1, 6 + 2*L**3)
        Rooted maximal-tree atlas returned by Step 01.

    Returns
    -------
    response : np.ndarray, shape (2, 8*L**3, 8*Nphys), float
        Abelian and connection pieces M0, M1 of the reduced Gauss response.
    coefficients : np.ndarray, shape (3, 8*Nphys, 8*Nphys), float
        Symmetric g**0, g**1, and g**2 nonlocal interaction coefficients.
    momenta : np.ndarray, shape (8*Nphys,), float
        Rotated physical-link momenta, where Nphys=(L-1)**2*(2*L+1).

    Raises
    ------
    ValueError
        If side_length is not a Python int or NumPy integer in [2, 5]
        (bool excluded), or frame_index is not a Python int or NumPy integer
        in {0, 1, 2} (bool excluded); if lattice_spacing is not finite and
        positive; or if the atlas has the wrong shape, nonfinite entries,
        or an invalid incidence tree.
    '''
    return np.empty((2, 0, 0)), np.empty((3, 0, 0)), np.empty(0)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_build_reduced_nonabelian_gauss_response(
    side_length,
    lattice_spacing,
    frame_index,
    tree_atlas,
):
    import numpy as np

    if isinstance(side_length, (bool, np.bool_)) or not isinstance(
        side_length, (int, np.integer)
    ):
        raise ValueError("side_length must be an ordinary integer")
    if isinstance(frame_index, (bool, np.bool_)) or not isinstance(
        frame_index, (int, np.integer)
    ):
        raise ValueError("frame_index must be an ordinary integer")
    L = int(side_length)
    frame = int(frame_index)
    try:
        a = float(lattice_spacing)
    except Exception as exc:
        raise ValueError("invalid lattice spacing") from exc
    if L < 2 or L > 5 or not np.isfinite(a) or a <= 0.0:
        raise ValueError("invalid lattice")
    if frame not in (0, 1, 2):
        raise ValueError("invalid residual frame")

    V = L ** 3
    try:
        raw_atlas = np.asarray(tree_atlas)
        if np.iscomplexobj(raw_atlas):
            if not np.all(np.isfinite(raw_atlas)) or np.any(raw_atlas.imag != 0.0):
                raise ValueError("tree atlas must be finite and real")
            raw_atlas = raw_atlas.real
        atlas = np.asarray(raw_atlas, dtype=float)
    except Exception as exc:
        raise ValueError("invalid tree atlas") from exc
    if atlas.shape != (V - 1, 6 + 2 * V) or not np.all(np.isfinite(atlas)):
        raise ValueError("invalid tree atlas")

    def flat(x, y, z):
        return x + L * (y + L * z)

    expected = []
    for z in range(L - 1):
        for y in range(L):
            for x in range(L):
                expected.append((3, flat(x, y, z), flat(x, y, z + 1)))
    for y in range(L - 1):
        for x in range(L):
            expected.append((2, flat(x, y, 0), flat(x, y + 1, 0)))
    for x in range(L - 1):
        expected.append((1, flat(x, 0, 0), flat(x + 1, 0, 0)))
    expected = np.asarray(expected, dtype=float)
    if not np.array_equal(atlas[:, :3], expected):
        raise ValueError("atlas endpoints or family order are inconsistent")

    B = atlas[:, 6 : 6 + V]
    wanted_B = np.zeros_like(B)
    row_numbers = np.arange(V - 1)
    wanted_B[row_numbers, expected[:, 1].astype(int)] = -1.0
    wanted_B[row_numbers, expected[:, 2].astype(int)] = 1.0
    if not np.array_equal(B, wanted_B):
        raise ValueError("atlas incidence block is inconsistent")

    lam = np.zeros((8, 3, 3), dtype=complex)
    lam[0, 0, 1] = lam[0, 1, 0] = 1.0
    lam[1, 0, 1] = -1.0j
    lam[1, 1, 0] = 1.0j
    lam[2] = np.diag([1.0, -1.0, 0.0])
    lam[3, 0, 2] = lam[3, 2, 0] = 1.0
    lam[4, 0, 2] = -1.0j
    lam[4, 2, 0] = 1.0j
    lam[5, 1, 2] = lam[5, 2, 1] = 1.0
    lam[6, 1, 2] = -1.0j
    lam[6, 2, 1] = 1.0j
    lam[7] = np.diag([1.0, 1.0, -2.0]) / np.sqrt(3.0)

    structure = np.zeros((8, 8, 8), dtype=float)
    for r in range(8):
        for s in range(8):
            commutator = lam[r] @ lam[s] - lam[s] @ lam[r]
            for t in range(8):
                structure[r, s, t] = float(
                    np.real(np.trace(commutator @ lam[t]) / (4.0j))
                )

    theta = np.roll(
        np.array([0.29, -0.17, 0.23, 0.11, -0.19, 0.13, -0.07, 0.31]),
        2 * frame,
    )
    theta[1::2] *= (-1.0) ** frame
    H = np.einsum("r,rij->ij", 0.5 * theta, lam)
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    U = (eigenvectors * np.exp(1.0j * eigenvalues)) @ eigenvectors.conj().T
    rotation = np.empty((8, 8), dtype=float)
    for colour_out in range(8):
        for colour_in in range(8):
            rotation[colour_out, colour_in] = 0.5 * float(
                np.real(
                    np.trace(
                        lam[colour_out]
                        @ U
                        @ lam[colour_in]
                        @ U.conj().T
                    )
                )
            )

    tree_edges = {
        (int(round(row[1])), int(round(row[2])))
        for row in atlas
    }
    links = []
    for direction in (0, 1):
        for z in range(L):
            for y in range(L):
                for x in range(L):
                    coordinate = (x, y)[direction]
                    if coordinate >= L - 1:
                        continue
                    head_xyz = [x, y, z]
                    head_xyz[direction] += 1
                    tail = flat(x, y, z)
                    head = flat(*head_xyz)
                    if (tail, head) not in tree_edges:
                        links.append((direction, x, y, z, tail, head))

    Nphys = (L - 1) ** 2 * (2 * L + 1)
    if len(links) != Nphys:
        raise ValueError("atlas leaves an inconsistent physical-link count")
    response = np.zeros((2, 8 * V, 8 * Nphys), dtype=float)
    momenta = np.zeros(8 * Nphys, dtype=float)
    colours = np.arange(8, dtype=float)

    for link_index, (direction, x, y, z, tail, head) in enumerate(links):
        X = (x + 0.5) / L
        Y = (y + 0.5) / L
        Z = (z + 0.5) / L
        gauge = (
            0.41 * np.sin(
                np.pi * ((colours + 1.0) * X + 0.17 * (direction + 1) * Z)
            )
            + 0.23 * np.cos(
                np.pi
                * (
                    ((colours % 3.0) + 1.0) * Y
                    - 0.13 * (direction + 1) * X
                )
            )
        )
        momentum = (
            np.cos(
                np.pi
                * (
                    ((colours % 4.0) + 1.0) * Z
                    + 0.11 * (direction + 1) * Y
                )
            )
            + 0.37 * np.sin(
                np.pi
                * (
                    (colours + 2.0) * Y
                    - 0.19 * (direction + 1) * X
                )
            )
        )
        gauge = rotation @ gauge
        momentum = rotation @ momentum
        col0 = 8 * link_index
        momenta[col0 : col0 + 8] = momentum
        for b in range(8):
            response[0, b * V + tail, col0 + b] = 1.0
            response[0, b * V + head, col0 + b] = -1.0
            for r in range(8):
                response[1, b * V + tail, col0 + r] = np.dot(
                    structure[r, b], gauge
                )

    # Work at unit spacing, then restore the known powers of a. The physical
    # kernel or its inverse may overflow while the returned tuple is finite.
    K = B.T @ B
    kernel_values, kernel_vectors = np.linalg.eigh(K)
    tolerance = 1e-11 * float(kernel_values[-1])
    if (
        abs(float(kernel_values[0])) > tolerance
        or float(kernel_values[1]) <= tolerance
    ):
        raise ValueError("atlas incidence does not define one connected tree")
    inverse_values = np.zeros_like(kernel_values)
    inverse_values[1:] = 1.0 / kernel_values[1:]
    green = (kernel_vectors * inverse_values) @ kernel_vectors.T
    green -= np.mean(green, axis=0, keepdims=True)
    green -= np.mean(green, axis=1, keepdims=True)
    colour_green = np.kron(np.eye(8), green)
    M0, M1 = response
    coefficients = np.stack(
        (
            0.5 * (M0.T @ colour_green @ M0),
            0.5 * (M0.T @ colour_green @ M1 + M1.T @ colour_green @ M0),
            0.5 * (M1.T @ colour_green @ M1),
        )
    )
    coefficients = 0.5 * (coefficients + coefficients.transpose(0, 2, 1))
    coefficients[1] *= a
    coefficients[2] *= a
    coefficients[2] *= a
    response[0] /= a
    if not (
        np.all(np.isfinite(response))
        and np.all(np.isfinite(coefficients))
        and np.all(np.isfinite(momenta))
    ):
        raise ValueError("nonfinite reduced response")
    return response, coefficients, momenta

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': '# case: boundary\nL=2\na=1.0\nj=0\nT=_oracle_build_maximal_tree_reduction_atlas(L)\n',
  'call': 'build_reduced_nonabelian_gauss_response(L,a,j,T)',
  'gold_call': '_oracle_build_reduced_nonabelian_gauss_response(L,a,j,T)'},
 {'setup': '# case: normal\nL=3\na=0.41\nj=2\nT=_oracle_build_maximal_tree_reduction_atlas(L)\n',
  'call': 'build_reduced_nonabelian_gauss_response(L,a,j,T)',
  'gold_call': '_oracle_build_reduced_nonabelian_gauss_response(L,a,j,T)'},
 {'setup': '# case: normal\nL=3\na=0.73\nj=1\nT=_oracle_build_maximal_tree_reduction_atlas(L)\n',
  'call': 'build_reduced_nonabelian_gauss_response(L,a,j,T)',
  'gold_call': '_oracle_build_reduced_nonabelian_gauss_response(L,a,j,T)'},
 {'setup': '# case: edge\nL=4\na=1.7\nj=0\nT=_oracle_build_maximal_tree_reduction_atlas(L)\n',
  'call': 'build_reduced_nonabelian_gauss_response(L,a,j,T)[0]',
  'gold_call': '_oracle_build_reduced_nonabelian_gauss_response(L,a,j,T)[0]'},
 {'setup': '# case: edge\nL=5\na=0.37\ng=0.91\nj=2\nT=_oracle_build_maximal_tree_reduction_atlas(L)\n',
  'call': '(lambda '
          'z:z[2]@(z[1][0]+g*z[1][1]+g*g*z[1][2])@z[2])(build_reduced_nonabelian_gauss_response(L,a,j,T))',
  'gold_call': '(lambda '
               'z:z[2]@(z[1][0]+g*z[1][1]+g*g*z[1][2])@z[2])(_oracle_build_reduced_nonabelian_gauss_response(L,a,j,T))'},
 {'setup': '# case: edge\n'
           'import numpy as np\n'
           'L=3\n'
           'T=build_maximal_tree_reduction_atlas(L)\n'
           'Tbad=T.copy(); Tbad[0,6]=0.0\n'
           'def status(fn):\n'
           '    try: fn(); return 0\n'
           '    except ValueError: return 1\n'
           '    except Exception: return 2\n',
  'call': 'tuple((status(fn) for fn in (lambda: build_reduced_nonabelian_gauss_response(True, 0.4, 0, T), '
          'lambda: build_reduced_nonabelian_gauss_response(1, 0.4, 0, T), lambda: '
          'build_reduced_nonabelian_gauss_response(6, 0.4, 0, T), lambda: '
          'build_reduced_nonabelian_gauss_response(3, 0.0, 0, T), lambda: '
          'build_reduced_nonabelian_gauss_response(3, np.nan, 0, T), lambda: '
          'build_reduced_nonabelian_gauss_response(3, 0.4, True, T), lambda: '
          'build_reduced_nonabelian_gauss_response(3, 0.4, 3, T), lambda: '
          'build_reduced_nonabelian_gauss_response(3, 0.4, 0, T[:1]), lambda: '
          'build_reduced_nonabelian_gauss_response(3, 0.4, 0, Tbad))))',
  'gold_call': 'tuple((status(fn) for fn in (lambda: _oracle_build_reduced_nonabelian_gauss_response(True, '
               '0.4, 0, T), lambda: _oracle_build_reduced_nonabelian_gauss_response(1, 0.4, 0, T), lambda: '
               '_oracle_build_reduced_nonabelian_gauss_response(6, 0.4, 0, T), lambda: '
               '_oracle_build_reduced_nonabelian_gauss_response(3, 0.0, 0, T), lambda: '
               '_oracle_build_reduced_nonabelian_gauss_response(3, np.nan, 0, T), lambda: '
               '_oracle_build_reduced_nonabelian_gauss_response(3, 0.4, True, T), lambda: '
               '_oracle_build_reduced_nonabelian_gauss_response(3, 0.4, 3, T), lambda: '
               '_oracle_build_reduced_nonabelian_gauss_response(3, 0.4, 0, T[:1]), lambda: '
               '_oracle_build_reduced_nonabelian_gauss_response(3, 0.4, 0, Tbad))))'}]
