"""
Expand all three Cartesian transition-dipole components in the initial Hagedorn basis.

Multiplication by a nuclear coordinate contains raising and lowering contributions. Polynomial dipoles therefore populate lower shells as well as their highest degree.

Returns
-------
return labels, c
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def dipole_coefficients(q0, powers, coefficients, hbar):
    r"""For the real SPD initial Q0 and P0=i*inv(Q0), expand
    mu_a(y)*phi_0 = sum_k c[k,a]*phi_k for a=x,y,z, y=q-q_initial.
    The basis is orthonormal, with coordinate operator
    y_i=sqrt(hbar/2)*sum_j Q0[i,j]*(a_j + a_j^dagger),
    a_j|k>=sqrt(k_j)|k-e_j>, a_j^dagger|k>=sqrt(k_j+1)|k+e_j>.
    mu_a(y)=sum_r coefficients[r,a]*product_i y_i**powers[r,i].
    There are no hidden factorials. Repeated terms add. Include every occupation
    label with total degree <= max_r sum_i powers[r,i], including zero coefficients.
    Sort labels first by total degree, then by the tuple in ascending lexicographic
    order. This order is (0,0), (0,1), (1,0), (0,2), (1,1), (2,0) for D=2, degree=2.

    Parameters
    ----------
    q0 : real SPD array, shape (D,D)
    powers : nonnegative integer-valued array, shape (R,D), R>=1, degree<=6
    coefficients : finite real or complex array, shape (R,3)
    hbar : finite positive float

    Returns
    -------
    labels : integer array, shape (L,D)
    c : complex array, shape (L,3), unnormalized

    Raises
    ------
    ValueError
        For invalid or nonfinite shapes; q0 not SPD or symmetry error above 1e-12;
        negative, noninteger or degree>6 powers; or nonfinite/nonpositive hbar.
    """
    return labels, c

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_dipole_coefficients(q0, powers, coefficients, hbar):
    import numpy as np
    from itertools import product
    q0 = np.asarray(q0, dtype=float)
    raw = np.asarray(powers)
    coefficients = np.asarray(coefficients, dtype=complex)
    if (q0.ndim != 2 or q0.shape[0] == 0 or q0.shape[0] != q0.shape[1]
            or raw.ndim != 2 or raw.shape[0] == 0 or raw.shape[1] != q0.shape[0]
            or coefficients.shape != (raw.shape[0], 3)
            or not all(np.all(np.isfinite(a)) for a in (q0, raw, coefficients))
            or np.any(raw < 0) or np.any(raw != np.floor(raw))
            or not np.allclose(q0, q0.T, atol=1e-12, rtol=0)
            or np.min(np.linalg.eigvalsh(q0)) <= 0
            or not np.isfinite(hbar) or hbar <= 0):
        raise ValueError("Invalid dipole inputs")
    powers = raw.astype(int)
    degree = int(np.max(powers.sum(axis=1)))
    if degree > 6:
        raise ValueError("Dipole degree exceeds six")
    d = q0.shape[0]
    labels = np.array(sorted(
        (k for k in product(range(degree + 1), repeat=d) if sum(k) <= degree),
        key=lambda k: (sum(k), k),
    ), dtype=int)
    index = {tuple(k): i for i, k in enumerate(labels)}
    output = np.zeros((len(labels), 3), dtype=complex)
    scale = np.sqrt(hbar / 2.0) * q0
    vacuum = (0,) * d
    for alpha, polar_coefficients in zip(powers, coefficients):
        state = {vacuum: 1.0}
        for axis, exponent in enumerate(alpha):
            for _ in range(exponent):
                updated = {}
                for k, amplitude in state.items():
                    for mode in range(d):
                        up = list(k)
                        up[mode] += 1
                        up = tuple(up)
                        updated[up] = updated.get(up, 0.0) + amplitude * scale[axis, mode] * np.sqrt(k[mode] + 1)
                        if k[mode]:
                            down = list(k)
                            down[mode] -= 1
                            down = tuple(down)
                            updated[down] = updated.get(down, 0.0) + amplitude * scale[axis, mode] * np.sqrt(k[mode])
                state = updated
        for k, amplitude in state.items():
            output[index[k]] += amplitude * polar_coefficients
    return labels, output

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "name": 'case_1',
            "setup": """import numpy as np
data={'mass': [[1.2, 0.12, -0.08], [0.12, 1.5, 0.1], [-0.08, 0.1, 0.9]], 'ground_hessian': [[1.4, 0.23, -0.14], [0.23, 2.1, 0.17], [-0.14, 0.17, 0.95]], 'dipole_powers': [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [2, 0, 0], [1, 1, 0], [0, 1, 1], [0, 0, 2], [3, 0, 0], [1, 1, 1], [0, 2, 1]], 'dipole_coefficients': [[0.32, -0.18, 0.11], [0.75, 0.12, -0.24], [-0.28, 0.6, 0.16], [0.18, -0.32, 0.55], [0.2, -0.1, 0.08], [-0.17, 0.22, -0.12], [0.09, -0.16, 0.13], [-0.11, 0.07, 0.2], [0.065, -0.04, 0.015], [-0.045, 0.035, 0.055], [0.03, -0.05, 0.025]], 'hbar': 0.7}
Q,_,_=_oracle_initial_frame(data["mass"],data["ground_hessian"],data["hbar"])
""",
            "call": "dipole_coefficients(Q, data['dipole_powers'], data['dipole_coefficients'], data['hbar'])",
            "gold_call": "_oracle_dipole_coefficients(Q, data['dipole_powers'], data['dipole_coefficients'], data['hbar'])",
        },
        {
            "name": 'case_2',
            "setup": """import numpy as np
Q=np.array([[.9]]);powers=np.array([[0]]);a=np.array([[1.,-2.,.5]])
""",
            "call": 'dipole_coefficients(Q, powers, a, 0.7)',
            "gold_call": '_oracle_dipole_coefficients(Q, powers, a, 0.7)',
        },
        {
            "name": 'case_3',
            "setup": """import numpy as np
Q=np.eye(2);powers=np.array([[2,1],[2,1]]);a=np.array([[1.,2.,3.],[-1.,-2.,-3.]])
""",
            "call": 'dipole_coefficients(Q, powers, a, 1.0)',
            "gold_call": '_oracle_dipole_coefficients(Q, powers, a, 1.0)',
        },
        {
            "name": 'case_4',
            "setup": """import numpy as np
Q=np.array([[.8,.15],[.15,1.1]]);powers=np.array([[6,0],[1,3],[0,2]]);a=np.array([[.1,.03j,-.2],[-.2,.07,.1],[.3,-.1,.2]])
""",
            "call": 'dipole_coefficients(Q, powers, a, 0.8)',
            "gold_call": '_oracle_dipole_coefficients(Q, powers, a, 0.8)',
        },
        {
            "name": 'case_5',
            "setup": """import numpy as np
Q=np.eye(1);powers=np.array([[7]]);a=np.ones((1,3))

def _status(fn, *args):
    try:
        fn(*args)
    except ValueError:
        return 1
    raise AssertionError("Expected ValueError")
""",
            "call": '_status(dipole_coefficients, Q, powers, a, 1.0)',
            "gold_call": '_status(_oracle_dipole_coefficients, Q, powers, a, 1.0)',
        },
    ]
