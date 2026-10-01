"""
Inside the modulated segment the field is written in the generalised Floquet form, a carrier at the driving frequency multiplied by a series in the modulation phase. Harmonic n of that series then sits at frequency omega + n omega_m and at wavenumber kappa + n kappa_m, where kappa is the single unknown shared by the whole series. Substituting that form into the equation of motion and collecting the terms belonging to each harmonic leaves one algebraic equation per retained harmonic, and the set of them is the eigenvalue problem this stage assembles.

The structure follows from where the wavenumber enters. The axial stress is the modulus times the gradient, and taking the gradient of the field multiplies harmonic n by its own wavenumber; multiplying by the modulus then mixes neighbouring harmonics through the coupling operator of the previous step; and taking the gradient once more multiplies the result at harmonic q by the wavenumber of harmonic q. The wavenumber therefore appears twice, once on each side of the coupling operator, and the operator on the left of the harmonic balance is the product of a diagonal of shifted wavenumbers, the coupling matrix, and a second diagonal of shifted wavenumbers. Writing the shifted wavenumber of harmonic n as kappa plus n kappa_m and expanding that product in powers of kappa gives a matrix polynomial of degree two, whose coefficient at the square of kappa is the coupling matrix itself, whose coefficient at the first power is the sum of the coupling matrix with the order diagonal on either side, and whose constant term carries the order diagonal on both sides less the inertia.

Two features of the result matter later. The polynomial is quadratic in the wavenumber because the wavenumber enters the harmonic balance twice, once on each side of the coupling operator. Posing the same medium the other way round, with the wavenumber prescribed and the frequency sought, does not give a simpler problem: the inertia term carries the shifted frequencies, and writing that diagonal as omega times the identity plus omega_m times the order diagonal leaves a polynomial that is quadratic in omega and whose coefficient of the first power of omega is minus twice rho0 omega_m times the order diagonal, which does not vanish. The two postures are different problems rather than one problem in two guises, and which is appropriate is decided by the physics of the bounded segment, whose spatial interfaces prescribe frequency. And every coefficient matrix is real and symmetric, the coupling matrix by construction and the order diagonals because they commute with transposition in the combination that appears here, which is what later permits the group velocity to be obtained from the same eigenvector on both sides without a separate left eigenproblem.

The sign of kappa_m is not restricted. Reversing it reverses the direction in which the modulus travels, which is exactly how the second direction of incidence is posed.

Returns
-------
dict holding float64 arrays a2, a1 and a0, each of shape (2 n_order + 1, 2 n_order + 1), being the coefficient matrices of the quadratic matrix polynomial in the wavenumber, ordered from the square of the wavenumber down to the constant term.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def floquet_quadratic_operator(
    omega: float,
    coupling: np.ndarray,
    rho0: float,
    kappa_m: float,
    omega_m: float,
) -> dict:
    """Assemble the quadratic matrix polynomial in the wavenumber for the modulated segment at a prescribed frequency.

    Parameters
    ----------
    omega : float
        Driving angular frequency in radians per second, finite and above zero.
    coupling : np.ndarray
        Harmonic coupling operator of the modulus, shape (2 n_order + 1, 2 n_order + 1).
    rho0 : float
        Mass density in kilogram per cubic metre, above zero.
    kappa_m : float
        Modulation wavenumber in radians per metre, finite and not zero.
    omega_m : float
        Modulation angular frequency in radians per second, finite and not zero.

    Returns
    -------
    dict
        Under the keys a2, a1 and a0, each a float64 array of shape
        (2 n_order + 1, 2 n_order + 1), the coefficient matrices of the quadratic
        matrix polynomial in the wavenumber from the square term down to the
        constant term.

    Raises
    ------
    ValueError
        If coupling is not a square array of odd side length or is not finite, or if
        omega, rho0, kappa_m or omega_m is not finite or violates its stated sign.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_floquet_quadratic_operator(
    omega: float,
    coupling: np.ndarray,
    rho0: float,
    kappa_m: float,
    omega_m: float,
) -> dict:
    coupling = np.asarray(coupling, dtype=np.float64)
    if coupling.ndim != 2 or coupling.shape[0] != coupling.shape[1]:
        raise ValueError("coupling must be a square matrix")
    if coupling.shape[0] % 2 != 1:
        raise ValueError("coupling must have an odd side length")
    if not np.all(np.isfinite(coupling)):
        raise ValueError("coupling must be finite")
    omega = float(omega)
    rho0 = float(rho0)
    kappa_m = float(kappa_m)
    omega_m = float(omega_m)
    if not np.isfinite(omega) or omega <= 0.0:
        raise ValueError("omega must be finite and above zero")
    if not np.isfinite(rho0) or rho0 <= 0.0:
        raise ValueError("rho0 must be finite and above zero")
    if not np.isfinite(kappa_m) or kappa_m == 0.0:
        raise ValueError("kappa_m must be finite and not zero")
    if not np.isfinite(omega_m) or omega_m == 0.0:
        raise ValueError("omega_m must be finite and not zero")

    n_order = (coupling.shape[0] - 1) // 2
    orders = np.arange(-n_order, n_order + 1, dtype=np.float64)
    shift = np.diag(orders * kappa_m)
    inertia = np.diag((omega + orders * omega_m) ** 2)

    a2 = coupling.copy()
    a1 = shift @ coupling + coupling @ shift
    a0 = shift @ coupling @ shift - rho0 * inertia
    return {"a2": a2, "a1": a1, "a0": a0}

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np\nC = np.array([[1.0, 0.15, 0.0], [0.15, 1.0, 0.15], [0.0, 0.15, 1.0]])\ndef pack(d):\n    arrays = [np.asarray(d[k]) for k in ('a2', 'a1', 'a0')]\n    shape = np.array([value for array in arrays for value in array.shape], dtype=float)\n    return np.concatenate([shape] + [array.real.ravel() for array in arrays] + [array.imag.ravel() for array in arrays])\n",
            "call": 'pack(floquet_quadratic_operator(15.0,np.array(C, copy=True), 1.0, 10.0, 20.0))',
            "gold_call": 'pack(_oracle_floquet_quadratic_operator(15.0,np.array(C, copy=True), 1.0, 10.0, 20.0))',
        },
        {
            "setup": "import numpy as np\nC = np.array([[1.0, 0.15, 0.0], [0.15, 1.0, 0.15], [0.0, 0.15, 1.0]])\ndef pack(d):\n    arrays = [np.asarray(d[k]) for k in ('a2', 'a1', 'a0')]\n    shape = np.array([value for array in arrays for value in array.shape], dtype=float)\n    return np.concatenate([shape] + [array.real.ravel() for array in arrays] + [array.imag.ravel() for array in arrays])\n",
            "call": 'pack(floquet_quadratic_operator(15.0,np.array(C, copy=True), 1.0, -10.0, 20.0))',
            "gold_call": 'pack(_oracle_floquet_quadratic_operator(15.0,np.array(C, copy=True), 1.0, -10.0, 20.0))',
        },
        {
            "setup": "import numpy as np\nI = np.eye(1)\ndef pack(d):\n    arrays = [np.asarray(d[k]) for k in ('a2', 'a1', 'a0')]\n    shape = np.array([value for array in arrays for value in array.shape], dtype=float)\n    return np.concatenate([shape] + [array.real.ravel() for array in arrays] + [array.imag.ravel() for array in arrays])\n",
            "call": 'pack(floquet_quadratic_operator(4.0,np.array(I, copy=True), 2.0, 1.0, 3.0))',
            "gold_call": 'pack(_oracle_floquet_quadratic_operator(4.0,np.array(I, copy=True), 2.0, 1.0, 3.0))',
        },
        {
            "setup": "import numpy as np\nC = np.eye(3)\nBAD = np.ones((3, 4))\nEVEN = np.eye(4)\nNAN = np.array([[1.0, 0.0, 0.0], [0.0, float('nan'), 0.0], [0.0, 0.0, 1.0]])\ndef verdict(fn, c=C, w=15.0, r=1.0, km=10.0, wm=20.0):\n    try:\n        fn(w, np.array(c, copy=True), r, km, wm)\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n    return 0\n",
            "call": '(verdict(floquet_quadratic_operator, c=BAD), verdict(floquet_quadratic_operator, c=EVEN), verdict(floquet_quadratic_operator, c=NAN), verdict(floquet_quadratic_operator, w=0.0), verdict(floquet_quadratic_operator, r=-1.0), verdict(floquet_quadratic_operator, km=0.0), verdict(floquet_quadratic_operator))',
            "gold_call": '(verdict(_oracle_floquet_quadratic_operator, c=BAD), verdict(_oracle_floquet_quadratic_operator, c=EVEN), verdict(_oracle_floquet_quadratic_operator, c=NAN), verdict(_oracle_floquet_quadratic_operator, w=0.0), verdict(_oracle_floquet_quadratic_operator, r=-1.0), verdict(_oracle_floquet_quadratic_operator, km=0.0), verdict(_oracle_floquet_quadratic_operator))',
        },
    ]
