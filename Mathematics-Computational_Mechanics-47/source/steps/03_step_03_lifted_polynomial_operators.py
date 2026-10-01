"""
Assemble the constant, linear, quadratic, input, and bilinear operators of the lifted at-most-quadratic system.

Adjoining the pointwise square of the temperature as a second state block

removes the cubic sink and leaves a system whose semi-discrete right-hand side

is a fixed polynomial of degree two in the joint state.

Returns
-------
tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] holding the constant vector of shape (2n,), the linear operator of shape (2n, 2n), the quadratic operator of shape (2n, 4n**2), the input operator of shape (2n, 2), and the bilinear operator of shape (2n, 4n).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

from numbers import Integral, Real

import math
import numpy as np


def lifted_polynomial_operators(
    n_nodes: int,
    domain_length: float,
    diffusivity: float,
    boundary_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the operators of the lifted quadratic system.

    The joint state is x = [q; w], where q is the temperature on the n
    interior nodes x_i = i * dx of (0, domain_length), dx = domain_length /
    (n + 1), and w is the auxiliary field defined pointwise by w = q * q. The
    operators must satisfy

        dx/dt = C + A x + F (x kron x) + B u + N (u kron x)

    for every state that lies on w = q * q, where the temperature block
    reproduces the same central-difference diffusion, cubic sink and two-lobe
    source used by the unlifted model, the auxiliary block is the exact time
    derivative implied by the definition of w, both blocks impose
    boundary_values[0] at x = 0 and boundary_values[1] at x = domain_length
    through ghost nodes, and

        u(t) = [a * sin(2 * pi * t), b * sin(4 * pi * t)]

    multiplies the spatial profiles 1 / (1 + 100 * (x / L - 1 / 4)**2) and
    1 / (1 + 100 * (x / L - 3 / 4)**2) respectively, with L = domain_length.

    Storage conventions. Columns of F follow numpy's Kronecker layout, so the
    monomial x_p * x_q occupies column p * (2 n) + q with zero-based p and q;
    each monomial is placed entirely in the single column whose first index is
    the smaller of the two, leaving every column with p > q equal to zero.
    Columns of N likewise follow numpy's layout, so u_k * x_p occupies column
    k * (2 n) + p.

    Parameters
    ----------
    n_nodes : int
        Positive number of interior spatial nodes.
    domain_length : float
        Finite strictly positive length of the spatial domain.
    diffusivity : float
        Finite strictly positive diffusion coefficient.
    boundary_values : np.ndarray
        Finite real array of shape (2,) holding the Dirichlet values at
        x = 0 and x = domain_length, in that order.

    Returns
    -------
    constant_operator : np.ndarray
        Float array of shape (2 n,).
    linear_operator : np.ndarray
        Float array of shape (2 n, 2 n).
    quadratic_operator : np.ndarray
        Float array of shape (2 n, 4 n**2).
    input_operator : np.ndarray
        Float array of shape (2 n, 2).
    bilinear_operator : np.ndarray
        Float array of shape (2 n, 4 n).

    Raises
    ------
    ValueError
        If n_nodes is not a positive integer, if boundary_values has the wrong
        shape or a non-finite entry, or if domain_length or diffusivity is not
        finite and positive.
    """
    return (
        np.empty(0, dtype=float),
        np.empty((0, 0), dtype=float),
        np.empty((0, 0), dtype=float),
        np.empty((0, 0), dtype=float),
        np.empty((0, 0), dtype=float),
    )

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_lifted_polynomial_operators(
    n_nodes: int,
    domain_length: float,
    diffusivity: float,
    boundary_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reference implementation."""
    from numbers import Integral, Real

    import math
    import numpy as np

    def _is_number(value) -> bool:
        return isinstance(value, Real) and not isinstance(value, bool)

    def _is_integer(value) -> bool:
        return isinstance(value, Integral) and not isinstance(value, bool)

    if not _is_integer(n_nodes) or int(n_nodes) < 1:
        raise ValueError("n_nodes must be a positive integer")
    bounds = np.asarray(boundary_values, dtype=float)
    if bounds.ndim != 1 or bounds.shape[0] != 2 or not np.all(np.isfinite(bounds)):
        raise ValueError("boundary_values must be a finite 1-D array of length 2")
    if not _is_number(domain_length) or not math.isfinite(float(domain_length)) or float(domain_length) <= 0.0:
        raise ValueError("domain_length must be a finite positive real scalar")
    if not _is_number(diffusivity) or not math.isfinite(float(diffusivity)) or float(diffusivity) <= 0.0:
        raise ValueError("diffusivity must be a finite positive real scalar")

    count = int(n_nodes)
    width = 2 * count
    length = float(domain_length)
    spacing = length / (count + 1.0)
    scale = float(diffusivity) / spacing ** 2
    scaled = spacing * np.arange(1, count + 1, dtype=float) / length
    profile = np.stack([1.0 / (1.0 + 100.0 * (scaled - 0.25) ** 2),
                        1.0 / (1.0 + 100.0 * (scaled - 0.75) ** 2)], axis=1)

    constant = np.zeros(width, dtype=float)
    linear = np.zeros((width, width), dtype=float)
    quadratic = np.zeros((width, width * width), dtype=float)
    inputs = np.zeros((width, 2), dtype=float)
    bilinear = np.zeros((width, 2 * width), dtype=float)

    def _deposit(row: int, first: int, second: int, value: float) -> None:
        low, high = (first, second) if first <= second else (second, first)
        quadratic[row, low * width + high] += value

    for node in range(count):
        upper = count + node
        # Temperature block: diffusion, ghost values, sink written as -q*w, source.
        linear[node, node] += -2.0 * scale
        if node - 1 >= 0:
            linear[node, node - 1] += scale
        else:
            constant[node] += scale * bounds[0]
        if node + 1 <= count - 1:
            linear[node, node + 1] += scale
        else:
            constant[node] += scale * bounds[1]
        _deposit(node, node, upper, -1.0)
        inputs[node, :] = profile[node, :]
        # Auxiliary block: twice the temperature times the temperature equation.
        _deposit(upper, node, node, -4.0 * scale)
        if node - 1 >= 0:
            _deposit(upper, node, node - 1, 2.0 * scale)
        else:
            linear[upper, node] += 2.0 * scale * bounds[0]
        if node + 1 <= count - 1:
            _deposit(upper, node, node + 1, 2.0 * scale)
        else:
            linear[upper, node] += 2.0 * scale * bounds[1]
        _deposit(upper, upper, upper, -2.0)
        for channel in range(2):
            bilinear[upper, channel * width + node] += 2.0 * profile[node, channel]

    return constant, linear, quadratic, inputs, bilinear

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return a list of test case specifications."""
    reducer = (
        "import numpy as np\n"
        "def _sig(value, scale):\n"
        "    a = np.asarray(value, dtype=float)\n"
        "    b = np.concatenate([np.asarray([a.ndim, *a.shape], dtype=float), a.ravel()])\n"
        "    k = np.arange(1.0, b.size + 1.0)\n"
        "    return float((np.sum(np.abs(b)) + np.sum(b * np.cos(k))) / scale)\n"
        "def _sigt(parts, scale):\n"
        "    return float(sum((i + 1.0) * _sig(p, 1.0) for i, p in enumerate(parts)) / scale)\n"
        "def _rate(ops, state, moment, amps):\n"
        "    c, a, f, b, nn = ops\n"
        "    u = np.array([amps[0] * np.sin(2.0 * np.pi * moment),\n"
        "                  amps[1] * np.sin(4.0 * np.pi * moment)])\n"
        "    return c + a @ state + f @ np.kron(state, state) + b @ u + nn @ np.kron(u, state)\n"
    )
    return [
        # Case 1: full operator signature on a small grid.
        {
            "setup": reducer + (
                "n_nodes = 4\n"
                "domain_length = 1.0\n"
                "diffusivity = 0.005\n"
                "boundary_values = np.array([0.0, 1.0])\n"
            ),
            "call": (
                "_sigt(lifted_polynomial_operators(n_nodes, domain_length, diffusivity, "
                "boundary_values), 1e3)"
            ),
            "gold_call": (
                "_sigt(_oracle_lifted_polynomial_operators(n_nodes, domain_length, diffusivity, "
                "boundary_values), 1e3)"
            ),
        },
        # Case 2: a single interior node, where both ghost values act on one stencil.
        {
            "setup": reducer + (
                "n_nodes = 1\n"
                "domain_length = 2.0\n"
                "diffusivity = 0.02\n"
                "boundary_values = np.array([-0.5, 1.25])\n"
            ),
            "call": (
                "_sigt(lifted_polynomial_operators(n_nodes, domain_length, diffusivity, "
                "boundary_values), 1e0)"
            ),
            "gold_call": (
                "_sigt(_oracle_lifted_polynomial_operators(n_nodes, domain_length, diffusivity, "
                "boundary_values), 1e0)"
            ),
        },
        # Case 3: asymmetric boundary data with a coarser grid.
        {
            "setup": reducer + (
                "n_nodes = 5\n"
                "domain_length = 1.5\n"
                "diffusivity = 0.05\n"
                "boundary_values = np.array([0.3, -0.8])\n"
            ),
            "call": (
                "_sigt(lifted_polynomial_operators(n_nodes, domain_length, diffusivity, "
                "boundary_values), 1e3)"
            ),
            "gold_call": (
                "_sigt(_oracle_lifted_polynomial_operators(n_nodes, domain_length, diffusivity, "
                "boundary_values), 1e3)"
            ),
        },
        # --- Decisive: on the lifting manifold the auxiliary block must equal
        #     twice the temperature times the temperature block, and the
        #     temperature block must reproduce the unlifted right-hand side.
        #     A boundary term booked as a constant instead of a linear term, or
        #     a sink written with the wrong pair of state indices, breaks this.
        {
            "setup": reducer + (
                "def consistent():\n"
                "    n_nodes, dl, kap = 6, 1.0, 0.005\n"
                "    bv = np.array([0.4, 1.3])\n"
                "    amps = np.array([1.5, 0.5])\n"
                "    ops = lifted_polynomial_operators(n_nodes, dl, kap, bv)\n"
                "    rng = np.random.default_rng(5)\n"
                "    q = 0.4 + 0.6 * rng.random(n_nodes)\n"
                "    st = np.concatenate([q, q * q])\n"
                "    r = _rate(ops, st, 0.37, amps)\n"
                "    dx = dl / (n_nodes + 1.0)\n"
                "    s = dx * np.arange(1, n_nodes + 1) / dl\n"
                "    lap = np.empty(n_nodes)\n"
                "    lap[0] = bv[0] - 2.0 * q[0] + q[1]\n"
                "    lap[1:-1] = q[:-2] - 2.0 * q[1:-1] + q[2:]\n"
                "    lap[-1] = q[-2] - 2.0 * q[-1] + bv[1]\n"
                "    src = (amps[0] * np.sin(2.0 * np.pi * 0.37) / (1.0 + 100.0 * (s - 0.25) ** 2)\n"
                "           + amps[1] * np.sin(4.0 * np.pi * 0.37) / (1.0 + 100.0 * (s - 0.75) ** 2))\n"
                "    expected = kap / dx ** 2 * lap - q ** 3 + src\n"
                "    ok_q = np.max(np.abs(r[:n_nodes] - expected)) < 1e-10\n"
                "    ok_w = np.max(np.abs(r[n_nodes:] - 2.0 * q * r[:n_nodes])) < 1e-10\n"
                "    return int(ok_q) + 2 * int(ok_w)\n"
            ),
            "call": "consistent()",
            "gold_call": "3",
        },
        # --- Decisive: the quadratic operator must be stored upper-triangular in
        #     the index pair, and the bilinear operator must act on the input
        #     Kronecker state rather than on the state Kronecker input.
        {
            "setup": reducer + (
                "def layout():\n"
                "    n_nodes = 4\n"
                "    width = 2 * n_nodes\n"
                "    ops = lifted_polynomial_operators(n_nodes, 1.0, 0.005, np.array([0.0, 1.0]))\n"
                "    f = ops[2]\n"
                "    nn = ops[4]\n"
                "    lower = 0.0\n"
                "    for p in range(width):\n"
                "        for q in range(p):\n"
                "            lower += float(np.sum(np.abs(f[:, p * width + q])))\n"
                "    u = np.array([1.0, 0.0])\n"
                "    st = np.concatenate([np.ones(n_nodes), np.zeros(n_nodes)])\n"
                "    first = nn @ np.kron(u, st)\n"
                "    swapped = nn @ np.kron(st, u)[:nn.shape[1]]\n"
                "    return int(lower == 0.0) + 2 * int(np.max(np.abs(first - swapped)) > 1e-8)\n"
            ),
            "call": "layout()",
            "gold_call": "3",
        },
        # Case 6: invalid node count.
        {
            "setup": reducer + (
                "def run_model():\n"
                "    try:\n"
                "        lifted_polynomial_operators(0, 1.0, 0.005, np.array([0.0, 1.0]))\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "def run_gold():\n"
                "    try:\n"
                "        _oracle_lifted_polynomial_operators(0, 1.0, 0.005, np.array([0.0, 1.0]))\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2"
            ),
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # Case 7: invalid boundary_values shape.
        {
            "setup": reducer + (
                "def run_model():\n"
                "    try:\n"
                "        lifted_polynomial_operators(3, 1.0, 0.005, np.array([0.0]))\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "def run_gold():\n"
                "    try:\n"
                "        _oracle_lifted_polynomial_operators(3, 1.0, 0.005, np.array([0.0]))\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2"
            ),
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
