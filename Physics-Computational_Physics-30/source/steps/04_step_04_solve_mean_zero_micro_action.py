"""
Weighted mean-zero micro-resolvent action.

Sections 5.5-5.8 Eqs. (62)-(79): set `T=sigma_t*M`, `S_m=solve(T,B_m-(sigma_t/epsilon)*M)`, `S=block_diag(S_m)`, `Q=(I_omega-ones(N_omega,1)*weights[None,:]) kron I_N`, and `K=stack_m(C_m-L_m)` vertically. Here `B_m=data[m,1]` is the full transport form, `C_m=data[m,2]` is the broken gradient map, and `lifting[m]` is L_m. Solve `(I+epsilon*Q*S*Q)Y=K` and return the ordered N-by-N blocks Y_m. The identity acts on the full angular space; never invert the singular projection Q. For consistent predecessor outputs, K and Y have zero weighted angular mean. Do not reverse the angular blocks at this stage: reversal acts on the returned solved action in the next step. All products are matrix products, and no intermediate is rounded.

Returns
-------
micro_action : np.ndarray, shape (N_omega, N, N), float The ordered blocks Y_m of the solved micro action. The micro matrix is required to be nonsingular.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def solve_mean_zero_micro_action(data: np.ndarray, lifting: np.ndarray, epsilon: float, sigma_t: float, omegas: np.ndarray, weights: np.ndarray) -> np.ndarray:
    '''Weighted mean-zero micro-resolvent action.

    Parameters
    ----------
    data : np.ndarray, shape (N_omega, 3, N, N)
        Directional data from step 02 in the supplied angular order. All mass
        slices are the same nonempty symmetric positive-definite matrix;
        transport slices are nonsingular and all entries are finite.
    lifting : np.ndarray, shape (N_omega, N, N)
        Finite lifting maps from step 03 on the same geometry and angles.
    epsilon, sigma_t : float
        Positive finite diffusive scale and unscaled total coefficient.
    omegas : np.ndarray, shape (N_omega, 3)
        Finite unit directions with the shared moments and equal-weight opposite pairs.
    weights : np.ndarray, shape (N_omega,)
        Positive finite angular weights, summing to one.

    Returns
    -------
    micro_action : np.ndarray, shape (N_omega, N, N), float
        The ordered blocks Y_m of the solved micro action. The micro matrix is
        required to be nonsingular.

    Raises
    ------
    ValueError
        If scalar ranges, matching finite array shapes, common positive-definite
        mass, angular moments or pairing, or nonsingular transport or micro
        matrix conditions are violated.
    '''
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_solve_mean_zero_micro_action(data: np.ndarray, lifting: np.ndarray, epsilon: float, sigma_t: float, omegas: np.ndarray, weights: np.ndarray) -> np.ndarray:
    import numpy as np

    def _real_array(value):
        """Convert real numeric entries without silently discarding imaginary parts."""
        from numbers import Number
        try:
            array = np.asarray(value)
            if array.dtype.kind == "O":
                if any(not isinstance(x, Number) or x.imag != 0 for x in array.flat):
                    raise ValueError("real numeric entries required")
                array = np.array([x.real for x in array.flat]).reshape(array.shape)
            elif array.dtype.kind not in "buifc" or np.any(array.imag != 0):
                raise ValueError("real numeric entries required")
            if array.dtype.kind == "c":
                array = array.real.copy(order="K")
            return np.asarray(array.real, dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("real numeric entries required") from exc

    def _require_positive(*values):
        normalized = []
        for value in values:
            scalar = _real_array(value)
            if (scalar.ndim != 0 or isinstance(np.asarray(value).item(), (bool, np.bool_))
                    or not np.isfinite(scalar) or scalar <= 0):
                raise ValueError("positive finite real scalars, excluding bool, required")
            # Preserve real scalar arithmetic; normalize real-valued complex/object storage.
            normalized.append(value if np.asarray(value).dtype.kind in "iuf" else float(scalar))
        return tuple(normalized)

    def _validate_angles(omegas, weights):
        om, wt = _real_array(omegas), _real_array(weights)
        if (om.ndim != 2 or om.shape[1] != 3 or len(om) == 0 or wt.shape != (len(om),)
                or not np.all(np.isfinite(om)) or not np.all(np.isfinite(wt)) or np.any(wt <= 0)):
            raise ValueError("three-dimensional positive angular quadrature required")
        close = lambda a, b: np.allclose(a, b, atol=1e-12, rtol=0)
        if (not close(np.sum(om * om, axis=1), 1) or not close(wt.sum(), 1)
                or not close(wt @ om, 0) or not close((om.T * wt) @ om, np.eye(3) / 3)):
            raise ValueError("unit directions and spherical moments required")
        todo, reverse = set(range(len(om))), np.empty(len(om), dtype=int)
        while todo:
            m = min(todo)
            todo.remove(m)
            matches = [j for j in sorted(todo) if close(om[j], -om[m]) and close(wt[j], wt[m])]
            if not matches:
                raise ValueError("equal-weight opposite pairs required")
            j = matches[0]
            todo.remove(j)
            reverse[m], reverse[j] = j, m
        return om, wt, reverse

    def _require_spd(matrix):
        a = _real_array(matrix)
        if (a.ndim != 2 or a.shape[0] != a.shape[1] or not len(a)
                or not np.all(np.isfinite(a)) or not np.allclose(a, a.T, atol=1e-12, rtol=1e-10)):
            raise ValueError("finite symmetric positive-definite matrix required")
        try:
            np.linalg.cholesky(a)
        except np.linalg.LinAlgError as exc:
            raise ValueError("positive-definite matrix required") from exc
        return a

    def _validate_directional_data(data, weights):
        data, wt = _real_array(data), _real_array(weights)
        if (data.ndim != 4 or data.shape[1] != 3 or data.shape[2] != data.shape[3]
                or data.shape[2] == 0 or len(data) == 0 or wt.shape != (len(data),)
                or not np.all(np.isfinite(data)) or not np.all(np.isfinite(wt))
                or np.any(wt <= 0) or not np.isclose(wt.sum(), 1, atol=1e-12, rtol=0)):
            raise ValueError("finite directional data and normalized positive weights required")
        mass = _require_spd(data[0, 0])
        if not np.allclose(data[:, 0], mass, atol=1e-12, rtol=0):
            raise ValueError("common mass matrix required")
        return data, wt

    def _micro_inputs(data, lifting, epsilon, sigma_t, omegas, weights):
        """Validate the common arrays and form T, S_m and K_m without solving for Y."""
        epsilon, sigma_t = _require_positive(epsilon, sigma_t)
        data, weights = _validate_directional_data(data, weights)
        omegas, weights, reverse = _validate_angles(omegas, weights)
        count, size = len(data), data.shape[2]
        if len(omegas) != count:
            raise ValueError("matching angular count required")
        lifting = _real_array(lifting)
        if lifting.shape != (count, size, size) or not np.all(np.isfinite(lifting)):
            raise ValueError("matching finite lifting maps required")
        mass, transports, gradients = data[0, 0], data[:, 1], data[:, 2]
        try:
            for transport in transports:
                np.linalg.solve(transport, np.eye(size))
        except np.linalg.LinAlgError as exc:
            raise ValueError("nonsingular directional forms required") from exc
        collision = sigma_t * mass
        streaming = np.stack([
            np.linalg.solve(collision, transport - collision / epsilon)
            for transport in transports
        ])
        coupling = gradients - lifting
        return mass, collision, streaming, coupling, weights, reverse

    def _solve_micro_action(data, lifting, epsilon, sigma_t, omegas, weights):
        epsilon, sigma_t = _require_positive(epsilon, sigma_t)
        mass, collision, streaming, coupling, weights, reverse = _micro_inputs(
            data, lifting, epsilon, sigma_t, omegas, weights
        )
        count, size = coupling.shape[:2]
        streaming_block = np.zeros((count * size, count * size))
        for m in range(count):
            block = slice(m * size, (m + 1) * size)
            streaming_block[block, block] = streaming[m]
        # Q projects out the weighted angular mean; it is never inverted.
        mean_zero = np.kron(
            np.eye(count) - np.ones((count, 1)) * weights, np.eye(size)
        )
        resolvent = np.eye(count * size) + epsilon * mean_zero @ streaming_block @ mean_zero
        try:
            solved = np.linalg.solve(resolvent, coupling.reshape(count * size, size))
        except np.linalg.LinAlgError as exc:
            raise ValueError("nonsingular micro block required") from exc
        return solved.reshape(count, size, size)

    return _solve_micro_action(data, lifting, epsilon, sigma_t, omegas, weights)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    imports = 'import numpy as np\n'
    angles = 'om=np.concatenate([np.eye(3),-np.eye(3),np.array([[x,y,z] for x in (-1.,1.) for y in (-1.,1.) for z in (-1.,1.)])/np.sqrt(3)]);wt=np.r_[np.full(6,1/12),np.full(8,1/16)]\n'
    parameters = 'n=12;seed=1;eps=1/768;st=1.\n'
    # Seeded stand-ins for the step 02 and 03 outputs: a common SPD mass, nonsingular transport forms
    # a gradient map odd in the direction and a lifting with an even part, and K has zero weighted angular mean.
    generator = ("def synthetic_inputs(n, om, st, eps, seed):\n"
                 "    rng = np.random.default_rng(seed)\n"
                 "    X = rng.standard_normal((n, n))\n"
                 "    M = X @ X.T / n + 0.02 * np.eye(n)\n"
                 "    E, P, G, H = rng.standard_normal((4, 3, n, n))\n"
                 "    V = np.einsum('mk,ml->mkl', om, om) - np.eye(3) / 3\n"
                 "    A = np.einsum('mk,kij->mij', om, E - E.transpose(0, 2, 1)) + np.einsum('mk,kij->mij', abs(om), P @ P.transpose(0, 2, 1) / n)\n"
                 "    B = st / eps * M + 0.4 * A\n"
                 "    data = np.stack([np.broadcast_to(M, B.shape), B, 20 * np.einsum('mk,kij->mij', om, G)], axis=1)\n"
                 "    return data, 20 * np.einsum('mk,kij->mij', om, H) + 10 * np.einsum('mkl,klij->mij', V, rng.standard_normal((3, 3, n, n)))\n")
    inputs = 'data,lifting=synthetic_inputs(n,om,st,eps,seed)\n'
    domain_wrapper = 'def check(f):\n try:\n  f();return 0\n except ValueError:return 1\n except Exception:return 2\n'
    step_call = 'solve_mean_zero_micro_action(data,lifting,eps,st,om,wt)'
    call_1 = '#case:normal\n' + step_call
    call_2 = '_oracle_' + step_call
    call_5 = '#case:edge\ncheck(lambda:' + step_call + ')'
    call_6 = 'check(lambda:_oracle_' + step_call + ')'
    base = imports + angles + parameters + generator
    return [
        {'setup': base + inputs, 'call': call_1, 'gold_call': call_2},
        {'setup': base + 'seed=2;eps=1/8192\n' + inputs, 'call': '#case:boundary\n' + step_call, 'gold_call': call_2},
        {'setup': base + 'n=4;seed=3;eps=1/128\n' + inputs, 'call': '#case:edge\n' + step_call, 'gold_call': call_2},
        {'setup': base + 'n=24;seed=4;eps=1/2048\n' + inputs, 'call': call_1, 'gold_call': call_2},
        {'setup': base + 'seed=5;eps=1/32;st=2.5\n' + inputs, 'call': call_1, 'gold_call': call_2},
        {'setup': base + 'seed=6;a=.23;b=.37;om=om@np.array([[np.cos(a),-np.sin(a),0.],[np.sin(a),np.cos(a),0.],[0.,0.,1.]])@np.array([[np.cos(b),0.,np.sin(b)],[0.,1.,0.],[-np.sin(b),0.,np.cos(b)]]);wt=np.full(14,1/14)\n' + inputs, 'call': call_1, 'gold_call': call_2},
        {'setup': base + 'n=4;seed=7\n' + inputs + 'eps=0\n' + domain_wrapper, 'call': call_5, 'gold_call': call_6},
        {'setup': base + 'n=4;seed=8\n' + inputs + 'data=data.copy();data[1,0]*=1.1\n' + domain_wrapper, 'call': call_5, 'gold_call': call_6},
    ]
