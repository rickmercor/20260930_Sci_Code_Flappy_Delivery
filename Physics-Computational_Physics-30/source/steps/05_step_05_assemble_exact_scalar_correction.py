"""
Reversal-weighted exact scalar correction form.

Section 5.7 Eq. (75), with the mean jump form of Eq. (22): use the supplied micro_action as Y, set `T=sigma_t*M`, `S_m=solve(T,B_m-(sigma_t/epsilon)*M)`, `K_m=C_m-L_m`, and `J=T*sum_m weights[m]*S_m`. Assemble `A_exact=J+epsilon*sigma_a*M-epsilon*sum_m weights[m]*K_m.T*T*Y_reverse(m)`. Equivalently, the last term is `epsilon*K.T*W*R*Y` with `W=diag(weights) kron T` and R the equal-weight angular reversal. Apply reversal to the supplied Y blocks in their supplied angular order and return the N-by-N form. With repeated equal-weight copies, pair each lowest unpaired index with the lowest eligible opposite index.

Returns
-------
A_exact : np.ndarray, shape (N, N), float The reversal-weighted exact scalar correction form.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def assemble_exact_scalar_correction(data: np.ndarray, lifting: np.ndarray, micro_action: np.ndarray, epsilon: float, sigma_t: float, sigma_a: float, omegas: np.ndarray, weights: np.ndarray) -> np.ndarray:
    '''Reversal-weighted exact scalar correction form.

    Parameters
    ----------
    data : np.ndarray, shape (N_omega, 3, N, N)
        Directional data from step 02 in the supplied angular order. All mass
        slices are the same nonempty symmetric positive-definite matrix;
        transport slices are nonsingular and all entries are finite.
    lifting : np.ndarray, shape (N_omega, N, N)
        Finite lifting maps from step 03 on the same geometry and angles.
    micro_action : np.ndarray, shape (N_omega, N, N)
        Finite ordered Y_m blocks in the same basis and angular order as data.
        Assemble with the supplied values.
    epsilon : float
        Positive finite diffusive scale, less than sqrt(sigma_t/sigma_a).
    sigma_t, sigma_a : float
        Positive finite unscaled total and absorption coefficients.
    omegas : np.ndarray, shape (N_omega, 3)
        Finite unit directions with the shared moments and equal-weight opposite pairs.
    weights : np.ndarray, shape (N_omega,)
        Positive finite angular weights, summing to one.

    Returns
    -------
    A_exact : np.ndarray, shape (N, N), float
        The reversal-weighted exact scalar correction form.

    Raises
    ------
    ValueError
        If scalar ranges, matching finite array shapes, common positive-definite
        mass, angular moments or pairing, or nonsingular transport conditions
        are violated.
    '''
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_assemble_exact_scalar_correction(data: np.ndarray, lifting: np.ndarray, micro_action: np.ndarray, epsilon: float, sigma_t: float, sigma_a: float, omegas: np.ndarray, weights: np.ndarray) -> np.ndarray:
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

    def _exact_correction(data, lifting, micro_action, epsilon, sigma_t, sigma_a, omegas, weights):
        epsilon, sigma_t, sigma_a = _require_positive(epsilon, sigma_t, sigma_a)
        if epsilon >= np.sqrt(sigma_t / sigma_a):
            raise ValueError("positive scattering required")
        mass, collision, streaming, coupling, weights, reverse = _micro_inputs(
            data, lifting, epsilon, sigma_t, omegas, weights
        )
        micro_action = _real_array(micro_action)
        if micro_action.shape != coupling.shape or not np.all(np.isfinite(micro_action)):
            raise ValueError("matching finite micro-action blocks required")
        exact = collision @ np.einsum("m,mij->ij", weights, streaming) + epsilon * sigma_a * mass
        # Reverse the solved action, not K before the resolvent. Supplied Y enters this product.
        exact -= epsilon * sum(
            weights[m] * coupling[m].T @ collision @ micro_action[reverse[m]]
            for m in range(len(weights))
        )
        return exact

    return _exact_correction(data, lifting, micro_action, epsilon, sigma_t, sigma_a, omegas, weights)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    imports = 'import numpy as np\n'
    angles = 'om=np.concatenate([np.eye(3),-np.eye(3),np.array([[x,y,z] for x in (-1.,1.) for y in (-1.,1.) for z in (-1.,1.)])/np.sqrt(3)]);wt=np.r_[np.full(6,1/12),np.full(8,1/16)]\n'
    parameters = 'n=12;seed=1;eps=1/768;st=1.;sa=1.\n'
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
    # The supplied Y is drawn independently of the data, so a recomputed action cannot pass.
    micro = 'micro_action=20*np.random.default_rng(seed+100).standard_normal(lifting.shape)\n'
    step_call = 'assemble_exact_scalar_correction(data,lifting,micro_action,eps,st,sa,om,wt)'
    call_1 = '#case:normal\n' + step_call
    call_2 = '_oracle_' + step_call
    call_5 = '#case:edge\ncheck(lambda:' + step_call + ')'
    call_6 = 'check(lambda:_oracle_' + step_call + ')'
    base = imports + angles + parameters + generator
    return [
        {'setup': base + inputs + micro, 'call': call_1, 'gold_call': call_2},
        {'setup': base + 'seed=2;eps=1/8192\n' + inputs + micro, 'call': '#case:boundary\n' + step_call, 'gold_call': call_2},
        {'setup': base + 'n=4;seed=3;eps=1/128\n' + inputs + micro, 'call': '#case:edge\n' + step_call, 'gold_call': call_2},
        {'setup': base + 'n=24;seed=4;eps=1/2048\n' + inputs + micro, 'call': call_1, 'gold_call': call_2},
        {'setup': base + 'seed=5;eps=1/32;st=2.5;sa=.7\n' + inputs + micro, 'call': call_1, 'gold_call': call_2},
        {'setup': base + 'seed=6;a=.23;b=.37;om=om@np.array([[np.cos(a),-np.sin(a),0.],[np.sin(a),np.cos(a),0.],[0.,0.,1.]])@np.array([[np.cos(b),0.,np.sin(b)],[0.,1.,0.],[-np.sin(b),0.,np.cos(b)]]);wt=np.full(14,1/14)\n' + inputs + micro, 'call': call_1, 'gold_call': call_2},
        {'setup': base + inputs + micro + 'micro_action=.9*micro_action\n', 'call': call_1, 'gold_call': call_2},
        {'setup': base + inputs + micro + 'order=np.array([6,0,10,3,8,1,12,4,7,2,11,5,9,13]);om=om[order];wt=wt[order]\ndata=data[order];lifting=lifting[order];micro_action=micro_action[order]\n', 'call': call_1, 'gold_call': call_2},
        {'setup': base + 'n=4;seed=9\n' + inputs + micro + 'sa=-1\n' + domain_wrapper, 'call': call_5, 'gold_call': call_6},
        {'setup': base + 'n=4;seed=10\n' + inputs + micro + 'micro_action=micro_action[:,:-1,:-1]\n' + domain_wrapper, 'call': call_5, 'gold_call': call_6},
    ]
