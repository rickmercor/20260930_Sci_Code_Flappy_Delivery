"""
Differentiate the quantum covariance and pair response along a matrix and thermal perturbation.

At fixed physical coordinates, let $\Phi(t)=\Phi+tE$ and

$\theta(t)=\theta+t\tau$, with positive-definite $\Phi$ and symmetric $E$.

The covariance is $\Psi(\Phi,\theta)$ and the symmetric-pair response obeys

$\operatorname{pack}(D_\Phi\Psi[F])=2\Lambda\operatorname{pack}(F)$

for every symmetric $F$.

Return the derivatives



$$

\dot\Psi=\left.\frac{d}{dt}\Psi(\Phi(t),\theta(t))\right|_{t=0},

\qquad

\dot\Lambda=\left.\frac{d}{dt}\Lambda(\Phi(t),\theta(t))\right|_{t=0}.

$$



The matrix direction need not commute with $\Phi$; changes in mode polarizations

therefore contribute to both derivatives.

Repeated and nearly repeated squared frequencies require continuous limits,

including when three spectral arguments coincide in the second matrix response.

Pair coordinates retain the fixed physical-frame Frobenius-orthonormal basis.

At $\theta=0$, use the right thermal derivative of the gapped quantum covariance;

a nonnegative $\tau$ is required and the thermal contribution vanishes.

Return the exact directional derivative to the stated numerical tolerance;

a converged equivalent method is acceptable, with no prescribed differencing step.

Returns
-------
Two float64 symmetric matrices giving the covariance and pair-response directional derivatives.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def quantum_response_tangent(
    auxiliary: np.ndarray,
    temperature: float,
    auxiliary_direction: np.ndarray,
    temperature_direction: float,
) -> tuple:
    r"""Differentiate the quantum covariance and pair response along a matrix and thermal perturbation.

    Parameters
    ----------
    auxiliary : np.ndarray, shape (d, d)
        Finite real symmetric positive-definite $\Phi$, with $1\le d\le5$.
    temperature : float
        Finite nonnegative thermal energy $\theta$.
    auxiliary_direction : np.ndarray, shape (d, d)
        Finite real symmetric direction $E$, not necessarily commuting with $\Phi$.
    temperature_direction : float
        Finite signed thermal direction $\tau$, nonnegative when $\theta=0$.

    Returns
    -------
    covariance_direction : np.ndarray, shape (d, d)
        Symmetric derivative $\dot\Psi$ in the original frame.
    pair_direction : np.ndarray, shape (p, p)
        Symmetric derivative $\dot\Lambda$, with $p=d(d+1)/2$.

    Raises
    ------
    ValueError
        If shapes, finiteness, symmetry, positive definiteness or thermal directions
        violate the documented domain.
    RuntimeError
        If a finite directional response cannot be resolved.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from decimal import Context, Decimal, localcontext


def _scalar_direction(value, name):
    value = _real(value, name)
    if value.ndim != 0:
        raise ValueError(f"{name} must be a scalar")
    return float(value)


def _oracle_quantum_response_tangent(
    auxiliary: np.ndarray,
    temperature: float,
    auxiliary_direction: np.ndarray,
    temperature_direction: float,
) -> tuple:
    auxiliary = _symmetric(auxiliary, "auxiliary", positive=True)
    direction = _symmetric(auxiliary_direction, "auxiliary_direction")
    if len(auxiliary) > 5 or direction.shape != auxiliary.shape:
        raise ValueError(
            "auxiliary dimension must be one to five and directions must match"
        )
    temperature = _temperature(temperature)
    rate = _scalar_direction(temperature_direction, "temperature_direction")
    if temperature == 0 and rate < 0:
        raise ValueError(
            "a zero-temperature directional path cannot enter negative temperature"
        )
    values, vectors = np.linalg.eigh(auxiliary)
    basis = _basis(len(auxiliary))
    rotated_basis = np.einsum("ji,ajk,kl->ail", vectors, basis, vectors)
    rotated_direction = vectors.T @ direction @ vectors
    with localcontext(Context(prec=80)):
        thermal = Decimal.from_float(temperature)
        nodes = [Decimal.from_float(float(value)) for value in values]

        def _scalar_values(x):
            omega = x.sqrt()
            if temperature == 0:
                occupation = Decimal(0)
            else:
                ratio = omega / thermal
                decay = (-ratio).exp() if ratio < 100000 else Decimal(0)
                occupation = decay / (1 - decay)
            factor = 1 + 2 * occupation
            product = occupation * (1 + occupation)
            value = factor / (2 * omega)
            first_value = -factor / (4 * omega**3)
            second_value = 3 * factor / (8 * omega**5)
            thermal_value = Decimal(0)
            thermal_first_value = Decimal(0)
            if temperature > 0:
                first_value -= product / (2 * thermal * omega**2)
                second_value += 3 * product / (4 * thermal * omega**4)
                second_value += product * factor / (4 * thermal**2 * omega**3)
                thermal_value = product / thermal**2
                thermal_first_value = -product * factor / (2 * thermal**3 * omega)
            return value, first_value, second_value, thermal_value, thermal_first_value

        scalar_values = {x: _scalar_values(x) for x in nodes}

        def _first(x, y, thermal_only=False):
            if x == y:
                return scalar_values[x][4 if thermal_only else 1]
            index = 3 if thermal_only else 0
            return (scalar_values[y][index] - scalar_values[x][index]) / (y - x)

        def _second(x, y, z):
            x, y, z = sorted((x, y, z))
            if x == z:
                return scalar_values[x][2] / 2
            return (_first(y, z) - _first(x, y)) / (z - x)

        dimension = len(values)
        first = np.empty((dimension, dimension))
        thermal_first = np.empty_like(first)
        second = np.empty((dimension, dimension, dimension))
        thermal_values = np.asarray([float(scalar_values[x][3]) for x in nodes])
        for i in range(dimension):
            for j in range(dimension):
                first[i, j] = float(_first(nodes[i], nodes[j]))
                thermal_first[i, j] = float(_first(nodes[i], nodes[j], True))
                for k in range(dimension):
                    second[i, k, j] = float(_second(nodes[i], nodes[k], nodes[j]))
    covariance_direction = (
        vectors
        @ (first * rotated_direction + rate * np.diag(thermal_values))
        @ vectors.T
    )
    columns = []
    for entry in rotated_basis:
        action = np.einsum("ikj,ik,kj->ij", second, rotated_direction, entry)
        action += np.einsum("ikj,ik,kj->ij", second, entry, rotated_direction)
        action += rate * thermal_first * entry
        columns.append(0.5 * np.einsum("aij,ij->a", rotated_basis, action))
    pair_direction = np.column_stack(columns)
    if not np.all(np.isfinite(covariance_direction)) or not np.all(
        np.isfinite(pair_direction)
    ):
        raise RuntimeError("quantum response tangent is not finite")
    return (covariance_direction + covariance_direction.T) * 0.5, (
        pair_direction + pair_direction.T
    ) * 0.5

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return deterministic normal, boundary and declared invalid-input cases."""
    return [
        {
            "setup": "import numpy as np\nauxiliary = np.array([[.9,.13,-.07],[.13,1.4,.18],[-.07,.18,2.2]])\ntemperature = .7\ndirection = np.array([[.2,-.3,.17],[-.3,-.1,.23],[.17,.23,.4]])\nrate = .6\ndef _flatten(response):\n    a,b=response\n    d=len(auxiliary)\n    assert np.asarray(a).shape == (d,d)\n    assert np.asarray(b).shape == (d*(d+1)//2,)*2\n    return np.concatenate([np.asarray(a).ravel(),np.asarray(b).ravel()])\n",
            "call": "_flatten(quantum_response_tangent(auxiliary.copy(),temperature,direction.copy(),rate))",
            "gold_call": "_flatten(_oracle_quantum_response_tangent(auxiliary.copy(),temperature,direction.copy(),rate))",
            "tol": 1e-09,
        },
        {
            "setup": "import numpy as np\nauxiliary = np.array([[.9,.13,-.07],[.13,1.4,.18],[-.07,.18,2.2]])\ntemperature = .7\ndirection = np.array([[.2,-.3,.17],[-.3,-.1,.23],[.17,.23,.4]])\nrate = .6\ndef _flatten(response):\n    a,b=response\n    d=len(auxiliary)\n    assert np.asarray(a).shape == (d,d)\n    assert np.asarray(b).shape == (d*(d+1)//2,)*2\n    return np.concatenate([np.asarray(a).ravel(),np.asarray(b).ravel()])\nauxiliary = np.eye(3)*.64\n",
            "call": "_flatten(quantum_response_tangent(auxiliary.copy(),temperature,direction.copy(),rate))",
            "gold_call": "_flatten(_oracle_quantum_response_tangent(auxiliary.copy(),temperature,direction.copy(),rate))",
            "tol": 1e-09,
        },
        {
            "setup": "import numpy as np\nauxiliary = np.array([[.9,.13,-.07],[.13,1.4,.18],[-.07,.18,2.2]])\ntemperature = .7\ndirection = np.array([[.2,-.3,.17],[-.3,-.1,.23],[.17,.23,.4]])\nrate = .6\ndef _flatten(response):\n    a,b=response\n    d=len(auxiliary)\n    assert np.asarray(a).shape == (d,d)\n    assert np.asarray(b).shape == (d*(d+1)//2,)*2\n    return np.concatenate([np.asarray(a).ravel(),np.asarray(b).ravel()])\nq,_ = np.linalg.qr(np.array([[1.,2.,3.],[3.,-.2,1.],[.1,2.,-.3]]))\nauxiliary = q @ np.diag([.64,.64+2e-13,1.7]) @ q.T\n",
            "call": "_flatten(quantum_response_tangent(auxiliary.copy(),temperature,direction.copy(),rate))",
            "gold_call": "_flatten(_oracle_quantum_response_tangent(auxiliary.copy(),temperature,direction.copy(),rate))",
            "tol": 1e-09,
        },
        {
            "setup": "import numpy as np\nauxiliary = np.array([[.9,.13,-.07],[.13,1.4,.18],[-.07,.18,2.2]])\ntemperature = .7\ndirection = np.array([[.2,-.3,.17],[-.3,-.1,.23],[.17,.23,.4]])\nrate = .6\ndef _flatten(response):\n    a,b=response\n    d=len(auxiliary)\n    assert np.asarray(a).shape == (d,d)\n    assert np.asarray(b).shape == (d*(d+1)//2,)*2\n    return np.concatenate([np.asarray(a).ravel(),np.asarray(b).ravel()])\ntemperature = 0.0\n",
            "call": "_flatten(quantum_response_tangent(auxiliary.copy(),temperature,direction.copy(),rate))",
            "gold_call": "_flatten(_oracle_quantum_response_tangent(auxiliary.copy(),temperature,direction.copy(),rate))",
            "tol": 1e-09,
        },
        {
            "setup": "import numpy as np\nauxiliary = np.array([[.9,.13,-.07],[.13,1.4,.18],[-.07,.18,2.2]])\ntemperature = .7\ndirection = np.array([[.2,-.3,.17],[-.3,-.1,.23],[.17,.23,.4]])\nrate = .6\ndef _flatten(response):\n    a,b=response\n    d=len(auxiliary)\n    assert np.asarray(a).shape == (d,d)\n    assert np.asarray(b).shape == (d*(d+1)//2,)*2\n    return np.concatenate([np.asarray(a).ravel(),np.asarray(b).ravel()])\ndirection[:] = 0.0\nrate = 0.0\n",
            "call": "_flatten(quantum_response_tangent(auxiliary.copy(),temperature,direction.copy(),rate))",
            "gold_call": "_flatten(_oracle_quantum_response_tangent(auxiliary.copy(),temperature,direction.copy(),rate))",
            "tol": 1e-09,
        },
        {
            "setup": "import numpy as np\nauxiliary = np.array([[.9,.13,-.07],[.13,1.4,.18],[-.07,.18,2.2]])\ntemperature = .7\ndirection = np.array([[.2,-.3,.17],[-.3,-.1,.23],[.17,.23,.4]])\nrate = .6\ndef _flatten(response):\n    a,b=response\n    d=len(auxiliary)\n    assert np.asarray(a).shape == (d,d)\n    assert np.asarray(b).shape == (d*(d+1)//2,)*2\n    return np.concatenate([np.asarray(a).ravel(),np.asarray(b).ravel()])\nauxiliary = np.diag([.02,.021,3.])\ntemperature = .4\n",
            "call": "_flatten(quantum_response_tangent(auxiliary.copy(),temperature,direction.copy(),rate))",
            "gold_call": "_flatten(_oracle_quantum_response_tangent(auxiliary.copy(),temperature,direction.copy(),rate))",
            "tol": 1e-09,
        },
        {
            "setup": "import numpy as np\nauxiliary = np.array([[.9,.13,-.07],[.13,1.4,.18],[-.07,.18,2.2]])\ntemperature = .7\ndirection = np.array([[.2,-.3,.17],[-.3,-.1,.23],[.17,.23,.4]])\nrate = .6\ndef _flatten(response):\n    a,b=response\n    d=len(auxiliary)\n    assert np.asarray(a).shape == (d,d)\n    assert np.asarray(b).shape == (d*(d+1)//2,)*2\n    return np.concatenate([np.asarray(a).ravel(),np.asarray(b).ravel()])\nrate = -.6\n",
            "call": "_flatten(quantum_response_tangent(auxiliary.copy(),temperature,direction.copy(),rate))",
            "gold_call": "_flatten(_oracle_quantum_response_tangent(auxiliary.copy(),temperature,direction.copy(),rate))",
            "tol": 1e-09,
        },
        {
            "setup": "import numpy as np\nauxiliary = np.array([[.9,.13,-.07],[.13,1.4,.18],[-.07,.18,2.2]])\ntemperature = .7\ndirection = np.array([[.2,-.3,.17],[-.3,-.1,.23],[.17,.23,.4]])\nrate = .6\ndef _flatten(response):\n    a,b=response\n    d=len(auxiliary)\n    assert np.asarray(a).shape == (d,d)\n    assert np.asarray(b).shape == (d*(d+1)//2,)*2\n    return np.concatenate([np.asarray(a).ravel(),np.asarray(b).ravel()])\ntemperature=0.0\nrate=-1.0\ndef _raises(function,*args):\n    try:\n        function(*args)\n    except ValueError:\n        return 1.0\n    return 0.0\n",
            "call": "_raises(quantum_response_tangent,auxiliary.copy(),temperature,direction.copy(),rate)",
            "gold_call": "_raises(_oracle_quantum_response_tangent,auxiliary.copy(),temperature,direction.copy(),rate)",
            "tol": 1e-09,
        },
    ]
