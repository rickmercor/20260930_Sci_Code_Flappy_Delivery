"""
Evaluate a channel-weighted inverse-Gram variance and its gradient and Hessian in the two design parameters.

Let $$A$$ be a real transmission matrix, $$G=A^{\mathsf T}A$$ and $$w$$ a fixed nonnegative weight for each frequency channel. The graded quantity is $$F=\sum_n w_n[G^+]_{nn}$$, where the plus denotes the Moore-Penrose pseudoinverse. At unit detector-noise variance this is the weighted sum of the minimum-norm reconstruction variances of the individual frequency channels; null-space bias is excluded. The matrix has full rank equal to the smaller of its row and column counts throughout a neighborhood of the evaluation point.

The returned orders are the exact ordinary partial derivatives of $$F$$ with respect to the coupling $$q$$ and the detuning $$s$$ along the constant-rank path, driven by the supplied intensity derivatives, with no factorial normalization. They follow the intensity jet's convention: value, $$\partial_q$$, $$\partial_s$$, $$\partial_q^2$$, $$\partial_q\partial_s$$, $$\partial_s^2$$.

Returns
-------
np.ndarray: Real vector [F, dF/dq, dF/ds, d2F/dq2, d2F/dqds, d2F/ds2], shape (6,).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def inverse_gram_jet(transmission: "np.ndarray", weights: "np.ndarray") -> "np.ndarray":
    r"""Compute F, its gradient and its Hessian in the two design parameters.
    
    Parameters
    ----------
    transmission : np.ndarray
        Finite real array (6, M, N), orders A, dA/dq, dA/ds, d2A/dq2,
        d2A/dqds, d2A/ds2, M,N >= 1. A is full rank min(M,N), with that rank
        constant nearby, and its condition number, the ratio of its largest
        to smallest nonzero singular value, is at most 50. Negative entries
        in derivative slices are allowed.
    weights : np.ndarray
        Finite nonnegative real vector of length N, one weight per frequency
        channel, indexed like the columns of A. Weights are independent of the
        design parameters.
    
    Returns
    -------
    result : np.ndarray
        Real vector [F, dF/dq, dF/ds, d2F/dq2, d2F/dqds, d2F/ds2], shape (6,).
        The weighted sum runs over frequency channels; it is not divided by M,
        N or the weight sum.
    
    Notes
    -----
    Inputs satisfy the stated domain and must remain unchanged. Square,
    overdetermined and underdetermined matrices use the same scientific definition.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_inverse_gram_jet(transmission: "np.ndarray", weights: "np.ndarray") -> "np.ndarray":
    a, aq, as_, aqq, aqs, ass = transmission
    w = np.asarray(weights, dtype=float)
    first = (aq, as_)
    second = ((aqq, aqs), (aqs, ass))
    if a.shape[0] < a.shape[1]:
        # Underdetermined: (A^T A)^+ = A^T (A A^T)^-2 A, so F = Tr[(A A^T)^-2 A diag(w) A^T].
        pair = lambda u, v: u@v.T
        weighted = lambda u, v: (u*w)@v.T
    else:
        # Full column rank: G^+ = G^-1 and F = Tr[diag(w) G^-1].
        pair = lambda u, v: u.T@v
        weighted = None
    if weighted is None:
        gram = pair(a, a)
        gram_first = [pair(first[i], a)+pair(a, first[i]) for i in range(2)]
        inverse = np.linalg.solve(gram, np.eye(gram.shape[0]))
        inv_first = [-inverse@gram_first[i]@inverse for i in range(2)]
        value = float(w@np.diag(inverse))
        out = [value]+[float(w@np.diag(inv_first[i])) for i in range(2)]
        for i, j in ((0, 0), (0, 1), (1, 1)):
            gram_second = (pair(second[i][j], a)+pair(first[i], first[j])
                           +pair(first[j], first[i])+pair(a, second[i][j]))
            inv_second = (-inverse@gram_second@inverse
                          +inverse@gram_first[i]@inverse@gram_first[j]@inverse
                          +inverse@gram_first[j]@inverse@gram_first[i]@inverse)
            out.append(float(w@np.diag(inv_second)))
        return np.array(out)
    port = pair(a, a)
    band = weighted(a, a)
    port_first = [pair(first[i], a)+pair(a, first[i]) for i in range(2)]
    band_first = [weighted(first[i], a)+weighted(a, first[i]) for i in range(2)]
    inverse = np.linalg.solve(port, np.eye(port.shape[0]))
    inv_first = [-inverse@port_first[i]@inverse for i in range(2)]
    kernel = inverse@inverse
    kernel_first = [inv_first[i]@inverse+inverse@inv_first[i] for i in range(2)]
    out = [np.trace(kernel@band)]
    out += [np.trace(kernel_first[i]@band+kernel@band_first[i]) for i in range(2)]
    for i, j in ((0, 0), (0, 1), (1, 1)):
        port_second = (pair(second[i][j], a)+pair(first[i], first[j])
                       +pair(first[j], first[i])+pair(a, second[i][j]))
        band_second = (weighted(second[i][j], a)+weighted(first[i], first[j])
                       +weighted(first[j], first[i])+weighted(a, second[i][j]))
        inv_second = (-inverse@port_second@inverse
                      +inverse@port_first[i]@inverse@port_first[j]@inverse
                      +inverse@port_first[j]@inverse@port_first[i]@inverse)
        kernel_second = (inv_second@inverse+inv_first[i]@inv_first[j]
                         +inv_first[j]@inv_first[i]+inverse@inv_second)
        out.append(np.trace(kernel_second@band+kernel_first[i]@band_first[j]
                            +kernel_first[j]@band_first[i]+kernel@band_second))
    return np.array([float(v) for v in out])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return explicit differential cases for Studio contract validation."""
    return [
        {
            "setup": 'import numpy as np\nr=np.random.default_rng(13)\ndef draw(shape):\n    return (r.uniform(.1,1,shape), r.standard_normal(shape)*.4, r.standard_normal(shape)*.35,\n            r.standard_normal(shape)*.3, r.standard_normal(shape)*.25, r.standard_normal(shape)*.3)\njet=np.stack(draw((4,9))); w=r.uniform(.2,2.,9)\ndef measure(fn):\n    j=jet.copy(); v=w.copy(); out=np.asarray(fn(j,v))\n    assert out.shape==(6,) and np.array_equal(j,jet) and np.array_equal(v,w)\n    return out\n',
            "call": "measure(inverse_gram_jet)",
            "gold_call": "measure(_oracle_inverse_gram_jet)",
            "tol": 1e-10
        },
        {
            "setup": 'import numpy as np\nr=np.random.default_rng(13)\ndef draw(shape):\n    return (r.uniform(.1,1,shape), r.standard_normal(shape)*.4, r.standard_normal(shape)*.35,\n            r.standard_normal(shape)*.3, r.standard_normal(shape)*.25, r.standard_normal(shape)*.3)\njet=np.stack(draw((4,9))); w=r.uniform(.2,2.,9)\ndef measure(fn):\n    j=jet.copy(); v=w.copy(); out=np.asarray(fn(j,v))\n    assert out.shape==(6,) and np.array_equal(j,jet) and np.array_equal(v,w)\n    return out\nw=np.zeros(9); w[2:6]=1.\n',
            "call": "measure(inverse_gram_jet)",
            "gold_call": "measure(_oracle_inverse_gram_jet)",
            "tol": 1e-10
        },
        {
            "setup": 'import numpy as np\nr=np.random.default_rng(13)\ndef draw(shape):\n    return (r.uniform(.1,1,shape), r.standard_normal(shape)*.4, r.standard_normal(shape)*.35,\n            r.standard_normal(shape)*.3, r.standard_normal(shape)*.25, r.standard_normal(shape)*.3)\njet=np.stack(draw((4,9))); w=r.uniform(.2,2.,9)\ndef measure(fn):\n    j=jet.copy(); v=w.copy(); out=np.asarray(fn(j,v))\n    assert out.shape==(6,) and np.array_equal(j,jet) and np.array_equal(v,w)\n    return out\nw=np.ones(9)\n',
            "call": "measure(inverse_gram_jet)",
            "gold_call": "measure(_oracle_inverse_gram_jet)",
            "tol": 1e-10
        },
        {
            "setup": 'import numpy as np\nr=np.random.default_rng(13)\ndef draw(shape):\n    return (r.uniform(.1,1,shape), r.standard_normal(shape)*.4, r.standard_normal(shape)*.35,\n            r.standard_normal(shape)*.3, r.standard_normal(shape)*.25, r.standard_normal(shape)*.3)\njet=np.stack(draw((4,9))); w=r.uniform(.2,2.,9)\ndef measure(fn):\n    j=jet.copy(); v=w.copy(); out=np.asarray(fn(j,v))\n    assert out.shape==(6,) and np.array_equal(j,jet) and np.array_equal(v,w)\n    return out\nw=np.zeros(9); w[4]=1.\n',
            "call": "measure(inverse_gram_jet)",
            "gold_call": "measure(_oracle_inverse_gram_jet)",
            "tol": 1e-10
        },
        {
            "setup": 'import numpy as np\nr=np.random.default_rng(13)\ndef draw(shape):\n    return (r.uniform(.1,1,shape), r.standard_normal(shape)*.4, r.standard_normal(shape)*.35,\n            r.standard_normal(shape)*.3, r.standard_normal(shape)*.25, r.standard_normal(shape)*.3)\njet=np.stack(draw((4,9))); w=r.uniform(.2,2.,9)\ndef measure(fn):\n    j=jet.copy(); v=w.copy(); out=np.asarray(fn(j,v))\n    assert out.shape==(6,) and np.array_equal(j,jet) and np.array_equal(v,w)\n    return out\nw=np.zeros(9); w[0]=2.5; w[8]=.5\n',
            "call": "measure(inverse_gram_jet)",
            "gold_call": "measure(_oracle_inverse_gram_jet)",
            "tol": 1e-10
        },
        {
            "setup": 'import numpy as np\nr=np.random.default_rng(13)\ndef draw(shape):\n    return (r.uniform(.1,1,shape), r.standard_normal(shape)*.4, r.standard_normal(shape)*.35,\n            r.standard_normal(shape)*.3, r.standard_normal(shape)*.25, r.standard_normal(shape)*.3)\njet=np.stack(draw((4,9))); w=r.uniform(.2,2.,9)\ndef measure(fn):\n    j=jet.copy(); v=w.copy(); out=np.asarray(fn(j,v))\n    assert out.shape==(6,) and np.array_equal(j,jet) and np.array_equal(v,w)\n    return out\njet=np.stack((jet[0],)+tuple(np.zeros((4,9)) for _ in range(5)))\n',
            "call": "measure(inverse_gram_jet)",
            "gold_call": "measure(_oracle_inverse_gram_jet)",
            "tol": 1e-10
        },
        {
            "setup": 'import numpy as np\nr=np.random.default_rng(13)\ndef draw(shape):\n    return (r.uniform(.1,1,shape), r.standard_normal(shape)*.4, r.standard_normal(shape)*.35,\n            r.standard_normal(shape)*.3, r.standard_normal(shape)*.25, r.standard_normal(shape)*.3)\njet=np.stack(draw((4,9))); w=r.uniform(.2,2.,9)\ndef measure(fn):\n    j=jet.copy(); v=w.copy(); out=np.asarray(fn(j,v))\n    assert out.shape==(6,) and np.array_equal(j,jet) and np.array_equal(v,w)\n    return out\njet=np.stack(draw((5,5))); w=r.uniform(.2,2.,5)\n',
            "call": "measure(inverse_gram_jet)",
            "gold_call": "measure(_oracle_inverse_gram_jet)",
            "tol": 1e-10
        },
        {
            "setup": 'import numpy as np\nr=np.random.default_rng(13)\ndef draw(shape):\n    return (r.uniform(.1,1,shape), r.standard_normal(shape)*.4, r.standard_normal(shape)*.35,\n            r.standard_normal(shape)*.3, r.standard_normal(shape)*.25, r.standard_normal(shape)*.3)\njet=np.stack(draw((4,9))); w=r.uniform(.2,2.,9)\ndef measure(fn):\n    j=jet.copy(); v=w.copy(); out=np.asarray(fn(j,v))\n    assert out.shape==(6,) and np.array_equal(j,jet) and np.array_equal(v,w)\n    return out\njet=np.stack(draw((7,4))); w=r.uniform(.2,2.,4)\n',
            "call": "measure(inverse_gram_jet)",
            "gold_call": "measure(_oracle_inverse_gram_jet)",
            "tol": 1e-10
        }
    ]
