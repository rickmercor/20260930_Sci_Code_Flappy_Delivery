"""
Sum closed Fourier transforms of the propagated Gaussian components.

Analytic Hagedorn-Gaussian Fourier transforms evaluate the coherent mesh-free ensemble directly on the shifted reference momentum nodes.

Returns
-------
np.ndarray: finite coherent momentum field with unit wave-number-weighted L2 norm.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

from numbers import Real
import numpy as np

def reconstruct_tgwp_momentum(
    x: np.ndarray, y: np.ndarray, coefficients: np.ndarray,
    q: np.ndarray, p: np.ndarray, Q: np.ndarray,
    P: np.ndarray, S: np.ndarray,
    hbar: float = 1.0,
) -> np.ndarray:
    """Return the normalized TGWP momentum field on shifted FFT nodes.

    Finite, strictly monotone, uniform real-space axes, each with at least
    two points and nonzero spacing, define
    ``k=fftshift(2*pi*fftfreq(n,d=spacing))`` in each direction. For each packet
    let ``C=P@inv(Q)`` and ``delta=p-hbar*k``; its Fourier factor is
    ``pi**(-1/2)*(2*pi*hbar)/sqrt(det(Q)*det(-1j*C))`` times
    ``exp(-0.5j*delta@inv(C)@delta/hbar - 1j*q@k + 1j*S/hbar)``. Sum these
    factors with ``coefficients`` and normalize with
    ``sqrt(sum(abs(wave)**2)*abs(dkx*dky))``. Use principal complex square roots,
    which remain on one branch for the specified short-time benchmark.

    Returns
    -------
    np.ndarray
        Complex momentum field of shape ``(len(x),len(y))``.

    Raises
    ------
    ValueError
        If axes are invalid, the Gaussian arrays are misaligned or non-finite,
        ``hbar`` is not positive finite, a required width is singular or its
        numerical inverse/determinant is invalid, or the reconstructed field
        has zero or non-finite norm.
    """
    return momentum

# =============================================================================
# GOLD SOLUTION
# =============================================================================

from numbers import Real
import numpy as np

def _oracle_reconstruct_tgwp_momentum(
    x: np.ndarray, y: np.ndarray, coefficients: np.ndarray,
    q: np.ndarray, p: np.ndarray, Q: np.ndarray,
    P: np.ndarray, S: np.ndarray,
    hbar: float = 1.0,
) -> np.ndarray:
    """Reference closed-form Gaussian Fourier reconstruction."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    c = np.asarray(coefficients, complex)
    q, p, Q, P, S = map(np.asarray, (q, p, Q, P, S))
    if c.ndim != 1 or c.size < 1:
        raise ValueError("coefficients must be a nonempty vector")
    n = c.size
    if x.ndim != 1 or y.ndim != 1 or min(x.size, y.size) < 2:
        raise ValueError("x and y must be one-dimensional axes")
    if q.shape != (n, 2) or p.shape != q.shape or Q.shape != (n, 2, 2) or P.shape != Q.shape or S.shape != (n,):
        raise ValueError("Gaussian arrays are not aligned")
    for axis in (x, y):
        if not np.all(np.isfinite(axis)):
            raise ValueError("axes must be finite")
        spacing = np.diff(axis)
        if (not np.all(np.isfinite(spacing))
                or not (np.all(spacing > 0) or np.all(spacing < 0))
                or not np.allclose(spacing, spacing[0])):
            raise ValueError("axes must be strictly monotone and uniform with nonzero spacing")
    if isinstance(hbar, bool) or not isinstance(hbar, Real) or not np.isfinite(hbar) or hbar <= 0:
        raise ValueError("hbar must be positive and finite")
    if not np.all(np.isfinite(np.r_[c.real, c.imag, q.ravel(), p.ravel(), Q.real.ravel(), Q.imag.ravel(), P.real.ravel(), P.imag.ravel(), S])):
        raise ValueError("Gaussian arrays must be finite")
    try:
        invq = np.linalg.inv(Q)
        width = np.einsum("nij,njk->nik", P, invq)
        invwidth = np.linalg.inv(width)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Q and the width matrices must be invertible") from exc
    detq = np.linalg.det(Q)
    detc = np.linalg.det(-1j * width)
    if (not np.all(np.isfinite(invq)) or not np.all(np.isfinite(invwidth))
            or not np.all(np.isfinite(detq * detc)) or np.any(detq * detc == 0)):
        raise ValueError("width matrices have invalid numerical inverses or determinants")
    kx = np.fft.fftshift(2 * np.pi * np.fft.fftfreq(len(x), d=np.diff(x)[0]))
    ky = np.fft.fftshift(2 * np.pi * np.fft.fftfreq(len(y), d=np.diff(y)[0]))
    gx, gy = np.meshgrid(kx, ky, indexing="ij")
    points = np.stack((gx.ravel(), gy.ravel()), axis=-1)
    delta = p[:, None, :] - float(hbar) * points[None, :, :]
    quad = np.einsum("nmi,nij,nmj->nm", delta, invwidth, delta)
    phase = np.exp(-.5j * quad / float(hbar) - 1j * np.einsum("ni,mi->nm", q, points) + 1j * S[:, None] / float(hbar))
    amplitude = np.pi ** (-.5) * (2 * np.pi * float(hbar)) / np.sqrt(detq * detc)
    wave = np.sum(c[:, None] * amplitude[:, None] * phase, axis=0).reshape(len(x), len(y))
    norm = np.sqrt(np.sum(np.abs(wave) ** 2) * abs(np.diff(kx)[0] * np.diff(ky)[0]))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("momentum reconstruction has invalid norm")
    return wave / norm

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return complete differential tests for normal and boundary inputs."""
    return [{'setup': 'import numpy as np\n'
               'from numbers import Real\n'
               'x=np.linspace(-2,2,9);y=np.linspace(-1,1,7);z=np.array([[-.5,.2,1.,0.],[.4,-.1,.7,.2]]);g=np.diag([2.,3.]);q=z[:,:2].copy();p=z[:,2:].copy();Q=np.broadcast_to(np.diag(1/np.sqrt(np.diag(g))),(2,2,2)).copy();P=np.broadcast_to(1j*np.diag(np.sqrt(np.diag(g))),(2,2,2)).copy();S=np.zeros(2);c=np.array([.7+.1j,-.2+.4j])',
      'call': '(lambda '
              'a:float(np.dot(a.real.ravel(),np.linspace(.1,1.,63))+2*np.dot(a.imag.ravel(),np.linspace(1.,.1,63))))(reconstruct_tgwp_momentum(x,y,c,q,p,Q,P,S))',
      'gold_call': '(lambda '
                   'a:float(np.dot(a.real.ravel(),np.linspace(.1,1.,63))+2*np.dot(a.imag.ravel(),np.linspace(1.,.1,63))))(_oracle_reconstruct_tgwp_momentum(x,y,c,q,p,Q,P,S))'},
     {'setup': 'import numpy as np\n'
               'from numbers import Real\n'
               'x=np.linspace(-1,1,5);y=np.linspace(-2,2,6);z=np.array([[0.,0.,0.,0.]]);q=z[:,:2].copy();p=z[:,2:].copy();Q=np.array([np.eye(2)*np.sqrt(.5)]);P=np.array([1j*np.eye(2)*np.sqrt(.5)]);S=np.zeros(1);c=np.array([1.+0j])',
      'call': '(lambda '
              'a:float(np.dot(a.real.ravel(),np.linspace(.1,1.,30))+2*np.dot(a.imag.ravel(),np.linspace(1.,.1,30))))(reconstruct_tgwp_momentum(x,y,c,q,p,Q,P,S,.5))',
      'gold_call': '(lambda '
                   'a:float(np.dot(a.real.ravel(),np.linspace(.1,1.,30))+2*np.dot(a.imag.ravel(),np.linspace(1.,.1,30))))(_oracle_reconstruct_tgwp_momentum(x,y,c,q,p,Q,P,S,.5))'},
     {'setup': 'import numpy as np\n'
               'from numbers import Real\n'
               'x=np.linspace(-1,1,5);y=np.linspace(-1,1,5);z=np.array([[0.,0.,0.,0.]]);q=z[:,:2].copy();p=z[:,2:].copy();Q=np.array([np.eye(2)]);P=np.array([1j*np.eye(2)]);S=np.zeros(1);c=np.array([1.+0j])\n'
               'def status(fn):\n'
               '    try: fn(); return 0\n'
               '    except ValueError: return 1\n'
               '    except Exception: return 2',
      'call': 'status(lambda: reconstruct_tgwp_momentum(x,y,c,q,p,Q,P,S,0.0))',
      'gold_call': 'status(lambda: _oracle_reconstruct_tgwp_momentum(x,y,c,q,p,Q,P,S,0.0))'},
     {'setup': 'import numpy as np\n'
               'from numbers import Real\n'
               'x=np.linspace(-1,1,5); y=np.linspace(-1,1,5); c=np.ones(1,dtype=complex); '
               'q=np.zeros((1,2)); p=np.zeros((1,2)); Q=np.eye(2,dtype=complex)[None]; '
               'P=1j*np.eye(2)[None]; S=np.zeros(1)\n'
               'Q=np.zeros((1,2,2),complex)\n'
               'def status(fn):\n'
               '    try:\n'
               '        fn()\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2',
      'call': 'status(lambda: reconstruct_tgwp_momentum(x,y,c,q,p,Q,P,S))',
      'gold_call': 'status(lambda: _oracle_reconstruct_tgwp_momentum(x,y,c,q,p,Q,P,S))'},
     {'setup': 'import numpy as np\n'
               'from numbers import Real\n'
               'x=np.linspace(-1,1,5); y=np.linspace(-1,1,5); c=np.ones(1,dtype=complex); '
               'q=np.zeros((1,2)); p=np.zeros((1,2)); Q=np.eye(2,dtype=complex)[None]; '
               'P=1j*np.eye(2)[None]; S=np.zeros(1)\n'
               'P=np.zeros((1,2,2),complex)\n'
               'def status(fn):\n'
               '    try:\n'
               '        fn()\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2',
      'call': 'status(lambda: reconstruct_tgwp_momentum(x,y,c,q,p,Q,P,S))',
      'gold_call': 'status(lambda: _oracle_reconstruct_tgwp_momentum(x,y,c,q,p,Q,P,S))'},
     {'setup': 'import numpy as np\n'
               'from numbers import Real\n'
               'x=np.linspace(-1,1,5); y=np.linspace(-1,1,5); c=np.ones(1,dtype=complex); '
               'q=np.zeros((1,2)); p=np.zeros((1,2)); Q=np.eye(2,dtype=complex)[None]; '
               'P=1j*np.eye(2)[None]; S=np.zeros(1)\n'
               'x=np.zeros(5)\n'
               'def status(fn):\n'
               '    try:\n'
               '        fn()\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2',
      'call': 'status(lambda: reconstruct_tgwp_momentum(x,y,c,q,p,Q,P,S))',
      'gold_call': 'status(lambda: _oracle_reconstruct_tgwp_momentum(x,y,c,q,p,Q,P,S))'},
     {'setup': 'import numpy as np\n'
               'from numbers import Real\n'
               'x=np.linspace(-1,1,5); y=np.linspace(-1,1,5); c=np.ones(1,dtype=complex); '
               'q=np.zeros((1,2)); p=np.zeros((1,2)); Q=np.eye(2,dtype=complex)[None]; '
               'P=1j*np.eye(2)[None]; S=np.zeros(1)\n'
               'Q=1e8*np.eye(2)[None]; P=1e-8j*np.eye(2)[None]',
      'call': 'float(np.sum(reconstruct_tgwp_momentum(x,y,c,q,p,Q,P,S).real))',
      'gold_call': 'float(np.sum(_oracle_reconstruct_tgwp_momentum(x,y,c,q,p,Q,P,S).real))'}]
