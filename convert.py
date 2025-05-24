import numpy as np

def kep2cart(kep, mu=1.32712440018e+20):
    """
    Inputs:
       kep: Orbit list. Numpy array with last dimension in format [a, e, i, w, RAAN, theta]. Can also be a tuple or a list.
       mu: Gravitational parameter of central body. Defaults to MU_SUN (1.32712440018e+20) in m3/s2
    Outputs:
       cart: Orbit list in Cartesian vectors [rx, ry, rz, vx, vy, vz], along the last dimension with same size as kep.
    """
    if isinstance(kep, list) or isinstance(kep, tuple):
        kep = np.array(kep, np.float64)

    shape = kep.shape
    cart = np.zeros(shape, np.float64)

    a = np.copy(kep[..., [0]])
    e = np.copy(kep[..., [1]])
    i = np.copy(kep[..., [2]])
    w = np.copy(kep[..., [3]])
    RAAN = np.copy(kep[..., [4]])
    theta = np.copy(kep[..., [5]])

    sinE = (np.sqrt(1 - e**2)*np.sin(theta))/(1 + e*np.cos(theta))
    cosE = (e + np.cos(theta))/(1 + e*np.cos(theta))

    rc = a*(1 - e*cosE)

    ox = rc*np.cos(theta)
    oy = rc*np.sin(theta)

    odx = (np.sqrt(mu*a)/rc)*(-sinE)
    ody = (np.sqrt(mu*a)/rc)*(np.sqrt(1 - e**2)*cosE)

    cosi = np.cos(i)
    sini = np.sin(i)
    cosw = np.cos(w)
    sinw = np.sin(w)
    cosRAAN = np.cos(RAAN)
    sinRAAN = np.sin(RAAN)

    cart[..., [0]] = ox*(cosw*cosRAAN - sinw*cosi*sinRAAN) - oy*(sinw*cosRAAN + cosw*cosi*sinRAAN)
    cart[..., [1]] = ox*(cosw*sinRAAN + sinw*cosi*cosRAAN) + oy*(cosw*cosi*cosRAAN - sinw*sinRAAN)
    cart[..., [2]] = ox*sinw*sini + oy*cosw*sini

    cart[..., [3]] = odx*(cosw*cosRAAN - sinw*cosi*sinRAAN) - ody*(sinw*cosRAAN + cosw*cosi*sinRAAN)
    cart[..., [4]] = odx*(cosw*sinRAAN + sinw*cosi*cosRAAN) + ody*(cosw*cosi*cosRAAN - sinw*sinRAAN)
    cart[..., [5]] = odx*sinw*sini + ody*cosw*sini

    return cart

def cart2kep(cart, mu=1.32712440018e+20):
    """
    Inputs:
       cart: Orbit list. Numpy array with last dimension in format [rx, ry, rz, vx, vy, vz]. Can also be a tuple or a list.
       mu: Gravitational parameter of central body. Defaults to MU_SUN (1.32712440018e+20) in m3/s2
    Outputs:
       kep: Orbit list in Keplerian elements [a, e, i, w, RAAN, theta], along the last dimension with same size as cart.
    """
    if isinstance(cart, list) or isinstance(cart, tuple):
        cart = np.array(cart, np.float64)

    shape = cart.shape
    kep = np.zeros(shape, np.float64)

    r_vec = np.copy(cart[..., 0:3])
    v_vec = np.copy(cart[..., 3:6])

    
    r = np.linalg.norm(r_vec, axis=-1, keepdims=True)
    v = np.linalg.norm(v_vec, axis=-1, keepdims=True)

    h_vec = np.cross(r_vec, v_vec)
    h = np.linalg.norm(h_vec, axis=-1, keepdims=True)
    e_vec = np.cross(v_vec, h_vec)/mu - r_vec/r

    kep[..., [1]] = e = np.linalg.norm(e_vec, axis=-1, keepdims=True)

    kep[..., [2]] = np.arccos(h_vec[..., [2]]/h)

    n_vec = np.cross([0.0, 0.0, 1.0], h_vec)
    n = np.linalg.norm(n_vec, axis=-1, keepdims=True)

    rdotvsign = np.sign(np.vecdot(r_vec, v_vec, keepdims=True))
    kep[..., [5]] = (np.pi - rdotvsign*np.pi + rdotvsign*np.arccos(np.vecdot(e_vec, r_vec, keepdims=True)/(e*r)))%(2*np.pi)

    nysign = np.sign(n_vec[..., [1]])
    kep[..., [4]] = (np.pi - nysign*np.pi + nysign*np.arccos(n_vec[..., [0]]/n))%(2*np.pi)

    ezsign = np.sign(e_vec[..., [2]])
    kep[..., [3]] = (np.pi - ezsign*np.pi + ezsign*np.arccos(np.vecdot(n_vec, e_vec, keepdims=True)/(n*e)))%(2*np.pi)

    kep[..., [0]] = 1/((2/r) - (v**2)/mu)

    return kep

def kep2mee(kep, mu=1.32712440018e+20):
    """
    Inputs:
       kep: Orbit list. Numpy array with last dimension in format [a, e, i, w, RAAN, theta]. Can also be a tuple or a list.
       mu: Gravitational parameter of central body. Defaults to MU_SUN (1.32712440018e+20) in m3/s2
    Outputs:
       mee: Orbit list in modified equinoctial elements [p, f, g, h, k, L], along the last dimension with same size as kep.
    """
    if isinstance(kep, list) or isinstance(kep, tuple):
        kep = np.array(kep, np.float64)

    shape = kep.shape
    mee = np.zeros(shape, np.float64)

    a = np.copy(kep[..., [0]])
    e = np.copy(kep[..., [1]])
    i = np.copy(kep[..., [2]])
    w = np.copy(kep[..., [3]])
    RAAN = np.copy(kep[..., [4]])
    theta = np.copy(kep[..., [5]])

    mee[..., [0]] = a*(1 - e**2)
    mee[..., [1]] = e*np.cos(w + RAAN)
    mee[..., [2]] = e*np.sin(w + RAAN)
    mee[..., [3]] = np.tan(i/2)*np.cos(RAAN)
    mee[..., [4]] = np.tan(i/2)*np.sin(RAAN)
    mee[..., [5]] = w + RAAN + theta

    return mee

def mee2cart(mee, mu=1.32712440018e+20):
    """
    Inputs:
       mee: Orbit list. Numpy array with last dimension in format [p, f, g, h, k, L]. Can also be a tuple or a list.
       mu: Gravitational parameter of central body. Defaults to MU_SUN (1.32712440018e+20) in m3/s2
    Outputs:
       cart: Orbit list in Cartesian vectors [rx, ry, rz, vx, vy, vz], along the last dimension with same size as kep.
    """
    if isinstance(mee, list) or isinstance(mee, tuple):
        mee = np.array(mee, np.float64)

    shape = mee.shape
    cart = np.zeros(shape, np.float64)

    p = mee[..., [0]]
    f = mee[..., [1]]
    g = mee[..., [2]]
    h = mee[..., [3]]
    k = mee[..., [4]]
    L = mee[..., [5]]

    alph2 = h**2 - k**2
    s2 = 1 + h**2 + k**2
    cosL = np.cos(L)
    sinL = np.sin(L)
    w = 1 + f*cosL + g*sinL
    r = p/w
    hk2 = 2*h*k
    sqrtmup = np.sqrt(mu/p)

    cart[..., [0]] = (r/s2)*(cosL + alph2*cosL + hk2*sinL)
    cart[..., [1]] = (r/s2)*(sinL - alph2*sinL + hk2*cosL)
    cart[..., [2]] = (2*r/s2)*(h*sinL - k*cosL)

    cart[..., [3]] = (-1/s2)*sqrtmup*(sinL + alph2*sinL - hk2*cosL + g - hk2*f + alph2*g)
    cart[..., [4]] = (-1/s2)*sqrtmup*(-cosL + alph2*cosL + hk2*sinL - f + hk2*g + alph2*f)
    cart[..., [5]] = (2/s2)*sqrtmup*(h*cosL + k*sinL + f*h + g*k)
    return cart

def theta2E(theta, e):
    """
    Inputs:
       theta: True anomaly list. Numpy array. Can also be a tuple or a list.
       e: Eccentricity list. Numpy array. Can also be a tuple or a list.
    Outputs:
       E: Eccentric anomaly list.
    """
    if isinstance(theta, list) or isinstance(theta, tuple):
        theta = np.array(theta, np.float64)
    if isinstance(e, list) or isinstance(e, tuple):
        e = np.array(e, np.float64)

    sinE = (np.sqrt(1 - e**2)*np.sin(theta))/(1 + e*np.cos(theta))
    cosE = (e + np.cos(theta))/(1 + e*np.cos(theta))
    E = np.arctan2(sinE, cosE)
    return E


def E2theta(E, e):
    """
    Inputs:
       E: Eccentric anomaly list. Numpy array. Can also be a tuple or a list.
       e: Eccentricity list. Numpy array. Can also be a tuple or a list.
    Outputs:
       theta: True anomaly list.
    """
    if isinstance(E, list) or isinstance(theta, tuple):
        E = np.array(E, np.float64)
    if isinstance(e, list) or isinstance(e, tuple):
        e = np.array(e, np.float64)

    sintheta = (np.sqrt(1 - e**2)*np.sin(E))/(1 - e*np.cos(E))
    costheta = (np.cos(E) - e)/(1 - e*np.cos(E))
    theta = np.arctan2(sintheta, costheta)
    return theta

def E2M(E, e):
    """
    Inputs:
       E: Eccentric anomaly list. Numpy array. Can also be a tuple or a list.
       e: Eccentricity list. Numpy array. Can also be a tuple or a list.
    Outputs:
       M: Mean anomaly list.
    """
    if isinstance(E, list) or isinstance(E, tuple):
        E = np.array(E, np.float64)
    if isinstance(e, list) or isinstance(e, tuple):
        e = np.array(e, np.float64)

    M = E - e*np.sin(E)
    return M