import numpy as np

def kep2cart(kep, mu=1.32712440018e+20):
    """
    Inputs:
       kep: Orbit list. 2D or 3D numpy array with last dimension in format [a, e, i, w, RAAN, theta]. Can also be a tuple or a list.
       mu: Gravitational parameter of central body. Defaults to MU_SUN (1.32712440018e+20) in m3/s2
    Outputs:
       cart: Orbit list in Cartesian vectors [rx, ry, rz, vx, vy, vz], along the last dimension with same size as kep.
    """
    if isinstance(kep, list) or isinstance(kep, tuple):
        kep = np.array(kep, np.float64)
        
    if len(kep.shape) == 3:
        a = np.copy(kep[:, :, [0]])
        e = np.copy(kep[:, :, [1]])
        i = np.copy(kep[:, :, [2]])
        w = np.copy(kep[:, :, [3]])
        RAAN = np.copy(kep[:, :, [4]])
        theta = np.copy(kep[:, :, [5]])
    elif len(kep.shape) == 2:
        a = np.copy(kep[:, [0]])
        e = np.copy(kep[:, [1]])
        i = np.copy(kep[:, [2]])
        w = np.copy(kep[:, [3]])
        RAAN = np.copy(kep[:, [4]])
        theta = np.copy(kep[:, [5]])

    cart = np.zeros(kep.shape, np.float64)

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

    if len(kep.shape) == 3:
        cart[:, :, [0]] = ox*(cosw*cosRAAN - sinw*cosi*sinRAAN) - oy*(sinw*cosRAAN + cosw*cosi*sinRAAN)
        cart[:, :, [1]] = ox*(cosw*sinRAAN + sinw*cosi*cosRAAN) + oy*(cosw*cosi*cosRAAN - sinw*sinRAAN)
        cart[:, :, [2]] = ox*sinw*sini + oy*cosw*sini

        cart[:, :, [3]] = odx*(cosw*cosRAAN - sinw*cosi*sinRAAN) - ody*(sinw*cosRAAN + cosw*cosi*sinRAAN)
        cart[:, :, [4]] = odx*(cosw*sinRAAN + sinw*cosi*cosRAAN) + ody*(cosw*cosi*cosRAAN - sinw*sinRAAN)
        cart[:, :, [5]] = odx*sinw*sini + ody*cosw*sini
    elif len(kep.shape) == 2:
        cart[:, [0]] = ox*(cosw*cosRAAN - sinw*cosi*sinRAAN) - oy*(sinw*cosRAAN + cosw*cosi*sinRAAN)
        cart[:, [1]] = ox*(cosw*sinRAAN + sinw*cosi*cosRAAN) + oy*(cosw*cosi*cosRAAN - sinw*sinRAAN)
        cart[:, [2]] = ox*sinw*sini + oy*cosw*sini

        cart[:, [3]] = odx*(cosw*cosRAAN - sinw*cosi*sinRAAN) - ody*(sinw*cosRAAN + cosw*cosi*sinRAAN)
        cart[:, [4]] = odx*(cosw*sinRAAN + sinw*cosi*cosRAAN) + ody*(cosw*cosi*cosRAAN - sinw*sinRAAN)
        cart[:, [5]] = odx*sinw*sini + ody*cosw*sini
    return cart