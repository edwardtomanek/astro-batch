import numpy as np
import convert
import properties

def ephem_2body(orb, T, param='kep', mu=1.32712440018e+20):
    """
    Inputs:
       orb: Orbit list at epoch 0. Last dimension is either [a, e, i, w, RAAN, theta], [p, f, g, h, k, L], [rx, ry, rz, vx, vy, vz]. Numpy array, can also be a tuple or a list.
       T: Epoch (secs).
       param: String defining parametrisation of input. Valid options are: 'kep', 'cart', 'mee'. Default 'kep'
       mu: Gravitational parameter of central body. Defaults to MU_SUN (1.32712440018e+20) in m3/s2
    Outputs:
       orb_T: Orbit state list in same parametrisation as input, at time T.
    """
    if isinstance(orb, list) or isinstance(orb, tuple):
        orb = np.array(orb, np.float64)

    if param == 'mee':
        orb = convert.mee2kep(orb.copy())
    elif param == 'cart':
        orb = convert.cart2kep(orb.copy())
    
    theta0 = orb[..., [-1]]
    e = orb[..., [1]]
    a = orb[..., [0]]
    

    M0 = convert.E2M(convert.theta2E(theta0.copy(), e.copy()), e.copy())
    period = properties.period(a.copy(), 'a', mu)
    
    Mf = ((T/period)*(2*np.pi) + M0)%(2*np.pi)

    thetaf = convert.E2theta(convert.M2E(Mf.copy(), e.copy()), e.copy())
    orb_T = orb.copy()
    orb_T[..., [-1]] = thetaf

    if param == 'mee':
        orb_T = convert.kep2mee(orb_T.copy())
    elif param == 'cart':
        orb_T = convert.kep2cart(orb_T.copy())
    return orb_T