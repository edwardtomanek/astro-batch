import numpy as np

def period(inp, param='a', mu=1.32712440018e+20):
    """
    Inputs:
       inp: Orbit list or semimajor axis list. Last dimension is either [a], [a, e, i, w, RAAN, theta], [p, f, g, h, k, L], [rx, ry, rz, vx, vy, vz].  Numpy array, can also be a tuple or a list.
       param: String defining parametrisation of input. Valid options are: 'kep', 'cart', 'mee', 'a'. Default 'a'
       mu: Gravitational parameter of central body. Defaults to MU_SUN (1.32712440018e+20) in m3/s2
    Outputs:
       T: List of periods of orbits. Keeps dimensions of input. If input is in format 'kep', 'cart' or 'mee' the last dimension is of size 1.
    """
    if isinstance(inp, list) or isinstance(inp, tuple):
        inp = np.array(inp, np.float64)
    
    
    if param == 'kep':
        a = inp[..., [0]].copy()
    elif param == 'mee':
        a = inp[..., [0]]/(1 - inp[..., [1]]**2 - inp[..., [2]]**2)
    elif param == 'cart':
        r = np.linalg.norm(inp[..., 0:3], axis=-1, keepdims=True)
        v = np.linalg.norm(inp[..., 3:6], axis=-1, keepdims=True)
        a = 1/((2/r) - (v**2)/mu)
    elif param == 'a':
        a = inp.copy()
    else:
        raise ValueError("Parametrisation not recognised.")

    return 2*np.pi*np.sqrt((a**3)/mu)

def rperi(inp1, inp2=None, param='kep', mu=1.32712440018e+20):
    """
    Inputs:
       inp1: Main input. Last dimension is either [a], [a, e, i, w, RAAN, theta], [p, f, g, h, k, L], [rx, ry, rz, vx, vy, vz]. Numpy array, can also be a tuple or a list.
       inp2: Additional input for [e] if inp1 is [a]. Otherwise it is ignored.
       param: String defining parametrisation of input. Valid options are: 'kep', 'cart', 'mee', 'ae'. Default 'kep'
       mu: Gravitational parameter of central body. Defaults to MU_SUN (1.32712440018e+20) in m3/s2
    Outputs:
       rperi: List of periapsis radii of orbits. Keeps dimensions of input. If input is in format 'kep', 'cart' or 'mee' the last dimension is of size 1.
    """
    if isinstance(inp1, list) or isinstance(inp1, tuple):
        inp1 = np.array(inp1, np.float64)
    if isinstance(inp2, list) or isinstance(inp2, tuple):
        inp2 = np.array(inp2, np.float64)

    if param == 'ae':
        a = inp1.copy()
        e = inp2.copy()
    elif param == 'kep':
        a = inp1[..., [0]].copy()
        e = inp1[..., [1]].copy()
    elif param == 'mee':
        e = np.sqrt(inp1[..., [1]]**2 + inp1[..., [2]]**2)
        a = inp1[..., [0]]/(1 - e**2)
    elif param == 'cart':
        r_vec = inp1[..., 0:3].copy()
        v_vec = inp1[..., 3:6].copy()

        r = np.linalg.norm(r_vec, axis=-1, keepdims=True)
        v = np.linalg.norm(v_vec, axis=-1, keepdims=True)

        h_vec = np.cross(r_vec, v_vec)
        e_vec = np.cross(v_vec, h_vec)/mu - r_vec/r

        e = np.linalg.norm(e_vec, axis=-1, keepdims=True)
        a = 1/((2/r) - (v**2)/mu)
    else:
        raise ValueError("Parametrisation not recognised.")
    
    return a*(1 - e)

def rapo(inp1, inp2=None, param='kep', mu=1.32712440018e+20):
    """
    Inputs:
       inp1: Main input. Last dimension is either [a], [a, e, i, w, RAAN, theta], [p, f, g, h, k, L], [rx, ry, rz, vx, vy, vz]. Numpy array, can also be a tuple or a list.
       inp2: Additional input for [e] if inp1 is [a]. Otherwise it is ignored.
       param: String defining parametrisation of input. Valid options are: 'kep', 'cart', 'mee', 'ae'. Default 'kep'
       mu: Gravitational parameter of central body. Defaults to MU_SUN (1.32712440018e+20) in m3/s2
    Outputs:
       rperi: List of apoapsis radii of orbits. Keeps dimensions of input. If input is in format 'kep', 'cart' or 'mee' the last dimension is of size 1.
    """
    if isinstance(inp1, list) or isinstance(inp1, tuple):
        inp1 = np.array(inp1, np.float64)
    if isinstance(inp2, list) or isinstance(inp2, tuple):
        inp2 = np.array(inp2, np.float64)

    if param == 'ae':
        a = inp1.copy()
        e = inp2.copy()
    elif param == 'kep':
        a = inp1[..., [0]].copy()
        e = inp1[..., [1]].copy()
    elif param == 'mee':
        e = np.sqrt(inp1[..., [1]]**2 + inp1[..., [2]]**2)
        a = inp1[..., [0]]/(1 - e**2)
    elif param == 'cart':
        r_vec = inp1[..., 0:3].copy()
        v_vec = inp1[..., 3:6].copy()

        r = np.linalg.norm(r_vec, axis=-1, keepdims=True)
        v = np.linalg.norm(v_vec, axis=-1, keepdims=True)

        h_vec = np.cross(r_vec, v_vec)
        e_vec = np.cross(v_vec, h_vec)/mu - r_vec/r

        e = np.linalg.norm(e_vec, axis=-1, keepdims=True)
        a = 1/((2/r) - (v**2)/mu)
    else:
        raise ValueError("Parametrisation not recognised.")
    
    return a*(1 + e)