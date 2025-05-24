import convert
import numpy as np
import pykep as pk

au = pk.AU
lb = np.array([2.7*au, 0.0, -6*np.pi/180, 0, 0, 0], dtype=np.float64)
ub = np.array([3.3*au, 0.1, 0*np.pi/180, 2*np.pi, 2*np.pi, 2*np.pi])

kep = np.random.rand(1000000, 6)*(ub - lb) + lb