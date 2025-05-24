import convert
import numpy as np
import pykep as pk

au = pk.AU
lb = np.array([2.7*au, 0.0, -1*np.pi/180, 0, 0, 0], dtype=np.float64)
ub = np.array([3.3*au, 0.1, 6*np.pi/180, 2*np.pi, 2*np.pi, 2*np.pi])

kep = np.random.rand(1000000, 6)*(ub - lb) + lb
import time
time3 = time.time()
kepout2 = convert.kep2cart(kep, pk.MU_SUN)
time4 = time.time()
print(time4 - time3)
