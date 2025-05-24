import convert
import numpy as np
import pykep as pk

au = pk.AU
lb = np.array([2.7*au, 0.0, -6*np.pi/180, 0, 0, 0], dtype=np.float64)
ub = np.array([3.3*au, 0.1, 0*np.pi/180, 2*np.pi, 2*np.pi, 2*np.pi])

kep = np.random.rand(1000000, 6)*(ub - lb) + lb
import time
time3 = time.time()
cartout2 = convert.kep2cart(kep, pk.MU_SUN)
time4 = time.time()
print(time4 - time3)
time5 = time.time()
kepout = convert.cart2kep(cartout2, pk.MU_SUN)
time6 = time.time()
print(time6 - time5)
print(kepout/kep)
keppk = kepout*0
for i in range(1000000):
    keppk[i] = pk.ic2par(cartout2[i, 0:3], cartout2[i, 3:6], pk.MU_SUN)
b = keppk[..., [3]]
keppk[..., [3]] = np.copy(keppk[..., [4]])
keppk[..., [4]] = np.copy(b)
print(kepout/keppk)
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('TkAgg')
plt.scatter(kepout[:, 4], kep[:, 4])
plt.show()