import convert
import properties
import numpy as np
import calc

# Some example calculations

au = 149597870691.0

lb = np.array([2.7*au, 0.0, 0*np.pi/180, 0, 0, 0], dtype=np.float64)
ub = np.array([3.3*au, 0.1, 90*np.pi/180, 2*np.pi, 2*np.pi, 2*np.pi])

kep = np.random.rand(10000, 6)*(ub - lb) + lb

cart = convert.kep2cart(kep)

mee = convert.kep2mee(kep)

kep_withE = kep.copy()
kep_withE[..., [-1]] = convert.theta2E(kep[..., [-1]], kep[..., [1]])

periods = properties.period(kep, 'kep')

T = 365*24*3600
calc.ephem_2body(kep, T, 'kep')