
import time
import numpy as np
import matplotlib.pyplot as plt

from dedalus import public as de

import logging
logger = logging.getLogger(__name__)


# Parameters
Lz = 136
Nz = 32
alpha = 8
beta = 1.3e3
phi_0 = 0.1
rho_0 = 0.08
dt = 1e-4

snapshot_iter = 100
stop_iteration = 1000
print_iter = 10

ncc_cutoff = 1e-10
tolerance = 1e-10

# Derived parameters
a = phi_0 / rho_0 / beta * alpha / (alpha + 2 * phi_0 * (1 - rho_0))
b = (1 - phi_0 + phi_0 * rho_0) / beta
c = (2 - 2 * rho_0 + rho_0 * alpha) * (phi_0 / alpha / beta / rho_0)
wg_0 = 1 + (phi_0 * (1 - phi_0) * (1 - rho_0))

# Build domain
z_basis = de.Chebyshev('z', Nz, interval=(0, Lz), dealias=2)
domain = de.Domain([z_basis], grid_dtype=np.float64)

phi = domain.new_field()
phi_rho_g = domain.new_field()
z = domain.grid(0)

phi['g'] = phi_0 + (1 - phi_0) * a * z
rho_g_init = rho_0 * np.exp(-z * b / rho_0)
phi_rho_g['g'] = phi['g'] * rho_g_init

# Setup problem
problem = de.NLBVP(domain, variables=['w_g', 'w_m', 'wz_m'], ncc_cutoff=ncc_cutoff)
problem.parameters['alpha'] = alpha
problem.parameters['beta'] = beta
problem.parameters['B2'] = 1
problem.parameters['B3'] = a
problem.parameters['phi'] = phi
problem.parameters['phi_rho_g'] = phi_rho_g
problem.add_equation("alpha*(w_m - w_g) = beta*dz(phi_rho_g) + phi_rho_g - beta*phi_rho_g * dz(phi) / phi", tau=False)
problem.add_equation("(4/3)*alpha*(dz(wz_m)) - alpha*(w_m - w_g) = (phi - phi_rho_g) + (4/3)*alpha*(1+phi**2)*dz(phi)/phi*wz_m - phi*(phi - phi_rho_g)", tau=True)
problem.add_equation("wz_m - dz(w_m) = 0", tau=True)
problem.add_bc("left(w_m) = B2")
problem.add_bc("left(wz_m) = B3")

# Setup initial guess
solver = problem.build_solver()
logger.info('Solver built')

# Initial conditions
w_g = solver.state['w_g']
w_m = solver.state['w_m']
wz_m = solver.state['wz_m']

w_g['g'] = wg_0 * (1 + a * z)
w_m['g'] = 1 + a*z

# Iterations
pert = solver.perturbations.data
pert.fill(1+tolerance)
start_time = time.time()
while np.sum(np.abs(pert)) > tolerance:
    solver.newton_iteration()
    logger.info('Perturbation norm: {}'.format(np.sum(np.abs(pert))))
end_time = time.time()
logger.info('-'*20)
logger.info('Iterations: {}'.format(solver.iteration))
logger.info('Run time: %.2f sec' %(end_time-start_time))


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(1, 2)
w_g.set_scales(1)
w_m.set_scales(1)
ax1.plot(w_g['g'], z, '.-', label='w_g')
ax1.plot(w_m['g'], z, '.-', label='w_m')
ax1.legend()
ax2.plot(w_m['g']-w_g['g'], z, '.-', label='w_m - w_g')
ax2.legend()
plt.savefig("background.pdf")