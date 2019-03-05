"""Conduit IVP simulation."""

import numpy as np
from mpi4py import MPI
import time

from dedalus import public as de
from dedalus.extras import flow_tools
import parameters as params
import background

import logging
logger = logging.getLogger(__name__)
np.seterr(all='raise')


# Build domain
z_basis = de.Chebyshev('z', params.Nz, interval=(0, params.Lz), dealias=2)
domain = de.Domain([z_basis], grid_dtype=np.float64)

# Solve for background
bg_solver = background.compute_background(domain, params)

# Setup problem
problem = de.IVP(domain, variables=['phi', 'phi_rho_g', 'w_g', 'w_m', 'wz_m'], ncc_cutoff=params.bvp_ncc_cutoff)
problem.parameters['alpha'] = params.alpha
problem.parameters['beta'] = params.beta
problem.parameters['B0'] = params.phi_0
problem.parameters['B1'] = params.phi_0 * params.rho_0
problem.parameters['B2'] = 1
problem.parameters['B3'] = params.a
problem.parameters['omega'] = params.omega
problem.parameters['amp'] = params.amp
problem.add_equation("- dt(phi) + dz(w_m) = dz(phi * w_m)", tau=True)
problem.add_equation("dt(phi_rho_g) + dz(phi_rho_g) = dz(phi_rho_g) - dz(phi_rho_g * w_g)", tau=True)
problem.add_equation("alpha*(w_m - w_g) - beta*dz(phi_rho_g) - phi_rho_g = - beta*phi_rho_g * dz(phi) / phi", tau=False)
problem.add_equation("(4/3)*alpha*(dz(wz_m)) - alpha*(w_m - w_g) - (phi - phi_rho_g) = (4/3)*alpha*(1+phi**2)*dz(phi)/phi*wz_m - phi*(phi - phi_rho_g)", tau=True)
problem.add_equation("wz_m - dz(w_m) = 0", tau=True)
problem.add_bc("left(phi) = B0 * (1 + amp*sin(omega*t))")
problem.add_bc("left(phi_rho_g) = B1 * (1 + amp*sin(omega*t))")
problem.add_bc("left(w_m) = B2")
problem.add_bc("left(wz_m) = B3")

# Build solver
solver = problem.build_solver(de.timesteppers.SBDF1)
logger.info('Solver built')

# Initial conditions
for var in problem.variables:
    solver.state[var]['c'] = bg_solver.state[var]['c']

# Integration parameters
solver.stop_sim_time = np.inf
solver.stop_wall_time = np.inf
solver.stop_iteration = params.stop_iteration

# Analysis
snapshots = solver.evaluator.add_file_handler('snapshots', iter=params.snapshot_iter, max_writes=50)
snapshots.add_system(solver.state)
snapshots.add_task("phi_rho_g / phi", name="rho_g")

coefficients = solver.evaluator.add_file_handler('coefficients', iter=params.snapshot_iter, max_writes=50)
coefficients.add_system(solver.state, layout='c')
coefficients.add_task("phi_rho_g / phi", name="rho_g", layout='c')

# Main loop
dt = params.dt
try:
    logger.info('Starting loop')
    start_time = time.time()
    while solver.ok:
        solver.step(dt)
        if (solver.iteration-1) % params.print_iter == 0:
            logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    end_time = time.time()
    logger.info('Iterations: %i' %solver.iteration)
    logger.info('Sim end time: %f' %solver.sim_time)
    logger.info('Run time: %.2f sec' %(end_time-start_time))
    logger.info('Run time: %f cpu-hr' %((end_time-start_time)/60/60*domain.dist.comm_cart.size))
