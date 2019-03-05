"""Conduit background solvers."""

import time
import numpy as np
import matplotlib.pyplot as plt

from dedalus import public as de

import logging
logger = logging.getLogger(__name__)


def compute_background(domain, params, plot=False):

    # Setup problem
    problem = de.NLBVP(domain, variables=['phi', 'phi_rho_g', 'w_g', 'w_m', 'wz_m'], ncc_cutoff=params.bvp_ncc_cutoff)
    problem.parameters['alpha'] = params.alpha
    problem.parameters['beta'] = params.beta
    problem.parameters['B0'] = params.phi_0
    problem.parameters['B1'] = params.phi_0 * params.rho_0
    problem.parameters['B2'] = 1
    problem.parameters['B3'] = params.a
    problem.substitutions['dt(A)'] = "0*A"
    problem.add_equation("- dt(phi) + dz(w_m) = dz(phi * w_m)", tau=True)
    problem.add_equation("dt(phi_rho_g) + dz(phi_rho_g) = dz(phi_rho_g) - dz(phi_rho_g * w_g)", tau=True)
    problem.add_equation("alpha*(w_m - w_g) - beta*dz(phi_rho_g) - phi_rho_g = - beta*phi_rho_g * dz(phi) / phi", tau=False)
    problem.add_equation("(4/3)*alpha*(dz(wz_m)) - alpha*(w_m - w_g) - (phi - phi_rho_g) = (4/3)*alpha*(1+phi**2)*dz(phi)/phi*wz_m - phi*(phi - phi_rho_g)", tau=True)
    problem.add_equation("wz_m - dz(w_m) = 0", tau=True)
    problem.add_bc("left(phi) = B0")
    problem.add_bc("left(phi_rho_g) = B1")
    problem.add_bc("left(w_m) = B2")
    problem.add_bc("left(wz_m) = B3")

    # Setup initial guess
    solver = problem.build_solver()
    logger.info('Solver built')

    # Initial conditions
    z = domain.grid(0)
    phi = solver.state['phi']
    phi_rho_g = solver.state['phi_rho_g']
    w_g = solver.state['w_g']
    w_m = solver.state['w_m']
    wz_m = solver.state['wz_m']

    phi['g'] = params.phi_0 + (1 - params.phi_0) * params.a * z
    rho_g_init = params.rho_0 * np.exp(-z * params.b / params.rho_0)
    phi_rho_g['g'] = phi['g'] * rho_g_init
    w_g['g'] = params.wg_0 * (1 + params.a * z)
    w_m['g'] = 1 + params.a*z

    # Iterations
    pert = solver.perturbations.data
    pert.fill(1+params.tolerance)
    start_time = time.time()
    while np.sum(np.abs(pert)) > params.tolerance:
        solver.newton_iteration()
        logger.info('Perturbation norm: {}'.format(np.sum(np.abs(pert))))
    end_time = time.time()
    logger.info('-'*20)
    logger.info('Iterations: {}'.format(solver.iteration))
    logger.info('Run time: %.2f sec' %(end_time-start_time))

    if plot:
        rho_g = (phi_rho_g / phi).evaluate()

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
        phi.set_scales(1)
        rho_g.set_scales(1)
        w_g.set_scales(1)
        w_m.set_scales(1)
        ax1.plot(phi['g'], z, '.-')
        ax1.set_title('phi')
        ax2.plot(rho_g['g'], z, '.-')
        ax2.set_title('rho_g')
        ax3.plot(w_m['g'], z, '.-')
        ax3.set_title('w_m')
        ax4.plot(w_g['g'], z, '.-')
        ax4.set_title('w_g')
        plt.savefig("background.pdf")

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        ax1.plot(np.abs(phi['c']), '.-')
        ax1.set_title('phi')
        ax1.set_yscale('log')
        ax2.plot(np.abs(rho_g['c']), '.-')
        ax2.set_title('rho_g')
        ax2.set_yscale('log')
        ax3.plot(np.abs(w_m['c']), '.-')
        ax3.set_title('w_m')
        ax3.set_yscale('log')
        ax4.plot(np.abs(w_g['c']), '.-')
        ax4.set_title('w_g')
        ax4.set_yscale('log')
        plt.savefig("background_coeffs.pdf")

    return solver

if __name__=='__main__':

    import parameters as params

    z_basis = de.Chebyshev('z', params.Nz, interval=(0, params.Lz), dealias=2)
    domain = de.Domain([z_basis], grid_dtype=np.float64)

    compute_background(domain, params, plot=True)
