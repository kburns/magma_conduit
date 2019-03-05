"""Simulation parameters."""

import numpy as np

# Parameters
Nz = 64
Lz = 136
alpha = 8
beta = 1.3e3
phi_0 = 0.1
rho_0 = 0.08

# Derived parameters
a = phi_0 / rho_0 / beta * alpha / (alpha + 2 * phi_0 * (1 - rho_0))
b = (1 - phi_0 + phi_0 * rho_0) / beta
c = (2 - 2 * rho_0 + rho_0 * alpha) * (phi_0 / alpha / beta / rho_0)
wg_0 = 1 + (phi_0 * (1 - phi_0) * (1 - rho_0))

# NLBVP parameters
bvp_ncc_cutoff = 1e-10
tolerance = 1e-10

# IVP parameters
amp = 1e-1
omega = 2 * np.pi * 0.1
ivp_ncc_cutoff = 1e-6
dt = 2e-4
snapshot_iter = 100
stop_iteration = 10000
print_iter = 100
