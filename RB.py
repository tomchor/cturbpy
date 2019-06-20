"""
To run, merge, and plot using 4 processes, for instance, you could use:
    $ mpiexec -n 4 python3 script.py
    $ mpiexec -n 4 python3 merge.py snapshots
    $ mpiexec -n 4 python3 plot_2d_series.py snapshots/*.h5
"""
import numpy as np
from mpi4py import MPI
import time

from dedalus import public as de
from dedalus.extras import flow_tools

import logging
logger = logging.getLogger(__name__)


# Parameters
Lx, Lz = (78.3, 26.1)
ν = 1e-5 # Viscosity
χ = 1e-5 # Thermal diffusivity
Prandtl = ν/χ 
R = 287.058 #
g = -9.81 # m/s**2
Cp = 1.003 # kJ/kg*K
Cv = .7176 # kJ/kg*K
γ = 5/3
tstop = 100       # simulation stop time
tstop_wall = 24 * 60 * 60

# Create bases and domain
N = 128
x_basis = de.Fourier('x', N, interval=(0, Lx), dealias=3/2)
z_basis = de.Chebyshev('z', N, interval=(0, Lz), dealias=3/2)
domain = de.Domain([x_basis, z_basis], grid_dtype=np.float64)

#--------
# Define variables
problem = de.IVP(domain, variables=['u', 'uz', 'w', 'wz', 'Y', 'T', 'Sp', 'Qz'])
#--------

#--------
# IC
z = domain.grid(1)
T0 = 1
ρ0 = 1
P0 = 1
H = 1
n = 2 # polytropic index
u0, w0, uz0, wz0 = 0, 0, 0, 0
#--------

#--------
# Define parameters
problem.parameters['g']  = g
problem.parameters['ν'] = ν
problem.parameters['χ'] = χ
problem.parameters['R'] = R
problem.parameters['γ'] = γ
problem.parameters['Cp'] = Cp
problem.parameters['Cv'] = Cv
problem.parameters['T0'] = T0
problem.parameters['P0'] = P0
problem.parameters['ρ0'] = ρ0
problem.parameters['Lz'] = Lz
problem.parameters['H'] = H
problem.parameters['n'] = n
#--------

#--------
problem.substitutions["T_mean"] = "T0" #"T0*(Lz+H-z)/H"
problem.substitutions["ρ_mean"] = "ρ0*((Lz+H-z)/H)**n"
problem.substitutions["P_mean"] = "P0*((Lz+H-z)/H)**(n-1)"
problem.substitutions["lnρ"] = "log(ρ_mean)"
#--------

#--------
# Substitutions
problem.substitutions["D1p1"] = "dx(dx(w)) + dz(wz) + 2*dz(lnρ)*wz + 1/3*(dx(uz) + dz(wz)) - 2/3*dz(lnρ)*(dx(u) + wz)"
problem.substitutions["D1p2"] = "uz*dx(Y) + 2*wz*dz(Y) + dx(w)*dx(Y) - 2/3*dz(Y)*(dx(u) + wz)"

problem.substitutions["D2p1"] = "dx(dx(u)) + dz(uz) + dz(lnρ)*(uz + dx(w)) + 1/3*(dx(dx(u)) + dx(wz))"
problem.substitutions["D2p2"] = "2*dx(u)*dx(Y) + dx(w)*dz(Y) + uz*dz(Y) - 2/3*dx(Y)*(dx(u) + wz)"

problem.substitutions["D4p1"] = "dx(dx(T)) - dz(Qz) - Qz*dz(lnρ)"
problem.substitutions["D4p2"] = "dx(T)*dx(Y) - Qz*dz(Y)"
problem.substitutions["D4p3"] = "2*(dx(u))**2 + (dx(w))**2 + uz**2 + 2*(wz**2) + 2*uz*dx(w) - 2/3*(dx(u) + wz)**2"
#--------

#--------
# Eq D1
problem.add_equation("dt(w) + dz(T) + T_mean*dz(Y) + T*dz(lnρ) - ν*(D1p1) \
                     = -T*dz(Y) - u*dx(w) - w*wz   + ν*(D1p2)")
# Eq D2
problem.add_equation("dt(u) + dx(T) + T_mean*dx(Y) - ν*(D2p1) \
                     = -T*dx(Y) - u*dx(u) - w*uz + ν*(D2p2)")
# Eq D3
problem.add_equation("dt(Y) + w*dz(lnρ) + dx(u) + wz \
                     = - u*dx(Y) - w*dz(Y)", tau=False)
# Eq D4
problem.add_equation("dt(T) + w*dz(T_mean) + (γ-1)*T_mean*(dx(u) + wz) - χ/Cv*(D4p1) \
                     = - u*dx(T) - w*dz(T) - (γ-1)*T*(dx(u) + wz)      + χ/Cv*(D4p2) + \
                     (ν/Cv)*(D4p3)")
# Eq D5
problem.add_equation("Qz + dz(T) = 0")
# Eq D6
problem.add_equation("Sp/Cp + T/(γ*T_mean) + (1/Cp)*Y \
                     = (1/γ)*(log(1+T/T_mean) - T/T_mean)")
# Eq D7
problem.add_equation("wz - dz(w) = 0")
# Eq D8
problem.add_equation("uz - dz(u) = 0")
#--------

# Boundary conditions
#problem.add_bc("left(Y) = 1")
problem.add_bc("left(dz(T)) = 0")
problem.add_bc("right(dz(T)) = 0")
problem.add_bc("left(u) = 0")
problem.add_bc("right(u) = 0")
problem.add_bc("left(w) = 0")
problem.add_bc("right(w) = 0")



# Build solver
solver = problem.build_solver(de.timesteppers.RK443)
logger.info('Solver built')

x = domain.grid(0)
z = domain.grid(1)
Y = solver.state['Y']
T = solver.state['T']
u = solver.state['u']
uz = solver.state['uz']
w = solver.state['w']
wz = solver.state['wz']


# Initial conditions
u['g'] = u0
w['g'] = w0
Y['g'] = 1
T['g'] = 0 + np.random.randn(*T['g'].shape)*1e-3
uz['g'] = uz0
wz['g'] = wz0


logger.info("Y = {:g} -- {:g}".format(np.min(Y['g']), np.max(Y['g'])))
logger.info("T = {:g} -- {:g}".format(np.min(T['g']), np.max(T['g'])))


# Initial timestep
dt = 1e-4

# Integration parameters
solver.stop_sim_time = tstop
solver.stop_wall_time = tstop_wall
solver.stop_iteration = np.inf

# Analysis
snapshots = solver.evaluator.add_file_handler('snaps', iter=1, max_writes=50)
snapshots.add_task('T')
snapshots.add_task('u')
snapshots.add_task('w')
snapshots.add_task('Y')
snapshots.add_task('uz')
snapshots.add_task('wz')

#------
# CFL
CFL = flow_tools.CFL(solver, initial_dt=dt, cadence=10, safety=1,
                     max_change=1.5, min_change=0.5, max_dt=0.125, threshold=0.05)
CFL.add_velocities(('u', 'w'))
#------

#------
# Flow properties
flow = flow_tools.GlobalFlowProperty(solver, cadence=10)
flow.add_property("u*u + w*w", name='KE')
#------


# Main loop
try:
    logger.info('Starting loop')
    start_time = time.time()
    while solver.ok:
        dt = CFL.compute_dt()
        dt = solver.step(dt)
        if (solver.iteration-1) % 1 == 0:
            logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))
            logger.info('Max KE = %f' %flow.max('KE'))
        if np.isnan(flow.max('KE')):
            break
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    end_time = time.time()
    logger.info('Iterations: %i' %solver.iteration)
    logger.info('Sim end time: %f' %solver.sim_time)
    logger.info('Run time: %.2f sec' %(end_time-start_time))
    logger.info('Run time: %f cpu-hr' %((end_time-start_time)/60/60*domain.dist.comm_cart.size))
    logger.info('Final KE: %f' %(flow.max('KE')))


