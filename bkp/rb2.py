

import numpy as np
from mpi4py import MPI
import time

from dedalus import public as de
from dedalus.extras import flow_tools

import logging
logger = logging.getLogger(__name__)


# Parameters
Lx, Lz = (1., 2.)
nu = 0.00001  # problem parameters
Prandtl = 1 
gamm = 5.0/3.0
R = 0.8
Cv = R/(gamm-1)
g = -1.0
tstop = 100       # simulation stop time
tstop_wall = 200 # max walltime limit in hours

# Create bases and domain
x_basis = de.Fourier('x', 128, interval=(0, Lx), dealias=3/2)
z_basis = de.Chebyshev('z', 256, interval=(0, Lz), dealias=3/2)
domain = de.Domain([x_basis, z_basis], grid_dtype=np.float64)

# 1D compressible NS
problem = de.IVP(domain, variables=['w', 'w_z','u', 'u_z', 'log_rho','T', 'T_z', 'rho_1'])


problem.parameters['g']  = g
problem.parameters['Pr']  = Prandtl
problem.parameters['dim_cof'] = 2.0/3.0
problem.parameters['nu'] = nu
problem.parameters['Cv'] = Cv
problem.parameters['kappa'] = Prandtl*nu
problem.parameters['gamm'] = gamm


problem.substitutions["t_xx"] = ("dim_cof*(2*dx(u) - w_z)")
problem.substitutions["t_xz"] =("dx(w) + u_z")
problem.substitutions["t_zz"] =("dim_cof*(2*w_z - dx(u))")


problem.add_equation("dz(u) - u_z = 0")
problem.add_equation("dz(w) - w_z = 0")
problem.add_equation("dz(T) - T_z = 0")
problem.add_equation("rho_1 = exp(log_rho)")

problem.add_equation("dt(log_rho) + w_z + dx(u) = -w*dz(log_rho) - u*dx(log_rho)")
problem.add_equation("dt(u) + dx(T) - nu*dx(t_xx) - nu*dz(t_xz) = - w*dz(u) - u*dx(u) - T*dx(log_rho)   + nu*t_xx*dx(log_rho) + nu*t_xz*dz(log_rho)")
problem.add_equation("dt(w) + T_z   - nu*dz(t_zz) - nu*dx(t_xz) = - w*w_z - u*dx(w) - T*dz(log_rho) + g + nu*t_zz*dz(log_rho) + nu*t_xz*dx(log_rho)")
problem.add_equation("dt(T) - 1/Cv*(kappa*(dz(T_z) + dx(dx(T)))) = - w*T_z - u*dx(T) - (gamm-1)*T*(w_z + dx(u)) + 1/Cv*(kappa*(T_z*dz(log_rho) + dx(T)*dx(log_rho)) + nu*t_zz*w_z + nu*t_xx*dx(u) + nu*t_xz*u_z + nu*t_xz*dx(w))")

#problem.add_bc("right(dz(rho_1)) = 0")
#problem.add_bc("left(dz(rho_1)) = 0")
problem.add_bc("left(T) = 270.001")
problem.add_bc("right(T) = 270.00")
problem.add_bc("left(dz(u)) = 0")
problem.add_bc("right(dz(u)) = 0")
problem.add_bc("left(w) = 0")
problem.add_bc("right(w) = 0")



# Build solver
solver = problem.build_solver(de.timesteppers.RK443)
logger.info('Solver built')

# Initial conditions
x = domain.grid(0)
z = domain.grid(1)
log_rho = solver.state['log_rho']
T = solver.state['T']
T_z = solver.state['T_z']
rho_1 = solver.state['rho_1']

solver.evaluator.vars['Lx'] = Lx
solver.evaluator.vars['Lz'] = Lz

# initially UNstable stratification

tanh_width = Lz/20.0
tanh_center = 0.5*Lz
rho_h = 5.0
rho_l = 4.0
#phi =  rho_l + (rho_h-rho_l)*0.5*(1-np.tanh((z-tanh_center)/tanh_width))
#rho = rho_l + (rho_h-rho_l)*0.5*(1+np.tanh((z-tanh_center)/tanh_width))
rho = 4.0

A_u = 1
sigma = 0.2
amp = -0.02
log_rho['g'] = np.log(rho)
rho_1['g'] = rho
#Press = 30.0 + 0.5*(rho_h+rho_l)*z + 0.5*tanh_width*(rho_l-rho_h)*np.log(np.cosh((z-tanh_center)/tanh_width))
Press = 50 + rho*g*z
#Press = 50.0 + g*(0.5*(rho_h+rho_l)*z + 0.5*tanh_width*(rho_l-rho_h)*np.log(np.cosh((z-tanh_center)/tanh_width)))
T['g'] = Press/(rho*R)
T.differentiate('z',out=T_z)

logger.info("Au = {:g}".format(A_u))
logger.info("log_rho = {:g} -- {:g}".format(np.min(np.exp(log_rho['g'])), np.max(np.exp(log_rho['g']))))
logger.info("T = {:g} -- {:g}".format(np.min(T['g']), np.max(T['g'])))


# Initial timestep
dt = 0.00125

# Integration parameters
solver.stop_sim_time = tstop
solver.stop_wall_time = tstop_wall
solver.stop_iteration = np.inf

# Analysis
snapshots = solver.evaluator.add_file_handler('snapshots', iter=1, max_writes=50)
snapshots.add_task('T')
snapshots.add_task('w')
snapshots.add_task('w_z')
snapshots.add_task('0.8*T*exp(log_rho)', name='P')
snapshots.add_task('rho_1', name='rho')
snapshots.add_task('u', name='u')
# CFL
CFL = flow_tools.CFL(solver, initial_dt=dt, cadence=10, safety=1,
                     max_change=1.5, min_change=0.5, max_dt=0.125, threshold=0.05)
CFL.add_velocities(('u', 'w'))

# Flow properties
flow = flow_tools.GlobalFlowProperty(solver, cadence=10)
flow.add_property("T", name='u_rms')

# Main loop
try:
    logger.info('Starting loop')
    start_time = time.time()
    while solver.ok:
        dt = CFL.compute_dt()
        dt = solver.step(dt)
        if (solver.iteration-1) % 1 == 0:
            logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))
            logger.info('Max u)rms = %f' %flow.max('u_rms'))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    end_time = time.time()
    logger.info('Iterations: %i' %solver.iteration)
    logger.info('Sim end time: %f' %solver.sim_time)
    logger.info('Run time: %.2f sec' %(end_time-start_time))
    logger.info('Run time: %f cpu-hr' %((end_time-start_time)/60/60*domain.dist.comm_cart.size))


