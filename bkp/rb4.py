import numpy as np
from mpi4py import MPI
import time

from dedalus import public as de
from dedalus.extras import flow_tools

import logging
logger = logging.getLogger(__name__)


# Parameters
Lx, Lz = (2, 1)
ν = 1e-5
Prandtl = 1 
R = 287.058
g = -9.81
tstop = 100       # simulation stop time
tstop_wall = 50 

# Create bases and domain
x_basis = de.Fourier('x', 256, interval=(0, Lx), dealias=3/2)
z_basis = de.Chebyshev('z', 256, interval=(0, Lz), dealias=3/2)
domain = de.Domain([x_basis, z_basis], grid_dtype=np.float64)

# Compressible NS
problem = de.IVP(domain, variables=['w','wz','u','uz','T','Tz','ρ','ρz','P'])


problem.parameters['g']  = g
problem.parameters['ν'] = ν
problem.parameters['κ'] = Prandtl*ν
problem.parameters['R'] = R


problem.substitutions["Πxx"] = "2*dx(u) - 2/3*(dx(u) + wz)"
problem.substitutions["Πzz"] = "2*dz(w) - 2/3*(dx(u) + wz)"
problem.substitutions["Πxz"] = "dx(u) + wz"
problem.substitutions["Πzx"] = "dx(u) + wz"


problem.add_equation("dz(u) - uz = 0")
problem.add_equation("dz(w) - wz = 0")
problem.add_equation("dz(T) - Tz = 0")
problem.add_equation("dz(ρ) - ρz = 0")
problem.add_equation("P = R*ρ*T")

problem.add_equation("dt(ρ)                                 = - dz(w*ρ) - dx(u*ρ)")
problem.add_equation("dt(u) - ν*dx(Πxx) - ν*dz(Πxz)         = - u*dx(u) - w*dz(u)   + dx(P)     + ν*Πxx*dx(ρ)/ρ + ν*Πxz*dz(ρ)/ρ")
problem.add_equation("dt(w) + ν*dz(Πzx) - ν*dx(Πzz)         = - u*dx(w) - w*dz(w)   + dz(P)     + ν*Πzx*dx(ρ)/ρ + ν*Πzz*dz(ρ)/ρ")
problem.add_equation("dt(T) - κ*(dz(Tz) + dx(dx(T)))        = - u*dx(T) - w*dz(T)   + κ*(Tz*dz(ρ)/ρ + dx(T)*dx(ρ)/ρ) \
                                                                                    + ν*Πzz*wz + ν*Πxx*dx(u) + ν*Πxz*uz + ν*Πxz*dx(w)")

# Boundary conditions
problem.add_bc("left(ρ) = 1.2")
problem.add_bc("left(T) = 270.01")
problem.add_bc("right(T) = 270.000")
problem.add_bc("left(u) = 0")
problem.add_bc("right(u) = 0")
problem.add_bc("left(w) = 0")
problem.add_bc("right(w) = 0")



# Build solver
solver = problem.build_solver(de.timesteppers.RK443)
logger.info('Solver built')

x = domain.grid(0)
z = domain.grid(1)
ρ = solver.state['ρ']
ρz = solver.state['ρz']
T = solver.state['T']
Tz = solver.state['Tz']
u = solver.state['u']
uz = solver.state['uz']
w = solver.state['w']
wz = solver.state['wz']
P = solver.state['P']

solver.evaluator.vars['Lx'] = Lx
solver.evaluator.vars['Lz'] = Lz

# initially Unstable stratification
rho = 1.2
P0 = 50 + rho*g*z
u0, w0, uz0, wz0 = 0, 0, 0, 0

# Initial conditions
u['g'] = u0
w['g'] = w0
ρ['g'] = rho
T['g'] = P0/(rho*R)
P['g'] = P0
ρz['g'] = 0
T.differentiate('z', out=Tz)
uz['g'] = 0
wz['g'] = 0


logger.info("ρ = {:g} -- {:g}".format(np.min(ρ['g']), np.max(ρ['g'])))
logger.info("T = {:g} -- {:g}".format(np.min(T['g']), np.max(T['g'])))


# Initial timestep
dt = 1e-4

# Integration parameters
solver.stop_sim_time = tstop
solver.stop_wall_time = tstop_wall
solver.stop_iteration = np.inf

# Analysis
snapshots = solver.evaluator.add_file_handler('snapshots', iter=1, max_writes=50)
snapshots.add_task('T')
snapshots.add_task('w')
snapshots.add_task('wz')
snapshots.add_task('P', name='P')
snapshots.add_task('ρ', name='ρ')
snapshots.add_task('u', name='u')

# CFL
CFL = flow_tools.CFL(solver, initial_dt=dt, cadence=10, safety=1,
                     max_change=1.5, min_change=0.5, max_dt=0.125, threshold=0.05)
CFL.add_velocities(('u', 'w'))

# Flow properties
flow = flow_tools.GlobalFlowProperty(solver, cadence=10)

# Main loop
try:
    logger.info('Starting loop')
    start_time = time.time()
    while solver.ok:
        dt = CFL.compute_dt()
        dt = solver.step(dt)
        if (solver.iteration-1) % 1 == 0:
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


