import numpy as np
from mpi4py import MPI
import time

from dedalus import public as de
from dedalus.extras import flow_tools

import logging
logger = logging.getLogger(__name__)


# Parameters
Lx, Lz = (2, 1)
ν = 0.00001  # problem parameters
Prandtl = 1 
γ = 5.0/3.0
R = 0.8
Cv = R/(γ-1)
g = -1.0
tstop = 100       # simulation stop time
tstop_wall = 200 # max walltime limit in hours

# Create bases and domain
x_basis = de.Fourier('x', 256, interval=(0, Lx), dealias=3/2)
z_basis = de.Chebyshev('z', 256, interval=(0, Lz), dealias=3/2)
domain = de.Domain([x_basis, z_basis], grid_dtype=np.float64)

# Compressible NS
problem = de.IVP(domain, variables=['w','wz','u','uz','T','Tz','ρ','ρz','P'])


problem.parameters['g']  = g
problem.parameters['Pr']  = Prandtl
problem.parameters['ν'] = ν
problem.parameters['Cv'] = Cv
problem.parameters['κ'] = Prandtl*ν
problem.parameters['γ'] = γ


problem.substitutions["Πxx"] = "2*dx(u) - 2/3*(dx(u) + wz)"
problem.substitutions["Πzz"] = "2*dz(w) - 2/3*(dx(u) + wz)"
problem.substitutions["Πxz"] = "dx(w) + uz"
problem.substitutions["Πzx"] = "dx(w) + uz"


problem.add_equation("dz(u) - uz = 0")
problem.add_equation("dz(w) - wz = 0")
problem.add_equation("dz(T) - Tz = 0")
problem.add_equation("dz(ρ) - ρz = 0")
problem.add_equation("P = ρ*T")

#problem.add_equation("dt(ρ) + ρ*wz + ρ*dx(u) = -w*dz(ρ) - u*dx(ρ)")
#problem.add_equation("dt(u) + dx(T) - ν*dx(t_xx) - ν*dz(t_xz) = - w*dz(u) - u*dx(u) - T*dx(log_rho)   + ν*t_xx*dx(log_rho) + ν*t_xz*dz(log_rho)")
#problem.add_equation("dt(w) + Tz   - ν*dz(t_zz) - ν*dx(t_xz) = - w*wz - u*dx(w) - T*dz(log_rho) + g + ν*t_zz*dz(log_rho) + ν*t_xz*dx(log_rho)")
#problem.add_equation("dt(T) - 1/Cv*(κ*(dz(Tz) + dx(dx(T)))) = - w*Tz - u*dx(T) - (γ-1)*T*(wz + dx(u)) + 1/Cv*(κ*(Tz*dz(log_rho) + dx(T)*dx(log_rho)) + ν*t_zz*wz + ν*t_xx*dx(u) + ν*t_xz*uz + ν*t_xz*dx(w))")
#problem.add_equation("dt(T) - 1/Cv*(κ*(dz(Tz) + dx(dx(T)))) = - w*Tz    - u*dx(T) - (γ-1)*T*(wz + dx(u)) + 1/Cv*(κ*(Tz*dz(ρ)/ρ + dx(T)*dx(ρ)/ρ) + ν*Πzz*wz + ν*Πxx*dx(u) + ν*Πxz*uz + ν*Πxz*dx(w))")
problem.add_equation("dt(ρ)                                 = - dz(w*ρ) - dx(u*ρ)")
problem.add_equation("dt(u) - ν*dx(Πxx) - ν*dz(Πxz)         = - u*dx(u) - w*dz(u)   + dx(P)     + ν*Πxx*dx(ρ)/ρ + ν*Πxz*dz(ρ)/ρ")
problem.add_equation("dt(w) + ν*dz(Πzx) - ν*dx(Πzz)         = - u*dx(w) - w*dz(w)   + dz(P)     + ν*Πzx*dx(ρ)/ρ + ν*Πzz*dz(ρ)/ρ")
problem.add_equation("dt(T) - κ*(dz(Tz) + dx(dx(T)))        = - u*dx(T) - w*dz(T)   + κ*(Tz*dz(ρ)/ρ + dx(T)*dx(ρ)/ρ) \
                                                                                    + ν*Πzz*wz + ν*Πxx*dx(u) + ν*Πxz*uz + ν*Πxz*dx(w)")

#problem.add_bc("right(dz(rho_1)) = 0")
problem.add_bc("left(dz(ρ)) = 0")
problem.add_bc("left(T) = 270.01")
problem.add_bc("right(T) = 270.000")
problem.add_bc("left(u) = 0")
problem.add_bc("right(u) = 0")
problem.add_bc("left(w) = 0")
problem.add_bc("right(w) = 0")



# Build solver
solver = problem.build_solver(de.timesteppers.RK443)
logger.info('Solver built')

# Initial conditions
x = domain.grid(0)
z = domain.grid(1)
ρ = solver.state['ρ']
T = solver.state['T']
Tz = solver.state['Tz']

solver.evaluator.vars['Lx'] = Lx
solver.evaluator.vars['Lz'] = Lz

# initially UNstable stratification

tanh_width = Lz/20.0
tanh_center = 0.5*Lz
rho = 1.2

A_u = 1
sigma = 0.2
amp = -0.02
ρ['g'] = rho
Press = 50 + rho*g*z
T['g'] = Press/(rho*R)
T.differentiate('z',out=Tz)

logger.info("Au = {:g}".format(A_u))
logger.info("ρ = {:g} -- {:g}".format(np.min(ρ['g']), np.max(ρ['g'])))
logger.info("T = {:g} -- {:g}".format(np.min(T['g']), np.max(T['g'])))


# Initial timestep
dt = 0.001

# Integration parameters
solver.stop_sim_time = tstop
solver.stop_wall_time = tstop_wall
solver.stop_iteration = np.inf

# Analysis
snapshots = solver.evaluator.add_file_handler('snapshots', iter=1, max_writes=50)
snapshots.add_task('T')
snapshots.add_task('w')
snapshots.add_task('wz')
snapshots.add_task('T*ρ', name='P')
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


