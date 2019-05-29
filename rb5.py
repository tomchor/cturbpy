import numpy as np
from mpi4py import MPI
import time

from dedalus import public as de
from dedalus.extras import flow_tools

import logging
logger = logging.getLogger(__name__)


# Parameters
Lx, Lz = (78.3, 26.1)
ν = 1e-5 #
χ = 1e-5 # Thermal diffusivity
Prandtl = 1 
R = 287.058 #
g = -9.81 # m/s**2
Cp = 1.003 # kJ/kg*K
γ = 5/3
tstop = 100       # simulation stop time
tstop_wall = 50 

# Create bases and domain
x_basis = de.Fourier('x', 256, interval=(0, Lx), dealias=3/2)
z_basis = de.Chebyshev('z', 256, interval=(0, Lz), dealias=3/2)
domain = de.Domain([x_basis, z_basis], grid_dtype=np.float64)

#--------
# Define variables
problem = de.IVP(domain, variables=['u', 'uz', 'w', 'wz', 'Y', 'T','Sp', 'Qz', 'lnρ', 'ρ'])
problem.add_equation("ρ = exp(lnρ)")
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
problem.parameters['T0'] = T0
problem.parameters['ρ0'] = ρ0
problem.parameters['Lz'] = Lz
problem.parameters['H'] = H
#--------

#--------
problem.substitutions["T_mean"] = "T0*(Lz+H-z)/H"
problem.substitutions["ρ_mean"] = ρ0*((Lz+H-z)/H)**n
problem.substitutions["P_mean"] = P0*((Lz+H-z)/H)**(n-1)
#--------


#--------
# Eq D1
problem.substitutions["D1p1"] = "dx(dx(w)) + dz(wz) + 2*dz(lnρ)*wz + 1/3*(dx(uz) + dz(wz)) - 2/3*dz(lnρ)*(dx(u) + wz)"
problem.substitutions["D1p2"] = "dz(dx(Y)) + 2*wz*dz(Y) + dx(w)*dx(Y) - 2/3*dz(Y)*(dx(u) + wz)"
problem.add_equation("dt(w) + dz(T) + T_mean*dz(Y) + T*dz(lnρ) - ν*(D1p1) \
                     = -T*dz(Y) - u*dx(w) - w*wz   + ν*(D1p2)")
#--------

#--------
# Eq D2
problem.substitutions["D2p1"] = "dx(dx(u)) + dz(uz) + dz(lnρ)*(uz + dx(w)) + 1/3*(dx(dx(u)) + dx(wz))"
problem.substitutions["D2p2"] = "2*dx(u)dx(Y) + dx(w)*dz(Y) + uz*dz(Y) - 2/3*dx(Y)*(dx(u) + wz)"
problem.add_equation("dt(u) + dx(T) + T_mean*dx(Y) - ν*(D2p1) \
                     = -T*dx(Y) - u*dx(u) - w*uz + ν*(D2p2)")
#--------

#--------
# Eq D3
problem.add_equation("dt(Y) + w*dz(lnρ) + dx(u) + wz \
                     = - u*dx(Y) - w*dz(Y)")
#--------

#--------
# Eq D4
problem.substitutions["D4p1"] = "dx(dx(T)) - dz(Qz) - Qz*dz(lnρ)"
problem.substitutions["D4p2"] = "dx(T)*dx(Y) - Qz*dz(Y)"
problem.substitutions["D4p3"] = "2*(dx(u))**2 + (dx(w))**2 + uz**2 + 2*wz + 2*uz*dx(w) - 2/3*(dx(u) + wz)**2"
problem.add_equation("dt(T) + w*dz(T_mean) + (γ-1)*T_mean*(dx(u) + wz) - χ/Cv*(D4p1) \
                     = - u*dx(T) - w*dz(T) - (γ-1)*T*(dx(u) + wz) + χ/Cv*(D4p2) + \
                     (ν/Cv)*(D4p3)")
#--------

#--------
# Eq D5
problem.add_equation("Qz + dz(T) = 0")
#--------

#--------
# Eq D6
problem.add_equation("Sp/Cp + T/(γ*T_mean) + 1/Cp*Y \
                     = 1/γ*(log(1+T/T_mean) - T/T_mean)")
#--------

#--------
# Eq D7
problem.add_equation("wz - dz(w) = 0")
#--------

#--------
# Eq D8
problem.add_equation("uz - dz(u) = 0")
#--------

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


# Initial conditions
u['g'] = u0
w['g'] = w0
ρ['g'] = ρ0
T['g'] = P0/(ρ0*R)
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


