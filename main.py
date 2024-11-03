import numpy as np
import matplotlib.pyplot as plt

from casadi import *
from dynamics import Dynamics
from multiple_shooting import *
from qp import *

# Define parameters
params = {
    'm':0.1,
    'Ma':1.0,
    'l':0.5,
    'g':9.81,
    'Ts':0.1
}

integrator = "FRK4"
cart_pole_frk4_integrator = Dynamics(params, integrator)


# Simulate system with random control inputs
T = 10
Ts = params['Ts']
x0 = [0.0, np.pi, 0.0, 0.0] # Initial state
u_random = np.random.uniform(-2, 2, T) # Control input

x_traj = cart_pole_frk4_integrator.simulate_integrated_dynamics(x0, u_random, T)
cart_pole_frk4_integrator.plot_trajectory(x_traj, T)

#Multiple shooting
u0 = [0]
w_opt, nlp_multiple_shooting = multiple_shooting(cart_pole_frk4_integrator, T, x0, u0)
plot_multiple_shooting(w_opt, Ts, T)

#NLP without IPOPT
H, d_g_x, d_phi_x, g_x = prepare_qp(nlp_multiple_shooting)

w = DM.zeros(54, 1)  # Initial guess for w
lambd = DM.zeros(44, 1)

for n in range(10):
    w, lambd = run_newton_step(H, d_g_x, d_phi_x, g_x, w, lambd)
plot_multiple_shooting(w, Ts, T)