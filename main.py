import numpy as np
import matplotlib.pyplot as plt

from dynamics import Dynamics
from multiple_shooting import *

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

u0 = [0]
w_opt = multiple_shooting(cart_pole_frk4_integrator, T, x0, u0)
plot_multiple_shooting(w_opt, Ts, T)