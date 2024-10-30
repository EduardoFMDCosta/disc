from dynamics import Dynamics

params = {
    'm':0.1,
    'Ma':1.0,
    'l':0.5,
    'g':9.81,
    'T':10,
    'Ts':0.1
}

integrator = "FRK4"
cart_pole_frk4_integrator = Dynamics(params, integrator)
print(cart_pole_frk4_integrator.integrated_dynamics)