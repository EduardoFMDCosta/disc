from casadi import *
import numpy as np
import matplotlib.pyplot as plt

def multiple_shooting(int_dynamics, T, x0, u0):

    # Integrated dynamics
    Ts = int_dynamics.Ts
    F = int_dynamics.integrated_dynamics['F']

    # Start with an empty NLP
    w = []
    w0 = []
    J = 0.0
    g=[]
    lbg = []
    ubg = []

    # "Lift" initial conditions
    Xk = MX.sym('X0', 4)
    w += [Xk]
    w0 += [*x0]

    # Initial condition constraint
    #x_init = MX.sym('x_init', 4)
    g += [Xk - x0]
    lbg += [0, 0, 0, 0]
    ubg += [0, 0, 0, 0]

    # Formulate the NLP
    for k in range(T):
        # New NLP variable for the control
        Uk = MX.sym('U_' + str(k))
        w   += [Uk]
        w0  += [*u0]

        # Integrate till the end of the interval
        Fk = F(x0 = Xk, p = Uk)
        Xk_end = Fk['xf']
        J = J + Fk['qf']

        # New NLP variable for state at end of interval
        Xk = MX.sym('X_' + str(k+1), 4)
        w   += [Xk]
        w0  += [0, 0, 0, 0]

        # Add equality constraint
        g += [Xk_end - Xk]
        lbg += [0, 0, 0, 0]
        ubg += [0, 0, 0, 0]

    # Create an NLP solver
    nlp = {'f': J, 'x': vertcat(*w), 'g': vertcat(*g)}
    solver = nlpsol('solver', 'ipopt', nlp)

    # Solve the NLP
    sol = solver(x0 = w0, lbg = lbg, ubg = ubg)
    w_opt = sol['x'].full().flatten()

    return w_opt, nlp

def plot_multiple_shooting(w_opt, Ts, T, title):
    # Plot the solution
    x1_opt = w_opt[0::5]
    x2_opt = w_opt[1::5]
    x3_opt = w_opt[2::5]
    x4_opt = w_opt[3::5]
    u_opt = w_opt[4::5]

    tgrid = [Ts * k for k in range(T + 1)]
    plt.figure(1)
    plt.clf()
    plt.plot(tgrid, x1_opt, '-..')
    plt.plot(tgrid, x2_opt, '--.')
    plt.plot(tgrid, x3_opt, '-..')
    plt.plot(tgrid, x4_opt, '--.')
    plt.plot(tgrid, vertcat(DM.nan(1), u_opt), '-.')
    plt.xlabel('t')
    plt.legend(['x', 'theta', 'x_dot', 'theta_dot', 'u'])
    plt.title(title)
    plt.grid()
    plt.show()

    return w_opt