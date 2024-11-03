from casadi import *

def prepare_qp(nlp):

    # Extract NLP components
    w = nlp['x']       # Decision variables
    g = nlp['g']       # Constraints
    phi = nlp['f']     # Objective function

    ng = g.size1()  # Number of constraints
    nw = w.size1()  # Number of decision variables

    # Define dual variables (Lagrange multipliers) as symbolic variables
    lambd = MX.sym('lambda', ng)  # Dual variables

    # Define CasADi Functions for constraints and objective
    g_x = Function('c', [w], [g], ['w'], ['g'])
    phi_x = Function('phi_x', [w], [phi], ['w'], ['phi'])
    d_g_x = Function('d_g_x', [w], [jacobian(g, w)], ['w'], ['d_g_x'])
    d_phi_x = Function('d_phi_x', [w], [jacobian(phi, w)], ['w'], ['Jx'])

    # Lagrangian function: phi + y^T * g
    L = Function('L', [w, lambd], [phi + mtimes(lambd.T, g)], ['w', 'lambda'], ['L'])

    # Hessian of the Lagrangian with respect to w
    H = Function('H', [w, lambd], [jacobian(jacobian(L(w, lambd), w), w)], ['w', 'lambda'], ['H'])

    return H, d_g_x, d_phi_x, g_x

def run_newton_step(H, d_g_x, d_phi_x, g_x, w, lambd):

    #Evaluate functions at points
    H_ = H(w, lambd)
    d_g_x_ = d_g_x(w)
    d_phi_x_ = d_phi_x(w)
    g_x_ = g_x(w)

    # Set up the KKT system
    KKT_matrix = vertcat(
        horzcat(H_, d_g_x_.T),
        horzcat(d_g_x_, DM.zeros(d_g_x_.size1(), d_g_x_.size1()))
    )

    KKT_rhs = vertcat(-d_phi_x_.T, -g_x_)

    # Solve the linear system for the Newton step
    delta = solve(KKT_matrix, KKT_rhs)

    # Extract step for w and y
    delta_w = delta[:w.size1()]
    delta_lambd = delta[w.size1():]

    # Update estimates of w and y
    w_new = w + delta_w
    lambd_new = lambd + delta_lambd

    return w_new, lambd_new