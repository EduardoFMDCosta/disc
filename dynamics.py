from casadi import *
import numpy as np
import matplotlib.pyplot as plt

class Dynamics:
    def __init__(self, params, integrator):
        self.m = params['m']
        self.Ma = params['Ma']
        self.l = params['l']
        self.g = params['g']
        self.Ts = params['Ts']

        self.integrator = integrator
        self.dynamics_elements = self.get_dynamics_elements()
        self.integrated_dynamics = self.get_integrated_dynamics()

    def get_dynamics_elements(self):
        # Define symbolic variables
        x = SX.sym('x')
        theta = SX.sym('theta')
        x_dot = SX.sym('x_dot')
        theta_dot = SX.sym('theta_dot')

        # Define state and control signal
        state = vertcat(x, theta, x_dot, theta_dot)
        u = SX.sym('u')

        # Model equations
        dstate = vertcat(
            x_dot,
            theta_dot,
            (u + self.m * self.l * theta_dot ** 2 * sin(theta) - self.m * self.g * sin(theta) * cos(theta)) / (
                        self.Ma + self.m - self.m * cos(theta) ** 2),
            (-u * cos(theta) - self.m * self.l * theta_dot ** 2 * sin(theta) * cos(theta) + (self.Ma + self.m) * self.g * sin(theta)) / (
                        self.l * (self.Ma + self.m - self.m * cos(theta) ** 2))
        )

        # Objective term
        L = x ** 2 + x_dot ** 2 + theta ** 2 + theta_dot ** 2 + u ** 2

        return state, dstate, u, L


    def get_integrated_dynamics(self):

        state = self.dynamics_elements[0]
        dstate = self.dynamics_elements[1]
        u = self.dynamics_elements[2]
        L = self.dynamics_elements[3]

        # Continuous time dynamics
        f = Function('f', [state, u], [dstate, L])

        if self.integrator == "FRK4":

            # Build integrator using RK4
            M = 4  # RK4 steps per interval
            DT = self.Ts / M
            X0 = MX.sym('X0', 4)
            U = MX.sym('U')
            X = X0
            Q = 0
            for j in range(M):
                k1, k1_q = f(X, U)
                k2, k2_q = f(X + DT / 2 * k1, U)
                k3, k3_q = f(X + DT / 2 * k2, U)
                k4, k4_q = f(X + DT * k3, U)
                X = X + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
                Q = Q + DT / 6 * (k1_q + 2 * k2_q + 2 * k3_q + k4_q)

            FRK4 = Function('FRK4', [X0, U], [X, Q], ['x0', 'p'], ['xf', 'qf'])

            # Return a dictionary representing the system
            sys = {'x': state, 'u': u, 'dx': dstate, 'L': L, 'f': f, 'F': FRK4}
            return sys

        elif self.integrator == "CVODES":

            # Build integrator using CVODES
            ode = {'x': state, 'p': u, 'ode': dstate, 'quad': L}
            opts = {'tf': self.Ts}
            F = integrator('F', 'cvodes', ode, opts)

            # Return a dictionary representing the system
            sys = {'x': state, 'u': u, 'dx': dstate, 'L': L, 'f': f, 'F': F}
            return sys

        else:
            raise Exception("Not implemented")


    def simulate_integrated_dynamics(self, x0, u, T):

        F = self.integrated_dynamics['F']

        # Simulate the trajectory
        x_traj = [x0]
        x_current = x0
        for t in range(T):
            res = F(x0=x_current, p=u[t])
            x_next = res['xf'].full().flatten()
            x_traj.append(x_next)
            x_current = x_next

        # Convert trajectory to numpy array for easy plotting
        x_traj = np.array(x_traj)
        return x_traj

    def plot_trajectory(self, x_traj, T):
        # Plot the trajectory
        plt.figure(figsize=(10, 5))
        plt.plot(np.arange(T + 1) * self.Ts, x_traj[:, 0], label='x')
        plt.plot(np.arange(T + 1) * self.Ts, x_traj[:, 1], label='theta')
        plt.plot(np.arange(T + 1) * self.Ts, x_traj[:, 2], label='x_dot')
        plt.plot(np.arange(T + 1) * self.Ts, x_traj[:, 3], label='theta_dot')
        plt.xlabel('Time [s]')
        plt.ylabel('State')
        plt.title('System Trajectory')
        plt.legend()
        plt.grid(True)
        plt.show()