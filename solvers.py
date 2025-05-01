import numpy as np
from numpy import float64
import matplotlib.pyplot as plt


'''
This is the file for the solvers, including the ones for the transport equation, Poisson equation, and the incompressible Navier-Stokes equation.
'''


class TransportSolver:
    '''
    This is the class for solving the 1D transport equation.
    '''
    def __init__(self, a, b, dt, dx, initial_condition, c, T, discretization_method="central"):
        self.a = a
        self.b = b
        self.dt = dt
        self.dx = dx
        self.c = c
        self.T = T

        self.N = int((b - a) / self.dx)
        self.x_grid = np.linspace(self.a, self.b, self.N+1)

        self.u_total = [initial_condition(self.x_grid)]
        self.t_total = np.linspace(0, T, int(T/dt)+1)

        self.discretization_method = discretization_method

        print("dt is " + str(self.dt))
        print("dx is " + str(self.dx))

    def solve(self):
        if self.discretization_method == "central":
            u_final = self.solve_central()
        elif self.discretization_method == "upwind":
            u_final = self.solve_upwind()

        return u_final

    def solve_central(self):
        for i in range(len(self.t_total) - 1):
            u_previous = self.u_total[-1]
            u_new = np.zeros(len(self.x_grid))

            # Update the interior points using central difference
            u_derivative = (-u_previous[:-2] + u_previous[2:]) / (2 * self.dx)
            u_new[1:-1] = u_previous[1:-1] - self.c * self.dt * u_derivative

            # Update the boundary points using periodic boundary condition
            u_new[0] = u_previous[0] - self.c * self.dt * (u_previous[1] - u_previous[-2]) / (2 * self.dx)
            u_new[-1] = u_new[0]

            self.u_total.append(u_new)

        return self.u_total[-1]
    

    def solve_upwind(self):
        for i in range(len(self.t_total) - 1):
            u_previous = self.u_total[-1]
            u_new = np.zeros(len(self.x_grid))

            # Update the interior points using central difference
            u_derivative = (-u_previous[:-1] + u_previous[1:]) / (self.dx)
            u_new[1:] = u_previous[1:] - self.c * self.dt * u_derivative

            # Update the boundary points using periodic boundary condition
            u_new[0] = u_previous[0] - self.c * self.dt * (u_previous[0] - u_previous[-2]) / (self.dx)

            self.u_total.append(u_new)

        return self.u_total[-1]
    

    def visualize(self, idx_list):
        plt.plot(self.x_grid, self.u_total[0])
        legend_list = ["Initial condition"]
        for idx in idx_list:
            plt.plot(self.x_grid, self.u_total[idx])
            legend_list.append('Time ' + str(round(self.t_total[idx], 2)))
        plt.xlabel("x")
        plt.ylabel("u")
        plt.title("Advection equation with " + self.discretization_method + " difference scheme", fontsize=12)
        plt.legend(legend_list, fontsize=12)
        plt.show()






    def get_t_total(self):
        return self.t_total

    def get_u_total(self):
        return self.u_total
    
    def get_x_grid(self):
        return self.x_grid





class LidDrivenCavitySolver:
    '''
    This is the class for solving the lid driven cavity problem.
    TODO: Implement the solver.
    '''
    def __init__(self, Re, Nx, Ny, dt, dx, dy, initial_condition, T):
        self.Re = Re
        self.Nx = Nx
        self.Ny = Ny
        self.dt = dt
        self.dx = dx




