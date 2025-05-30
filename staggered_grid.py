import numpy as np
import matplotlib.pyplot as plt
import importlib
import operators

importlib.reload(operators)
from operators import vorticity, divergence, gradient



'''
Generate the staggered grid.
The position of the variables have been verified.
'''
class StaggeredGrid:


    def __init__(self, Lx, Ly, Nx, Ny, initial_condition_velocity, initial_condition_pressure):
        '''
        Initialize the staggered grid.
        In this implementation, it follows the convention of Python indexing, 
        so that the horizontally it mimics that x is increasing from left to right,
        and the vertically it mimics that y is increasing from top to bottom.

        To vary the x value, change the second argument in the mesh_grid.
        To vary the y value, change the first argument in the mesh_grid.

        We assume the grid is uniform in this implementation
        '''
        self.Lx = Lx
        self.Ly = Ly
        self.Nx = Nx
        self.Ny = Ny
        self.dx = Lx / Nx
        self.dy = Ly / Ny



        '''
        Staggered grid initialization
        '''
        # Define the mesh grid for the spatial points
        self.spatial_mesh_grid = np.meshgrid(np.linspace(0, Lx, Nx+1), np.linspace(0, Ly, Ny+1))
        # Define the mesh grid for the center points
        self.pressure_mesh_grid = np.meshgrid(np.linspace(self.dx/2, Lx-self.dx/2, Nx), np.linspace(self.dy/2, Ly-self.dy/2, Ny))
        # Define the mesh grid for the velocity points
        # The velocity components are only defined at the inner points, 
        # since the boundary points are not unknown variables.
        self.u_mesh_grid = np.meshgrid(np.linspace(0, Lx, Nx+1), np.linspace(self.dy/2, Ly-self.dy/2, Ny))
        self.v_mesh_grid = np.meshgrid(np.linspace(self.dx/2, Lx-self.dx/2, Nx), np.linspace(0, Ly, Ny+1))
        self.u_mesh_grid_inner = np.meshgrid(np.linspace(self.dx, Lx-self.dx, Nx-1), np.linspace(self.dy/2, Ly-self.dy/2, Ny))
        self.v_mesh_grid_inner = np.meshgrid(np.linspace(self.dx/2, Lx-self.dx/2, Nx), np.linspace(self.dy, Ly-self.dy, Ny-1))
        # Define the mesh grid for the vorticity points
        self.vorticity_mesh_grid = np.meshgrid(np.linspace(self.dx, Lx-self.dx, Nx-1), np.linspace(self.dy, Ly-self.dy, Ny-1))

        # Define the pressure at the cell centers using the initial condition
        # We pin the pressure at the bottom left corner to 0, so that it's always 0.
        self.pressure = initial_condition_pressure(self.pressure_mesh_grid)
        self.pressure[0, 0] = 0                                             # Manually enforce the pressure at the bottom left corner to be 0.
        self.pressure_vector = self.pressure_2D_to_1D()                     # Vectorized representation of the pressure field.
        self.pressure_pointer = self.pointer_pressure()                     # Pointer that maps the 2D index of the pressure field to the 1D index of the pressure field.

        # Define the velocity field at the faces using the initial condition
        # self.u, self.v, and self.velocity contains the stacked u, v at the NON-BOUNDARY faces!!
        self.u, self.v = initial_condition_velocity(self.u_mesh_grid_inner, self.v_mesh_grid_inner)
        self.velocity = self.velocity_2D_to_1D()                            # Vectorized representation of the velocity field.  
        self.u_pointer, self.v_pointer = self.pointer_velocity()            # Pointer that maps the 2D index of the velocity field to the 1D index of the velocity field.




    ###############################
    # Compute discrete quantities #
    ###############################
    def compute_divergence(self):
        '''
        Compute the divergence of the velocity field
        '''
        return divergence(self.u, self.v, self.pressure_mesh_grid)
    
    def compute_gradient(self):
        '''
        Compute the gradient of the pressure field
        '''
        return gradient(self.pressure, self.pressure_mesh_grid)
    
    def compute_vorticity(self):
        '''
        Compute the vorticity of the velocity field
        '''
        return vorticity(self.u, self.v, self.vorticity_mesh_grid)


    #################
    # Get functions #
    #################
    # Get the grid points
    def get_grid(self, inner_grid=True):
        if inner_grid:
            return self.spatial_mesh_grid, self.pressure_mesh_grid, self.u_mesh_grid_inner, self.v_mesh_grid_inner, self.vorticity_mesh_grid
        else:
            return self.spatial_mesh_grid, self.pressure_mesh_grid, self.u_mesh_grid, self.v_mesh_grid, self.vorticity_mesh_grid
    # Get the pressures at the cell centers
    def get_pressure(self):
        return self.pressure, self.pressure_vector, self.pressure_pointer
    # Get the velocities at the faces
    def get_velocity(self):
        return self.u, self.v, self.velocity, self.u_pointer, self.v_pointer
    # Get the vorticity at the cell vertices
    def get_vorticity(self):
        return self.vorticity
    


    #####################
    # Pointer functions #
    #####################
    def pointer_velocity(self):
        '''
        Generate a map that maps the 2D index of the 
        velocity field to the 1D index of the velocity field.
        '''
        idx = 0
        u_pointer = np.zeros(self.u.shape)
        v_pointer = np.zeros(self.v.shape)

        for i in range(self.u.shape[0]):
            for j in range(self.u.shape[1]):
                u_pointer[i, j] = idx
                idx += 1
        # We do NOT reset the idx in between.
        for i in range(self.v.shape[0]):
            for j in range(self.v.shape[1]):
                v_pointer[i, j] = idx
                idx += 1

        return u_pointer.astype(int), v_pointer.astype(int)
    
    def pointer_pressure(self):
        '''
        Generate a map that maps the 2D index of the 
        pressure field to the 1D index of the pressure field.
        '''
        idx = 0
        pressure_pointer = np.zeros(self.pressure.shape)
        for i in range(self.pressure.shape[0]):
            for j in range(self.pressure.shape[1]):
                if i == 0 and j == 0:
                    # Pinned pressure value at the bottom left corner.
                    # We set it to NaN, so that we'll not use it in the computation by mistake.
                    pressure_pointer[i, j] = np.nan
                else:
                    pressure_pointer[i, j] = idx
                    idx += 1
        return pressure_pointer.astype(int)


    ###################################################
    # Synchronization of 2D and 1D pressure, velocity #
    ###################################################
    def pressure_2D_to_1D(self):
        '''
        After changing the 2D representation of the pressure field,
        we need to update the 1D representation of the pressure field.
        '''
        # The bottom left corner is pinned to 0, so that we 
        # don't need to solve it.
        return self.pressure.reshape(-1)[1:]
    
    def velocity_2D_to_1D(self):
        '''
        After changing the 2D representation of the pressure field,
        we need to update the 1D representation of the velocity field.
        '''
        # The boundary velocities are specified, so that we
        # don't need to solve them.
        return np.concatenate((self.u.reshape(-1), self.v.reshape(-1)))
    
    def pressure_1D_to_2D(self):
        '''
        After changing the 1D representation of the pressure field,
        we need to update the 2D representation of the pressure field.
        '''
        pressure = np.insert(self.pressure_vector, 0, 0)
        return pressure.reshape(self.Nx, self.Ny)
    
    def velocity_1D_to_2D(self):
        '''
        After changing the 1D representation of the velocity field,
        we need to update the 2D representation of the velocity field.
        '''
        u = self.velocity[:self.u_pointer.size].reshape(self.u.shape)
        v = self.velocity[self.u_pointer.size:].reshape(self.v.shape)
        return u, v

    ###########################
    # Visualization functions #
    ###########################
    def visualize_velocity(self, scale=0.5):
        '''
        Visualize the x and y components of the velocity field, on the inner grids.
        In order to do the visualization, we need to make sure that the velocity field is defined at the vertices of the cell.
        Thus interpolation of the velocity field is done here, so that the pressure_mesh_grid is used.
        '''
        X, Y = self.vorticity_mesh_grid
        # Interpolate the x-component of the velocity field to the pressure mesh grid
        u_interpolated = (self.u[:-1, :] + self.u[1:, :]) / 2
        v_interpolated = (self.v[:, :-1] + self.v[:, 1:]) / 2


        plt.quiver(X, Y, u_interpolated, v_interpolated, color='blue', scale=scale, scale_units='xy')
        plt.title('Staggered Grid Velocity Field', fontsize=16)
        plt.xlabel('x', fontsize=16)
        plt.ylabel('y', fontsize=16)
        plt.xticks(np.round(X[0, :], 2), fontsize=16)
        plt.yticks(np.round(Y[:, 0], 2), fontsize=16)
        plt.axis('equal')
        plt.grid(axis='both')

    def visualize_vorticity(self):
        X, Y = self.vorticity_mesh_grid
        vorticity = self.compute_vorticity()
        plt.contourf(X, Y, vorticity, cmap='bwr')
        plt.colorbar(label='Vorticity')
        plt.title('Vorticity Field', fontsize=16)
        plt.xticks(np.round(X[0, :], 2), fontsize=12)
        plt.yticks(np.round(Y[:, 0], 2), fontsize=12)
        plt.axis('equal')
        plt.show()

    def visualize_pressure(self):
        X, Y = self.pressure_mesh_grid
        plt.contourf(X, Y, self.pressure, cmap='bwr')
        plt.colorbar(label='Pressure')
        plt.title('Pressure Field', fontsize=16)
        plt.xticks(np.round(X[0, :], 2), fontsize=12)
        plt.yticks(np.round(Y[:, 0], 2), fontsize=12)
        plt.axis('equal')
        plt.show()