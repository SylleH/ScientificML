"""
Author: Sylle Hoogeveen
Solve steady state Navier Stokes equations around cylinder

  u . nabla(u)) + nabla(p) - nu(div(grad(u))= f
                                 div(u) = 0

Generate flow data set by varying cylinder location
"""

from __future__ import print_function
from fenics import *
import numpy as np
import matplotlib.pyplot as plt
from boxfield import *
from mshr import *
from random import random

#Define constants
N = 100              # size data set
nu = 1              # kinematic viscosity

#Define constant boundaries

left = 0.1
right = 1.4
top = 0.4
bottom = 0.1

#Define cylinder locations
cylinder_loc_x = 0.25
cylinder_loc_y = 0.25
cylinder_loc_list = [(cylinder_loc_x, cylinder_loc_y)]

# generate random coordinates
for i in range(N):
    value_x = random()
    cylinder_loc_x = left + (value_x * (right - left)) #x coordinate in between 0.1 and 1.4
    value_y = random()
    cylinder_loc_y = bottom + (value_y * (top - bottom)) #y coordinate in between 0.1 and 0.4
    cylinder_loc_new = (cylinder_loc_x, cylinder_loc_y)
    cylinder_loc_list.append(cylinder_loc_new)
    #print(cylinder_loc_list)


def Create_geometry(cylinder_loc):
    # Create mesh
    channel = Rectangle(Point(0, 0), Point(1.5,0.5))
    cylinder = Circle(Point(cylinder_loc), 0.05)
    domain = channel - cylinder
    mesh = generate_mesh(domain, 100)

    # Define function spaces
    P2 = VectorElement("P", mesh.ufl_cell(), 2)
    P1 = FiniteElement("P", mesh.ufl_cell(), 1)
    TH = MixedElement([P2, P1])
    W = FunctionSpace(mesh, TH)

    # Define boundary conditions
    #x_min = cylinder_loc[0]-0.05
    #x_max = cylinder_loc[0]+0.05
    #y_min = cylinder_loc[1]-0.05
    #y_max = cylinder_loc[1]+0.05
    #cylinder = 'on_boundary && x[0]>x_min && x[0]<x_max && x[1]>y_min && x[1]<y_max'
    return W, mesh

def boundary_conditions(mesh, cylinder_loc):
    inflow_profile = ('4.0*1.5*x[1]*(0.41 - x[1]) / pow(0.41, 2)', '0')
    #inflow = 'near(x[0], 0)'
    outflow = 'near(x[0], 1.5)'
    #walls = 'near(x[1], 0) || near(x[1], 0.5)'

    center = Point(cylinder_loc)
    # Construct facet markers
    bndry = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
    for f in facets(mesh):
        mp = f.midpoint()
        if near(mp[0], 0.0):  # inflow
            bndry[f] = 1
        elif near(mp[0], 1.5):  # outflow
            bndry[f] = 2
        elif near(mp[1], 0.0) or near(mp[1], 0.5):  # walls
            bndry[f] = 3
        elif mp.distance(center) <= 0.05:  # cylinder
            bndry[f] = 5


    # Define boundary conditions
    bcu_inflow = DirichletBC(W.sub(0), Expression(inflow_profile, degree=2), bndry,1)
    bcu_walls = DirichletBC(W.sub(0), (0, 0), bndry, 3)
    bcu_cylinder = DirichletBC(W.sub(0), (0, 0), bndry, 5)
    bcp_outflow = DirichletBC(W.sub(1), Constant(0), outflow)

    #bcu = [bcu_walls, bcu_cylinder,bcu_inflow]
    #bcp = [bcp_outflow]
    bcs = [bcu_walls, bcu_cylinder, bcu_inflow, bcp_outflow]
    return bcs

def Var_problem(W, bcs, nu):
    #define test and trial functions
    v, q = TestFunctions(W)
    w = Function(W)
    u, p = split(w)

    # Define expressions used in variational forms
    f = Constant((0, 0))
    nu = Constant(nu)

    # Define variational form PDE
    F = nu * inner(grad(u), grad(v)) * dx + dot(dot(grad(u), u), v) * dx \
        - p * div(v) * dx - q * div(u) * dx

    # Solve problem
    solve(F == f, w, bcs)
    u,p = w.split()
    return u

def GenerateImages(u,i,N, cylinder_loc):
    # call the plot command from dolfin
    p = plot(u[0])
    plt.axis('off')
    # set colormap
    p.set_cmap("gray")
    p.set_clim(0.0, 5.0 )
    # add a title to the plot
    #plot_title = 'cylinder_loc= (%f,%f)' % (cylinder_loc[0], cylinder_loc[1])
    #plt.title(plot_title)
    # add a colorbar
    #plt.colorbar(p);

    # save image to disk
    if i < (0.8 * N):  # 80% trainingdata
        plot_name = 'Flow_data/TrainingData/data_cylinder/plot_%d.png' % (i)
        plt.savefig(plot_name, bbox_inches = 'tight', transparent = True, pad_inches = 0)

    else:  # 20% validationdata
        plot_name = 'Flow_data/ValidationData/data_cylinder/plot_%d.png' % (i)
        plt.savefig(plot_name, bbox_inches = 'tight', transparent = True, pad_inches = 0)


for i in range(N):
    cylinder_loc = cylinder_loc_list[i]
    W, mesh = Create_geometry(cylinder_loc)
    bcs = boundary_conditions(mesh, cylinder_loc)
    u = Var_problem(W, bcs, nu)
    GenerateImages(u, i, N, cylinder_loc)
    plt.show()
