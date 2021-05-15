import matplotlib.pyplot as plt
import matplotlib.colors as clr
from fenics import *
from boxfield import *
import random
import numpy as np

# Define constants
Nx = 3
Ny = 2
N = 100

# Create mesh and define function space
k,l = 100,100
mesh = UnitSquareMesh(k, l)
coordinates = mesh.coordinates()
V = FunctionSpace(mesh, 'P', 1)

# Define boundary condition
u_D = Constant(0)
def boundary(x, on_boundary):
    return on_boundary
bc = DirichletBC(V, u_D, boundary)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
a = dot(grad(u), grad(v)) * dx

def GenerateFunctions(Nx,Ny):
    a_n = []
    b_n = []
    for j in range(0, Nx):
        # any random integer from 1 to 100
        a_n.append(random.randint(1, 100))
    for j in range(0, Ny):
        # any random integer from 1 to 100
        b_n.append(random.randint(1, 100))
    print(a_n)
    print(b_n)
    xx = Expression('x[0]', degree=1)
    yy = Expression('x[1]', degree=1)

    fx = 0
    fy = 0

    for n in range(Nx):
        fx += a_n[n] * sin(pi * n * xx)
    for n in range(Ny):
        fy += b_n[n] * sin(pi * n * yy)
    f = fx * fy
    return f , a_n, b_n

def ComputeSolution(f,v):
    L = f*v*dx
    u = Function(V)
    solve(a == L, u, bc)
    return u

def GenerateImages(u, a_n, b_n, i, N):
    # Create plot with certain resolution
    u_box = FEniCSBoxField(u, (k, l))
    u_ = u_box.values
    # save plot to image
    #cmap_gray = plt.gray # a colormap grey
    cmap_rgba = plt.cm.jet # a colormap jet
    norm = plt.Normalize(vmin=u_.min(), vmax=u_.max())  # map the normalized data to colors
    image = cmap_rgba(norm(u_))  # image is now RGBA

    plot_title = 'a_n = %s,b_n = %s' % (a_n, b_n)
    plt.title(plot_title)  # possibility to make the title of the plots reference the a_n and b_n
    if i<(0.8*N):   #80% trainingdata
        plot_name = 'data/TrainingData/data_sines/plot_%d.png' % (i)
        plt.imsave(plot_name, image)
    else:           #20% validationdata
        plot_name = 'data/ValidationData/data_sines/plot_%d.png' % (i)
        plt.imsave(plot_name, image)

def PlotShow():
    plt.imshow(u_, interpolation='nearest')
    plt.show()

def ErrorAnalysis(u_D, u):
    # Compute error in L2 norm
    error_L2 = errornorm(u_D, u, 'L2')
    # Compute maximum error at vertices
    vertex_values_u_D = u_D.compute_vertex_values(mesh)
    vertex_values_u = u.compute_vertex_values(mesh)

    error_max = np.max(np.abs(vertex_values_u_D - vertex_values_u))
    # Print errors
    print('error_L2 =', error_L2)
    print('error_max =', error_max)


for i in range(N):
    f, a_n, b_n  = GenerateFunctions(Nx,Ny)
    u = ComputeSolution(f,v)
    GenerateImages(u, a_n,b_n, i, N)



