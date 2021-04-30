import matplotlib.pyplot as plt
from fenics import *
from boxfield import *

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

# Define constants
Nx = 3
Ny = 2
import random
a = random.sample(range(1,Nx+1),Nx)
b = random.sample(range(1,Ny+1),Ny)
# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
xx = Expression('x[0]', degree=1)
yy = Expression('x[1]', degree=1)
fx = 0
fy = 0

for n in range(Nx):
    fx +=a[n]*sin(m.pi*n*xx)
for n in range(Ny):
    fy +=b[n]*sin(m.pi*n*yy)
f = fx*fy
#f = Expression('(7*sin(pi*x[0])+6*sin(2*pi*x[0])+4*sin(3*pi*x[0]))*(2*sin(pi*x[1])+5*sin(2*pi*x[1]))', degree=1)
a = dot(grad(u), grad(v))*dx
L = f*v*dx


# Compute solution
u = Function(V)
solve(a == L, u, bc)

# Create plot with certain resolution
u_box = FEniCSBoxField(u, (k, l))
u_ = u_box.values

#save plot to image
    # a colormap and a normalization instance
cmap = plt.cm.jet

    # map the normalized data to colors
norm = plt.Normalize(vmin=u_.min(), vmax=u_.max())

    # image is now RGBA (512x512x4)
image = cmap(norm(u_))

    # save the image
plt.imsave('data10.png', image)


# Plot solution and mesh
plt.imshow(u_, interpolation='nearest')
#plot(mesh)
plt.show()
# Save solution to file in VTK format
#vtkfile = File('poisson/Nx1Ny1.pvd')
#vtkfile << u

# Compute error in L2 norm
error_L2 = errornorm(u_D, u, 'L2')
# Compute maximum error at vertices
vertex_values_u_D = u_D.compute_vertex_values(mesh)
vertex_values_u = u.compute_vertex_values(mesh)

import numpy as np
error_max = np.max(np.abs(vertex_values_u_D - vertex_values_u))
# Print errors
print('error_L2 =', error_L2)
print('error_max =', error_max)
