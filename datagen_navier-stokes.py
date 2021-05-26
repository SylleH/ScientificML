

# Data generation flow using https://fenicsproject.org/pub/tutorial/pdf/fenics-tutorial-vol1.pdf section3.4

from dolfin import *
import matplotlib.pyplot as plt
from fenics import *


# Print log messages only from the root process in parallel
parameters["std_out_all_processes"] = False;

# Load mesh from file
k,l=100,100
mesh = UnitSquareMesh(k, l)
coordinates = mesh.coordinates()

# Define function spaces (P2-P1)
V = VectorFunctionSpace(mesh, 'P', 2)
Q = FunctionSpace(mesh, 'P', 1)

# Define trial and test functions
u = TrialFunction(V)
v = TestFunction(V)
p = TrialFunction(Q)
q = TestFunction(Q)

# Define functions for solutions at previous and current time steps
u_n = Function(V)
u_  = Function(V)
p_n = Function(Q)
p_  = Function(Q)

# Define boundaries
inflow = 'near(x[0],0)'
outflow = 'near(x[0],1)'
walls = 'near(x[1],0) || near(x[1],1)'

#Define boundary conditions
bcu_noslip = DirichletBC(V, Constant((0, 0)), walls)
bcp_inflow = DirichletBC(Q, Constant(8), inflow)
bcp_outflow = DirichletBC(Q, Constant(0), outflow)
bcu = [bcu_noslip]
bcp = [bcp_inflow, bcp_outflow]

# Set parameter values
rho = 1
mu = 1

#Implement IPCS scheme = Incremetal pressure correction scheme
U = 0.5*(u_n + u)
n = FacetNormal(mesh)
f = Constant((0, 0))
mu = Constant(mu)
rho = Constant(rho)

# Define strain-rate tensor
def epsilon(u):
    return sym(nabla_grad(u))

# Define stress tensor
def sigma(u, p):
    return 2*mu*epsilon(u) - p*Identity(len(u))

# Define variational problem for step 1
F1 = rho*dot(dot(u_, nabla_grad(u_)), v)*dx \
   + inner(sigma(U, p_), epsilon(v))*dx \
   + dot(p_*n, v)*ds - dot(mu*nabla_grad(U)*n, v)*ds \
   - dot(f, v)*dx
a1 = lhs(F1)
L1 = rhs(F1)

# Define variational problem for step 2
a2 = dot(nabla_grad(p), nabla_grad(q))*dx
L2 = dot(nabla_grad(p_), nabla_grad(q))*dx

# Define variational problem for step 3
a3 = dot(u, v)*dx
L3 = dot(u_, v)*dx

# Assemble matrices
A1 = assemble(a1)
A2 = assemble(a2)
A3 = assemble(a3)

# Apply boundary conditions to matrices
[bc.apply(A1) for bc in bcu]
[bc.apply(A2) for bc in bcp]

# Step 1: Tentative velocity step
b1 = assemble(L1)
[bc.apply(b1) for bc in bcu]
solve(A1, u_.vector(), b1)

# Step 2: Pressure correction step
b2 = assemble(L2)
[bc.apply(b2) for bc in bcp]
solve(A2, p_.vector(), b2)

# Step 3: Velocity correction step
b3 = assemble(L3)
solve(A3, u_.vector(), b3)


# Plot solution
u_inter = interpolate(u_,V)
uplot =plot(u_inter, title="Velocity")
uplot.set_cmap = ('gray')
plt.savefig("velocity.png")

