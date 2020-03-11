import numpy as np
from scipy.optimize import linprog
from numpy.linalg import solve

A_eq = np.array([[1,1,1]])
b_eq = np.array([999])

A_ub = np.array([[1, 4, 8],
                 [40,30,20],
                 [3,2,4]])

b_ub = np.array([4500, 36000,2700])
c    = np.array([70, 80, 85])

res  = linprog(c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub,
bounds=(0, None))
print('Optimal value:', res.fun, '\nX:', res.x)

#------------------------------------------------------------------------------
print("\n")

A = np.array([[1, 1, 1, 0, 0, 0],
              [1, 4, 8, 1, 0, 0],
              [40, 30, 20, 0, 1, 0],
              [3, 2, 4, 0, 0, 1]])

b = np.array([999, 4500, 36000, 2700])
c = np.array([70, 80, 85, 0, 0, 0])

res = linprog(c, A_eq=A, b_eq=b, bounds=(0, None))
print('Optimal value:', res.fun, '\nX:', res.x)

#------------------------------------------------------------------------------
print("\n")

res = linprog(-b, A_ub=A.T, b_ub=c, bounds=[(None,None), (None,None),
(None,None), (None,None)])
y = res.x
print('Optimal value:', -res.fun, '\nY:', y)

u = c - A.T.dot(y)
Ar = A[:, np.abs(u)< 1e-10]
x_1 = solve(Ar, b)
print(x_1)
x = np.array([0.0] * len(c))
x[np.abs(u)< 1e-10] = x_1
x[np.abs(u)> 1e-10] = 0
print('Primal solution from the dual:', x)
