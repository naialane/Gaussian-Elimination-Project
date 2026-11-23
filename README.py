# Gaussian-Elimination-Project
#Uni project:
#Includes matrix, plt code, time code and 3 solvers: Numpy, Gemini and mine (with keys to where I used AI to debug it)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 15:23:48 2026

@author: naialane
"""

import numpy as np
import time
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True)

A = np.array([[-1.5, 2.3, 0.3, 0.3, 0.5],
              [-1.4, 1.5, 2.1, -0.2, -1.6],
              [-0.1, -0.3, 2.3, 0.9, 0.4],
              [0.6, 0.8, 1.7, -2.4, 1.7],
              [-0.2, 0.4, -0.2, 1.1, -0.9]])
b = np.array([[-7.7], [-13.9], [-34.4], [-43.5], [12.0]])
#print(A)

#print(b)

M = np.concatenate((A, b), axis=1)
#print(M)

M0 = np.divide(M, 0.8)
M1 = np.round(M0, 2)
#print(M1)

def upper_triangular(matrix):
    
    matrix = np.array(matrix, dtype=float, copy=True)
    n = len(matrix)
    m = len(matrix[0])
        
    if matrix[0, 0] != 1:
        factor0 = matrix[0, 0]
        for j in range(m):
            matrix[0, j] /= factor0
    
    for i in range(1, n):
       for k in range(i):
           scale = matrix[i, k]
           for j in range(m):
               matrix[i, j] -= scale * matrix[k, j]
       factor = matrix[i, i]
       matrix[i] /= factor
        
        
    return np.round(matrix, 2)

result = upper_triangular(M1)
#print(result)

x = np.array([[0.00], [0.00], [0.00], [0.00], [0.00]])

#x[i] = result[i, -1] - sum(result[i, -2]*x[i+1] + result[i, -3]*x[i])
x[4] = result[4, 5]
x[3] = result[3, 5] - ((result[3, 4])*(x[4]))
x[2] = result[2, 5] - ((result[2, 4])*(x[4])) - ((result[2, 3])*(x[3]))
x[1] = result[1, 5] - ((result[1, 4])*x[4]) - (result[1, 3]*x[3]) - (result[1, 2]*x[2])
x[0] = result[0, 5] - (result[0, 4]*x[4]) - (result[0, 3]*x[3]) - (result[0, 2]*x[2]) - (result[0, 1]*x[1])

def backsolver(ut_matrix):
    zero_list = []
    n = len(ut_matrix)
    for xi in range(n):
        zero_list.append([0.00])
    #print(zero_list)
    x = np.array(zero_list)
    
    #print(x)
    for i in reversed(range(n)):
        x[i] = ut_matrix[i, -1]
        for k in range(i+1, n):
            x[i] -= ut_matrix[i, k]*x[k]

    return x

x_M1 = backsolver(result)
#print(x_M1)
                
        
#print(x)

def linear_solver(Matrix):
    Matrix = np.array(Matrix, dtype=float, copy=True)
    
    n = len(Matrix)
    m = len(Matrix[0])
        
    if Matrix[0, 0] != 1:
        factor0 = Matrix[0, 0]
        for j in range(m):
            Matrix[0, j] /= factor0
    
    for i in range(1, n):
       for k in range(i):
           scale = Matrix[i, k]
           for j in range(m):
               Matrix[i, j] -= scale * Matrix[k, j]
       factor = Matrix[i, i]
       Matrix[i] /= factor
       
    zero_list = []
    n = len(Matrix)
    for xi in range(n): #debugged 
        zero_list.append([0.00])
    x = np.array(zero_list)
    
    for i in reversed(range(n)):
         x[i] = Matrix[i, -1]
         for k in range(i+1, n):    #debugged
             x[i] -= Matrix[i, k]*x[k] 
    #print(Matrix)
    return x
   
combined = linear_solver(M1)
#print(combined)

numpysolve = np.linalg.solve(A, b)
#print(numpysolve)

import numpy as np


def solve_augmented_matrix(matrix: np.ndarray, tol: float = 1e-10) -> np.ndarray: #Gemini
    """Solve a linear system Ax = b given an augmented matrix [A | b].

    Parameters
    ----------
    matrix : np.ndarray
        2D NumPy array of shape (m, n + 1), where the first n columns form A
        and the last column is b.
    tol : float, optional
        Numerical tolerance for zero/rank checks (default: 1e-10).

    Returns
    -------
    np.ndarray
        1D NumPy array x of length n solving A @ x = b.

    Raises
    ------
    ValueError
        If the matrix has fewer than 2 columns, is inconsistent (no solution),
        or has infinitely many solutions (underdetermined / rank-deficient).
    """
    aug = np.asarray(matrix, dtype=float)
    if aug.ndim != 2 or aug.shape[1] < 2:
        raise ValueError("matrix must be a 2D array with at least 2 columns.")

    m, total_cols = aug.shape
    n = total_cols - 1

    # Gaussian elimination with partial pivoting to Row Echelon Form (REF)
    ref = aug.copy()
    pivot_row = 0
    pivot_cols = []

    for col in range(n):
        if pivot_row >= m:
            break

        max_rel = np.argmax(np.abs(ref[pivot_row:, col]))
        max_row = pivot_row + max_rel

        if abs(ref[max_row, col]) <= tol:
            continue

        if max_row != pivot_row:
            ref[[pivot_row, max_row]] = ref[[max_row, pivot_row]]

        pivot_cols.append(col)

        # Eliminate entries below the pivot
        for r in range(pivot_row + 1, m):
            factor = ref[r, col] / ref[pivot_row, col]
            ref[r, col:] -= factor * ref[pivot_row, col:]
            ref[r, col] = 0.0

        pivot_row += 1

    # Check for inconsistent equations: [0 0 ... 0 | c] with c != 0
    for r in range(pivot_row, m):
        if abs(ref[r, -1]) > tol:
            raise ValueError("System is inconsistent (no solution).")

    # Check for free variables (infinitely many solutions)
    rank = len(pivot_cols)
    if rank < n:
        raise ValueError(
            f"System has infinitely many solutions (rank {rank} < {n} unknowns)."
        )

    # Back-substitution for a unique solution
    x = np.zeros(n, dtype=float)
    for i in range(n - 1, -1, -1):
        row = i
        col = pivot_cols[i]
        rhs = ref[row, -1] - np.dot(ref[row, col + 1 : n], x[col + 1 : n])
        x[col] = rhs / ref[row, col]

    return x

    
    
print(solve_augmented_matrix(M))
seed = 99
np.random.seed(seed)

#random_A1 = np.random.standard_normal((2, 2))
#random_b1 = np.random.standard_normal((2, 1))
#A1_b1 = np.concatenate((random_A1, random_b1), axis=1)

#print(linear_solver(A1_b1))
#print(np.linalg.solve(random_A1, random_b1))

#random_A2 = np.random.standard_normal((3, 3))
#random_b2 = np.random.standard_normal((3, 1))
#A2_b2 = np.concatenate((random_A2, random_b2), axis=1)

#print(linear_solver(A2_b2))
#print(np.linalg.solve(random_A2, random_b2))

#random_A3 = np.random.standard_normal((4, 4))
#random_b3 = np.random.standard_normal((4, 1))
#A3_b3 = np.concatenate((random_A3, random_b3), axis=1)

#print(linear_solver(A3_b3))
#print(np.linalg.solve(random_A3, random_b3))


sarray = np.array([10, 12, 15, 23, 31, 40, 85, 98, 170, 290, 374, 408])
#print("Size array = ", sarray)

t_solve_m = []
t_solve_n = []
t_solve_g = []
for size in sarray:
    matrix_A = np.random.standard_normal((size, size))
    vec_b = np.random.standard_normal((size, 1))
    
    t_0m = time.perf_counter()
    linear_solver(np.concatenate((matrix_A, vec_b), axis=1))
    t_1m = time.perf_counter()
    t_m = t_1m-t_0m
    t_solve_m.append(t_m)
    
    t_0n = time.perf_counter()
    np.linalg.solve(matrix_A, vec_b)
    t_1n = time.perf_counter()
    t_n = t_1n - t_0n 
    t_solve_n.append(t_n)
    
    t_0g = time.perf_counter()
    solve_augmented_matrix(M)
    t_1g = time.perf_counter()
    t_g = t_1g - t_0g
    t_solve_g.append(t_g)
    
t_solve_lin = np.array(t_solve_m)
t_solve_np = np.array(t_solve_n)
t_solve_gem = np.array(t_solve_g)

#print("Numpy time = ", t_solve_np)


fig, ax = plt.subplots(figsize=(7, 5))
ax.set_xscale = ("log")
ax.set_yscale = ("log")

#ax.plot(sarray, t_solve_lin, "o-", label = "My Solver")
ax.plot(sarray, t_solve_np, "o-", label = "Numpy", color="orange")
ax.plot(sarray, t_solve_g, "o-", label = "Gemini", color="green")

c = t_solve_np[0] / (sarray[0] ** 3)
# Plot the reference curve
##ax.plot(sarray, c * sarray**3, "--", label="cn³ (reference)")

# Figure labels
ax.set_xlabel("System size N",fontsize=16)
ax.set_ylabel("Time (seconds)",fontsize=16)
# Add a grid
ax.grid(True, which="both", ls=":")
# Include a legend
ax.legend(fontsize=14)
# Avoid having white space around the edge
fig.tight_layout()

# The following line saves the figure as a png file
fig.savefig("time_compnpgem.png", dpi=300, bbox_inches="tight", transparent=False)

N=100
A=np.random.standard_normal((N,N))
b=np.random.standard_normal((N))
x=np.linalg.solve(A,b)

# Compute the error
er=np.linalg.norm(A @ x - b)

# Print the error
print(er)
