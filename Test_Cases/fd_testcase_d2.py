import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.axes as ax

### Parameters

degree = 5
poly_degree = 3
start = -1
stop = 1
num_pts = 50
num_knn = (2*poly_degree) + 1
xvals = np.linspace(start, stop, num=num_pts)  # creates x values into a np.array

start_time = time.time()

def true_func(x):
    return np.sin(2*np.pi * x)

def true_func_dxx(x):
    return -4*(np.pi*np.pi) * np.sin(2*np.pi * x)

def phs(r):
    return (abs(r) ** (degree))

def deriv_phs(r):
    return degree * (np.abs(r) ** (degree - 1))

def deriv2_phs(r):
    return degree * (degree - 1) * (np.abs(r) ** (degree - 2))

def list_to_matrix(list):
      return np.matrix(np.array(list))

def knn_matrix_creator(xvals,n_knn):
  matrix = np.zeros((len(xvals),len(xvals)))
  distances = []
  indices = []
  for i in range(len(xvals)):
    for k in range(len(xvals)):
      distances.append(abs(xvals[k] - xvals[i]))
    index = np.argsort(distances)
    index = index[:n_knn]
    indices.append(index)
    for j in range(len(xvals)):
      if j in index:
        matrix[i][j] = 1
    distances = []
  return matrix, indices


# Example usage:
poly_degree = 3

def local_rbf_fd_d2(xvals = xvals, poly_degree = poly_degree):
    """
    Calculates the SECOND derivative (D2) matrix.
    Written in the same style as the original local_rbf_fd.
    """
    num_knn = (2*poly_degree) + 1  # creates x values into a np.array
    
    # Kept your original (inefficient) way of calling this twice
    knn_matrix = knn_matrix_creator(xvals,num_knn)[0]
    indices = knn_matrix_creator(xvals,num_knn)[1]
    length = len(xvals)
    final_matrix = np.zeros(len(xvals)**2).reshape(len(xvals),len(xvals))

    # Kept these unused variables
    X, Y = np.meshgrid(xvals, xvals)
    final_matrices = []
    local_diff_matrix =[]
    center = 0

    for i in range(length):
        knn = []
        for j in range(num_knn):
            index = indices[i][j]
            knn.append(xvals[index])

        X, Y = np.meshgrid(knn, knn)
        r = X-Y
        phs_matrix= np.ones(len(r))

        #Creates the matrix of distances between the sample points
        for z in range(0,len(r)):
            row = phs(r[z])
            phs_matrix = np.vstack((phs_matrix,row))
        phs_matrix = phs_matrix[1:]



        sol_vec = np.array(deriv2_phs(r[0]))


        #Creates the matrix of added polynomial terms
        poly_matrix = []
        for z in range(len(knn)):
            row = []
            row.append(1)
            for k in range(1,poly_degree+1):
                row.append(knn[z]**(k))
            poly_matrix.append(row)
        

        phs_matrix = list_to_matrix(phs_matrix)
        poly_matrix = list_to_matrix(poly_matrix)
        poly_matrix_t = poly_matrix.getT() #getT gets the transpose of the matrix
        zeroes = np.matrix(np.array([0]*(poly_degree+1)**2).reshape(poly_degree+1,poly_degree+1))
        
        top_half_A = np.hstack((phs_matrix.getA(),poly_matrix.getA()))
        bottom_half_A = np.hstack((poly_matrix_t.getA(), zeroes.getA()))
        A_matrix = np.matrix(np.vstack((top_half_A,bottom_half_A)))

        poly_deriv_vec = []
        center_node_x = knn[0] # The center of the stencil
    
        for k in range(poly_degree + 1):
            if k == 0 or k == 1:
                poly_deriv_vec.append(0.0)  
            else:

                poly_deriv_vec.append(k * (k - 1) * (center_node_x ** (k - 2)))

                
        sol_vec = np.append(sol_vec, poly_deriv_vec)

        finite_weights = np.linalg.solve(A_matrix, sol_vec)
        final_matrix[i][indices[i]] = finite_weights[0:num_knn]
    return final_matrix


# Graphs The Derivative Approximation vs The True Derivative

# NO BOUNDARY CONDITIONS

# xvals_graph = np.linspace(start, stop, num=num_pts)

# true_vals = true_func_dx(xvals)

# plt.scatter(xvals_graph, local_rbf_fd(), color = 'red', label = 'Approx.') # Data Points
# plt.scatter(xvals_graph, true_vals, color = 'blue', label = 'Exact') # Data Points
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.title('Derivative Approximations vs True Derivative')
# plt.grid(True)
# plt.legend()
# plt.show()


# Graphs the Error of Derivative Approximations vs n points (Different Polynomial Degrees)
poly_degree_3_vec = []
poly_degree_5_vec = []
poly_degree_7_vec = []
poly_degree_9_vec = []
xvals_graph = []

for k in range(1,5): # Iterates over polynomial degrees
   for j in range(1,21): # Iterates over number of sample points
      num_pts = 30 + 10*j
      xvals = np.linspace(start, stop, num=num_pts)  # creates x values into a np.array
      true_d2_vals = true_func_dxx(xvals)
      true_vals = true_func(xvals)
      est_deriv = local_rbf_fd_d2(xvals, (k*2)+1) @ true_vals
      error = np.log(np.max(np.abs(est_deriv - true_d2_vals)))
      if k == 1:
         poly_degree_3_vec.append(error)
         xvals_graph = np.append(xvals_graph, num_pts)
      elif k == 2:
         poly_degree_5_vec.append(error)
      elif k == 3:
         poly_degree_7_vec.append(error)
      elif k == 4:
         poly_degree_9_vec.append(error)

xvals_graph = np.log10(xvals_graph)

plt.scatter(xvals_graph, poly_degree_3_vec, color = 'red', label = 'Poly Degree 3', s = 25, marker = 'o') # Data Points
plt.scatter(xvals_graph, poly_degree_5_vec, color = 'purple', label = 'Poly Degree 5', s = 25, marker = 'D') # Data Points
plt.scatter(xvals_graph, poly_degree_7_vec, color = 'green', label = 'Poly Degree 7', s = 25, marker = '^') # Data Points
#plt.scatter(xvals_graph, poly_degree_9_vec, color = 'orange', label = 'Poly Degree 9', s = 12) # Data Points
plt.xlabel(r'$\log_{10}(n)$')
plt.ylabel(r'$\log_{10} ||\vec{\Gamma}||_{\infty}$')
plt.title('Base 10 Log of Max Error vs. Base 10 Log of n')
plt.grid(False)
plt.legend()
plt.savefig(r"c:\Users\lucas\RBF_Research\RBF_Research\Test_Cases\2nd_deriv_error_vs_n.png", dpi=300, bbox_inches="tight")
plt.show()
