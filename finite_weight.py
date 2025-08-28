import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes as ax
import math
from finite_weight_parameters import *
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
import time

start_time = time.time()

def list_to_matrix(list):
      return np.matrix(np.array(list))

def y_func(x):
    return np.sin((x)*(2*np.pi)) #x**(1/3) #true y_function

def y_func_dx(x):
    return (2*np.pi)*np.cos(x*2*np.pi)#(1/3)*(x)**(-2/3)

def PH_Spline(r):
    return abs((r)**degree)

def PH_Spline_dx(r):
    return (degree*(r)**(degree-1))


#Derivative appended polynomial function

def poly_dx(num, poly_degree):

  # poly degree must be 2 or more
  deriv = [0,1]

  for i in range(2,poly_degree+1):
    deriv.append(i*(num)**(i-1))

  return deriv

#Parameters of original equation

step = (stop-start)/u

num_pts = int((((stop-start)/step)+1))

#x value creation

xvals = np.linspace(start,stop, num = num_pts) #creates x values into a np.array

true_y = y_func(xvals)

size = len(xvals)

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



def local_differentiation(xvals, p_degree):
    num_knn = ((p_degree+1)*2)
    knn_matrix = knn_matrix_creator(xvals,num_knn)[0]
    indices = knn_matrix_creator(xvals,num_knn)[1]
    length = len(xvals)
    final_matrix = np.zeros(len(xvals)**2).reshape(len(xvals),len(xvals))


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
          row = PH_Spline(r[z])
          phs_matrix = np.vstack((phs_matrix,row))
      phs_matrix = phs_matrix[1:]



      #Creation the matrix of distances between sample points (with DERIVED PHS)
      phs_matrix_dx = np.ones(len(r))

      for z in range(0,len(r)):
          row = PH_Spline_dx(r[z])
          phs_matrix_dx = np.vstack((phs_matrix_dx,row))
      phs_matrix_dx = phs_matrix_dx[1:]

      #Creates the matrix of added polynomial terms
      poly_matrix = []
      for z in range(len(knn)):
          row = []
          row.append(1)
          for k in range(1,p_degree+1):
              row.append(knn[z]**(k))
          poly_matrix.append(row)

      #Creates the matrix of added derivative polynomial terms

      poly_matrix_dx = []
      for z in range(len(knn)):
        poly_matrix_dx.append(poly_dx(knn[z],p_degree))

      deriv_phs_vector = []

      for z in range(len(knn)):
        deriv_phs_vector.append(PH_Spline_dx(knn[z]-xvals[i]))


      deriv_poly_vector = poly_dx(xvals[i], p_degree)
      for z in range(len(deriv_poly_vector)):
        deriv_phs_vector.append(deriv_poly_vector[z])














      #Creation of A matrix (Collocation matrix)

      phs_matrix = list_to_matrix(phs_matrix)
      poly_matrix = list_to_matrix(poly_matrix)
      poly_matrix_t = poly_matrix.getT() #getT gets the transpose of the matrix
      zeroes = np.matrix(np.array([0]*(p_degree+1)**2).reshape(p_degree+1,p_degree+1))


      top_half_A = np.hstack((phs_matrix.getA(),poly_matrix.getA())) #getA makes the matrix an array which allows us to stack it
      bottom_half_A = np.hstack((poly_matrix_t.getA(), zeroes.getA()))
      A_matrix = np.matrix(np.vstack((top_half_A,bottom_half_A)))






      # Attempt to use sparse solver for large matrices
      A_sparse = scipy.sparse.csc_matrix(A_matrix)
      finite_weights = np.matrix(scipy.sparse.linalg.spsolve(A_sparse, deriv_phs_vector))
      #finite_weights = np.matmul(A_matrix_inv, deriv_phs_vector)


      finite_weights = np.array(finite_weights[0][:len(knn)-1 - (p_degree+1)])



      for j in range(num_knn):

        index = indices[i][j]

        final_matrix[i][index] = finite_weights[0][j]











    return final_matrix



poly_degree_2_vec = []
poly_degree_3_vec = []
poly_degree_4_vec = []
poly_degree_5_vec = []
xvals_graph = []
for i in range(1,16): # Iterates over number of sample points
  u = 50*i
  step = (stop-start)/u
  num_pts = int((((stop-start)/step)+1))
  xvals_ = np.linspace(start,stop, num = num_pts) #creates x values into a np.array
  true_y = y_func(xvals_)

  xvals_graph.append(u)
  for k in range(1,5): # Iterates over polynomial degrees
    global_diff_matrix = list_to_matrix(local_differentiation(xvals_, k+1))
    est_deriv = np.matmul(global_diff_matrix, true_y)
    true_deriv = [y_func_dx(xvals_)]
    diff = abs(true_deriv - est_deriv)/true_deriv
    diff = np.array(diff)
    infinity_norm = max(diff[0])
    if k == 1:
      poly_degree_2_vec.append(np.log(infinity_norm))
    if k == 2:
      poly_degree_3_vec.append(np.log(infinity_norm))
    if k == 3:
      poly_degree_4_vec.append(np.log(infinity_norm))
    if k == 4:
      poly_degree_5_vec.append(np.log(infinity_norm))






plt.scatter(xvals_graph, poly_degree_2_vec, color = 'red', label = '2') # Data Points
plt.scatter(xvals_graph, poly_degree_3_vec, color = 'blue', label = '3') # Data Points
plt.scatter(xvals_graph, poly_degree_4_vec, color = 'green', label = '4') # Data Points
plt.scatter(xvals_graph, poly_degree_5_vec, color = 'orange', label = '5') # Data Points
plt.xlabel('n')
plt.ylabel('infinity norm')
plt.title('Infinity norm vs. n')
plt.grid(True)
plt.legend()
plt.show()

end_time = time.time()

print("Execution Time: ", end_time - start_time, "seconds")