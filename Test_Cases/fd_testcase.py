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

def true_func_dx(x):
    return 2*np.pi * np.cos(2*np.pi * x)

def phs(r):
    return (abs(r) ** (degree))

def deriv_phs(r):
    return degree * (np.abs(r) ** (degree - 1))

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

### Base Functions

def poly_terms_and_dt(var1, t, poly_degree):
    """
    Returns:
        terms: list of polynomial terms evaluated at (x, t)
        dterms_dt: list of derivatives of those terms with respect to t
    """
    terms = []
    dterms_dt = []
    # Generate all monomials: x^i * t^j where i + j <= poly_degree
    for i in range(poly_degree + 1):
        for j in range(poly_degree + 1 - i):
            term = (var1 ** i) * (t ** j)
            # Derivative w.r.t t: j * x^i * t^(j-1) (if j > 0), else 0
            if j == 0:
                dterm_dt = 0
            else:
                dterm_dt = j * (var1 ** i) * (t ** (j - 1))
                print("i: " + str(i) + " j: " + str(j))
            terms.append(term)
            dterms_dt.append(dterm_dt)
    return terms, dterms_dt

# Example usage:
poly_degree = 3

def local_rbf_fd(xvals = xvals, poly_degree = poly_degree):
    
    num_knn = (2*poly_degree) + 1
   
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
          row = phs(r[z])
          phs_matrix = np.vstack((phs_matrix,row))
      phs_matrix = phs_matrix[1:]

      sol_vec = np.array(deriv_phs(r[0]))
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


      top_half_A = np.hstack((phs_matrix.getA(),poly_matrix.getA())) #getA makes the matrix an array which allows us to stack it
      bottom_half_A = np.hstack((poly_matrix_t.getA(), zeroes.getA()))
      A_matrix = np.matrix(np.vstack((top_half_A,bottom_half_A)))

      poly_deriv_vec = []
      center_node_x = knn[0] # The center of the stencil
    
      for k in range(poly_degree + 1):
        if k == 0:
            poly_deriv_vec.append(0.0)  # d(1)/dx = 0
        else:
            # d(x^k)/dx = k * x^(k-1)
            poly_deriv_vec.append(k * (center_node_x ** (k - 1)))
            
      # Append the correct derivative vector
      sol_vec = np.append(sol_vec, poly_deriv_vec)

      finite_weights = np.linalg.solve(A_matrix, sol_vec)
      final_matrix[i][indices[i]] = finite_weights[0:num_knn]
    d1_matrix = final_matrix
    u = true_func(xvals)
    u_dx_approx = d1_matrix @ u
    return u_dx_approx

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
      true_vals = true_func_dx(xvals)
      est_deriv = local_rbf_fd(xvals, (k*2)+1)
      error = np.log(np.max(np.abs(est_deriv - true_vals)))
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

plt.scatter(xvals_graph, poly_degree_3_vec, color = 'red', label = 'Poly Degree 3', s = 12, marker = 'o') # Data Points
plt.scatter(xvals_graph, poly_degree_5_vec, color = 'purple', label = 'Poly Degree 5', s = 12, marker = 'D') # Data Points
plt.scatter(xvals_graph, poly_degree_7_vec, color = 'green', label = 'Poly Degree 7', s = 12, marker = '+') # Data Points
#plt.scatter(xvals_graph, poly_degree_9_vec, color = 'orange', label = 'Poly Degree 9', s = 12) # Data Points
plt.xlabel('Base 10 Log of n')
plt.ylabel('Base 10 Log of Max Error')
plt.title('Base 10 Log of Max Error vs. Base 10 Log of n')
plt.grid(True)
plt.legend()
plt.savefig(r"c:\Users\lucas\RBF_Research\RBF_Research\Test_Cases\deriv_error_vs_n.png", dpi=300, bbox_inches="tight")
plt.show()


# Do another example with the D2 directly calculated and D1*D1



    


         
