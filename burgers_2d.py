import numpy as np

### Parameters

degree = 5
poly_degree = 3
start = -1
stop = 1
num_pts = 140
num_knn = (2*poly_degree) + 1
xvals = np.linspace(start, stop, num=num_pts)  # creates x values into a np.array

def phs(r):
    return (abs(r) ** (degree))

def deriv_phs(r):
    return degree * (r ** (degree - 1))

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

### Burgers Equation Exact Solution
def burgers_exact_solution(x, t, v):
    component_a = -(x + 0.5 + 4.95*t)/(2*v)
    component_b = -(x + 0.5)/(4*v)
    component_c = -(x + 0.625 + 0.75*t)/(2*v)
    return ((0.1*np.exp(component_a) + 0.5*np.exp(component_b) + np.exp(component_c)) / (np.exp(component_a) + np.exp(component_b) + np.exp(component_c)))

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
t, v = 1.2, 0.004375
poly_degree = 3

def local_rbf_fd():
    poly_degree = 3
    start = -1
    stop = 1
    num_pts = 140
    num_knn = (2*poly_degree) + 1
    xvals = np.linspace(start, stop, num=num_pts)  # creates x values into a np.array
   
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

      sol_vec = np.append(sol_vec,[0]*(poly_degree+1))

      finite_weights = np.linalg.solve(A_matrix, sol_vec)
      final_matrix[i][indices[i]] = finite_weights[0:num_knn]
    print((final_matrix[70]))

local_rbf_fd()

## Numpy Functions

# Euclidean Distance: np.linalg.norm(a-b)

"""
Questions:

Absolute value in phs?

"""

