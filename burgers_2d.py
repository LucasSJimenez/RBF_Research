import numpy as np
import time

### Parameters

degree = 5
poly_degree = 3
start = -1
stop = 1
num_pts = 140
num_knn = (2*poly_degree) + 1
xvals = np.linspace(start, stop, num=num_pts)  # creates x values into a np.array

start_time = time.time()

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

### Burgers Equation Exact Solution
def burgers_exact_solution(x, t, v = 0.004375):

    aa = 0.05 * (x + 0.5 + 4.95 * t) / (2*v)
    
    bb = 0.25 * (x + 0.5 + 0.75 * t) / (4*v)
    
    cc = 0.5 * (x + 0.625) / (2*v)

    # Store the NEGATIVE exponents
    neg_aa = -aa
    neg_bb = -bb
    neg_cc = -cc

    # Find the maximum value among the exponents
    # We need to do this for each x value, so we stack them
    stacked_exponents = np.stack([neg_aa, neg_bb, neg_cc], axis=0)
    M = np.max(stacked_exponents, axis=0)

    # Rescale the exponents by subtracting the max value
    # This makes the largest exponent 0 (e.g., exp(0) = 1)
    # and all others negative, preventing underflow.
    exp_aa = np.exp(neg_aa - M)
    exp_bb = np.exp(neg_bb - M)
    exp_cc = np.exp(neg_cc - M)


    numerator = 0.1 * exp_aa + 0.5 * exp_bb + exp_cc
    denominator = exp_aa + exp_bb + exp_cc


    
    return numerator / denominator

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
t, v = 0.0, 0.004375
dt = 0.001
final_time = 1.2
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
    return final_matrix

def local_rbf_fd_d2():
    """
    Calculates the SECOND derivative (D2) matrix.
    Written in the same style as the original local_rbf_fd.
    """
    poly_degree = 3
    start = -1
    stop = 1
    num_pts = 140
    num_knn = (2*poly_degree) + 1
    xvals = np.linspace(start, stop, num=num_pts)  # creates x values into a np.array
    
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

d1_matrix = local_rbf_fd()
d2_matrix = local_rbf_fd_d2()

def rk4(V, t, k, F):
    k1 = F(V, t)
    k2 = F(V + 0.5 * k * k1, t + 0.5 * k)
    k3 = F(V + 0.5 * k * k2, t + 0.5 * k)
    k4 = F(V + k * k3, t + k)
    return V + (k / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

def burgers_starting_condition(xvals):
   return burgers_exact_solution(xvals, 0)

U = burgers_starting_condition(xvals)

def Function(u, t):
   new_u = np.copy(u)
   new_u[0] = burgers_exact_solution(-1, t)
   new_u[-1] = burgers_exact_solution(1, t)
   
   convection = -0.5 * (d1_matrix @ (new_u ** 2))

   diffusion = v * (d2_matrix @ new_u)

   # @ used for matrix vector multiplication

   fp = convection + diffusion

   return fp


while t < final_time:
    u = rk4(U, t, dt, Function)
    t += dt
    u[0] = burgers_exact_solution(-1, t)
    u[-1] = burgers_exact_solution(1, t)
    U = u
    

print("Final time: " + str(t))
print("Final Runtime: " +str(time.time() - start_time))
print(U)
         

## Numpy Functions

# Euclidean Distance: np.linalg.norm(a-b)


