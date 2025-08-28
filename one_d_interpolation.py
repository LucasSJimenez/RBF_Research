#RBF Code

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes as ax
import math
from one_d_parameters import *
import scipy.linalg
import time

start_time = time.time()
#import sklearn.neighbors as nn

# Nearest Neighbors'



# import sklearn.neighbors as nn



## Adjustable Parameters (parameters.py)

# degree
# poly_degree
# start
# stop
# u
# num_interpolation_pts


## Dependent Parameters

step = (stop-start)/u

num_pts = int((((stop-start)/step)+1))


#list to matrix function

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

# k-nearest neighbor search for 2-D and above

#test_arr = [[1,1],[34,2],[3,3],[7,8],[8,9]]
#nbr = nn.NearestNeighbors(n_neighbors=3)
#nbr.fit(test_arr)
#print(nbr.kneighbors_graph(test_arr).toarray())



#x value creation

xvals = np.linspace(start,stop, num = num_pts) #creates x values into a np.array

extended_xvals = np.linspace(start, stop, num = num_interpolation_pts+1)

true_y = y_func(xvals)

def interpolation_function(xvals,poly_degree):



    X, Y = np.meshgrid(xvals, xvals)

    big_X, big_Y = np.meshgrid(xvals, extended_xvals)

    #Distance between points (input into PH_Spline)

    r = X-Y
    r_ext = big_X - big_Y
    phs_matrix= np.ones(len(r))

    #Creates the matrix of distances between the sample points
    for i in range(0,len(r)):
        row = PH_Spline(r[i])
        phs_matrix = np.vstack((phs_matrix,row))
    phs_matrix = phs_matrix[1:]

    #Creation the matrix of distances between sample points (with DERIVED PHS)
    phs_matrix_dx = np.ones(len(r))

    for i in range(0,len(r)):
        row = PH_Spline_dx(r[i])
        phs_matrix_dx = np.vstack((phs_matrix_dx,row))
    phs_matrix_dx = phs_matrix_dx[1:]
    #Creates the extended matrix of interpolation points

    phs_matrix_ext = np.ones(len(r_ext[1]))

    for i in range(0,len(r_ext)):
        row = PH_Spline(r_ext[i])
        phs_matrix_ext = np.vstack((phs_matrix_ext,row))
    phs_matrix_ext = phs_matrix_ext[1:]


    #Creates the matrix of added polynomial terms
    poly_matrix = []
    for i in range(len(xvals)):
        row = []
        row.append(1)
        for k in range(1,poly_degree+1):
            row.append(xvals[i]**(k))
        poly_matrix.append(row)

    poly_matrix_ext = []
    for i in range(len(extended_xvals)):
        row = []
        row.append(1)
        for k in range(1,poly_degree+1):
            row.append(extended_xvals[i]**(k))
        poly_matrix_ext.append(row)


    #Creates the matrix of added derivative polynomial terms

    poly_matrix_dx = []
    for i in range(len(xvals)):
      poly_matrix_dx.append(poly_dx(xvals[i], poly_degree))






    #Creation of A matrix (Collocation matrix)

    phs_matrix = list_to_matrix(phs_matrix)
    poly_matrix = list_to_matrix(poly_matrix)
    poly_matrix_t = poly_matrix.getT() #getT gets the transpose of the matrix
    zeroes = np.matrix(np.array([0]*(poly_degree+1)**2).reshape(poly_degree+1,poly_degree+1))

    top_half_A = np.hstack((phs_matrix.getA(),poly_matrix.getA())) #getA makes the matrix an array which allows us to stack it
    bottom_half_A = np.hstack((poly_matrix_t.getA(), zeroes.getA()))
    A_matrix = np.matrix(np.vstack((top_half_A,bottom_half_A)))

    # Extended A Matrix

    phs_matrix_ext = list_to_matrix(phs_matrix_ext)

    poly_matrix_ext = list_to_matrix(poly_matrix_ext)
    poly_matrix_t_ext = poly_matrix_ext.getT() #getT gets the transpose of the matrix
    zeroes = np.matrix(np.array([0]*(poly_degree+1)**2).reshape(poly_degree+1,poly_degree+1))

    top_half_A = np.hstack((phs_matrix_ext.getA(),poly_matrix_ext.getA())) #getA makes the matrix an array which allows us to stack it
    bottom_half_A = np.hstack((poly_matrix_t.getA(), zeroes.getA()))
    A_matrix_ext = np.matrix(np.vstack((top_half_A,bottom_half_A)))




    #Solving for lambda

    function_vals = list_to_matrix(np.hstack((true_y, [0]*len(zeroes)))).getT()




    #1D KNN Matrix





    #lambda_weights = np.linalg.solve(final_matrix.getA(), function_vals)
    lambda_weights = np.matrix(scipy.linalg.solve(A_matrix, function_vals))


    #Removing zeroes from lambda weights
    #Solving for interpolation points
    interpolation_pts = np.matmul(A_matrix_ext, lambda_weights)
    interpolation_pts = interpolation_pts[:len(extended_xvals)]

    #plt.plot(xvals, true_y, 'bo', label="Original Points")  # Blue dots for original points
    #plt.plot(extended_xvals, interpolation_pts, 'ro', label="Interpolation Curve")

    return interpolation_pts




poly_degree_3_vec = []
poly_degree_5_vec = []
poly_degree_7_vec = []
poly_degree_9_vec = []



xvals_graph = []

for i in range(1,10): # Iterates over number of sample points
  u = 200*i
  step = (stop-start)/u
  num_pts = int((((stop-start)/step)+1))
  xvals_ = np.linspace(start,stop, num = num_pts) #creates x values into a np.array
  true_y = y_func(xvals_)
  xvals_graph.append(u)
  for k in range(1,5): # Iterates over polynomial degrees

    p_degree = 2*k + 1
    interpolant = interpolation_function(xvals_, p_degree)
    ext_true_y = y_func(extended_xvals)
    diff = []

    for i in range(len(extended_xvals)):
        diff.append(float((abs(interpolant[i] - ext_true_y[i])/ext_true_y[i]).item()))


    infinity_norm = max(diff)
    if k == 1:
      poly_degree_3_vec.append(np.log10(infinity_norm))
    if k == 2:
      poly_degree_5_vec.append(np.log10(infinity_norm))
    if k == 3:
      poly_degree_7_vec.append(np.log10(infinity_norm))
    if k == 4:
      poly_degree_9_vec.append(np.log10(infinity_norm))






plt.scatter(xvals_graph, poly_degree_3_vec, color = 'red', label = '3') # Data Points
plt.scatter(xvals_graph, poly_degree_5_vec, color = 'blue', label = '5') # Data Points
plt.scatter(xvals_graph, poly_degree_7_vec, color = 'green', label = '7') # Data Points
plt.scatter(xvals_graph, poly_degree_9_vec, color = 'orange', label = '9') # Data Points
plt.xlabel('n')
plt.ylabel('infinity norm')
plt.title('Infinity norm vs. n')
plt.grid(True)
plt.legend()
plt.show()
## Basic Plot
#plt.plot(xvals, true_y, 'bo', label="Original Points")  # Blue dots for original points
#plt.plot(extended_xvals, interpolation_pts, 'ro', label="Interpolation Curve")

##by interval of 10
#start_int_pts = 1
#stop_int_pts = 10

#for i in range(start_int_pts, stop_int_pts+1):
    #extended_xvals = np.linspace(start, stop, num = int(((stop-start)/step)*(i*10)+1) )
    #extended_true_y = y_func(extended_xvals)
    #infinity_norm.append(math.log10(float(abs(max(interpolation_function(i*10) - extended_true_y)))))

#fig, ax = plt.subplots()

#for k in range(3,9):
    #poly_degree = k
    #infinity_norm = []
    #for i in range(start_int_pts, stop_int_pts+1):
        #extended_xvals = np.linspace(start, stop, num = int(((stop-start)/step)*(i*10)+1) )
        #extended_true_y = y_func(extended_xvals)
        #infinity_norm.append(math.log10(float(abs(max(interpolation_function(i*10) - extended_true_y)))))
    #ax.plot(range(start_int_pts, stop_int_pts+1), infinity_norm, '.', label = f'Degree {k}')

#for i in range(1,10):
    #new_u = u + i*3
    #u = new_u

# for k in range(3,9):
#     poly_degree = k
#     infinity_norm = []
#     for i in range(1,10):
#         new_u = u + i*3
#         u = new_u
#         high_error = max(interpolation_function())
#         infinity_norm.append(math.log10(float(abs(high_error))))
#     ax.plot(range(3,28,3), infinity_norm, '.', label = f'Degree {k}')


#ax.legend(title = 'Polynomial Degree', loc = 'lower right', bbox_to_anchor = (1,0), fontsize = 'small')
#plt.xlabel("Number of Points")
#plt.ylabel("Log10 Infinity Norm")
#plt.title("Infinity Norm vs. Number of Points")
#plt.show()


end_time = time.time()
print(f"Runtime: {end_time - start_time} seconds")