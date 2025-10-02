#RBF Code

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes as ax
import scipy.linalg
import time

start_time = time.time()

#list to matrix function
def list_to_matrix(list):
      return np.matrix(np.array(list))

def y_func(x):
    return np.sin((x)*(2*np.pi))

def y_func_dx(x):
    return (2*np.pi)*np.cos(x*2*np.pi)

def gaussian_rbf(r, epsilon=3.0):
    return np.exp(-(epsilon * r)**2)




num_pts = 4

start= 1
stop = 6

xvals = [1,2,4,6]
extended_xvals = np.linspace(start, stop, num = 21)
print(extended_xvals)
true_y = [1,4,5,9]

def interpolation_function_gaussian(xvals, epsilon=0.7):
    X, Y = np.meshgrid(xvals, xvals)
    big_X, big_Y = np.meshgrid(xvals, extended_xvals)
    r = X-Y
    r_ext = big_X - big_Y
    rbf_matrix= np.ones(len(r))
    for i in range(0,len(r)):
        row = gaussian_rbf(r[i], epsilon)
        rbf_matrix = np.vstack((rbf_matrix,row))
    rbf_matrix = rbf_matrix[1:]
    rbf_matrix_ext = np.ones(len(r_ext[1]))
    for i in range(0,len(r_ext)):
        row = gaussian_rbf(r_ext[i], epsilon)
        rbf_matrix_ext = np.vstack((rbf_matrix_ext,row))
    rbf_matrix_ext = rbf_matrix_ext[1:]
    rbf_matrix = list_to_matrix(rbf_matrix)

    rbf_matrix_ext = list_to_matrix(rbf_matrix_ext)

    
    
    function_vals = list_to_matrix(true_y).getT()
    lambda_weights = np.matrix(scipy.linalg.solve(rbf_matrix, function_vals))
    interpolation_pts = np.matmul(rbf_matrix_ext, lambda_weights)

    return np.array(interpolation_pts).flatten()


xvals_graph = []




plt.scatter(extended_xvals, interpolation_function_gaussian(xvals), color = 'blue', label = 'Interpolation Points')
plt.scatter(xvals, true_y, color = 'red', label = 'Data Points')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('One-Dimensional Gaussian RBF Interpolation Example')
plt.grid(True)
plt.legend()
plt.savefig('Test_Cases/one_d_interpolation_gaussian_example.png', dpi=600)

end_time = time.time()
print(f"Runtime: {end_time - start_time} seconds")
