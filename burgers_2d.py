import numpy as np

### Parameters

degree = 5
poly_degree = 4
start = 0.001
stop = 1
u = 20
num_interpolation_pts = 100

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
x, t = 0.5, 0.1
poly_degree = 3
terms, dterms_dt = poly_terms_and_dt(x, t, poly_degree)
print("Polynomial terms:", terms)
print("Derivatives w.r.t t:", dterms_dt)


## Numpy Functions

# Euclidean Distance: np.linalg.norm(a-b)

"""
Questions:



"""

