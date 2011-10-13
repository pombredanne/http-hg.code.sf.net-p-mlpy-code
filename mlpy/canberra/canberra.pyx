import numpy as np
cimport numpy as np
cimport cython

cdef extern from "c_canberra.h":
    double c_canberra(double *x, double *y, long n)
    double c_canberra_location(long *x, long *y, long n, long k)
    double c_canberra_stability(long *x, long n, long p, long k)
    double c_canberra_expected(long n, long k)

def canberra(x, y):
    """Returns the Canberra distance between two P-vectors x and y:
    sum_i(abs(x_i - y_i) / (abs(x_i) + abs(y_i))).
    """

    cdef np.ndarray[np.float64_t, ndim=1] x_arr
    cdef np.ndarray[np.float64_t, ndim=1] y_arr
    
    x_arr = np.ascontiguousarray(x, dtype=np.float)
    y_arr = np.ascontiguousarray(y, dtype=np.float)

    if x_arr.shape[0] != y_arr.shape[0]:
        raise ValueError("x, y: shape mismatch")
    
    return c_canberra(<double *> x_arr.data, <double *> y_arr.data,
                       <long> x_arr.shape[0])


def canberra_location(x, y, k=None):
    """Returns the Canberra distance between two position lists.
    A position list of length P contains the position (from 0 to P-1)
    of P elements. If k is not None the function computes the
    distance between the lists including the elements from position
    0 to k-1.
    """

    cdef np.ndarray[np.int64_t, ndim=1] x_arr
    cdef np.ndarray[np.int64_t, ndim=1] y_arr
    
    x_arr = np.ascontiguousarray(x, dtype=np.int)
    y_arr = np.ascontiguousarray(y, dtype=np.int)

    if x_arr.shape[0] != y_arr.shape[0]:
        raise ValueError("x, y: shape mismatch")
    
    if k == None:
        k = x_arr.shape[0]

    if k <= 0 or k > x_arr.shape[0]:
        raise ValueError("k must be in [1, %i]" % x_arr.shape[0])
    

    return c_canberra_location(<long *> x_arr.data,
               <long *> y_arr.data, <long> x_arr.shape[0], <long> k)


def canberra_stability(x, k=None):
    """Returns the Canberra stability indicator between N position
    lists. A position list of length P contains the position (from 0 to P-1)
    of P elements. If k is not None the function computes the
    indicator between the lists including the elements from position
    0 to k-1.The lower the indicator value, the higher the stability of 
    the lists.

    Example:

    >>> import numpy as np
    >>> import mlpy
    >>> x = np.array([[2,4,1,3,0], [3,4,1,2,0], [2,4,3,0,1]])  # 3 position lists
    >>> mlpy.canberra_stability(x, 3) # stability indicator
    0.74862979571499755
    >>> mlpy.canberra_stability_max(x.shape[1], 3) # max value
    """

    cdef np.ndarray[np.int64_t, ndim=2] x_arr
        
    x_arr = np.ascontiguousarray(x, dtype=np.int)
    
    if k == None:
        k = x_arr.shape[1]

    if k <= 0 or k > x_arr.shape[1]:
        raise ValueError("k must be in [1, %i]" % x_arr.shape[1])
    
    return c_canberra_stability(<long *> x_arr.data, 
               <long> x_arr.shape[0], <long> x_arr.shape[1], <long> k)


def canberra_location_expected(p, k=None):
    """Returns the expected value of the Canberra location distance,
    where `p` is the number of elements and `k` is the number of 
    positions to consider.
    """

    if k == None:
        k = p

    return c_canberra_expected(p, k)


def canberra_location_max(p):
    """Return the approximated maximum value of Canberra
    location distance.
    """

    return (np.log(3.0) / 2.0) * p - (2.0 / 3.0)


def canberra_stability_max(p, k=None):
    """Returns approximated maximum value of the Canberra 
    stability indicator where `p` is the number of elements
    and `k` is the number of positions to consider.
    """
    

    if k == None:
        k = p

    return canberra_location_max(p) / \
        canberra_location_expected(p, k)
