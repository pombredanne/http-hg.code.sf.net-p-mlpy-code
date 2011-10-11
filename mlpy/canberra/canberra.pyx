import numpy as np
cimport numpy as np
cimport cython

cdef extern from "c_canberra.h":
    double c_canberra(double *x, double *y, long n)
    double c_canberra_location(long *x, long *y, long n, long k)
    double c_canberra_stability(long *x, long n, long p, long k)

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
    """Returns the Canberra distance between two top-k lists.
    A top-k list is the sublist including the elements ranking 
    from position 0 to k-1 of the original list.
    x and y must be integer 1d array_like objects with entries
    from 0 to P-1, where P is the lenght of x and y.
    If k=None then k will set to P.

    The lower the indicator value, the higher the stability of 
    the lists.
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
    """Returns the Canberra stability indicator between N top-k
    ranked lists. A top-k list is the sublist including the elements
    ranking from position 0 to k-1 of the original list.
    x must be an integer 2d array_like object (N, P) with entries
    from 0 to P-1, where N is the number of lists and P is the 
    number of elements for each list. If k=None then k will set to P.

    The lower the indicator value, the higher the stability of 
    the lists.

    Example:

    >>> import numpy as np
    >>> import mlpy
    >>> x = np.array([[2,4,1,3,0], [3,4,1,2,0], [2,4,3,0,1]])  # 3 lists with entries must be from 0 to 4!
    >>> mlpy.canberra_stability(x, 3) #stability indicator on top-3 sublist
    0.74862979571499755
    """

    cdef np.ndarray[np.int64_t, ndim=2] x_arr
        
    x_arr = np.ascontiguousarray(x, dtype=np.int)
    
    if k == None:
        k = x_arr.shape[1]

    if k <= 0 or k > x_arr.shape[1]:
        raise ValueError("k must be in [1, %i]" % x_arr.shape[1])
    
    return c_canberra_stability(<long *> x_arr.data, 
               <long> x_arr.shape[0], <long> x_arr.shape[1], <long> k)
