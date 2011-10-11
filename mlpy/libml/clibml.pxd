cdef extern from "src/ml.h":
    int TRUE
    int FALSE
    int SORT_ASCENDING
    int SORT_DESCENDING
    int DIST_SQUARED_EUCLIDEAN
    int DIST_EUCLIDEAN
    int SVM_KERNEL_LINEAR
    int SVM_KERNEL_GAUSSIAN
    int SVM_KERNEL_POLINOMIAL
    int BAGGING
    int AGGREGATE
    int ADABOOST

    ctypedef struct NearestNeighbor:
        int n
        int d
        double **x
        int *y
        int nclasses
        int *classes
        int k
        int dist
        
    int compute_nn(NearestNeighbor *nn, int n, int d, double *x[], int y[],
                   int k, int dist)
    int predict_nn(NearestNeighbor *nn, double x[],double **margin)
    
