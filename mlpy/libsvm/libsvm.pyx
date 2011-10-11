## This code is written by Davide Albanese, <albanese@fbk.eu>
## (C) 2010 mlpy Developers.

## This program is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.

## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.

## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <http://www.gnu.org/licenses/>.


import numpy as np
cimport numpy as np
from libc.stdlib cimport *

from clibsvm cimport *

   
cdef void print_null(char *s):
   pass

# array 1D to svm node
cdef svm_node *array1d_to_node(np.ndarray[np.float64_t, ndim=1] x):
    cdef int i, k
    cdef np.ndarray[np.int64_t, ndim=1] nz
    cdef svm_node *ret

    nz = np.nonzero(x)[0]
    ret = <svm_node*> malloc ((nz.shape[0]+1) * sizeof(svm_node))
            
    k = 0
    for i in nz:
        ret[k].index = i+1
        ret[k].value = x[i]
        k += 1
    ret[nz.shape[0]].index = -1
            
    return ret

# array 2D to svm node
cdef svm_node **array2d_to_node(np.ndarray[np.float64_t, ndim=2] x):
    cdef int i
    cdef svm_node **ret

    ret = <svm_node **> malloc \
        (x.shape[0] * sizeof(svm_node *))
    
    for i in range(x.shape[0]):
        ret[i] = array1d_to_node(x[i])
            
    return ret

# array 1D to vector
cdef double *array1d_to_vector(np.ndarray[np.float64_t, ndim=1] y):
     cdef int i
     cdef double *ret

     ret = <double *> malloc (y.shape[0] * sizeof(double))

     for i in range(y.shape[0]):
         ret[i] = y[i]

     return ret


cdef class LibSvm:
    cdef svm_problem problem
    cdef svm_parameter parameter
    cdef svm_model *model 
        
    cdef int learn_disabled

    SVM_TYPE = ['c_svc',
                'nu_svc',
                'one_class',
                'epsilon_svr',
                'nu_svr']

    KERNEL_TYPE = ['linear',
                   'poly',
                   'rbf',
                   'sigmoid']

    def __cinit__(self, svm_type='c_svc', kernel_type='linear', 
                  degree=3, gamma=0.001, coef0=0, C=1, nu=0.5,
                  eps=0.001, p=0.1, cache_size=100, shrinking=True,
                  probability=False, weight={}):
        """LibSvm.
        
        :Parameters:

            svm_type : string
                SVM type, can be one of: 'c_svc', 'nu_svc', 
                'one_class', 'epsilon_svr', 'nu_svr'. The method load_model()
                can overwrite this parameter 
            kernel_type : string
                kernel type, can be one of: 'linear' (u'*v),
                'poly' ((gamma*u'*v + coef0)^degree), 'rbf' 
                (exp(-gamma*|u-v|^2)), 'sigmoid'
                (tanh(gamma*u'*v + coef0)). The method load_model()
                can overwrite this parameter
            degree : int (for 'poly' kernel_type)
                degree in kernel. The method load_model()
                can overwrite this parameter
            gamma : float (for 'poly', 'rbf', 'sigmoid' kernel_type)
                gamma in kernel (e.g. 1 / number of features).
                The method load_model() can overwrite this parameter
            coef0 : float (for 'poly', 'sigmoid' kernel_type)
                coef0 in kernel. The method load_model()
                can overwrite this parameter
            C : float (for 'c_svc', 'epsilon_svr', 'nu_svr')
                cost of constraints violation
            nu : float (for 'nu_svc', 'one_class', 'nu_svr')
                nu parameter
            eps : float
                stopping criterion, usually 0.00001 in nu-SVC,
                0.001 in others
            p : float (for 'epsilon_svr')
                p is the epsilon in epsilon-insensitive loss function
                of epsilon-SVM regression
            cache_size : float [MB]
                size of the kernel cache, specified in megabytes
            shrinking : bool
                use the shrinking heuristics
            probability : bool
                predict probability estimates
            weight : dict 
                changes the penalty for some classes (if the weight for a
                class is not changed, it is set to 1). For example, to
                change penalty for classes 1 and 2 to 0.5 and 0.8
                respectively set weight={1:0.5, 2:0.8}
        """
        
        svm_set_print_string_function(&print_null)
        self.learn_disabled = 0

        try:
            self.parameter.svm_type = self.SVM_TYPE.index(svm_type)
        except ValueError:
            raise ValueError("invalid svm_type")
	
        try:
            self.parameter.kernel_type = self.KERNEL_TYPE.index(kernel_type)
        except ValueError:
            raise ValueError("invalid kernel_type")
	
        self.parameter.degree = degree
        self.parameter.gamma = gamma
        self.parameter.coef0 = coef0
        self.parameter.C = C
        self.parameter.nu = nu
        self.parameter.eps = eps
        self.parameter.p = p
        self.parameter.cache_size = cache_size
        self.parameter.shrinking = int(shrinking)
        self.parameter.probability = int(probability)
	
        # weight
        self.parameter.nr_weight = len(weight)
        self.parameter.weight_label = <int *> malloc \
            (len(weight) * sizeof(int))
        self.parameter.weight = <double *> malloc \
            (len(weight) * sizeof(double))
        try:
            for i, key in enumerate(weight):
                self.parameter.weight_label[i] = int(key)
                self.parameter.weight[i] = float(weight[key])
        except ValueError:
            raise ValueError("invalid weight")
        
        self.model = NULL
                    
    def __dealloc__(self):
        self._free_problem()
        self._free_model()
        self._free_param()

    def _load_problem(self, x, y):
        """Convert the data into libsvm svm_problem struct
        """

        xarr = np.ascontiguousarray(x, dtype=np.float64)
        yarr = np.ascontiguousarray(y, dtype=np.float64)
        
        if xarr.ndim != 2:
            raise ValueError("x must be a 2d array_like object")

        if yarr.ndim != 1:
            raise ValueError("y must be an 1d array_like object")

        if xarr.shape[0] != yarr.shape[0]:
            raise ValueError("x, y: shape mismatch")
        
        self.problem.x = array2d_to_node(xarr)
        self.problem.y = array1d_to_vector(yarr)
        self.problem.l = x.shape[0]
                
    def learn(self, x, y):
        """Constructs the model.
        
        :Parameters:
        
            x : 2d array_like object
                training data (samples x features)
            y : 1d array_like object
                target values
        """
        
        cdef char *ret

        if self.learn_disabled:
            raise ValueError("learn method is disabled (model from file)")
        
        self._free_problem()
        self._load_problem(x, y)
        ret = svm_check_parameter(&self.problem, &self.parameter)
	
        if ret != NULL:
            raise ValueError(ret)

        self._free_model()       
        self.model = svm_train(&self.problem, &self.parameter)
        
    def pred(self, x):
        """Does classification or regression on test vector(s) x.
                
        :Parameters:
        
            x : 1d (one sample) or 2d array_like object
                test data (samples x features)
            
        :Returns:
        
            p : for a classification model, the predicted class(es) for x is
                returned. For a regression model, the function value(s) of x
                calculated using the model is returned. For an one-class
                model, +1 or -1 is returned.
        """

        cdef int i
        cdef svm_node *test_node

        xarr = np.ascontiguousarray(x, dtype=np.float64)

        if xarr.ndim > 2:
            raise ValueError("x must be an 1d or a 2d array_like object")
        
        if self.model is NULL:
            raise ValueError("no model computed")

        if xarr.ndim == 1:
            test_node = array1d_to_node(xarr)
            p = svm_predict(self.model, test_node)
            free(test_node)
        else:
            p = np.empty(xarr.shape[0], dtype=np.float64)
            for i in range(xarr.shape[0]):
                test_node = array1d_to_node(xarr[i])
                p[i] = svm_predict(self.model, test_node)
                free(test_node)

        return p

    def pred_probability(self, x):
        """Does classification or regression on a test vector x
        given a model with probability information.

        For a classification model with probability information, this
        method computes 'number of classes' probability estimates. The
        class with the highest probability is returned. For regression
        / one-class SVM, the returned value is the same as that of
        pred().
        
        :Parameters:
        
            x : 1d (one sample) or 2d array_like object
                test data (samples x features)
            
        :Returns:
         
            p : for a classification model, the predicted class(es) for x is
                returned. For a regression model, the function value(s) of x
                calculated using the model is returned. For an one-class
                model, +1 or -1 is returned.
        """
        
        cdef svm_node *test_node
        cdef double *prob_estimates

        xarr = np.ascontiguousarray(x, dtype=np.float64)

        if xarr.ndim > 2:
            raise ValueError("x must be an 1d or a 2d array_like object")
        
        if self.model is NULL:
            raise ValueError("no model computed")
        
        ret = svm_check_probability_model(self.model)
        if ret == 0:
            raise ValueError("model does not contain required information"
                             " to do probability estimates. Set probability"
                             "=True")

        prob_estimates = <double*> malloc (self.model.nr_class * 
            sizeof(double))
        
        if xarr.ndim == 1:
            test_node = array1d_to_node(xarr)
            p = svm_predict_probability(self.model, test_node,
                prob_estimates)
            free(test_node)
        else:
            p = np.empty(xarr.shape[0], dtype=np.float64)
            for i in range(xarr.shape[0]):
                test_node = array1d_to_node(xarr[i])
                p[i] = svm_predict_probability(self.model, test_node,
                    prob_estimates)
                free(test_node)

        free(prob_estimates)
        return p
   
    def labels(self):
        """For a classification model, this method outputs the name of
        labels. For regression and one-class models, this method
        returns None.
        """
        
        if self.model is NULL:
            raise ValueError("no model computed")

        if self.model.label is NULL:
            return None
        else:
            ret = np.empty(self.model.nr_class, dtype=np.int32)
            for i in range(self.model.nr_class):
                ret[i] = self.model.label[i]
            return ret

    def nclasses(self):
        """Get the number of classes.
        = 2 in regression and in one class SVM
        """
        
        if self.model is NULL:
            raise ValueError("no model computed")

        return self.model.nr_class

    def nsv(self):
        """Get the total number of support vectors.
        """
        
        if self.model is NULL:
            raise ValueError("no model computed")

        return self.model.l

    def label_nsv(self):
        """Return a dictionary containing the number 
        of support vectors for each class (for classification).
        """

        cdef int i
        
        if self.model is NULL:
            raise ValueError("no model computed")

        nsv = {}

        if self.model.nSV is not NULL:            
            for i in range(self.model.nr_class):
                nsv[self.model.label[i]] = self.model.nSV[i]
            
        return nsv
    
    cdef void _free_problem(self):
        cdef int i
        
        if self.problem.x is not NULL:
            for i in range(self.problem.l):
                free(self.problem.x[i])
            free(self.problem.x)

        if self.problem.y is not NULL:
            free(self.problem.y)
        
    cdef void _free_model(self):
        svm_free_and_destroy_model(&self.model)

    cdef void _free_param(self):
        svm_destroy_param(&self.parameter)
    
    @classmethod
    def load_model(cls, filename):
        """Loads model from file. Returns a LibSvm object
        with the learn() method disabled.
        """
        
        ret = LibSvm()

        try:
            ret.model = svm_load_model(filename)
        except:
            raise ValueError("invalid filename")
        else:
            if ret.model is NULL:
                raise IOError("model file could not be loaded")
            
        ret.parameter.svm_type = ret.model.param.svm_type
        ret.parameter.kernel_type = ret.model.param.kernel_type
        ret.parameter.degree = ret.model.param.degree
        ret.parameter.gamma = ret.model.param.gamma
        ret.parameter.coef0 = ret.model.param.coef0
        ret.learn_disabled = 1
        
        return ret

    def save_model(self, filename):
        """Saves model to a file.
        """
        
        cdef int ret
        
        if self.model is NULL:
            raise ValueError("no model computed")
        
        ret = svm_save_model(filename, self.model)
        
        if ret == -1:
            raise IOError("problem with svm_save_model()")
