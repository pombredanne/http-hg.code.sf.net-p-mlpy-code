## This code is written by Davide Albanese, <albanese@fbk.eu>.
## (C) 2012 mlpy Developers.

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

__all__ = ['LibSvm']

import sys
if sys.version >= '3':
    from . import libsvm
else:
    import libsvm

import numpy as np
from kernel_class import *

class LibSvm:
    def __init__(self, svm_type='c_svc', C=1, nu=0.5, eps=0.001,
                 p=0.1, shrinking=True, probability=False, weight={},
                 kernel=None):
        """LibSvm.
        
        :Parameters:
           svm_type : string
               SVM type, can be one of: 'c_svc', 'nu_svc', 
               'one_class', 'epsilon_svr', 'nu_svr'.             
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
           shrinking : bool
               use the shrinking heuristics
           probability : bool
               predict probability estimates
           weight : dict 
               changes the penalty for some classes (if the weight for a
               class is not changed, it is set to 1). For example, to
               change penalty for classes 1 and 2 to 0.5 and 0.8
               respectively set weight={1:0.5, 2:0.8}
           kernel : None or mlpy.Kernel object.
               if kernel is None, K and Kt in .learn()
               and in .pred*() methods must be precomputed kernel 
               matricies, else K and Kt must be training (resp. 
               test) data in input space.
        """
    
        if kernel is not None:
            if not isinstance(kernel, Kernel):
                raise ValueError("kernel must be None or a mlpy.Kernel object")
            
        self._kernel = kernel
        self._libsvm = libsvm.LibSvmBase(svm_type=svm_type, C=C, nu=nu, 
            eps=eps, p=p, shrinking=shrinking, probability=probability,
            weight=weight)
        self._x = None
    
    def learn(self, K, y):
        """Learning method.
        For classification, y is an integer indicating the class label
        (multi-class is supported). For regression, y is the target
        value which can be any real number. For one-class SVM, it's not used
        so can be any number.

        :Parameters:
           K: 2d array_like object
              precomputed training kernel matrix (if kernel=None);
              training data in input space (if kernel is a Kernel object)
           y : 1d array_like object integer (N)
              target values (N)
        """

        Karr = np.asarray(K, dtype=np.float)
        if self._kernel is not None:
            self._x = Karr.copy()
            Karr = self._kernel.kernel(Karr, Karr)
            
        self._libsvm.learn(Karr, y)

    def pred(self, Kt):
        """Does classification or regression on test vector(s) Kt.
                
        :Parameters:
            Kt : 1d or 2d array_like object
               precomputed test kernel matrix. (if kernel=None);
               test data in input space (if kernel is a Kernel object).

        :Returns:
            p : for a classification model, the predicted class(es) for Kt is
                returned. For a regression model, the function value(s) of Kt
                calculated using the model is returned. For an one-class
                model, +1 or -1 is returned.
        """

        Ktarr = np.asarray(Kt, dtype=np.float)
        if self._kernel is not None:
            Ktarr = self._kernel.kernel(Ktarr, self._x)
            
        return self._libsvm.pred(Ktarr)

    def pred_values(self, Kt):
        """Returns D decision values for each test sample. 
        For a classification model with C classes, this method
        returns D=C*(C-1)/2 decision values for each test sample. 
        The order is l[0] vs. l[1], ..., l[0] vs. l[C-1], l[1] vs. 
        l[2], ..., l[C-2] vs. l[C-1], where l can be obtained 
        from the labels() method.
        
        For a one-class model, this method returns D=1 decision value 
        for each test sample.
        
        For a regression model, this method returns the predicted
        value as in pred().
                
        :Parameters:
             Kt : 1d or 2d array_like object
              precomputed test kernel matrix. (if kernel=None);
              test data in input space (if kernel is a Kernel object).

        :Returns:
            decision values : 1d (D) or 2d numpy array (M, D)
                decision values for each observation.
        """
    
        Ktarr = np.asarray(Kt, dtype=np.float)
        if self._kernel is not None:
            Ktarr = self._kernel.kernel(Ktarr, self._x)
            
        return self._libsvm.pred_values(Ktarr)
    
    def pred_probability(self, Kt):
        """Returns C (number of classes) probability estimates.
        For a 'c_svc' and 'nu_svc' classification models with probability 
        information, this method computes 'number of classes' probability 
        estimates.

        :Parameters:
            Kt : 1d or 2d array_like object
              precomputed test kernel matrix. (if kernel=None);
              test data in input space (if kernel is a Kernel object).
        
        :Returns:
            probability estimates : 1d (C) or 2d numpy array (M,C)
                probability estimates for each observation.
        """

        Ktarr = np.asarray(Kt, dtype=np.float)
        if self._kernel is not None:
            Ktarr = self._kernel.kernel(Ktarr, self._x)
            
        return self._libsvm.pred_probability(Ktarr)

    def labels(self):
        """For a classification model, this method outputs class
        labels. For regression and one-class models, this method
        returns None.
        """

        return self._libsvm.labels()

    def sv_idx(self):
        """Returns the support vector indexes.
        """
        
        return self._libsvm.sv_idx()
