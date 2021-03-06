## This code is written by Davide Albanese, <albanese@fbk.eu>.
## (C) 2011 mlpy Developers.

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


__all__ = ["Parzen"]

import numpy as np


class Parzen:
    """Parzen based classifier (binary).
    """

    def __init__(self, kernel=None):
        """Initialization.

        :Parameters:
           kernel : None or mlpy.Kernel object.
              if kernel is None, K and Kt in .learn()
              and in .pred() methods must be precomputed kernel 
              matricies, else K and Kt must be training (resp. 
              test) data in input space.
        """

        self._alpha = None
        self._b = None
        self._labels = None
        self._kernel = kernel
        self._x = None
        self._model = False
                
    def learn(self, K, y):
        """Learning method.

        Parameters:
           K: 2d array_like object
              precomputed training kernel matrix (if kernel=None);
              training data in input space (if kernel is a Kernel object)
           y : 1d array_like object (N)
              target values
        """

        K_arr = np.asarray(K, dtype=np.float)
        y_arr = np.asarray(y, dtype=np.int)

        if K_arr.ndim != 2:
            raise ValueError("K must be a 2d array_like object")

        if y_arr.ndim != 1:
            raise ValueError("y must be an 1d array_like object")
        
        if K_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("K, y shape mismatch")

        if self._kernel is None:
            if K_arr.shape[0] != K_arr.shape[1]:
                raise ValueError("K must be a square matrix")
        else:
            self._x = K_arr.copy()
            K_arr = self._kernel.kernel(K_arr, K_arr)
        
        self._labels = np.unique(y_arr)
        if self._labels.shape[0] != 2:
            raise ValueError("number of classes != 2")

        ynew = np.where(y_arr==self._labels[0], 1., -1.)
        n = K_arr.shape[0]
        
        # from Kernel Methods for Pattern Analysis
        # Algorithm 5.6
        
        nplus = np.sum(ynew==1)
        nminus = n - nplus
        alphaplus = np.where(ynew==1, nplus**-1, 0)
        alphaminus = np.where(ynew==-1, nminus**-1, 0)
        self._b = -0.5 * (np.dot(np.dot(alphaplus, K_arr), alphaplus) - \
                         np.dot(np.dot(alphaminus, K_arr), alphaminus))
        self._alpha = alphaplus - alphaminus
        
        self._model = True

    def pred_values(self, Kt):
        """Returns the decision value (d(Kt)) for eache test sample.

        :Parameters:	
           Kt : 1d or 2d array_like object
              precomputed test kernel matrix. (if kernel=None);
              test data in input space (if kernel is a Kernel object).
        :Returns:	
           decision values : 1d (1) or 2d numpy array (M, 1)
              decision values for each observation.
        """

        if not self._model:
            raise ValueError("no model computed")

        Ktarr = np.asarray(Kt, dtype=np.float)
        if self._kernel is not None:
            Ktarr = self._kernel.kernel(Ktarr, self._x)

        try:
            values = np.dot(self._alpha, Ktarr.T) + self._b
        except ValueError:
            raise ValueError("Kt, alpha: shape mismatch")

        if Ktarr.ndim == 1:
            return np.array([values])
        else:
            return values.reshape(-1, 1)

    def pred(self, Kt):
        """Does classification on test vector(s) `Kt`.
        Returns l_1 if g(Kt) > 0, l_2 otherwise.
      
        :Parameters:
           Kt : 1d or 2d array_like object
              precomputed test kernel matrix. (if kernel=None);
              test data in input space (if kernel is a Kernel object).
            
        :Returns:        
            p : integer or 1d numpy array
                the predicted class(es)
        """

        values = self.pred_values(Kt)

        if values.ndim == 1:
            values = values[0]
        else:
            values = np.ravel(values)

        return np.where(values > 0, self._labels[0], self._labels[1]) \
            .astype(np.int)
          
    def alpha(self):
        """Return alpha.
        """
        
        if not self._model:
            raise ValueError("no model computed")
        
        return self._alpha

    def b(self):
        """Return b.
        """
        
        if not self._model:
            raise ValueError("no model computed")
        
        return self._b
    
    def labels(self):
        """Returns the class labels.
        """
        
        if not self._model:
            raise ValueError("no model computed")

        return self._labels
                        
