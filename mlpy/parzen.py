## This code is written by Davide Albanese, <albanese@fbk.eu>.
## (C) 2011 Davide Albanese.

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

    def __init__(self):
        """Initialization.
        """

        self._alpha = None
        self._b = None
        self._labels = None
                
    def learn(self, K, y):
        """Compute alpha and b.

        Parameters:
           K: 2d array_like object (N, N)
              precomputed kernel matrix
           y : 1d array_like object (N)
              target values
        """

        K_arr = np.asarray(K, dtype=np.float)
        y_arr = np.asarray(y, dtype=np.int)

        if K_arr.ndim != 2:
            raise ValueError("K must be a 2d array_like object")

        if y_arr.ndim != 1:
            raise ValueError("y must be an 1d array_like object")
        
        if K_arr.shape[0] != K_arr.shape[1]:
            raise ValueError("K must be a square matrix")

        if K_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("K, y shape mismatch")
        
        self._labels = np.unique(y_arr)
        if self._labels.shape[0] != 2:
            raise ValueError("number of classes != 2")

        ynew = np.where(y_arr==self._labels[0], -1., 1.)
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

    def pred(self, Kt):
        """Compute the predicted class.

        :Parameters:
           Kt : 1d or 2d array_like object ([M], N)
              test kernel matrix. Precomputed inner products 
              (in feature space) between M testing and N 
              training points.

        :Returns:
           p : integer or 1d numpy array
              predicted class
        """

        if self._alpha is None:
            raise ValueError("no model computed; run learn() first")

        Kt_arr = np.asarray(Kt, dtype=np.float)

        try:
            s = np.sign(np.dot(self._alpha, Kt_arr.T) + self._b)
        except ValueError:
            raise ValueError("Kt, alpha: shape mismatch")

        return np.where(s==-1, self._labels[0], self._labels[1]) \
            .astype(np.int)
          
    def alpha(self):
        """Return alpha.
        """
        return self._alpha

    def b(self):
        """Return b.
        """
        return self._b
    
    def labels(self):
        """Outputs the name of labels.
        """
        
        return self._labels
                        