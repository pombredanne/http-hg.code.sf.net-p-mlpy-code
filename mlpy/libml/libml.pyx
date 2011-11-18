## This code is written by Davide Albanese, <albanese@fbk.eu>
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


import numpy as np
cimport numpy as np
from libc.stdlib cimport *

from clibml cimport *

np.import_array()


cdef class KNN:
    """k-Nearest Neighbor (euclidean distance).
    """
    cdef NearestNeighbor nn
    cdef int k

    def __cinit__(self, k):
        """Initialization.
        
        :Parameters:
           k : int
              number of nearest neighbors
        """

        self.nn.x = NULL
        self.nn.y = NULL
        self.nn.classes = NULL
        self.k = int(k)
        
    def learn(self, x, y):
        """Learn method.
        
        :Parameters:	
           x : 2d array_like object (N x P)
              training data 
           y : 1d array_like integer 
              class labels (-1 or 1 for binary classification,
              1,..., nclasses for multiclass classification)
        """

        cdef int ret
        cdef np.ndarray[np.float_t, ndim=2] xarr
        cdef np.ndarray[np.int32_t, ndim=1] yarr
        cdef double *xp
        cdef double **xpp
        cdef int i

        xarr = np.ascontiguousarray(x, dtype=np.float)
        yarr = np.ascontiguousarray(y, dtype=np.int32)
        
        if self.k > xarr.shape[0]:
            raise ValueError("k must be smaller than number of samples")

        yu = np.unique(yarr)
        if yu.shape[0] <= 1:
            raise ValueError("y: number of classes must be >=2")
        if yu.shape[0] == 2:
            if not np.all(yu == np.array([-1, 1])):
                raise ValueError("y: for binary classification"
                                 "classes must be -1, 1")
        else:
            if not np.all(yu == np.arange(1, yu.shape[0]+1)):
                raise ValueError("y: for %d-class classification"
                                 "classes must be 1, ...,%d" \
                                     % (yu.shape[0], yu.shape[0]))           

        xp = <double *> xarr.data
        xpp = <double **> malloc (xarr.shape[0] * sizeof(double*))
        for i in range(xarr.shape[0]):
            xpp[i] = xp + (i * xarr.shape[1])
        
        self._free()
        ret = compute_nn(&self.nn, <int> xarr.shape[0], <int> xarr.shape[1],
                          xpp, <int *> yarr.data, self.k, DIST_EUCLIDEAN)
        free(xpp)

        if ret == 1:
            raise MemoryError("out of memory")
        
    def pred(self, t):
        """Predict KNN model on a test point(s).
        
        :Parameters:
           t : 1d or 2d array_like object ([M,] P)
              test point(s)
              
        :Returns:
	   p : the predicted value(s) on success:
           -1 or 1 for binary classification, 1, ..., nclasses 
           for multiclass classification, 0 on succes with non 
           unique classification
        """
        
        cdef int i
        cdef np.ndarray[np.float_t, ndim=1] tiarr
        cdef double *margin
        cdef double *tdata
        

        if self.nn.x is NULL:
            raise ValueError("no model computed")
        
        tarr = np.ascontiguousarray(t, dtype=np.float)
        if tarr.ndim > 2:
            raise ValueError("t must be an 1d or a 2d array_like object")

        if tarr.shape[-1] != self.nn.d:
            raise ValueError("t, model: shape mismatch")

        if tarr.ndim == 1:
            tiarr = tarr
            p = predict_nn(&self.nn, <double *> tiarr.data, &margin)
            free(margin)
            if p == -2:
                raise MemoryError("out of memory")
        else:
            p = np.empty(tarr.shape[0], dtype=np.int)
            for i in range(tarr.shape[0]):
                tiarr = tarr[i]
                tdata = <double *> tiarr.data
                p[i] = predict_nn(&self.nn, tdata, &margin)
                free(margin)
            if -2 in p:
                raise MemoryError("out of memory")
        
        return p

    def nclasses(self):
        """Returns the number of classes.
        """
        
        if self.nn.x is NULL:
            raise ValueError("no model computed")

        return self.nn.nclasses

    def labels(self):
        """Outputs the name of labels.
        """
        
        if self.nn.x is NULL:
            raise ValueError("no model computed")
        
        ret = np.empty(self.nn.nclasses, dtype=np.int)
        for i in range(self.nn.nclasses):
            ret[i] = self.nn.classes[i]

        return ret

    def _free(self):
        if self.nn.x is not NULL:
            for i in range(self.nn.n):
                free(self.nn.x[i])
            free(self.nn.x)

        if self.nn.y is not NULL:
            free(self.nn.y)
        
        if self.nn.classes is not NULL:
            free(self.nn.classes)


    def __dealloc__(self):
        self._free()

