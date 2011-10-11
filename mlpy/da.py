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

import numpy as np

__all__ = ['LDA', 'DLDA']


class LDA:
    """Linear Discriminant Analysis classifier.
    """
    
    def __init__(self):
        """Initialization.
        """

        self._labels = None
        self._weights = None
        self._bias = None
      
    def learn(self, x, y):
        """Learning method.

        :Parameters:
           x : 2d array_like object
              training data (N, P)
           y : 1d array_like object integer
              target values (N)
        """
        
        xarr = np.asarray(x, dtype=np.float)
        yarr = np.asarray(y, dtype=np.int)
        
        if xarr.ndim != 2:
            raise ValueError("x must be a 2d array_like object")
        
        if yarr.ndim != 1:
            raise ValueError("y must be an 1d array_like object")

        self._labels = np.unique(yarr)
        k = self._labels.shape[0]

        if k < 2:
            raise ValueError("number of classes must be >= 2")     
        
        p = np.empty(k, dtype=np.float)
        mu = np.empty((k, xarr.shape[1]), dtype=np.float)
        cov = np.zeros((xarr.shape[1], xarr.shape[1]), dtype=np.float)

        for i in range(k):
            wi = (yarr == self._labels[i])
            p[i] = np.sum(wi) / float(xarr.shape[0])
            mu[i] = np.mean(xarr[wi], axis=0)
            xi = xarr[wi] / mu[i]
            cov += np.dot(xi.T, xi)
        cov /= float(xarr.shape[0] - k)
        covinv = np.linalg.inv(cov)
        
        self._weights = np.empty((k, xarr.shape[1]), dtype=np.float)
        self._bias = np.empty(k, dtype=np.float)

        for i in range(k):           
            self._weights[i] = np.dot(covinv, mu[i])
            self._bias[i] = - 0.5 * np.dot(mu[i], self._weights[i]) + \
                np.log(p[i])

    def labels(self):
        """Outputs the name of labels.
        """
        
        return self._labels
        
    def weights(self):
        """Returns the feature weights.
        For multiclass classification this method returns a 2d 
        numpy array where w[i] contains the weights of label i 
        (.labels()[i]). For binary classification an 1d numpy 
        array (w_label0 - w_label1) is returned.
        """
        
        if self._weights is None:
            raise ValueError("no model computed.")

        if self._labels.shape[0] == 2:
            return self._weights[0] - self._weights[1]
        else:
            return self._weights

    def bias(self):
        """Returns the bias."""
        
        if self._weights is None:
            raise ValueError("no model computed.")
        
        if self._labels.shape[0] == 2:
            return self._bias[0] - self._bias[1]
        else:
            return self._bias

    def pred(self, x):
        """Does classification on test vector(s) x.
      
        :Parameters:
            x : 1d (one sample) or 2d array_like object
                test data ([M,] P)
            
        :Returns:        
            p : int or 1d numpy array
                the predicted class(es) for x is returned.
        """
        
        if self._weights is None:
            raise ValueError("no model computed.")

        xarr = np.asarray(x, dtype=np.float)

        if xarr.ndim == 1:
            delta = np.empty(self._labels.shape[0], dtype=np.float)
            for i in range(self._labels.shape[0]):
                delta[i] = np.dot(xarr, self._weights[i]) + self._bias[i]
            return self._labels[np.argmax(delta)]
        else:
            delta = np.empty((xarr.shape[0], self._labels.shape[0]),
                        dtype=np.float)
            for i in range(self._labels.shape[0]):
                delta[:, i] = np.dot(xarr, self._weights[i]) + self._bias[i]
            return self._labels[np.argmax(delta, axis=1)]



class DLDA:
    """Diagonal Linear Discriminant Analysis classifier.
    The algorithm uses the procedure called Nearest Shrunken
    Centroids (NSC).
    """
    
    def __init__(self, delta):
        """Initialization.
        
        :Parameters:
           delta : float
              regularization parameter
        """

        self._delta = float(delta)
        self._xstd = None # s_j
        self._dprime = None # d'_kj
        self._xmprime = None # xbar'_kj
        self._p = None # class prior probability
        self._labels = None

    def learn(self, x, y):
        """Learning method.

        :Parameters:
           x : 2d array_like object
              training data (N, P)
           y : 1d array_like object integer
              target values (N)
        """
        
        xarr = np.asarray(x, dtype=np.float)
        yarr = np.asarray(y, dtype=np.int)
        
        if xarr.ndim != 2:
            raise ValueError("x must be a 2d array_like object")
        
        if yarr.ndim != 1:
            raise ValueError("y must be an 1d array_like object")

        self._labels = np.unique(yarr)
        k = self._labels.shape[0]

        if k < 2:
            raise ValueError("number of classes must be >= 2")
        
        xm = np.mean(xarr, axis=0)
        self._xstd = np.std(xarr, axis=0, ddof=1)
        s0 = np.median(self._xstd)
        self._dprime = np.empty((k, xarr.shape[1]), dtype=np.float)
        self._xmprime = np.empty((k, xarr.shape[1]), dtype=np.float)
        n = yarr.shape[0]
        self._p = np.empty(k, dtype=np.float)

        for i in range(k):
            yi = (yarr == self._labels[i])
            xim = np.mean(xarr[yi], axis=0)
            nk = np.sum(yi)
            mk = np.sqrt(nk**-1 - n**-1)
            d = (xim - xm) / (mk * (self._xstd + s0))
            
            # soft thresholding
            tmp = np.abs(d) - self._delta
            tmp[tmp<0] = 0.0
            self._dprime[i] = np.sign(d) * tmp
            
            self._xmprime[i] = xm + (mk * (self._xstd + s0) * self._dprime[i])
            self._p[i] = float(nk) / float(n)

    def labels(self):
        """Outputs the name of labels.
        """
        
        return self._labels
        
    def sel(self):
        """Returns the most important features (the features that 
        have a nonzero dprime for at least one of the classes).
        """

        return np.where(np.sum(self._dprime, axis=0) != 0)[0]

    def dprime(self):
        """Return the dprime d'_kj (C, P), where C is the
        number of classes.
        """
        
        return self._dprime

    def _score(self, x):
        """Return the discriminant score"""

        return - np.sum((x-self._xmprime)**2/self._xstd**2,
                        axis=1) + (2 * np.log(self._p))

    def _prob(self, x):
        """Return the probability estimates"""
        
        score = self._score(x)
        tmp = np.exp(score * 0.5)
        return tmp / np.sum(tmp)
        
    def pred(self, x):
        """Does classification on test vector(s) x.
      
        :Parameters:
           x : 1d (one sample) or 2d array_like object
              test data ([M,] P)
            
        :Returns:        
           p : int or 1d numpy array
              the predicted class(es) for x is returned.
        """
        
        if self._xmprime is None:
            raise ValueError("no model computed.")
        
        xarr = np.asarray(x, dtype=np.float)
        
        if xarr.ndim == 1:
            return self._labels[np.argmax(self._score(xarr))]
        else:
            ret = np.empty(xarr.shape[0], dtype=np.int)
            for i in range(xarr.shape[0]):
                ret[i] = self._labels[np.argmax(self._score(xarr[i]))]
            return ret
        
    def prob(self, x):
        """For each sample returns C (number of classes)
        probability estimates.
        """

        if self._xmprime is None:
            raise ValueError("no model computed.")
        
        xarr = np.asarray(x, dtype=np.float)

        if xarr.ndim == 1:
            return self._prob(xarr)
        else:
            ret = np.empty((xarr.shape[0], self._labels.shape[0]),
                dtype=np.float)
            for i in range(xarr.shape[0]):
                ret[i] = self._prob(xarr[i])
            return ret
