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
import scipy.linalg as spla
from ridge import ridge_base
from ols import ols_base
import kernel


__all__ = ['LDA', 'LDA_FAST', 'SRDA', 'KFDA', 'pca', 'kpca', 'pca_fast',
           'whiten']


def proj(u, v):
    """(<v, u> / <u, u>) u
    """

    return (np.dot(v, u) / np.dot(u, u)) * u


def gso(v, norm=False):
    """Gram-Schmidt orthogonalization.
    Vectors v_1, ..., v_k are stored by rows.
    """
    
    for j in range(v.shape[0]):
        for i in range(j):
            v[j] = v[j] - proj(v[i], v[j])
        
        if norm:
            v[j] /= np.linalg.norm(v[j])


def lda(xarr, yarr):
    """Linear Discriminant Analysis.
    
    Returns the transformation matrix `coeff` (P, C-1),
    where `x` is a matrix (N,P) and C is the number of
    classes. Each column of `x` represents a variable, 
    while the rows contain observations. Each column of 
    `coeff` contains coefficients for one transformation
    vector.
    
    Sample(s) can be embedded into the C-1 dimensional space
    by z = x coeff (z = np.dot(x, coeff)).

    :Parameters:
       x : 2d array_like object (N, P)
          data matrix
       y : 1d array_like object integer (N)
          class labels
    
    :Returns:
       coeff: 2d numpy array (P, P)
          transformation matrix.
    """

    n, p = xarr.shape[0], xarr.shape[1]
    labels = np.unique(yarr)
    
    sw = np.zeros((p, p), dtype=np.float)   
    for i in labels:
        idx = np.where(yarr==i)[0]
        sw += np.cov(xarr[idx], rowvar=0) * \
            (idx.shape[0] - 1)
    st = np.cov(xarr, rowvar=0) * (n - 1)

    sb = st - sw
    evals, evecs = spla.eig(sb, sw, overwrite_a=True,
                            overwrite_b=True)
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:, idx]
    evecs = evecs[:, :labels.shape[0]-1]
    
    return evecs


def srda(xarr, yarr, alpha):
    """Spectral Regression Discriminant Analysis.

    Returns the (P, C-1) transformation matrix, where 
    `x` is a matrix (N,P) and C is the number of classes.
    Each column of `x` represents a variable, while the 
    rows contain observations. `x` must be centered 
    (subtracting the empirical mean vector from each column 
    of`x`).

    Sample(s) can be embedded into the C-1 dimensional space
    by z = x coeff (z = np.dot(x, coeff)).

    :Parameters:
       x : 2d array_like object
          training data (N, P)
       y : 1d array_like object integer
          target values (N)
       alpha : float (>=0)
          regularization parameter

    :Returns:
       coeff : 2d numpy array (P, C-1)
          tranformation matrix
    """

    # Point 1 in section 4.2
    yu = np.unique(yarr)
    yk = np.zeros((yu.shape[0]+1, yarr.shape[0]), dtype=np.float)
    yk[0] = 1.
    for i in range(1, yk.shape[0]):
        yk[i][yarr==yu[i-1]] = 1.
    gso(yk, norm=False) # orthogonalize yk
    yk = yk[1:-1]
    
    # Point 2 in section 4.2
    ak = np.empty((yk.shape[0], xarr.shape[1]), dtype=np.float)
    for i in range(yk.shape[0]):
        ak[i] = ridge_base(xarr, yk[i], alpha)

    return ak.T


def pca(x):
    """Principal Component Analysis.
    
    Returns the principal component coefficients `coeff`
    (P, P) and the corresponding eigenvalues (P) of the 
    covariance matrix of `x` (N,P) sorted by decreasing 
    eigenvalue. Each column of `x` represents a variable,  
    while the rows contain observations. Each column of 
    `coeff` contains coefficients for one principal 
    component.
    
    Sample(s) can be embedded into the M (<=P) dimensional space
    by z = x coeff_M (z = np.dot(x, coeff[:, :M])).

    :Parameters:
       x : 2d numpy array (N, P)
          data matrix
    
    :Returns:
       coeff, evals : 2d numpy array (P, P), 1d numpy array (P)
          principal component coefficients (eigenvectors of
          the covariance matrix of x) and eigenvalues sorted by 
          decreasing eigenvalue
    """


    xarr = np.asarray(x, dtype=np.float)

    if xarr.ndim != 2:
        raise ValueError("x must be a 2d array_like object")

    C = np.cov(xarr, rowvar=0)
    evals, evecs = np.linalg.eigh(C)
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:, idx]
    evals = evals[idx]
    
    return evecs, evals


def pca_fast(x, m, eps=0.01):
    """Fast principal component analysis using the fixed-point
    algorithm.
    
    Returns the first `m` principal component coefficients
    `coeff` (P, M). Each column of `x` represents a variable,  
    while the rows contain observations. Each column of `coeff` 
    contains coefficients for one principal component.

    Sample(s) can be embedded into the m (<=P) dimensional space 
    by z = x coeff (z = np.dot(X,  coeff)).

    :Parameters:
       x : 2d numpy array (N, P)
          data matrix
       m : integer (0 < m <= P) 
          the number of principal axes or eigenvectors required
       eps : float (> 0)
          tolerance error
    
    :Returns:
       coeff : 2d numpy array (P, H)
          principal component coefficients
    """
    
    xarr = np.asarray(x, dtype=np.float)
    if xarr.ndim != 2:
        raise ValueError("x must be a 2d array_like object")

    m = int(m)

    np.random.seed(0)
    evecs = np.random.rand(m, xarr.shape[1])

    C = np.cov(xarr, rowvar=0)    
    for i in range(0, m):
        while True:
            evecs_old = np.copy(evecs[i])
            evecs[i] = np.dot(C, evecs[i])
            
            # Gram-Schmidt orthogonalization
            a = np.dot(evecs[i], evecs[:i].T).reshape(-1, 1)
            b = a  * evecs[:i]
            evecs[i] -= np.sum(b, axis=0) # if i=0 sum is 0
            
            # Normalization
            evecs[i] = evecs[i] / np.linalg.norm(evecs[i])
            
            # convergence criteria
            if np.abs(np.dot(evecs[i], evecs_old) - 1) < eps:
                break

    return evecs.T
      

def lda_fast(xarr, yarr):
    """Fast implementation of Linear Discriminant Analysis.
    
    Returns the (P, C-1) transformation matrix, where 
    `x` is a centered matrix (N,P) and C is the number of classes.
    Each column of `x` represents a variable, while the 
    rows contain observations. `x` must be centered 
    (subtracting the empirical mean vector from each column 
    of`x`).

    :Parameters:
       x : 2d array_like object
          training data (N, P)
       y : 1d array_like object integer
          target values (N)
    
    :Returns:
       A : 2d numpy array (P, C-1)
          tranformation matrix
    """

    yu = np.unique(yarr)
    yk = np.zeros((yu.shape[0]+1, yarr.shape[0]), dtype=np.float)
    yk[0] = 1.
    for i in range(1, yk.shape[0]):
        yk[i][yarr==yu[i-1]] = 1.
    gso(yk, norm=False) # orthogonalize yk
    yk = yk[1:-1]
    
    ak = np.empty((yk.shape[0], xarr.shape[1]), dtype=np.float)
    for i in range(yk.shape[0]):
        ak[i], _ = ols_base(xarr, yk[i], -1)

    return ak.T


def kpca(K):
    """Kernel Principal Component Analysis, PCA in 
    a kernel-defined feature space making use of the
    dual representation.
    
    Returns the kernel principal component coefficients 
    `coeff` (N, N) computed as :math:`\lambda^{-1/2} \mathbf{v}_j`
    where :math:`\lambda` and :math:`\mathbf{v}` are the ordered
    eigenvalues and the corresponding eigenvector of the centered 
    kernel matrix K.
    
    Sample(s) can be embedded into the G (<=N) dimensional space
    by z = K coeff_G (z = np.dot(K, coeff[:, :G])).

    :Parameters:
       K: 2d array_like object (N,N)
          precomputed centered kernel matrix
        
    :Returns:
       coeff, evals: 2d numpy array (N,N), 1d numpy array (N)
          kernel principal component coefficients, eigenvalues
          sorted by decreasing eigenvalue.
    """
    
    evals, evecs = np.linalg.eigh(K)
    idx = np.argsort(evals)
    idx = idx[::-1]
    evecs = evecs[:, idx]
    evals = evals[idx]
    
    for i in range(len(evals)):
        evecs[:, i] /= np.sqrt(evals[i])
   
    return evecs, evals


def kfda(Karr, yarr, lmb=0.001):
    """Kernel Fisher Discriminant Analysis.
    
    Returns the transformation matrix `coeff` (N,1),
    where `K` is a the kernel matrix (N,N) and y
    is the class labels (the alghoritm works only with 2
    classes).
   
    :Parameters:
       K: 2d array_like object (N, N)
          precomputed kernel matrix
       y : 1d array_like object integer (N)
          class labels
       lmb : float (>= 0.0)
          regularization parameter

    :Returns:
       coeff: 2d numpy array (N,1)
          kernel fisher coefficients.
    """  

    labels = np.unique(yarr)
    n = yarr.shape[0]

    idx1 = np.where(yarr==labels[0])[0]
    idx2 = np.where(yarr==labels[1])[0]
    n1 = idx1.shape[0]
    n2 = idx2.shape[0]
    
    K1, K2 = Karr[:, idx1], Karr[:, idx2]
    
    N1 = np.dot(np.dot(K1, np.eye(n1) - (1 / float(n1))), K1.T)
    N2 = np.dot(np.dot(K2, np.eye(n2) - (1 / float(n2))), K2.T)
    N = N1 + N2 + np.diag(np.repeat(lmb, n))

    M1 = np.sum(K1, axis=1) / float(n1)
    M2 = np.sum(K2, axis=1) / float(n2)
    M = M1 - M2
    
    coeff = np.linalg.solve(N, M).reshape(-1, 1)
            
    return coeff
 

def whiten(x):
    """Whitening.

    Returns whitening and dewhitening coefficients w (P, P)
    and the corresponding eigenvalues (P) sorted by decreasing 
    eigenvalue.Each column of `x` represents a variable, while
    the rows contain observations. 
    
    Sample(s) can be whitened by z = x w_M (z = np.dot(x, w[:, :M]))
    where Cov(z) = I (where Cov(z) = np.cov(z, rowvar=0)).
    
    :Parameters:
       x : 2d numpy array (N, P)
          data matrix
          
    :Returns:
       w, dw, evals : 2d numpy array (P, P), 2d numpy array (P,P), 1d numpy array (P)
          whitening coeffs, dewhitening coeff and eigenvalues sorted by 
          decreasing eigenvalue
    """

    xarr = np.asarray(x, dtype=np.float)

    if xarr.ndim != 2:
        raise ValueError("x must be a 2d array_like object")

    C = np.cov(xarr, rowvar=0)
    evals, evecs = np.linalg.eigh(C)
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:, idx]
    evals = evals[idx]
    
    w = np.dot(np.diag(evals**-0.5), evecs.T).T
    dw = np.dot(evecs, np.diag(np.sqrt(evals))).T # dewhitening coeffs

    return w, dw, evals



class LDA:
    """Linear Discriminant Analysis.
    """
    
    def __init__(self):
        """Initialization.
        """
        
        self._coeff = None
        self._mean = None

    def learn(self, x, y):
        """Computes the transformation matrix.
        `x` is a matrix (N,P) and `y` is a vector containing
        the class labels. Each column of `x` represents a 
        variable, while the rows contain observations.
        """

        xarr = np.asarray(x, dtype=np.float)
        yarr = np.asarray(y, dtype=np.int)
        
        if xarr.ndim != 2:
            raise ValueError("x must be a 2d array_like object")
        
        if yarr.ndim != 1:
            raise ValueError("y must be an 1d array_like object")
        
        if xarr.shape[0] != yarr.shape[0]:
            raise ValueError("x, y shape mismatch")
        
        self._mean = np.mean(xarr, axis=0)
        self._coeff = lda(xarr, yarr)

    def transform(self, t):
        """Embedded t (M,P) into the C-1 dimensional space.
        Returns a (M,C-1) matrix.
        """
        if self._coeff is None:
            raise ValueError("no model computed")

        tarr = np.asarray(t, dtype=np.float)
        
        try:
            return np.dot(tarr-self._mean, self._coeff)
        except:
            ValueError("t, coeff: shape mismatch")

    def coeff(self):
        """Returns the tranformation matrix (P,C-1), where
        C is the number of classes. Each column contains 
        coefficients for one transformation vector.
        """

        return self._coeff
 

class LDA_FAST:
    """Fast implementation of Linear Discriminant Analysis.
    """
    
    def __init__(self):
        """Initialization.
        """
        
        self._coeff = None
        self._mean = None

    def learn(self, x, y):
        """Computes the transformation matrix.
        `x` is a matrix (N,P) and `y` is a vector containing
        the class labels. Each column of `x` represents a 
        variable, while the rows contain observations.
        """

        xarr = np.asarray(x, dtype=np.float)
        yarr = np.asarray(y, dtype=np.int)
        
        if xarr.ndim != 2:
            raise ValueError("x must be a 2d array_like object")
        
        if yarr.ndim != 1:
            raise ValueError("y must be an 1d array_like object")
        
        if xarr.shape[0] != yarr.shape[0]:
            raise ValueError("x, y shape mismatch")
        
        self._mean = np.mean(xarr, axis=0)
        self._coeff = lda_fast(xarr-self._mean, yarr)

    def transform(self, t):
        """Embedded t (M,P) into the C-1 dimensional space.
        Returns a (M,C-1) matrix.
        """
        if self._coeff is None:
            raise ValueError("no model computed")

        tarr = np.asarray(t, dtype=np.float)
        
        try:
            return np.dot(tarr-self._mean, self._coeff)
        except:
            ValueError("t, coeff: shape mismatch")

    def coeff(self):
        """Returns the tranformation matrix (P,C-1), where
        C is the number of classes. Each column contains 
        coefficients for one transformation vector.
        """

        return self._coeff


class SRDA:
    """Spectral Regression Discriminant Analysis.
    """
    
    def __init__(self, alpha=0.001):
        """Initialization.

        :Parameters:
           alpha : float (>=0)
              regularization parameter
        """
        
        self._coeff = None
        self._mean = None
        self._alpha = alpha

    def learn(self, x, y):
        """Computes the transformation matrix.
        `x` is a matrix (N,P) and `y` is a vector containing
        the class labels. Each column of `x` represents a 
        variable, while the rows contain observations.
        """

        xarr = np.asarray(x, dtype=np.float)
        yarr = np.asarray(y, dtype=np.int)
        
        if xarr.ndim != 2:
            raise ValueError("x must be a 2d array_like object")
        
        if yarr.ndim != 1:
            raise ValueError("y must be an 1d array_like object")
        
        if xarr.shape[0] != yarr.shape[0]:
            raise ValueError("x, y shape mismatch")
        
        self._mean = np.mean(xarr, axis=0)
        self._coeff = srda(xarr-self._mean, yarr, self._alpha)

    def transform(self, t):
        """Embedded t (M,P) into the C-1 dimensional space.
        Returns a (M,C-1) matrix.
        """

        if self._coeff is None:
            raise ValueError("no model computed")

        tarr = np.asarray(t, dtype=np.float)
        
        try:
            return np.dot(tarr-self._mean, self._coeff)
        except:
            ValueError("t, coeff: shape mismatch")

    def coeff(self):
        """Returns the tranformation matrix (P,C-1), where
        C is the number of classes. Each column contains 
        coefficients for one transformation vector.
        """

        return self._coeff


class KFDA:
    """Kernel Fisher Discriminant Analysis.
    """
    
    def __init__(self, lmb=0.001):
        """Initialization.
        
        :Parameters:
           lmb : float (>= 0.0)
              regularization parameter
        """
        
        self._coeff = None
        self._lmb = lmb

    def learn(self, K, y):
        """Computes the transformation vector.
        `K` is a the kernel matrix (N,N) and `y` contains 
        the class labels (the algorithm works only with 2
        classes).
        """

        Karr = np.array(K, dtype=np.float)
        yarr = np.asarray(y, dtype=np.int)
        if yarr.ndim != 1:
            raise ValueError("y must be an 1d array_like object")

        labels = np.unique(yarr)
        if labels.shape[0] != 2:
            raise ValueError("number of classes must be = 2")
        
        self._coeff = kfda(Karr, yarr, self._lmb)

    def transform(self, Kt):
        """Embedded Kt (M,N) into the 1 dimensional space.
        Returns a (M,1) matrix.
        """

        if self._coeff is None:
            raise ValueError("no model computed")

        Ktarr = np.asarray(Kt, dtype=np.float)
        
        try:
            return np.dot(Kt, self._coeff)
        except:
            ValueError("Kt, coeff: shape mismatch")

    def coeff(self):
        """Returns the tranformation vector (N,1).
        """

        return self._coeff



    
            
