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


__all__ = ['lda', 'fastlda', 'pca', 'kpca', 'fastpca', 'srda',
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


def lda(x, y):
    """Linear Discriminant Analysis.
    
    Returns the transformation matrix `coeff` (P, P)
    and the corresponding eigenvalues (P) sorted by 
    decreasing eigenvalue. Each column of `x` represents 
    a variable, while the rows contain observations. 
    `x` (N,P) must be centered (subtracting the empirical mean 
    vector from each column of`x`). Each column of `coeff` 
    contains coefficients for one transformation vector.
    
    Sample(s) can be embedded into the M (<=P) dimensional space
    by z = x coeff_M (z = np.dot(x, coeff[:, :M])).

    :Parameters:
       x : 2d array_like object (N, P)
          data matrix
       y : 1d array_like object integer (N)
          class labels
    
    :Returns:
       coeff, evals : 2d numpy array (P, P), 1d numpy array (P)
          transformation matrix and the corresponding eigenvalues
          sorted by decreasing eigenvalue.
    """

    xarr = np.asarray(x, dtype=np.float)
    yarr = np.asarray(y, dtype=np.int)

    if xarr.ndim != 2:
        raise ValueError("x must be a 2d array_like object")
    
    if yarr.ndim != 1:
        raise ValueError("y must be an 1d array_like object")
    
    if xarr.shape[0] != yarr.shape[0]:
        raise ValueError("x, y shape mismatch")

    n, p = xarr.shape[0], xarr.shape[1]
    
    sw = np.zeros((p, p), dtype=np.float)  
    labels = np.unique(yarr)
    for i in labels:
        idx = np.where(y==i)[0]
        xi = xarr[idx]
        covi = np.cov(xi, rowvar=0)
        sw += idx.shape[0] / float(n) * covi
    
    st = np.cov(xarr, rowvar=0)
    sb = st - sw
    evals, evecs = spla.eig(sw, sb)
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:, idx]
    evals = evals[idx]
  
    return evecs, evals


def srda(x, y, alpha):
    """Spectral Regression Discriminant Analysis.

    Returns the (P, C-1) transformation matrix, where 
    `x` is a matrix (N,P) and C is the number of classes.
    Each column of `x` represents a variable, while the 
    rows contain observations. `x` must be centered 
    (subtracting the empirical mean vector from each column 
    of`x`).

    Sample(s) can be embedded into the C-1 dimensional space
    by z = z coeff (z = np.dot(x, coeff)).

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

    xarr = np.asarray(x, dtype=np.float)
    yarr = np.asarray(y, dtype=np.int)
    
    if xarr.ndim != 2:
        raise ValueError("x must be a 2d array_like object")
    
    if yarr.ndim != 1:
        raise ValueError("y must be an 1d array_like object")
    
    if xarr.shape[0] != yarr.shape[0]:
        raise ValueError("x, y shape mismatch")

    # Point 1 in section 4.2
    yu = np.unique(yarr)
    yk = np.zeros((yu.shape[0]+1, yarr.shape[0]), dtype=np.float)
    yk[0] = 1.
    for i in range(1, yk.shape[0]):
        yk[i][y==yu[i-1]] = 1.
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
    while the rows contain observations. `x` must be centered 
    (subtracting the empirical mean vector from each column 
    of`x`). Each column of `coeff` contains coefficients 
    for one principal component.
    
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


def fastpca(x, m, eps=0.01):
    """Fast principal component analysis using the fixed-point
    algorithm.
    
    Returns the first `m` principal component coefficients
    `coeff` (P, M). Each column of `x` represents a variable,  
    while the rows contain observations. `x` must be centered 
    (subtracting the empirical mean vector from each column 
    of`x`). Each column of `coeff` contains coefficients 
    for one principal component.

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
      

def fastlda(x, y):
    """Fast implementation of Linear Discriminant Analysis.
    
    Returns the (P, C-1) transformation matrix, where 
    `x` is a matrix (N,P) and C is the number of classes.
    Each column of `x` represents a variable, while the 
    rows contain observations. `x` must be centered 
    (subtracting the empirical mean vector from each column 
    of`x`).

    Sample(s) can be embedded into the C-1 dimensional space
    by z = z coeff (z = np.dot(x, coeff)).

    :Parameters:
       x : 2d array_like object
          training data (N, P)
       y : 1d array_like object integer
          target values (N)
    
    :Returns:
       A : 2d numpy array (P, C-1)
          tranformation matrix
    """

    xarr = np.asarray(x, dtype=np.float)
    yarr = np.asarray(y, dtype=np.int)
    
    if xarr.ndim != 2:
        raise ValueError("x must be a 2d array_like object")
    
    if yarr.ndim != 1:
        raise ValueError("y must be an 1d array_like object")
    
    if xarr.shape[0] != yarr.shape[0]:
        raise ValueError("x, y shape mismatch")
        
    yu = np.unique(yarr)
    yk = np.zeros((yu.shape[0]+1, yarr.shape[0]), dtype=np.float)
    yk[0] = 1.
    for i in range(1, yk.shape[0]):
        yk[i][y==yu[i-1]] = 1.
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
    evecs = evecs[:,idx]
    evals = evals[idx]
    
    for i in range(len(evals)):
        evecs[:, i] /= np.sqrt(evals[i])
   
    return evecs, evals


def whiten(x):
    """Whitening.

    Returns whitening and de-withening coefficients.

    z = x coeff_M (z = np.dot(x, coeff[:, :M])).

    Cov(z) = I (np.cov(z, rowvar=0))

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
    dw = np.dot(evecs, np.diag(np.sqrt(evals))).T

    return w, dw
