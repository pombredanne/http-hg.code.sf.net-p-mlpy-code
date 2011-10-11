## Cross Validation Submodule

## This code is written by Davide Albanese, <albanese@fbk.eu>.
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

__all__ = ['cv_kfold', 'cv_random', 'cv_all']


import itertools
import numpy as np


def cv_kfold(a, k, strat=None, seed=0):
    """Randomly splits the array `a` into train and test partitions
    for k-fold cross-validation.
    
    :Parameters:
       a : 1d array_like
          source array to be parttioned
       k : int (k > 1) 
          number of iterations (folds). The case `k` = n
          (where n is the number of elements in `a`) is known as 
          leave-one-out cross-validation.
       strat : None or  1d array_like integer
          labels for stratification. If `strat` is not None
          returns 'stratified' k-fold CV partitions, where
          each subsample has roughly the same label proportions.
          If strat is not of integer type a casting is
          performed.
       seed : int
          random seed

    :Returns:
       ap: list of tuples
          list of `k` tuples containing the train and 
          test elements of `a`
    
    Example

    >>> import mlpy
    >>> a = range(12)
    >>> ap = mlpy.cv_kfold(a, k=3)
    >>> for tr, ts in ap: tr, ts
    ... 
    (array([2, 8, 1, 7, 9, 3, 0, 5]), array([ 6, 11,  4, 10]))
    (array([ 6, 11,  4, 10,  9,  3,  0,  5]), array([2, 8, 1, 7]))
    (array([ 6, 11,  4, 10,  2,  8,  1,  7]), array([9, 3, 0, 5]))
    
    Stratification example:
    
    >>> strat = [1,1,1,1,2,2,2,2,2,2,2,2]
    >>> ap = mlpy.cv_kfold(a, k=3, strat=strat)
    >>> for tr, ts in ap: tr, ts
    ... 
    (array([ 1,  0,  9,  5, 10, 11,  7]), array([2, 3, 6, 4, 8]))
    (array([ 2,  3,  0,  6,  4,  8, 11,  7]), array([ 1,  9,  5, 10]))
    (array([ 2,  3,  1,  6,  4,  8,  9,  5, 10]), array([ 0, 11,  7]))

    k must be less or equal than the number of samples of a group:
    
    >>> ap = mlpy.cv_kfold(a, k=5, strat=strat) # ValueError: k must be <= 4
    """

    _a = np.asarray(a) 
    
    if k <= 1:
        raise ValueError("k must be > 1")

    if _a.ndim != 1:
        raise ValueError("a must be an 1d array_like object")

    if _a.shape[0] <= 1:
        raise ValueError("number of elements of a must be larger than 1")

    if strat is not None:
        _strat = np.asarray(strat, dtype=np.int)
        if _a.shape[0] != _strat.shape[0]:
            raise ValueError("a, strat: shape mismatch")
    else:
        _strat = np.zeros(_a.shape[0], dtype=np.int)
    
    labels = np.unique(_strat)   

    # check k
    nmin = np.min([_a[lab == _strat].shape[0] for lab in labels])
    if k > nmin:
        raise ValueError('k must be <= %d' % nmin)

    np.random.seed(seed)

    alab = []
    for lab in labels:
        tmp = _a[lab == _strat]
        np.random.shuffle(tmp)
        alab.append(np.array_split(tmp, k))

    ret = []
    for i in range(k):
        tr, ts = [], []
        for j in range(len(alab)):
            tr.extend(alab[j][:i] + alab[j][i+1:])
            ts.extend(alab[j][i])
        ret.append((np.concatenate(tr), np.asarray(ts)))
            
    return ret


def cv_random(a, k, p, strat=None, seed=0):
    """Randomly splits the array `a` into train and test partitions
    for random subsampling cross-validation. The proportion of the
    train/test split is not dependent on the number of iterations
    (`k`).
        
    :Parameters:
       a : 1d array_like
          source array to be partitioned
       k : int (k > 0) 
          number of iterations (folds)
       p : float (0 < `p` < 100) 
          percentage of elements in the test splits
       strat : None or  1d array_like integer
          labels for stratification. If `strat` is not None
          returns 'stratified' random subsampling CV 
          partitions, where each subsample has roughly the 
          same label proportions. If strat is not of integer
          type a casting is performed.
       seed : int
          random seed
          
    :Returns:
       ap : list of tuples
          list of `k` tuples containing the train and 
          test elements of `a`

    Example
    
    >>> import mlpy
    >>> a = range(12)
    >>> ap = mlpy.cv_random(a, k=4, p=30)
    >>> for tr, ts in ap: tr, ts
    ... 
    (array([ 6, 11,  4, 10,  2,  8,  1,  7,  9]), array([3, 0, 5]))
    (array([ 5,  2,  3,  4,  9,  0, 11,  7,  6]), array([ 1, 10,  8]))
    (array([ 6,  1, 10,  2,  7,  5, 11,  0,  3]), array([4, 9, 8]))
    (array([2, 4, 8, 9, 5, 6, 1, 0, 7]), array([10, 11,  3]))
    """

    _a = np.asarray(a)   
    
    if k < 1:
        raise ValueError("k must be > 0")
    
    if (p <= 0.0) or (p >= 100.0):
        raise ValueError("p must be > 0 and < 100)")

    if _a.ndim != 1:
        raise ValueError("a must be an 1d array_like object")

    if _a.shape[0] <= 1:
        raise ValueError("shape of a must be > 1")

    if strat is not None:
        _strat = np.asarray(strat, dtype=np.int)
        if _a.shape[0] != _strat.shape[0]:
            raise ValueError("a, strat: shape mismatch")
    else:
        _strat = np.zeros(_a.shape[0], dtype=np.int)       

    labels = np.unique(_strat)

    # check p
    nmin = np.min([_a[lab == _strat].shape[0] for lab in labels])
    if int(0.01*p*nmin) == 0:
        raise ValueError('p must be >= %.2f%%' % (nmin**-1 * 100))
            
    np.random.seed(seed)

    ret = []
    for _ in range(k):        
        tr, ts = [], []
        for lab in labels:
            tmp = _a[lab == _strat]
            n = tmp.shape[0] - int(0.01*p*tmp.shape[0])
            np.random.shuffle(tmp)
            tr.append(tmp[:n])
            ts.append(tmp[n:])
        
        ret.append((np.concatenate(tr), np.concatenate(ts)))

    return ret


def cv_all(a, p):
    """Randomly splits the array `a` into train and test partitions
    for all-combinations cross-validation.

    :Parameters:
       a : 1d array_like
          source array to be partitioned
       p : float (0 < `p` < 100) 
          percentage of elements in the test splits

    :Returns:
       ap : list of tuples
          list of `k` tuples containing the train and 
          test elements of `a`

    Example

    >>> import mlpy
    >>> a = range(4)
    >>> ap = mlpy.cv_all(a, 50)
    >>> for tr, ts in ap: tr, ts
    ... 
    (array([2, 3]), array([0, 1]))
    (array([1, 3]), array([0, 2]))
    (array([1, 2]), array([0, 3]))
    (array([0, 3]), array([1, 2]))
    (array([0, 2]), array([1, 3]))
    (array([0, 1]), array([2, 3]))
    >>> part = mlpy.cv_all(a, 10) # ValueError: p must be >= 25.00%
    """

    _a = np.asarray(a)   
        
    if (p <= 0.0) or (p >= 100.0):
        raise ValueError("p must be > 0 and < 100)")

    if _a.ndim != 1:
        raise ValueError("a must be an 1d array_like object")

    if _a.shape[0] <= 1:
        raise ValueError("shape of a must be > 1")
    
    # check p
    n = int(0.01*p*_a.shape[0])
    if n == 0:
        raise ValueError('p must be >= %.2f%%' % (_a.shape[0]**-1 * 100))
    
    tmp = np.asarray(list(itertools.combinations(_a, n)))

    ret = []
    for ts in tmp:
        tr = np.setdiff1d(_a, ts)
        ret.append((tr, ts))

    return ret
