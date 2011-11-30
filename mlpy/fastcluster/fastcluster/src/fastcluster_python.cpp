/*
  fastcluster: Fast hierarchical clustering routines for R and Python

  Copyright © 2011 Daniel Müllner
  <http://math.stanford.edu/~muellner>
*/

// for INT32_MAX in fastcluster.cpp
// This must be defined here since Python.h loads the header file pyport.h,
//  and from this stdint.h. INT32_MAX is defined in stdint.h, but only if
// __STDC_LIMIT_MACROS is defined.
#define __STDC_LIMIT_MACROS

#include <Python.h>
#include <numpy/arrayobject.h>


#include <limits> // for infinity()

#include <float.h>
#ifndef DBL_MANT_DIG
#error The constant DBL_MANT_DIG could not be defined.
#endif

//#include <cmath>
#include <algorithm>

#ifndef LONG_MAX
#include <limits.h>
#endif
#ifndef LONG_MAX
#error The constant LONG_MAX could not be defined.
#endif
#ifndef INT_MAX
#error The constant INT_MAX could not be defined.
#endif

#ifndef INT32_MAX
#define __STDC_LIMIT_MACROS
#include <stdint.h>
#endif

typedef int_fast32_t t_index;
#ifndef INT32_MAX
#define MAX_INDEX 0x7fffffffL
#else
#define MAX_INDEX INT32_MAX
#endif
#if (LONG_MAX < MAX_INDEX)
#error The integer format "t_index" must not have a greater range than "long int".
#endif
#if (INT_MAX > MAX_INDEX)
#error The integer format "int" must not have a greater range than "t_index".
#endif
typedef double t_float;
#define T_FLOAT_MANT_DIG DBL_MANT_DIG

enum method_codes {
  // non-Euclidean methods
  METHOD_METR_SINGLE           = 0,
  METHOD_METR_COMPLETE         = 1,
  METHOD_METR_AVERAGE          = 2,
  METHOD_METR_WEIGHTED         = 3,
  METHOD_METR_WARD             = 4,
  METHOD_METR_CENTROID         = 5,
  METHOD_METR_MEDIAN           = 6
};

enum {
  // Euclidean methods
  METHOD_VECTOR_SINGLE         = 0,
  METHOD_VECTOR_WARD           = 1,
  METHOD_VECTOR_CENTROID       = 2,
  METHOD_VECTOR_MEDIAN         = 3
};

enum {
   // Return values
  RET_SUCCESS        = 0,
  RET_MEMORY_ERROR   = 1,
  RET_STL_ERROR      = 2,
  RET_UNKNOWN_ERROR  = 3
 };

// self-destructing array pointer
template <typename type>
class auto_array_ptr{
private:
  type * ptr;
public:
  auto_array_ptr() { ptr = NULL; }
  template <typename index>
  auto_array_ptr(index const size) { init(size); }
  template <typename index, typename value>
  auto_array_ptr(index const size, value const val) { init(size, val); }
  ~auto_array_ptr() {
    delete [] ptr; }
  void free() {
    delete [] ptr;
    ptr = NULL;
  }
  template <typename index>
  void init(index const size) {
    ptr = new type [size];
  }
  template <typename index, typename value>
  void init(index const size, value const val) {
    init(size);
    for (index i=0; i<size; i++) ptr[i] = val;
  }
  operator type *() const { return ptr; }
};

struct node {
  t_index node1, node2;
  t_float dist;

  /*
  inline bool operator< (const node a) const {
    return this->dist < a.dist;
  }
  */

  inline friend bool operator< (const node a, const node b) {
    // Numbers are always smaller than NaNs.
    return a.dist < b.dist || (a.dist==a.dist && b.dist!=b.dist);
  }
};

class cluster_result {
private:
  auto_array_ptr<node> Z;
  t_index pos;

public:
  cluster_result(const t_index size)
    : Z(size)
  {
    pos = 0;
  }

  void append(const t_index node1, const t_index node2, const t_float dist) {
    Z[pos].node1 = node1;
    Z[pos].node2 = node2;
    Z[pos].dist  = dist;
    pos++;
  }

  node * operator[] (const t_index idx) const { return Z + idx; }

  void sqrt(const t_float) const {
    sqrt();
  }

  void sqrt() const {
    for (t_index i=0; i<pos; i++) {
      Z[i].dist = ::sqrt(Z[i].dist);
    }
  }

  void sqrtdouble(const t_float) const {
    for (t_index i=0; i<pos; i++) {
      Z[i].dist = ::sqrt(2*Z[i].dist);
    }
  }

  #ifdef R_pow
  #define my_pow R_pow
  #else
  #define my_pow pow
  #endif

  void power(const t_float p) const {
    t_float const q = 1/p;
    for (t_index i=0; i<pos; i++) {
      Z[i].dist = my_pow(Z[i].dist,q);
    }
  }

  void plusone(const t_float) const {
    for (t_index i=0; i<pos; i++) {
      Z[i].dist += 1;
    }
  }

  void divide(const t_float denom) const {
    for (t_index i=0; i<pos; i++) {
      Z[i].dist /= denom;
    }
  }
};

class doubly_linked_list {
  /*
    Class for a doubly linked list. Initially, the list is the integer range
    [0, size]. We provide a forward iterator and a method to delete an index
    from the list.

    Typical use: for (i=L.start; L<size; i=L.succ[I])
    or
    for (i=somevalue; L<size; i=L.succ[I])
  */
public:
  t_index start;
  auto_array_ptr<t_index> succ;

private:
  auto_array_ptr<t_index> pred;
  // Not necessarily private, we just do not need it in this instance.

public:
  doubly_linked_list(const t_index size)
    // Initialize to the given size.
    : succ(size+1), pred(size+1)
  {
    for (t_index i=0; i<size; i++) {
      pred[i+1] = i;
      succ[i] = i+1;
    }
    // pred[0] is never accessed!
    //succ[size] is never accessed!
    start = 0;
  }

  void remove(const t_index idx) {
    // Remove an index from the list.
    if (idx==start) {
      start = succ[idx];
    }
    else {
      succ[pred[idx]] = succ[idx];
      pred[succ[idx]] = pred[idx];
    }
    succ[idx] = 0; // Mark as inactive
  }

  bool is_inactive(t_index idx) const {
    return (succ[idx]==0);
  }
};

// Indexing functions
// D is the upper triangular part of a symmetric (NxN)-matrix
// We require r_ < c_ !
#define D_(r_,c_) ( D[(static_cast<ptrdiff_t>(2*N-3-(r_))*(r_)>>1)+(c_)-1] )
// Z is an ((N-1)x4)-array
#define Z_(_r, _c) (Z[(_r)*4 + (_c)])

/*
  Lookup function for a union-find data structure.

  The function finds the root of idx by going iteratively through all
  parent elements until a root is found. An element i is a root if
  nodes[i] is zero. To make subsequent searches faster, the entry for
  idx and all its parents is updated with the root element.
 */
class union_find {
private:
  auto_array_ptr<t_index> parent;
  t_index nextparent;

public:
  void init(const t_index size) {
    parent.init(2*size-1, 0);
    nextparent = size;
  }

  t_index Find (t_index idx) const {
    if (parent[idx] !=0 ) { // a → b
      t_index p = idx;
      idx = parent[idx];
      if (parent[idx] !=0 ) { // a → b → c
        do {
          idx = parent[idx];
        } while (parent[idx] != 0);
        do {
          t_index tmp = parent[p];
          parent[p] = idx;
          p = tmp;
        } while (parent[p] != idx);
      }
    }
    return idx;
  }

  void Union (const t_index node1, const t_index node2) {
    parent[node1] = parent[node2] = nextparent++;
  }
};

static void MST_linkage_core(const t_index N, const t_float * const D,
                             cluster_result & Z2) {
/*
    N: integer, number of data points
    D: condensed distance matrix N*(N-1)/2
    Z2: output data structure

    The basis of this algorithm is an algorithm by Rohlf:

    F. James Rohlf, Hierarchical clustering using the minimum spanning tree,
    The Computer Journal, vol. 16, 1973, p. 93–95.

    This implementation should handle Inf values correctly (designed to
    do so but not tested).

    This implementation avoids NaN if possible. It treats NaN as if it was
    greater than +Infinity, ie. whenever we find a non-NaN value, this is
    preferred in all the minimum-distance searches.
*/
  t_index i;
  t_index idx2;
  doubly_linked_list AR(N);
  auto_array_ptr<t_float> d(N);

  t_index prev_node;
  t_float min;

  // first iteration
  idx2 = 1;
  min = d[1] = D[0];
  for (i=2; min!=min && i<N; i++) {  // eliminate NaNs if possible
    min = d[i] = D[i-1];
    idx2 = i;
  }
  for ( ; i<N; i++) {
    d[i] = D[i-1];
    if (d[i] < min) {
      min = d[i];
      idx2 = i;
    }
  }
  Z2.append(0, idx2, min);

  for (t_index j=1; j<N-1; j++) {
    prev_node = idx2;
    AR.remove(prev_node);

    idx2 = AR.succ[0];
    min = d[idx2];
    for (i=idx2; min!=min && i<prev_node; i=AR.succ[i]) {
      min = d[i] = D_(i, prev_node);
      idx2 = i;
    }
    for ( ; i<prev_node; i=AR.succ[i]) {
      if (d[i] > D_(i, prev_node))
        d[i] = D_(i, prev_node);
      if (d[i] < min) {
        min = d[i];
        idx2 = i;
      }
    }
    for (; min!=min && i<N; i=AR.succ[i]) {
      min = d[i] = D_(prev_node, i);
      idx2 = i;
    }
    for (; i<N; i=AR.succ[i]) {
      if (d[i] > D_(prev_node, i))
        d[i] = D_(prev_node, i);
      if (d[i] < min) {
        min = d[i];
        idx2 = i;
      }
    }
    Z2.append(prev_node, idx2, min);
  }
}


inline static void f_single( t_float * const b, const t_float a ) {
  if (*b > a) *b = a;
}
inline static void f_complete( t_float * const b, const t_float a ) {
  if (*b < a) *b = a;
}
inline static void f_average( t_float * const b, const t_float a, const t_float s, const t_float t) {
  *b = s*a + t*(*b);
}
inline static void f_weighted( t_float * const b, const t_float a) {
  *b = (a+*b)/2;
}
inline static void f_ward( t_float * const b, const t_float a, const t_float c, const t_float s, const t_float t, const t_float v) {
  *b = ( (v+s)*a - v*c + (v+t)*(*b) ) / (s+t+v);
  //*b = a+(*b)-(t*a+s*(*b)+v*c)/(s+t+v);
}
inline static void f_centroid( t_float * const b, const t_float a, const t_float stc, const t_float s, const t_float t) {
  *b = s*a + t*(*b) - stc;
}
inline static void f_median( t_float * const b, const t_float a, const t_float c_4) {
  *b = (a+(*b))/2 - c_4;
}


template <const unsigned char method, typename t_members>
static void NN_chain_core(const t_index N, t_float * const D, t_members * const members, cluster_result & Z2) {
/*
    N: integer
    D: condensed distance matrix N*(N-1)/2
    Z2: output data structure

    This is the NN-chain algorithm, described on page 86 in the following book:

﻿   Fionn Murtagh, Multidimensional Clustering Algorithms,
    Vienna, Würzburg: Physica-Verlag, 1985.

    This implementation does not give defined results when NaN or Inf values
    are present in the array D.
*/
  t_index i;

  auto_array_ptr<t_index> NN_chain(N);
  t_index NN_chain_tip = 0;

  t_index idx1, idx2;

  t_float size1, size2;
  doubly_linked_list AR(N);

  t_float min;

  for (t_index j=0; j<N-1; j++) {
    if (NN_chain_tip <= 3) {
      NN_chain[0] = idx1 = AR.start;
      NN_chain_tip = 1;

      idx2 = AR.succ[idx1];
      min = D_(idx1,idx2);
      for (i=AR.succ[idx2]; i<N; i=AR.succ[i]) {
        if (D_(idx1,i) < min) {
          min = D_(idx1,i);
          idx2 = i;
        }
      }
    }  // a: idx1   b: idx2
    else {
      NN_chain_tip -= 3;
      idx1 = NN_chain[NN_chain_tip-1];
      idx2 = NN_chain[NN_chain_tip];
      min = idx1<idx2 ? D_(idx1,idx2) : D_(idx2,idx1);
    }  // a: idx1   b: idx2

    do {
      NN_chain[NN_chain_tip] = idx2;

      for (i=AR.start; i<idx2; i=AR.succ[i]) {
        if (D_(i,idx2) < min) {
          min = D_(i,idx2);
          idx1 = i;
        }
      }
      for (i=AR.succ[idx2]; i<N; i=AR.succ[i]) {
        if (D_(idx2,i) < min) {
          min = D_(idx2,i);
          idx1 = i;
        }
      }

      idx2 = idx1;
      idx1 = NN_chain[NN_chain_tip++];

    } while (idx2 != NN_chain[NN_chain_tip-2]);

    Z2.append(idx1, idx2, min);

    if (idx1>idx2) {
      t_index tmp = idx1;
      idx1 = idx2;
      idx2 = tmp;
    }

    if (method==METHOD_METR_AVERAGE ||
        method==METHOD_METR_WARD) {
      size1 = static_cast<t_float>(members[idx1]);
      size2 = static_cast<t_float>(members[idx2]);
      members[idx2] += members[idx1];
    }

    // Remove the smaller index from the valid indices (AR).
    AR.remove(idx1);

    switch (method) {
    case METHOD_METR_SINGLE:
      /*
      Single linkage.

      Characteristic: new distances are never longer than the old distances.
      */
      // Update the distance matrix in the range [start, idx1).
      for (i=AR.start; i<idx1; i=AR.succ[i])
        f_single(&D_(i, idx2), D_(i, idx1) );
      // Update the distance matrix in the range (idx1, idx2).
      for (; i<idx2; i=AR.succ[i])
        f_single(&D_(i, idx2), D_(idx1, i) );
      // Update the distance matrix in the range (idx2, N).
      for (i=AR.succ[idx2]; i<N; i=AR.succ[i])
        f_single(&D_(idx2, i), D_(idx1, i) );
      break;

    case METHOD_METR_COMPLETE:
      /*
      Complete linkage.

      Characteristic: new distances are never shorter than the old distances.
      */
      // Update the distance matrix in the range [start, idx1).
      for (i=AR.start; i<idx1; i=AR.succ[i])
        f_complete(&D_(i, idx2), D_(i, idx1) );
      // Update the distance matrix in the range (idx1, idx2).
      for (; i<idx2; i=AR.succ[i])
        f_complete(&D_(i, idx2), D_(idx1, i) );
      // Update the distance matrix in the range (idx2, N).
      for (i=AR.succ[idx2]; i<N; i=AR.succ[i])
        f_complete(&D_(idx2, i), D_(idx1, i) );
      break;

    case METHOD_METR_AVERAGE: {
      /*
      Average linkage.

      Shorter and longer distances can occur.
      */
      // Update the distance matrix in the range [start, idx1).
      t_float s = size1/(size1+size2);
      t_float t = size2/(size1+size2);
      for (i=AR.start; i<idx1; i=AR.succ[i])
        f_average(&D_(i, idx2), D_(i, idx1), s, t );
      // Update the distance matrix in the range (idx1, idx2).
      for (; i<idx2; i=AR.succ[i])
        f_average(&D_(i, idx2), D_(idx1, i), s, t );
      // Update the distance matrix in the range (idx2, N).
      for (i=AR.succ[idx2]; i<N; i=AR.succ[i])
        f_average(&D_(idx2, i), D_(idx1, i), s, t );
      break;
    }

    case METHOD_METR_WEIGHTED:
      /*
      Weighted linkage.

      Shorter and longer distances can occur.
      */
      // Update the distance matrix in the range [start, idx1).
      for (i=AR.start; i<idx1; i=AR.succ[i])
        f_weighted(&D_(i, idx2), D_(i, idx1) );
      // Update the distance matrix in the range (idx1, idx2).
      for (; i<idx2; i=AR.succ[i])
        f_weighted(&D_(i, idx2), D_(idx1, i) );
      // Update the distance matrix in the range (idx2, N).
      for (i=AR.succ[idx2]; i<N; i=AR.succ[i])
        f_weighted(&D_(idx2, i), D_(idx1, i) );
      break;

    case METHOD_METR_WARD:
      /*
      Ward linkage.

      Shorter and longer distances can occur, not smaller than min(d1,d2)
      but maybe bigger than max(d1,d2).
      */
      // Update the distance matrix in the range [start, idx1).
      //t_float v = static_cast<t_float>(members[i]);
      for (i=AR.start; i<idx1; i=AR.succ[i])
        f_ward(&D_(i, idx2), D_(i, idx1), min,
               size1, size2, static_cast<t_float>(members[i]) );
      // Update the distance matrix in the range (idx1, idx2).
      for (; i<idx2; i=AR.succ[i])
        f_ward(&D_(i, idx2), D_(idx1, i), min,
               size1, size2, static_cast<t_float>(members[i]) );
      // Update the distance matrix in the range (idx2, N).
      for (i=AR.succ[idx2]; i<N; i=AR.succ[i])
        f_ward(&D_(idx2, i), D_(idx1, i), min,
               size1, size2, static_cast<t_float>(members[i]) );
      break;
    }
  }
}

class binary_min_heap {
  /*
  Class for a binary min-heap. The data resides in an array A. The elements of A
  are not changed but two lists I and R of indices are generated which point to
  elements of A and backwards.

  The heap tree structure is

     H[2*i+1]     H[2*i+2]
         \            /
          \          /
           ≤        ≤
            \      /
             \    /
              H[i]

  where the children must be less or equal than their parent. Thus, H[0] contains
  the minimum. The lists I and R are made such that H[i] = A[I[i]] and R[I[i]] = i.

  This implementation avoids NaN if possible. It treats NaN as if it was
  greater than +Infinity, ie. whenever we find a non-NaN value, this is
  preferred in all comparisons.
  */
private:
  t_float * A;
  t_index size;
  auto_array_ptr<t_index> I;
  auto_array_ptr<t_index> R;

public:
  binary_min_heap(const t_index size)
    : I(size), R(size)
  { // Allocate memory and initialize the lists I and R to the identity. This does
    // not make it a heap. Call heapify afterwards!
    this->size = size;
    for (t_index i=0; i<size; i++)
      R[i] = I[i] = i;
  }

  binary_min_heap(const t_index size1, const t_index size2, const t_index start)
    : I(size1), R(size2)
  { // Allocate memory and initialize the lists I and R to the identity. This does
    // not make it a heap. Call heapify afterwards!
    this->size = size1;
    for (t_index i=0; i<size; i++) {
      R[i+start] = i;
      I[i] = i + start;
    }
  }

  void heapify(t_float * const A) {
    // Arrange the indices I and R so that H[i] := A[I[i]] satisfies the heap
    // condition H[i] < H[2*i+1] and H[i] < H[2*i+2] for each i.
    //
    // Complexity: Θ(size)
    // Reference: Cormen, Leiserson, Rivest, Stein, Introduction to Algorithms,
    // 3rd ed., 2009, Section 6.3 “Building a heap”
    t_index idx;
    this->A = A;
    for (idx=(size>>1); idx>0; ) {
      idx--;
      update_geq_(idx);
    }
  }

  inline operator t_index() const {
    // Return the minimal element.
    return I[0];
  }

  void heap_pop() {
    // Remove the minimal element from the heap.
    size--;
    I[0] = I[size];
    R[I[0]] = 0;
    update_geq_(0);
  }

  void remove(t_index idx) {
    // Remove an element from the heap.
    size--;
    R[I[size]] = R[idx];
    I[R[idx]] = I[size];
    if ( H(size)<=A[idx] || A[idx]!=A[idx] ) {
      update_leq_(R[idx]);
    }
    else {
      update_geq_(R[idx]);
    }
  }

  void replace ( const t_index idxold, const t_index idxnew, const t_float val) {
    R[idxnew] = R[idxold];
    I[R[idxnew]] = idxnew;
    if (val<=A[idxold] || A[idxold]!=A[idxold]) // avoid NaN! ????????????????????
      update_leq(idxnew, val);
    else
      update_geq(idxnew, val);
  }

  void update ( const t_index idx, const t_float val ) const {
    // Update the element A[i] with val and re-arrange the indices the preserve the
    // heap condition.
    if (val<=A[idx] || A[idx]!=A[idx]) // avoid NaN! ????????????????????
      update_leq(idx, val);
    else
      update_geq(idx, val);
  }

  void update_leq ( const t_index idx, const t_float val ) const {
    // Use this when the new value is not more than the old value.
    A[idx] = val;
    update_leq_(R[idx]);
  }

  void update_geq ( const t_index idx, const t_float val ) const {
    // Use this when the new value is not less than the old value.
    A[idx] = val;
    update_geq_(R[idx]);
  }

private:
  void update_leq_ (t_index i) const {
    t_index j;
    for ( ; (i>0) && ( H(i)<H(j=(i-1)>>1) || H(j)!=H(j) ); i=j)
      // avoid NaN!
      heap_swap(i,j);
  }

  void update_geq_ (t_index i) const {
    t_index j;
    for ( ; (j=2*i+1)<size; i=j) {
      if ( H(j)>=H(i) || H(j)!=H(j) ) {  // avoid Nan!
        j++;
        if ( j>=size || H(j)>=H(i) || H(j)!=H(j) ) break; // avoid NaN!
      }
      else if ( j+1<size && H(j+1)<H(j) ) j++;
      heap_swap(i, j);
    }
  }

  void heap_swap(const t_index i, const t_index j) const {
    // Swap two indices.
    t_index tmp = I[i];
    I[i] = I[j];
    I[j] = tmp;
    R[I[i]] = i;
    R[I[j]] = j;
  }

  inline t_float H(const t_index i) const {
    return A[I[i]];
  }

};

template <const unsigned char method, typename t_members>
static void generic_linkage(const t_index N, t_float * const D, t_members * const members, cluster_result & Z2) {
  /*
    N: integer, number of data points
    D: condensed distance matrix N*(N-1)/2
    Z2: output data structure

    This implementation does not give defined results when NaN or Inf values
    are present in the array D.
  */

  const t_index N_1 = N-1;
  t_index i, j; // loop variables
  t_index idx1, idx2; // row and column indices

  auto_array_ptr<t_index> n_nghbr(N_1); // array of nearest neighbors
  auto_array_ptr<t_float> mindist(N_1); // distances to the nearest neighbors
  auto_array_ptr<t_index> row_repr(N);// row_repr[i]: node number that the i-th row represents
  doubly_linked_list AR(N); // active rows
  binary_min_heap minidx(N_1); // minimum heap structure for the nearest neighbors

  t_index node1, node2;     // node numbers in the output
  t_float size1, size2;     // and their cardinalities

  t_float min; // minimum and row index for nearest-neighbor search
  t_index idx;

  for (i=0; i<N; i++)
    // Build a list of row ↔ node label assignments.
    // Initially i ↦ i
    row_repr[i] = i;

  // Initialize the minimal distances:
  // Find the nearest neighbor of each point.
  // n_nghbr[i] = argmin_{j>i} D(i,j) for i in range(N-1)
  t_float * DD = D;
  for (i=0; i<N_1; i++) {
    min = *(DD++);
    idx = j = i+1;
    while (j<N_1) {
      j++;
      if (*DD<min) {
        min = *DD;
        idx = j;
      }
      DD++;
    }
    mindist[i] = min;
    n_nghbr[i] = idx;
  }
  // Put the minimal distances into a heap structure to make the repeated global
  // minimum searches fast.
  minidx.heapify(mindist);

  // Main loop: We have N-1 merging steps.
  for (i=0; i<N_1; i++) {
    /*
      Here is a special feature that allows fast bookkeeping and updates of the
      minimal distances.

      mindist[i] stores a lower bound on the minimum distance of the point i to
      all points of higher index:

          mindist[i] ≥ min_{j>i} D(i,j)

      Normally, we have equality. However, this minimum may become invalid due to
      the updates in the distance matrix. The rules are:

      1) If mindist[i] is equal to D(i, n_nghbr[i]), this is the correct minimum
         and n_nghbr[i] is a nearest neighbor.

      2) If mindist[i] is smaller than D(i, n_nghbr[i]), this might not be the
         correct minimum. The minimum needs to be recomputed.

      3) mindist[i] is never bigger than the true minimum. Hence, we never miss the
         true minimum if we take the smallest mindist entry, re-compute the value if
         necessary (thus maybe increasing it) and looking for the now smallest
         mindist entry until a valid minimal entry is found. This step is done in the
         lines below.

      The update process for D below takes care that these rules are fulfilled. This
      makes sure that the minima in the rows D(i,i+1:)of D are re-calculated when
      necessary but re-calculation is avoided whenever possible.

      The re-calculation of the minima makes the worst-case runtime of this algorithm
      cubic in N. We avoid this whenever possible, and in most cases the runtime
      appears to be quadratic.
    */
    idx1 = minidx;
    if (method != METHOD_METR_SINGLE) {
      while ( D_(idx1, n_nghbr[idx1]) > mindist[idx1] ) {
        // Recompute the minimum mindist[idx1] and n_nghbr[idx1].
        n_nghbr[idx1] = j = AR.succ[idx1]; // exists, maximally N-1
        min = D_(idx1,j);
        for (j=AR.succ[j]; j<N; j=AR.succ[j]) {
          if (D_(idx1,j)<min) {
            min = D_(idx1,j);
            n_nghbr[idx1] = j;
          }
        }
        minidx.update_geq(idx1, min); // Update the heap with the new true minimum.
        idx1 = minidx; // And search for the (possibly different) minimal entry.
      }
    }

    minidx.heap_pop(); // Remove the current minimum from the heap.
    idx2 = n_nghbr[idx1];

    // Write the newly found minimal pair of nodes to the output array.
    node1 = row_repr[idx1];
    node2 = row_repr[idx2];

    if (method==METHOD_METR_AVERAGE ||
        method==METHOD_METR_WARD ||
        method==METHOD_METR_CENTROID) {
      size1 = static_cast<t_float>(members[idx1]);
      size2 = static_cast<t_float>(members[idx2]);
      members[idx2] += members[idx1];
    }
    Z2.append(node1, node2, mindist[idx1]);

    // Remove idx1 from the list of active indices (AR).
    AR.remove(idx1);
    // Index idx2 now represents the new (merged) node with label N+i.
    row_repr[idx2] = N+i;

    // Update the distance matrix
    switch (method) {
    case METHOD_METR_SINGLE:
      /*
        Single linkage.

        Characteristic: new distances are never longer than the old distances.
      */
      // Update the distance matrix in the range [start, idx1).
      for (j=AR.start; j<idx1; j=AR.succ[j]) {
        f_single(&D_(j, idx2), D_(j, idx1));
        if (n_nghbr[j] == idx1)
          n_nghbr[j] = idx2;
      }
      // Update the distance matrix in the range (idx1, idx2).
      for (; j<idx2; j=AR.succ[j]) {
        f_single(&D_(j, idx2), D_(idx1, j));
        // If the new value is below the old minimum in a row, update
        // the mindist and n_nghbr arrays.
        if (D_(j, idx2)<mindist[j]) {
          minidx.update_leq(j, D_(j, idx2));
          n_nghbr[j] = idx2;
        }
      }
      // Update the distance matrix in the range (idx2, N).
      // Recompute the minimum mindist[idx2] and n_nghbr[idx2].
      if (idx2<N_1) {
        min = mindist[idx2];
        for (j=AR.succ[idx2]; j<N; j=AR.succ[j]) {
          f_single(&D_(idx2, j), D_(idx1, j) );
          if (D_(idx2, j) < min) {
            n_nghbr[idx2] = j;
            min = D_(idx2, j);
          }
        }
        minidx.update_leq(idx2, min);
      }
      break;

    case METHOD_METR_COMPLETE:
      /*
        Complete linkage.

        Characteristic: new distances are never shorter than the old distances.
      */
      // Update the distance matrix in the range [start, idx1).
      for (j=AR.start; j<idx1; j=AR.succ[j]) {
        f_complete(&D_(j, idx2), D_(j, idx1) );
        if (n_nghbr[j] == idx1)
          n_nghbr[j] = idx2;
      }
      // Update the distance matrix in the range (idx1, idx2).
      for (; j<idx2; j=AR.succ[j])
        f_complete(&D_(j, idx2), D_(idx1, j) );
      // Update the distance matrix in the range (idx2, N).
      for (j=AR.succ[idx2]; j<N; j=AR.succ[j])
        f_complete(&D_(idx2, j), D_(idx1, j) );
      break;

    case METHOD_METR_AVERAGE: {
      /*
        Average linkage.

        Shorter and longer distances can occur.
      */
      // Update the distance matrix in the range [start, idx1).
      t_float s = size1/(size1+size2);
      t_float t = size2/(size1+size2);
      for (j=AR.start; j<idx1; j=AR.succ[j]) {
        f_average(&D_(j, idx2), D_(j, idx1), s, t);
        if (n_nghbr[j] == idx1)
          n_nghbr[j] = idx2;
      }
      // Update the distance matrix in the range (idx1, idx2).
      for (; j<idx2; j=AR.succ[j]) {
        f_average(&D_(j, idx2), D_(idx1, j), s, t);
        if (D_(j, idx2)<mindist[j]) {
          minidx.update_leq(j, D_(j, idx2));
          n_nghbr[j] = idx2;
        }
      }
      // Update the distance matrix in the range (idx2, N).
      if (idx2<N_1) {
        n_nghbr[idx2] = j = AR.succ[idx2]; // exists, maximally N-1
        f_average(&D_(idx2, j), D_(idx1, j), s, t);
        min = D_(idx2,j);
        for (j=AR.succ[j]; j<N; j=AR.succ[j]) {
          f_average(&D_(idx2, j), D_(idx1, j), s, t);
          if (D_(idx2,j)<min) {
            min = D_(idx2,j);
            n_nghbr[idx2] = j;
          }
        }
        minidx.update(idx2, min);
      }
      break;
    }

    case METHOD_METR_WEIGHTED:
      /*
        Weighted linkage.

        Shorter and longer distances can occur.
      */
      // Update the distance matrix in the range [start, idx1).
      for (j=AR.start; j<idx1; j=AR.succ[j]) {
        f_weighted(&D_(j, idx2), D_(j, idx1) );
        if (n_nghbr[j] == idx1)
          n_nghbr[j] = idx2;
      }
      // Update the distance matrix in the range (idx1, idx2).
      for (; j<idx2; j=AR.succ[j]) {
        f_weighted(&D_(j, idx2), D_(idx1, j) );
        if (D_(j, idx2)<mindist[j]) {
          minidx.update_leq(j, D_(j, idx2));
          n_nghbr[j] = idx2;
        }
      }
      // Update the distance matrix in the range (idx2, N).
      if (idx2<N_1) {
        n_nghbr[idx2] = j = AR.succ[idx2]; // exists, maximally N-1
        f_weighted(&D_(idx2, j), D_(idx1, j) );
        min = D_(idx2,j);
        for (j=AR.succ[j]; j<N; j=AR.succ[j]) {
          f_weighted(&D_(idx2, j), D_(idx1, j) );
          if (D_(idx2,j)<min) {
            min = D_(idx2,j);
            n_nghbr[idx2] = j;
          }
        }
        minidx.update(idx2, min);
      }
      break;

    case METHOD_METR_WARD:
      /*
        Ward linkage.

        Shorter and longer distances can occur, not smaller than min(d1,d2)
        but maybe bigger than max(d1,d2).
      */
      // Update the distance matrix in the range [start, idx1).
      for (j=AR.start; j<idx1; j=AR.succ[j]) {
        f_ward(&D_(j, idx2), D_(j, idx1), mindist[idx1],
               size1, size2, static_cast<t_float>(members[j]) );
        if (n_nghbr[j] == idx1)
          n_nghbr[j] = idx2;
      }
      // Update the distance matrix in the range (idx1, idx2).
      for (; j<idx2; j=AR.succ[j]) {
        f_ward(&D_(j, idx2), D_(idx1, j), mindist[idx1], size1, size2,
               static_cast<t_float>(members[j]) );
        if (D_(j, idx2)<mindist[j]) {
          minidx.update_leq(j, D_(j, idx2));
          n_nghbr[j] = idx2;
        }
      }
      // Update the distance matrix in the range (idx2, N).
      if (idx2<N_1) {
        n_nghbr[idx2] = j = AR.succ[idx2]; // exists, maximally N-1
        f_ward(&D_(idx2, j), D_(idx1, j), mindist[idx1],
               size1, size2, static_cast<t_float>(members[j]) );
        min = D_(idx2,j);
        for (j=AR.succ[j]; j<N; j=AR.succ[j]) {
          f_ward(&D_(idx2, j), D_(idx1, j), mindist[idx1],
                 size1, size2, static_cast<t_float>(members[j]) );
          if (D_(idx2,j)<min) {
            min = D_(idx2,j);
            n_nghbr[idx2] = j;
          }
        }
        minidx.update(idx2, min);
      }
      break;

    case METHOD_METR_CENTROID: {
      /*
        Centroid linkage.

        Shorter and longer distances can occur, not bigger than max(d1,d2)
        but maybe smaller than min(d1,d2).
      */
      // Update the distance matrix in the range [start, idx1).
      t_float s = size1/(size1+size2);
      t_float t = size2/(size1+size2);
      t_float stc = s*t*mindist[idx1];
      for (j=AR.start; j<idx1; j=AR.succ[j]) {
        f_centroid(&D_(j, idx2), D_(j, idx1), stc, s, t);
        if (D_(j, idx2)<mindist[j]) {
          minidx.update_leq(j, D_(j, idx2));
          n_nghbr[j] = idx2;
        }
        else if (n_nghbr[j] == idx1)
          n_nghbr[j] = idx2;
      }
      // Update the distance matrix in the range (idx1, idx2).
      for (; j<idx2; j=AR.succ[j]) {
        f_centroid(&D_(j, idx2), D_(idx1, j), stc, s, t);
        if (D_(j, idx2)<mindist[j]) {
          minidx.update_leq(j, D_(j, idx2));
          n_nghbr[j] = idx2;
        }
      }
      // Update the distance matrix in the range (idx2, N).
      if (idx2<N_1) {
        n_nghbr[idx2] = j = AR.succ[idx2]; // exists, maximally N-1
        f_centroid(&D_(idx2, j), D_(idx1, j), stc, s, t);
        min = D_(idx2,j);
        for (j=AR.succ[j]; j<N; j=AR.succ[j]) {
          f_centroid(&D_(idx2, j), D_(idx1, j), stc, s, t);
          if (D_(idx2,j)<min) {
            min = D_(idx2,j);
            n_nghbr[idx2] = j;
          }
        }
        minidx.update(idx2, min);
      }
      break;
    }

    case METHOD_METR_MEDIAN:
      /*
        Median linkage.

        Shorter and longer distances can occur, not bigger than max(d1,d2)
        but maybe smaller than min(d1,d2).
      */
      // Update the distance matrix in the range [start, idx1).
      t_float c_4 = mindist[idx1]/4;
      for (j=AR.start; j<idx1; j=AR.succ[j]) {
        f_median(&D_(j, idx2), D_(j, idx1), c_4 );
        if (D_(j, idx2)<mindist[j]) {
          minidx.update_leq(j, D_(j, idx2));
          n_nghbr[j] = idx2;
        }
        else if (n_nghbr[j] == idx1)
          n_nghbr[j] = idx2;
      }
      // Update the distance matrix in the range (idx1, idx2).
      for (; j<idx2; j=AR.succ[j]) {
        f_median(&D_(j, idx2), D_(idx1, j), c_4 );
        if (D_(j, idx2)<mindist[j]) {
          minidx.update_leq(j, D_(j, idx2));
          n_nghbr[j] = idx2;
        }
      }
      // Update the distance matrix in the range (idx2, N).
      if (idx2<N_1) {
        n_nghbr[idx2] = j = AR.succ[idx2]; // exists, maximally N-1
        f_median(&D_(idx2, j), D_(idx1, j), c_4 );
        min = D_(idx2,j);
        for (j=AR.succ[j]; j<N; j=AR.succ[j]) {
          f_median(&D_(idx2, j), D_(idx1, j), c_4 );
          if (D_(idx2,j)<min) {
            min = D_(idx2,j);
            n_nghbr[idx2] = j;
          }
        }
        minidx.update(idx2, min);
      }
      break;
    }
  }
}

/*
  Clustering methods for vector data
*/
template <typename t_dissimilarity>
static void MST_linkage_core_vector(const t_index N,
                                    t_dissimilarity & dist,
                                    cluster_result & Z2) {
/*
    N: integer, number of data points
    dist: function pointer to the metric
    Z2: output data structure

    The basis of this algorithm is an algorithm by Rohlf:

    F. James Rohlf, Hierarchical clustering using the minimum spanning tree,
    The Computer Journal, vol. 16, 1973, p. 93–95.

    This implementation should handle Inf values correctly (designed to
    do so but not tested).

    This implementation avoids NaN if possible. It treats NaN as if it was
    greater than +Infinity, ie. whenever we find a non-NaN value, this is
    preferred in all the minimum-distance searches.
*/
  t_index i;
  t_index idx2;
  doubly_linked_list AR(N);
  auto_array_ptr<t_float> d(N);

  t_index prev_node;
  t_float min;

  // first iteration
  idx2 = 1;
  min = d[1] = dist(0,1);
  for (i=2; min!=min && i<N; i++) { // eliminate NaNs if possible
    min = d[i] = dist(0,i);
    idx2 = i;
  }

  for ( ; i<N; i++) {
    d[i] = dist(0,i);
    if (d[i] < min) {
      min = d[i];
      idx2 = i;
    }
  }

  Z2.append(0, idx2, min);

  for (t_index j=1; j<N-1; j++) {
    prev_node = idx2;
    AR.remove(prev_node);

    idx2 = AR.succ[0];
    min = d[idx2];

    for (i=idx2; min!=min && i<N; i=AR.succ[i]) { // eliminate NaNs if possible
      min = d[i] = dist(i, prev_node);
      idx2 = i;
    }

    for ( ; i<N; i=AR.succ[i]) {
      t_float tmp = dist(i, prev_node);
      if (d[i] > tmp)
        d[i] = tmp;
      if (d[i] < min) {
        min = d[i];
        idx2 = i;
      }
    }
    Z2.append(prev_node, idx2, min);
  }
}

template <const unsigned char method, typename t_dissimilarity>
static void generic_linkage_vector(const t_index N,
                                   t_dissimilarity & dist,
                                   cluster_result & Z2) {
  /*
    N: integer, number of data points
    dist: function pointer to the metric
    Z2: output data structure

    This algorithm is valid for the distance update methods
    "Ward", "centroid" and "median" only!

    This implementation does not give defined results when NaN or Inf values
    are returned by the distance function.
  */
  const t_index N_1 = N-1;
  t_index i, j=0; // loop variables
  t_index idx1, idx2; // row and column indices

  auto_array_ptr<t_index> n_nghbr(2*N-2); // array of nearest neighbors
  auto_array_ptr<t_float> mindist(2*N-2); // distances to the nearest neighbors

  doubly_linked_list AR(N+N_1); // active rows
  binary_min_heap minidx(N_1, 2*N-2,1); // minimum heap structure for the nearest neighbors

  t_float min; // minimum for nearest-neighbor searches

  // Initialize the minimal distances:
  // Find the nearest neighbor of each point.
  // n_nghbr[i] = argmin_{j<i} D(i,j) for i in range(N-1)
  for (i=1; i<N; i++) {
    t_index idx = j = 0;
    switch (method) {
    case METHOD_METR_WARD:
      min = dist.ward(i,j);
      break;
    default:
      min = dist.sqeuclidean(i,j);
    }
    for(j++; min!=min && j<i; j++) { // eliminate NaN if possible
      switch (method) {
      case METHOD_METR_WARD:
        min = dist.ward(i,j);
        break;
      default:
        min = dist.sqeuclidean(i,j);
      }
      idx = j;
    }
    for( ; j<i; j++) {
      t_float tmp;
      switch (method) {
      case METHOD_METR_WARD:
        tmp = dist.ward(i,j);
        break;
      default:
        tmp = dist.sqeuclidean(i,j);
      }
      if (tmp<min) {
        min = tmp;
        idx = j;
      }
    }
    mindist[i] = min;
    n_nghbr[i] = idx;
  }

  // Put the minimal distances into a heap structure to make the repeated global
  // minimum searches fast.
  minidx.heapify(mindist);

  // Main loop: We have N-1 merging steps.
  for (i=N; i<N+N_1; i++) {
    /*
      The bookkeeping is different from the "stored matrix approach" algorithm
      generic_linkage.

      mindist[i] stores a lower bound on the minimum distance of the point i to
      all points of *lower* index:

          mindist[i] ≥ min_{j<i} D(i,j)

      Moreover, new nodes do not re-use one of the old indices, but they are given
      a new, unique index (SciPy convention: initial nodes are 0,…,N−1, new
      nodes are N,…,2N−2).

      Invalid nearest neighbors are not recognized by the fact that the stored
      distance is smaller than the actual distance, but the list AR maintains
      a flag whether a node is inactive. If n_nghbr[i] points to an active node,
      the entries minidx[i] and n_nghbr[i] are valid, otherwise they must be
      recomputed.
    */
    idx1 = minidx;
    while ( AR.is_inactive(n_nghbr[idx1]) ) {
      // Recompute the minimum mindist[idx1] and n_nghbr[idx1].
      n_nghbr[idx1] = j = AR.start;
      switch (method) {
      case METHOD_METR_WARD:
        min = dist.ward_extended(idx1,j);
        for (j=AR.succ[j]; j<idx1; j=AR.succ[j]) {
          t_float tmp = dist.ward_extended(idx1,j);
          if (tmp<min) {
            min = tmp;
            n_nghbr[idx1] = j;
          }
        }
        break;
      default:
        min = dist.sqeuclidean_extended(idx1,j);
        for (j=AR.succ[j]; j<idx1; j=AR.succ[j]) {
          t_float tmp = dist.sqeuclidean_extended(idx1,j);
          if (tmp<min) {
            min = tmp;
            n_nghbr[idx1] = j;
          }
        }
      }
      minidx.update_geq(idx1, min); // Update the heap with the new true minimum.
      idx1 = minidx; // And search for the (possibly different) minimal entry.
    }

    idx2 = n_nghbr[idx1];
    AR.remove(idx1);
    AR.remove(idx2);

    Z2.append(idx1, idx2, mindist[idx1]);

    if (i<2*N_1) {
      switch (method) {
      case METHOD_METR_WARD:
      case METHOD_METR_CENTROID:
        dist.merge(idx1, idx2, i);
        break;

      case METHOD_METR_MEDIAN:
        dist.merge_weighted(idx1, idx2, i);
        break;
      }

      n_nghbr[i] = AR.start;
      if (method==METHOD_METR_WARD) {
        /*
          Ward linkage.

          Shorter and longer distances can occur, not smaller than min(d1,d2)
          but maybe bigger than max(d1,d2).
        */
        min = dist.ward_extended(AR.start, i);
        // TBD: avoid NaN
        for (j=AR.succ[AR.start]; j<i; j=AR.succ[j]) {
          t_float tmp = dist.ward_extended(j, i);
          if (tmp<min) {
            min = tmp;
            n_nghbr[i] = j;
          }
        }
      }
      else {
        /*
          Centroid and median linkage.

          Shorter and longer distances can occur, not bigger than max(d1,d2)
          but maybe smaller than min(d1,d2).
        */
        min = dist.sqeuclidean_extended(AR.start, i);
        // TBD: avoid NaN
        for (j=AR.succ[AR.start]; j<i; j=AR.succ[j]) {
          t_float tmp = dist.sqeuclidean_extended(j, i);
          if (tmp<min) {
            min = tmp;
            n_nghbr[i] = j;
          }
        }
      }
      if (idx2<AR.start)  {
        minidx.remove(AR.start);
      } else {
        minidx.remove(idx2);
      }
      minidx.replace(idx1, i, min);
    }
  }
}


// backwards compatibility
#ifndef NPY_ARRAY_CARRAY_RO
#define NPY_ARRAY_CARRAY_RO NPY_CARRAY_RO
#endif

/*
  Convenience class for the output array: automatic counter.
*/
class linkage_output {
private:
  t_float * Z;
  t_index pos;

public:
  linkage_output(t_float * const Z) {
    this->Z = Z;
    pos = 0;
  }

  void append(const t_index node1, const t_index node2, const t_float dist, const t_float size) {
    if (node1<node2) {
      Z[pos++] = static_cast<t_float>(node1);
      Z[pos++] = static_cast<t_float>(node2);
    }
    else {
      Z[pos++] = static_cast<t_float>(node2);
      Z[pos++] = static_cast<t_float>(node1);
    }
    Z[pos++] = dist;
    Z[pos++] = size;
  }
};

/*
  Generate the Scipy-specific output format for a dendrogram from the
  clustering output.

  The list of merging steps can be sorted or unsorted.
*/
// The size of a node is either 1 (a single point) or is looked up from
// one of the clusters.
#define size_(r_) ( ((r_<N) ? 1 : Z_(r_-N,3)) )

template <bool sorted>
static void generate_SciPy_dendrogram(t_float * const Z, cluster_result & Z2, const t_index N) {
  // The array "nodes" is a union-find data structure for the cluster
  // identites (only needed for unsorted cluster_result input).
  union_find nodes;
  if (!sorted) {
    std::stable_sort(Z2[0], Z2[N-1]);
    nodes.init(N);
  }

  linkage_output output(Z);
  t_index node1, node2;

  for (t_index i=0; i<N-1; i++) {
    // Get two data points whose clusters are merged in step i.
    if (sorted) {
      node1 = Z2[i]->node1;
      node2 = Z2[i]->node2;
    }
    else {
      // Find the cluster identifiers for these points.
      node1 = nodes.Find(Z2[i]->node1);
      node2 = nodes.Find(Z2[i]->node2);
      // Merge the nodes in the union-find data structure by making them
      // children of a new node.
      nodes.Union(node1, node2);
    }
    output.append(node1, node2, Z2[i]->dist, size_(node1)+size_(node2));
  }
}

/*
  Python interface code
*/
static PyObject * linkage_wrap(PyObject * const self, PyObject * const args);
static PyObject * linkage_vector_wrap(PyObject * const self, PyObject * const args);

// List the C++ methods that this extension provides.
static PyMethodDef _fastclusterWrapMethods[] = {
  {"linkage_wrap", linkage_wrap, METH_VARARGS},
  {"linkage_vector_wrap", linkage_vector_wrap, METH_VARARGS},
  {NULL, NULL, 0, NULL}     /* Sentinel - marks the end of this structure */
};

// Tell Python about these methods.
PyMODINIT_FUNC init_fastcluster(void)  {
  (void) Py_InitModule("_fastcluster", _fastclusterWrapMethods);
  import_array();  // Must be present for NumPy. Called first after above line.
}

/*
  Interface to Python, part 1:
  The input is a dissimilarity matrix.
*/
static PyObject *linkage_wrap(PyObject * const self, PyObject * const args) {
  PyArrayObject * D, * Z;
  long int N = 0;
  unsigned char method;

  try{
    // Parse the input arguments
    if (!PyArg_ParseTuple(args, "lO!O!b",
                          &N,                // signed long integer
                          &PyArray_Type, &D, // NumPy array
                          &PyArray_Type, &Z, // NumPy array
                          &method)) {        // unsigned char
      return NULL; // Error if the arguments have the wrong type.
    }
    if (N < 1 ) {
      // N must be at least 1.
      PyErr_SetString(PyExc_ValueError,
                      "At least one element is needed for clustering.");
      return NULL;
    }

    // (1)
    // The biggest index used below is 4*(N-2)+3, as an index to Z. This must fit
    // into the data type used for indices.
    // (2)
    // The largest representable integer, without loss of precision, by a floating
    // point number of type t_float is 2^T_FLOAT_MANT_DIG. Here, we make sure that
    // all cluster labels from 0 to 2N-2 in the output can be accurately represented
    // by a floating point number.
    if (N > MAX_INDEX/4 ||
        (N-1)>>(T_FLOAT_MANT_DIG-1) > 0) {
      PyErr_SetString(PyExc_ValueError,
                      "Data is too big, index overflow.");
      return NULL;
    }

    t_float * const D_ = reinterpret_cast<t_float *>(D->data);
    cluster_result Z2(N-1);
    auto_array_ptr<t_index> members;
    // For these methods, the distance update formula needs the number of
    // data points in a cluster.
    if (method==METHOD_METR_AVERAGE ||
        method==METHOD_METR_WARD ||
        method==METHOD_METR_CENTROID) {
      members.init(N, 1);
    }
    // Operate on squared distances for these methods.
    if (method==METHOD_METR_WARD ||
        method==METHOD_METR_CENTROID ||
        method==METHOD_METR_MEDIAN) {
      for (ptrdiff_t i=0; i < static_cast<ptrdiff_t>(N)*(N-1)/2; i++)
        D_[i] *= D_[i];
    }

    switch (method) {
    case METHOD_METR_SINGLE:
      MST_linkage_core(N, D_, Z2);
      break;
    case METHOD_METR_COMPLETE:
      NN_chain_core<METHOD_METR_COMPLETE, t_index>(N, D_, NULL, Z2);
      break;
    case METHOD_METR_AVERAGE:
      NN_chain_core<METHOD_METR_AVERAGE, t_index>(N, D_, members, Z2);
      break;
    case METHOD_METR_WEIGHTED:
      NN_chain_core<METHOD_METR_WEIGHTED, t_index>(N, D_, NULL, Z2);
      break;
    case METHOD_METR_WARD:
      NN_chain_core<METHOD_METR_WARD, t_index>(N, D_, members, Z2);
      break;
    case METHOD_METR_CENTROID:
      generic_linkage<METHOD_METR_CENTROID, t_index>(N, D_, members, Z2);
      break;
    case METHOD_METR_MEDIAN:
      generic_linkage<METHOD_METR_MEDIAN, t_index>(N, D_, NULL, Z2);
      break;
    default:
      PyErr_SetString(PyExc_IndexError, "Invalid method index.");
      return NULL;
    }

    if (method==METHOD_METR_WARD ||
        method==METHOD_METR_CENTROID ||
        method==METHOD_METR_MEDIAN) {
      Z2.sqrt();
    }

    t_float * const Z_ = reinterpret_cast<t_float *>(Z->data);
    if (method==METHOD_METR_CENTROID ||
        method==METHOD_METR_MEDIAN) {
      generate_SciPy_dendrogram<true>(Z_, Z2, N);
    }
    else {
      generate_SciPy_dendrogram<false>(Z_, Z2, N);
    }

  } // try
  catch (std::bad_alloc&) {
    return PyErr_NoMemory();
  }
  catch(std::exception& e){
    PyErr_SetString(PyExc_StandardError, e.what());
    return NULL;
  }
  catch(...){
    PyErr_SetString(PyExc_StandardError,
                    "C++ exception (unknown reason). Please send a bug report.");
    return NULL;
  }
  Py_RETURN_NONE;
}

/*
   Part 2: Clustering on vector data
*/

/*
   Helper class: Throw this if calling the Python interpreter from within
   C returned an error.
*/
class pythonerror {};

enum {
  // metrics
  METRIC_EUCLIDEAN       =  0,
  METRIC_MINKOWSKI       =  1,
  METRIC_CITYBLOCK       =  2,
  METRIC_SEUCLIDEAN      =  3,
  METRIC_SQEUCLIDEAN     =  4,
  METRIC_COSINE          =  5,
  METRIC_HAMMING         =  6,
  METRIC_JACCARD         =  7,
  METRIC_CHEBYCHEV       =  8,
  METRIC_CANBERRA        =  9,
  METRIC_BRAYCURTIS      = 10,
  METRIC_MAHALANOBIS     = 11,
  METRIC_YULE            = 12,
  METRIC_MATCHING        = 13,
  METRIC_DICE            = 14,
  METRIC_ROGERSTANIMOTO  = 15,
  METRIC_RUSSELLRAO      = 16,
  METRIC_SOKALSNEATH     = 17,
  METRIC_KULSINSKI       = 18,
  METRIC_USER            = 19,
  METRIC_INVALID         = 20, // sentinel
  METRIC_JACCARD_BOOL    = 21, // separate function for Jaccard metric on Boolean
};                             // input data

/*
  This class handles all the information about the dissimilarity
  computation.
*/

class python_dissimilarity {
private:
  t_float * Xa;
  auto_array_ptr<t_float> Xnew;
  ptrdiff_t dim; // size_t saves many statis_cast<> in products
  t_index N;
  t_index * members;
  void (cluster_result::*postprocessfn) (const t_float) const;
  t_float postprocessarg;

  t_float (python_dissimilarity::*distfn) (const t_index, const t_index) const;

  // for user-defined metrics
  PyObject * X_Python;
  PyObject * userfn;

  auto_array_ptr<t_float> precomputed;
  t_float * precomputed2;

  PyArrayObject * V;
  const t_float * V_data;

public:
  python_dissimilarity (PyArrayObject * const Xarg,
                        t_index * const members,
                        const unsigned char method,
                        const unsigned char metric,
                        PyObject * const extraarg,
                        bool temp_point_array)
    : Xa(reinterpret_cast<t_float *>(Xarg->data)),
      dim(Xarg->dimensions[1]),
      N(Xarg->dimensions[0]),
      members(members),
      postprocessfn(NULL),
      V(NULL)
  {
    switch (method) {
    case METHOD_METR_SINGLE:
      postprocessfn = NULL; // default
      switch (metric) {
      case METRIC_EUCLIDEAN:
        set_euclidean();
        break;
      case METRIC_SEUCLIDEAN:
        if (extraarg==NULL) {
          PyErr_SetString(PyExc_TypeError,
                          "The 'seuclidean' metric needs a variance parameter.");
          throw pythonerror();
        }
        V  = reinterpret_cast<PyArrayObject *>(PyArray_FromAny(extraarg,
                                               PyArray_DescrFromType(NPY_DOUBLE),
                                               1, 1,
                                               NPY_ARRAY_CARRAY_RO,
                                               NULL));
        if (PyErr_Occurred()) {
          throw pythonerror();
        }
        if (V->dimensions[0]!=dim) {
          PyErr_SetString(PyExc_ValueError,
          "The variance vector must have the same dimensionality as the data.");
          throw pythonerror();
        }
        V_data = reinterpret_cast<t_float *>(V->data);
        distfn = &python_dissimilarity::seuclidean;
        postprocessfn = &cluster_result::sqrt;
        break;
      case METRIC_SQEUCLIDEAN:
        distfn = &python_dissimilarity::sqeuclidean;
        break;
      case METRIC_CITYBLOCK:
        set_cityblock();
        break;
      case METRIC_CHEBYCHEV:
        set_chebychev();
        break;
      case METRIC_MINKOWSKI:
        set_minkowski(extraarg);
        break;
      case METRIC_COSINE:
        distfn = &python_dissimilarity::cosine;
        postprocessfn = &cluster_result::plusone;
        // precompute norms
        precomputed.init(N);
        for (t_index i=0; i<N; i++) {
          t_float sum=0;
          for (t_index k=0; k<dim; k++) {
            sum += X(i,k)*X(i,k);
          }
          precomputed[i] = 1/sqrt(sum);
        }
        break;
      case METRIC_HAMMING:
        distfn = &python_dissimilarity::hamming;
        postprocessfn = &cluster_result::divide;
        postprocessarg = static_cast<t_float>(dim);
        break;
      case METRIC_JACCARD:
        distfn = &python_dissimilarity::jaccard;
        break;
      case METRIC_CANBERRA:
        distfn = &python_dissimilarity::canberra;
        break;
      case METRIC_BRAYCURTIS:
        distfn = &python_dissimilarity::braycurtis;
        break;
      case METRIC_MAHALANOBIS:
        if (extraarg==NULL) {
          PyErr_SetString(PyExc_TypeError,
            "The 'mahalanobis' metric needs a parameter for the inverse covariance.");
          throw pythonerror();
        }
        V = reinterpret_cast<PyArrayObject *>(PyArray_FromAny(extraarg,
              PyArray_DescrFromType(NPY_DOUBLE),
              2, 2,
              NPY_ARRAY_CARRAY_RO,
              NULL));
        if (PyErr_Occurred()) {
          throw pythonerror();
        }
        if (V->dimensions[0]!=N || V->dimensions[1]!=dim) {
          PyErr_SetString(PyExc_ValueError,
            "The inverse covariance matrix has the wrong size.");
          throw pythonerror();
        }
        V_data = reinterpret_cast<t_float *>(V->data);
        distfn = &python_dissimilarity::mahalanobis;
        postprocessfn = &cluster_result::sqrt;
        break;
      case METRIC_YULE:
        distfn = &python_dissimilarity::yule;
        break;
      case METRIC_MATCHING:
        distfn = &python_dissimilarity::matching;
        postprocessfn = &cluster_result::divide;
        postprocessarg = static_cast<t_float>(dim);
        break;
      case METRIC_DICE:
        distfn = &python_dissimilarity::dice;
        break;
      case METRIC_ROGERSTANIMOTO:
        distfn = &python_dissimilarity::rogerstanimoto;
        break;
      case METRIC_RUSSELLRAO:
        distfn = &python_dissimilarity::russellrao;
        postprocessfn = &cluster_result::divide;
        postprocessarg = static_cast<t_float>(dim);
        break;
      case METRIC_SOKALSNEATH:
        distfn = &python_dissimilarity::sokalsneath;
        break;
      case METRIC_KULSINSKI:
        distfn = &python_dissimilarity::kulsinski;
        postprocessfn = &cluster_result::plusone;
        precomputed.init(N);
        for (t_index i=0; i<N; i++) {
          t_index sum=0;
          for (t_index k=0; k<dim; k++) {
            sum += Xb(i,k);
          }
          precomputed[i] = -.5/static_cast<t_float>(sum);
        }
        break;
      case METRIC_USER:
        X_Python = reinterpret_cast<PyObject *>(Xarg);
        this->userfn = extraarg;
        distfn = &python_dissimilarity::user;
        break;
      case METRIC_JACCARD_BOOL:
        distfn = &python_dissimilarity::jaccard_bool;
        break;
      default:
        throw 0;
      }
      break;

    case METHOD_METR_WARD:
      postprocessfn = &cluster_result::sqrtdouble;
      break;

    default:
      postprocessfn = &cluster_result::sqrt;
    }

    if (temp_point_array) {
      Xnew.init((N-1)*dim);
    }
  }

  ~python_dissimilarity() {
    Py_XDECREF(V);
  }

  inline t_float operator () (const t_index i, const t_index j) const {
    return (this->*distfn)(i,j);
  }

  inline t_float X (const t_index i, const t_index j) const {
    return Xa[i*dim+j];
  }

  inline bool Xb (const t_index i, const t_index j) const {
    return  reinterpret_cast<bool *>(Xa)[i*dim+j];
  }

  inline t_float * Xptr(const t_index i, const t_index j) const {
    return Xa+i*dim+j;
  }

  void merge(const t_index i, const t_index j, const t_index newnode) const {
    t_float const * Pi = i<N ? Xa+i*dim : Xnew+(i-N)*dim;
    t_float const * Pj = j<N ? Xa+j*dim : Xnew+(j-N)*dim;
    for(t_index k=0; k<dim; k++) {
      Xnew[(newnode-N)*dim+k] = (Pi[k]*static_cast<t_float>(members[i]) +
                             Pj[k]*static_cast<t_float>(members[j])) /
        static_cast<t_float>(members[i]+members[j]);
    }
    members[newnode] = members[i]+members[j];
  }

  void merge_weighted(const t_index i, const t_index j, const t_index newnode) const {
    t_float const * Pi = i<N ? Xa+i*dim : Xnew+(i-N)*dim;
    t_float const * Pj = j<N ? Xa+j*dim : Xnew+(j-N)*dim;
    for(t_index k=0; k<dim; k++) {
      Xnew[(newnode-N)*dim+k] = (Pi[k]+Pj[k])/2.;
    }
  }

  void postprocess(cluster_result & Z2) const {
    if (postprocessfn!=NULL) {
        (Z2.*postprocessfn)(postprocessarg);
    }
  }

  inline t_float ward(const t_index i, const t_index j) const {
    t_float mi = static_cast<t_float>(members[i]);
    t_float mj = static_cast<t_float>(members[j]);
    return sqeuclidean(i,j)*mi*mj/(mi+mj);
  }

  inline t_float ward_extended(const t_index i, const t_index j) const {
    t_float mi = static_cast<t_float>(members[i]);
    t_float mj = static_cast<t_float>(members[j]);
    return sqeuclidean_extended(i,j)*mi*mj/(mi+mj);
  }

  t_float sqeuclidean(const t_index i, const t_index j) const {
    t_float sum = 0;
    /*
    for (t_index k=0; k<dim; k++) {
        t_float diff = X(i,k) - X(j,k);
        sum += diff*diff;
    }
    */
    // faster
    t_float const * Pi = Xa+i*dim;
    t_float const * Pj = Xa+j*dim;
    for (t_index k=0; k<dim; k++) {
      t_float diff = Pi[k] - Pj[k];
      sum += diff*diff;
    }
    return sum;
  }

  t_float sqeuclidean_extended(const t_index i, const t_index j) const {
    t_float sum = 0;
    t_float const * Pi = i<N ? Xa+i*dim : Xnew+(i-N)*dim; // TBD
    t_float const * Pj = j<N ? Xa+j*dim : Xnew+(j-N)*dim;
    for (t_index k=0; k<dim; k++) {
      t_float diff = Pi[k] - Pj[k];
      sum += diff*diff;
    }
    return sum;
  }

private:
  void set_minkowski(PyObject * extraarg) {
    if (extraarg==NULL) {
      PyErr_SetString(PyExc_TypeError,
                      "The Minkowski metric needs a parameter.");
      throw pythonerror();
    }
    postprocessarg = PyFloat_AsDouble(extraarg);
    if (PyErr_Occurred()) {
      throw pythonerror();
    }

    if (postprocessarg==std::numeric_limits<t_float>::infinity()) {
      set_chebychev();
    }
    else if (postprocessarg==1.0){
      set_cityblock();
    }
    else if (postprocessarg==2.0){
      set_euclidean();
    }
    else {
      distfn = &python_dissimilarity::minkowski;
      postprocessfn = &cluster_result::power;
    }
  }

  void set_euclidean() {
    distfn = &python_dissimilarity::sqeuclidean;
    postprocessfn = &cluster_result::sqrt;
  }

  void set_cityblock() {
    distfn = &python_dissimilarity::cityblock;
  }

  void set_chebychev() {
    distfn = &python_dissimilarity::chebychev;
  }

  t_float seuclidean(const t_index i, const t_index j) const {
    t_float sum = 0;
    for (t_index k=0; k<dim; k++) {
      t_float diff = X(i,k)-X(j,k);
      sum += diff*diff/V_data[k];
    }
    return sum;
  }

  t_float cityblock(const t_index i, const t_index j) const {
    t_float sum = 0;
    for (t_index k=0; k<dim; k++) {
      sum += fabs(X(i,k)-X(j,k));
    }
    return sum;
  }

  t_float minkowski(const t_index i, const t_index j) const {
    t_float sum = 0;
    for (t_index k=0; k<dim; k++) {
      sum += pow(fabs(X(i,k)-X(j,k)),postprocessarg);
    }
    return sum;
  }

  t_float chebychev(const t_index i, const t_index j) const {
    t_float max = 0;
    for (t_index k=0; k<dim; k++) {
      t_float diff = fabs(X(i,k)-X(j,k));
      if (diff>max) {
        max = diff;
      }
    }
    return max;
  }

  t_float cosine(const t_index i, const t_index j) const {
    t_float sum = 0;
    for (t_index k=0; k<dim; k++) {
      sum -= X(i,k)*X(j,k);
    }
    return sum*precomputed[i]*precomputed[j];
  }

  t_float hamming(const t_index i, const t_index j) const {
    t_float sum = 0;
    for (t_index k=0; k<dim; k++) {
      sum += (X(i,k)!=X(j,k));
    }
    return sum;
  }

  // Differs from scipy.spatial.distance: equal vectors correctly
  // return distance 0.
  t_float jaccard(const t_index i, const t_index j) const {
    t_index sum1 = 0;
    t_index sum2 = 0;
    for (t_index k=0; k<dim; k++) {
      sum1 += (X(i,k)!=X(j,k));
      sum2 += ((X(i,k)!=0) || (X(j,k)!=0));
    }
    return sum1==0 ? 0 : static_cast<t_float>(sum1) / static_cast<t_float>(sum2);
  }

  t_float canberra(const t_index i, const t_index j) const {
    t_float sum = 0;
    for (t_index k=0; k<dim; k++) {
      t_float numerator = fabs(X(i,k)-X(j,k));
      sum += numerator==0 ? 0 : numerator / (fabs(X(i,k)) + fabs(X(j,k)));
    }
    return sum;
  }

  t_float user(const t_index i, const t_index j) const {
    PyObject * u = PySequence_ITEM(X_Python, i);
    PyObject * v = PySequence_ITEM(X_Python, j);
    PyObject * result = PyObject_CallFunctionObjArgs(userfn, u, v, NULL);
    Py_DECREF(u);
    Py_DECREF(v);
    if (result==NULL) {
      throw pythonerror();
    }
    const t_float C_result = PyFloat_AsDouble(result);
    Py_DECREF(result);
    if (PyErr_Occurred()) {
      throw pythonerror();
    }
    return C_result;
  }

  t_float braycurtis(const t_index i, const t_index j) const {
    t_float sum1 = 0;
    t_float sum2 = 0;
    for (t_index k=0; k<dim; k++) {
      sum1 += fabs(X(i,k)-X(j,k));
      sum2 += fabs(X(i,k)+X(j,k));
    }
    return sum1/sum2;
  }

  t_float mahalanobis(const t_index i, const t_index j) const {
    // V_data contains the product X*VI
    t_float sum = 0;
    for (t_index k=0; k<dim; k++) {
      sum += (V_data[i*dim+k]-V_data[j*dim+k])*(X(i,k)-X(j,k));
    }
    return sum;
  }

  t_index mutable NTT; // 'local' variables
  t_index mutable NXO;
  t_index mutable NTF;
  #define NTFFT NTF
  #define NFFTT NTT

  void nbool_correspond(const t_index i, const t_index j) const {
    NTT = 0;
    NXO = 0;
    for (t_index k=0; k<dim; k++) {
      NTT += (Xb(i,k) &  Xb(j,k)) ;
      NXO += (Xb(i,k) ^  Xb(j,k)) ;
    }
  }

  void nbool_correspond_tfft(const t_index i, const t_index j) const {
    NTT = 0;
    NXO = 0;
    NTF = 0;
    for (t_index k=0; k<dim; k++) {
      NTT += (Xb(i,k) &  Xb(j,k)) ;
      NXO += (Xb(i,k) ^  Xb(j,k)) ;
      NTF += (Xb(i,k) & ~Xb(j,k)) ;
    }
    NTF *= (NXO-NTF); // NTFFT
    NTT *= (dim-NTT-NXO); // NFFTT
  }

  void nbool_correspond_xo(const t_index i, const t_index j) const {
    NXO = 0;
    for (t_index k=0; k<dim; k++) {
      NXO += (Xb(i,k) ^  Xb(j,k)) ;
    }
  }

  void nbool_correspond_tt(const t_index i, const t_index j) const {
    NTT = 0;
    for (t_index k=0; k<dim; k++) {
      NTT += (Xb(i,k) &  Xb(j,k)) ;
    }
  }

  // Caution: zero denominators can happen here!
  t_float yule(const t_index i, const t_index j) const {
    nbool_correspond_tfft(i, j);
    return static_cast<t_float>(2*NTFFT) / static_cast<t_float>(NTFFT + NFFTT);
  }

  // Prevent a zero denominator for equal vectors.
  t_float dice(const t_index i, const t_index j) const {
    nbool_correspond(i, j);
    return (NXO==0) ? 0 :
      static_cast<t_float>(NXO) / static_cast<t_float>(NXO+2*NTT);
  }

  t_float rogerstanimoto(const t_index i, const t_index j) const {
    nbool_correspond_xo(i, j);
    return static_cast<t_float>(2*NXO) / static_cast<t_float>(NXO+dim);
  }

  t_float russellrao(const t_index i, const t_index j) const {
    nbool_correspond_tt(i, j);
    return static_cast<t_float>(dim-NTT);
  }

  // Prevent a zero denominator for equal vectors.
  t_float sokalsneath(const t_index i, const t_index j) const {
    nbool_correspond(i, j);
    return (NXO==0) ? 0 :
      static_cast<t_float>(2*NXO) / static_cast<t_float>(NTT+2*NXO);
  }

  t_float kulsinski(const t_index i, const t_index j) const {
    nbool_correspond_tt(i, j);
    return static_cast<t_float>(NTT) * (precomputed[i] + precomputed[j]);
  }

  // 'matching' distance = Hamming distance
  t_float matching(const t_index i, const t_index j) const {
    nbool_correspond_xo(i, j);
    return static_cast<t_float>(NXO);
  }

  // Prevent a zero denominator for equal vectors.
  t_float jaccard_bool(const t_index i, const t_index j) const {
    nbool_correspond(i, j);
    return (NXO==0) ? 0 :
      static_cast<t_float>(NXO) / static_cast<t_float>(NXO+NTT);
  }
};

static PyObject *linkage_vector_wrap(PyObject * const self, PyObject * const args) {
  PyArrayObject * X, * Z;
  unsigned char method, metric;
  PyObject * extraarg;

  try{
    // Parse the input arguments
    if (!PyArg_ParseTuple(args, "O!O!bbO",
                          &PyArray_Type, &X, // NumPy array
                          &PyArray_Type, &Z, // NumPy array
                          &method,           // unsigned char
                          &metric,           // unsigned char
                          &extraarg )) {     // Python object
      throw pythonerror(); // Error if the arguments have the wrong type.
    }

    if (X->nd != 2) {
      PyErr_SetString(PyExc_ValueError,
                      "The input array must be two-dimensional.");
    }
    npy_intp const N = X->dimensions[0];
    if (N < 1 ) {
      // N must be at least 1.
      PyErr_SetString(PyExc_ValueError,
                      "At least one element is needed for clustering.");
      throw pythonerror();
    }

    npy_intp const dim = X->dimensions[1];
    if (dim < 1 ) {
      PyErr_SetString(PyExc_ValueError,
                      "Invalid dimension of the data set.");
      throw pythonerror();
    }

    // (1)
    // The biggest index used below is 4*(N-2)+3, as an index to Z. This must fit
    // into the data type used for indices.
    // (2)
    // The largest representable integer, without loss of precision, by a floating
    // point number of type t_float is 2^T_FLOAT_MANT_DIG. Here, we make sure that
    // all cluster labels from 0 to 2N-2 in the output can be accurately represented
    // by a floating point number.
    if (N > MAX_INDEX/4 ||
        (N-1)>>(T_FLOAT_MANT_DIG-1) > 0) {
      PyErr_SetString(PyExc_ValueError,
                      "Data is too big, index overflow.");
      throw pythonerror();
    }

    cluster_result Z2(N-1);

    auto_array_ptr<t_index> members;
    if (method==METHOD_METR_WARD || method==METHOD_METR_CENTROID) {
      members.init(2*N-1, 1);
    }

    if ((method!=METHOD_METR_SINGLE && metric!=METRIC_EUCLIDEAN) ||
        metric>=METRIC_INVALID) {
      PyErr_SetString(PyExc_IndexError, "Invalid metric index.");
      throw pythonerror();
    }

    if (PyArray_ISBOOL(X)) {
      if (metric==METRIC_HAMMING) {
        metric = METRIC_MATCHING; // Alias
      }
      if (metric==METRIC_JACCARD) {
        metric = METRIC_JACCARD_BOOL;
      }
    }

    if (extraarg!=Py_None &&
        metric!=METRIC_MINKOWSKI &&
        metric!=METRIC_SEUCLIDEAN &&
        metric!=METRIC_MAHALANOBIS &&
        metric!=METRIC_USER) {
      PyErr_SetString(PyExc_TypeError,
                      "No extra parameter is allowed for this metric.");
      throw pythonerror();
    }

    bool temp_point_array = (method!=METHOD_METR_SINGLE);

    python_dissimilarity dist(X, members, method, metric, extraarg,
                              temp_point_array);

    switch (method) {
    case METHOD_METR_SINGLE:
      MST_linkage_core_vector(N, dist, Z2);
      break;
    case METHOD_METR_WARD:
      generic_linkage_vector<METHOD_METR_WARD>(N, dist, Z2);
      break;
    case METHOD_METR_CENTROID:
      generic_linkage_vector<METHOD_METR_CENTROID>(N, dist, Z2);
      break;
    case METHOD_METR_MEDIAN:
      generic_linkage_vector<METHOD_METR_MEDIAN>(N, dist, Z2);
      break;
    default:
      PyErr_SetString(PyExc_IndexError, "Invalid method index.");
      throw pythonerror();
    }

    if (method==METHOD_METR_WARD ||
        method==METHOD_METR_CENTROID) {
      members.free();
    }

    dist.postprocess(Z2);

    t_float * const Z_ = reinterpret_cast<t_float *>(Z->data);
    if (method!=METHOD_METR_SINGLE) {
      generate_SciPy_dendrogram<true>(Z_, Z2, N);
    }
    else {
      generate_SciPy_dendrogram<false>(Z_, Z2, N);
    }

  } // try
  catch (std::bad_alloc&) {
    return PyErr_NoMemory();
  }
  catch(std::exception& e){
    PyErr_SetString(PyExc_StandardError, e.what());
    return NULL;
  }
  catch(pythonerror){
    return NULL;
  }
  catch(...){
    PyErr_SetString(PyExc_StandardError,
                    "C++ exception (unknown reason). Please send a bug report.");
    return NULL;
  }
  Py_RETURN_NONE;
}

