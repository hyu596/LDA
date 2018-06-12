from __future__ import print_function
import numpy as np
# cimport numpy as np
import cython
from scipy.sparse import *

cdef int fast_atoi(char *buff):
    cdef int c = 0, sign = 0, x = 0
    cdef char *p = buff
    while True:
        c = p[0]
        if c == 0:
            break
        if c == 45:
            sign = 1
        elif c > 47 and c < 58:
            x = x * 10 + c - 48
        p += 1
    return -x if sign else x

@cython.boundscheck(False)
@cython.wraparound(False)
def read_file(filename, docs=None):
    cdef int n, v, counts, N, V
    cdef char *n_, *v_, *counts_
    
    with open(filename, 'rb') as f:
      f_ = f.readlines()
      N, V = fast_atoi(f_[0]), fast_atoi(f_[1])
    #cdef int[:, ::1] X = np.zeros((N, V), np.int32)
    if docs is not None:
      assert docs <= N
      X = dok_matrix((docs, V), np.int32)
    else:
      X = dok_matrix((N, V), np.int32)
    

    j = 0
    with open(filename, 'rb') as f:
      for line in f:
        if j >= 3:
          n_, v_, counts_ = line.split(b' ')
          n, v, counts = fast_atoi(n_), fast_atoi(v_), fast_atoi(counts_)
          if docs is not None and n > docs: break
          X[n-1, v-1] += counts
        else:
          j += 1
    return X

@cython.boundscheck(False)
@cython.wraparound(False)
def assign_topics(X, K, samples):
  cdef int total_count = np.sum(X)
  # cdef int* WS = <int*> malloc(total_count * sizeof(int))
  # cdef int* DS = <int*> malloc(total_count * sizeof(int))
  # cdef int* local_params = <int*> malloc(total_count * sizeof(int))
  cdef int[:] WS = np.zeros((total_count), np.int32)
  cdef int[:] DS = np.zeros((total_count), np.int32)
  cdef int[:] local_params = np.zeros((total_count), np.int32)
  cdef int N, V, m, n, temp, i

  N, V = X.shape
  cdef int[:, ::1] global_vocab_topics = np.zeros((V, K), np.int32)
  cdef int[:, ::1] global_doc_topics = np.zeros((N, K), np.int32)

  m_, n_ = X.nonzero()
  i = 0
  for m, n in zip(m_, n_):
    temp = int(X[m, n])
    for _ in range(temp):
      new_top = next(samples)
      DS[i] = m
      WS[i] = n
      local_params[i] = new_top
      global_vocab_topics[n, new_top] += 1
      global_doc_topics[m, new_top] += 1
      i += 1
  return WS, DS, local_params, global_vocab_topics, global_doc_topics
