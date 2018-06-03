import numpy as np
# cimport numpy as np
import cython

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
def read_file(filename):
    cdef int n, v, counts
    cdef char *n_, *v_, *counts_
    f = open(filename, 'rb')
    f_ = f.readlines()

    N, V = fast_atoi(f_[0]), fast_atoi(f_[1])
    cdef int[:, ::1] X = np.zeros((N, V), np.int32)

    j = 0
    for line in f:
        if j >= 3:
          n_, v_, counts_ = line.split(' ')
          n, v, counts = fast_atoi(n_), fast_atoi(v_), fast_atoi(counts_)
          X[n, v] += counts
        else:
          j += 1
    f.close()
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
  cdef int N, V, i, j, m, n

  N, V = X.shape
  cdef int[:, ::1] global_vocab_topics = np.zeros((V, K), np.int32)
  cdef int[:, ::1] global_doc_topics = np.zeros((N, K), np.int32)

  i = 0
  for m in range(N):
      for n in range(V): # V
          j = 0
          while j < X[m, n]:
              new_top = next(samples)
              DS[i] = m
              WS[i] = n
              local_params[i] = new_top
              global_vocab_topics[n, new_top] += 1
              global_doc_topics[m, new_top] += 1
              j += 1
              i += 1
  return WS, DS, local_params, global_vocab_topics, global_doc_topics
