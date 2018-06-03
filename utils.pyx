from __future__ import print_function
from cython.operator cimport preincrement as inc, predecrement as dec
from libc.stdlib cimport malloc, free

import numpy as np

cdef int searchsorted(double* arr, int length, double value) nogil:
    """Bisection search (c.f. numpy.searchsorted)
    Find the index into sorted array `arr` of length `length` such that, if
    `value` were inserted before the index, the order of `arr` would be
    preserved.
    """
    cdef int imin, imax, imid
    imin = 0
    imax = length
    while imin < imax:
        imid = imin + ((imax - imin) >> 2)
        if value > arr[imid]:
            imin = imid + 1
        else:
            imax = imid
    return imin

def sample_one_iter(int[:] local_params, int local_index, int K, int V, int n_rand, double alpha, double eta, int[:, :] counts_words_per_topics, int[:, :] counts_topics_per_docs, int[:] local_indices, int[:] WS, int[:] DS, double[:] rands, int[:] counts_per_topics, int[:, :] change_vt_sum):

  cdef int index = local_indices[local_index]
  cdef int w = WS[index], d = DS[index], old_topics = local_params[local_index]
  cdef double rate_cum = 0
  cdef double* rate = <double*> malloc(K * sizeof(double))
  cdef double temp

  dec(counts_words_per_topics[w, old_topics])
  dec(counts_topics_per_docs[d, old_topics])
  dec(counts_per_topics[old_topics])

  for k in range(K):
    if k == old_topics:
      temp = (alpha + counts_topics_per_docs[d, k] - 1) \
               *(eta + counts_words_per_topics[w, k] - 1) \
                  / (V * eta + counts_per_topics[k] - 1)
    else:
      temp = (alpha + counts_topics_per_docs[d, k]) \
               *(eta + counts_words_per_topics[w, k]) \
                  / (V * eta + counts_per_topics[k])
    if temp > 0: rate_cum += temp
    rate[k] = rate_cum

  cdef double r = rands[local_index % n_rand] * rate_cum
  cdef int new_topics = searchsorted(rate, K, r)

  local_params[local_index] = new_topics
  inc(counts_words_per_topics[w, new_topics])
  inc(counts_topics_per_docs[d, new_topics])
  inc(counts_per_topics[new_topics])

  dec(change_vt_sum[w, old_topics])
  inc(change_vt_sum[w, new_topics])

  free(rate)

def sample_whole_iter(int[:] local_params, int K, int V, int n_rand, double alpha, double eta, int[:, :] counts_words_per_topics, int[:, :] counts_topics_per_docs, int[:] WS, int[:] DS, double[:] rands, int[:] counts_per_topics, int[:, :] change_vt_sum, N):

  # counts_per_docs = np.sum(counts_topics_per_docs)
  cdef int index, w, d, old_topics, new_topics, local_index
  cdef double rate_cum, temp, r
  cdef double* rate = <double*> malloc(K * sizeof(double))
  # for m in range(counts_topics_per_docs.shape[0]):
  #   for n in range(int(counts_per_docs[m])):
  for local_index in range(N):
      # local_index = np.sum(counts_per_docs[:m])+n
      w = WS[local_index]
      d = DS[local_index]
      old_topics = local_params[local_index]
      rate_cum = 0

      dec(counts_words_per_topics[w, old_topics])
      dec(counts_topics_per_docs[d, old_topics])
      dec(counts_per_topics[old_topics])

      for k in range(K):
        # if k == old_topics:
        #   temp = (alpha + counts_topics_per_docs[d, k] - 1) \
        #            *(eta + counts_words_per_topics[w, k] - 1) \
        #               / (V * eta + counts_per_topics[k] - 1)
        # else:
        #   temp = (alpha + counts_topics_per_docs[d, k]) \
        #            *(eta + counts_words_per_topics[w, k]) \
        #               / (V * eta + counts_per_topics[k])
        temp = (alpha + counts_topics_per_docs[d, k]) \
                 *(eta + counts_words_per_topics[w, k]) \
                    / (V * eta + counts_per_topics[k])
        if temp > 0: rate_cum += temp
        rate[k] = rate_cum

      r = rands[local_index % n_rand] * rate_cum
      new_topics = searchsorted(rate, K, r)

      local_params[local_index] = new_topics
      inc(counts_words_per_topics[w, new_topics])
      inc(counts_topics_per_docs[d, new_topics])
      inc(counts_per_topics[new_topics])

      dec(change_vt_sum[w, old_topics])
      inc(change_vt_sum[w, new_topics])

  free(rate)

cdef extern from "gamma.h":
    cdef double lda_lgamma(double x) nogil


cdef double lgamma(double x) nogil:
    if x <= 0:
        with gil:
            raise ValueError("x must be strictly positive")
    return lda_lgamma(x)

cpdef double _loglikelihood(int[:, :] nzw, int[:, :] ndz, int[:] nz, int[:] nd, double alpha, double eta) nogil:
    cdef int k, d
    cdef int D = ndz.shape[0]
    cdef int n_topics = ndz.shape[1]
    cdef int vocab_size = nzw.shape[1]

    cdef double ll = 0

    # calculate log p(w|z)
    cdef double lgamma_eta, lgamma_alpha
    with nogil:
        lgamma_eta = lgamma(eta)
        lgamma_alpha = lgamma(alpha)

        ll += n_topics * lgamma(eta * vocab_size)
        for k in range(n_topics):
            ll -= lgamma(eta * vocab_size + nz[k])
            for w in range(vocab_size):
                # if nzw[k, w] == 0 addition and subtraction cancel out
                if nzw[k, w] > 0:
                    ll += lgamma(eta + nzw[k, w]) - lgamma_eta

        # calculate log p(z)
        for d in range(D):
            ll += (lgamma(alpha * n_topics) -
                    lgamma(alpha * n_topics + nd[d]))
            for k in range(n_topics):
                if ndz[d, k] > 0:
                    ll += lgamma(alpha + ndz[d, k]) - lgamma_alpha
        return ll
