import tensorflow as tf
import numpy as np
import math
import re
from scipy.special import gammaln
import redis

import lda
import lda.datasets

import utils
import pre_proc

num_epochs = 1000

record_defaults = [[0.0] for _ in range(15)]
_CSV_COLS = [ "c" + str(i) for i in range(0, 15)]

POOL = redis.ConnectionPool(host='localhost', port=6379, db=0)
my_server = redis.Redis(connection_pool=POOL)

# At this point, still using the package dataset
#   1. X, valid, title: data
#   2. topics: latent variables
def input_fn(K, filename=None):
    if filename == None:
        X = lda.datasets.load_reuters()
        valid = list(lda.datasets.load_reuters_vocab())
        titles = list(lda.datasets.load_reuters_titles())
    return X, valid

def input_fn_ny(K):

    '''read data'''
    # contains index of words appearing in that document and the number of times they appear
    with open('nyt_data.txt') as f:
        documents = f.readlines()
    documents = [x.strip().strip('\n').strip("'") for x in documents]

    # contains vocabs with rows as index
    with open('nyt_vocab.dat') as f:
        vocabs = f.readlines()
    vocabs = [x.strip().strip('\n').strip("'") for x in vocabs]

    '''create matrix X'''
    numDoc = 8447
    numWord = 3012
    X = np.zeros([numWord,numDoc])

    for col in range(len(documents)):
        for row in documents[col].split(','):
            X[int(row.split(':')[0])-1,col] = int(row.split(':')[1])
    X = X.T.astype(int)
    return X, vocabs

K = np.int32(20)
num_clients = 2

X, valid = input_fn_ny(K)
# X = pre_proc.read_file('data/docword.nytimes.txt')
# X = np.asarray(X).astype(np.int32)

D, V = X.shape
D, V = np.int32(D), np.int32(V)
alpha, eta = .1, .01

total_docs = X.shape[0]

# counts_cum = [np.sum(np.sum(X, axis=1)[:i]) for i in range(X.shape[0])]
def generators(i):
    diff = int(np.ceil(X.shape[0]/num_clients))
    begin_index = i * diff
    if i != num_clients - 1:
        end_index = begin_index + diff
        return X[int(begin_index):int(end_index)]
    else:
        return X[int(begin_index):]

# def generators(i):
#     counts_cum = [np.sum(np.sum(X, axis=1)[:i]) for i in range(X.shape[0])]
#     diff = int(np.ceil(X.shape[0]/num_clients))
#     begin_index = i * diff
#     begin_acc_index = counts_cum[int(begin_index)]
#     if i != num_clients - 1:
#         end_acc_index = counts_cum[int(begin_index + diff)]
#         end_index = begin_index + diff
#         return np.copy(np.asarray(topics[int(begin_acc_index):int(end_acc_index)])), np.copy(np.asarray(indices_to_start[int(begin_acc_index):int(end_acc_index)])), np.copy(np.asarray(counts_topics_per_docs[int(begin_index):int(end_index)]))
#     else:
#         return np.copy(np.asarray(topics[int(begin_acc_index):])), np.copy(np.asarray(indices_to_start[int(begin_acc_index):])), np.copy(np.asarray(counts_topics_per_docs[int(begin_index):]))

def get_client_id():
    temp = int(my_server.get("id").decode())
    my_server.set("id", str(temp + 1).encode())
    # print(temp)
    return temp

def log_multinomial_beta(vec, s=None):

    if s:
        return s * gammaln(vec) - gammaln(s * vec)
    return np.sum(gammaln(vec)) - gammaln(np.sum(vec))

def model_calc_loglikelihood():

    # loglikelihood = 0
    # global_vt = np.fromstring(my_server.get("vt"), dtype=np.int32).reshape((V, K))
    # for k in range(K):
    #     loglikelihood += log_multinomial_beta(global_vt.T[k, :] + eta) \
    #                         - log_multinomial_beta(eta, V)
    #
    # for i in range(num_clients):
    #     temp_dt = np.fromstring(my_server.get("dt#"+str(i)), dtype=np.int32).reshape((-1, K))
    #     for n in range(len(temp_dt)):
    #         loglikelihood += log_multinomial_beta(temp_dt[n, :] + alpha) \
    #                             - log_multinomial_beta(alpha, K)

    global_vt = global_vt = np.fromstring(my_server.get("vt"), dtype=np.int32).reshape((V, K))
    global_vt = global_vt.astype(np.int32)

    global_dt = []
    for i in range(num_clients):
        global_dt.append(np.fromstring(my_server.get("dt#"+str(i)), dtype=np.int32).reshape((-1, K)))
    global_dt = np.vstack(global_dt).astype(np.int32)

    nd = np.sum(global_dt, axis=1).astype(np.int32)
    nt = np.sum(global_dt, axis=0).astype(np.int32)

    return utils._loglikelihood(global_vt.T, global_dt, nt, nd, alpha, eta)

# def evaluate_beta(counts_words_per_topics, counts_topics_per_docs, valid, K, V, eta):
#
#     beta_T = np.zeros((V, K))
#     counts_topics = np.sum(counts_topics_per_docs, axis=0)
#
#     for i, word in enumerate(valid):
#         beta_T[i] = (eta + counts_words_per_topics[valid.index(word)]) \
#                         / (V * eta + counts_topics)
#         beta_T[i] /= np.sum(beta_T[i])
#     return beta_T.T
#
# def most_frequent_10_words_num_topics(counts_words_per_topics, counts_topics_per_docs, valid, K, V, eta):
#
#     beta = evaluate_beta(counts_words_per_topics, counts_topics_per_docs, valid, K, V, eta)
#     valid_ = np.array(valid)
#
#
#     for k in range(K):
#         top = beta[k]
#         print("Topic ", k, " :", valid_[top.argsort()[-10:][::-1]])
