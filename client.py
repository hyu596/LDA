import os
from socket import *
import tensorflow as tf
import json
import io
import numpy as np
import redis
import pause
import datetime
from model import *
import re
import time
from scipy.special import gammaln
import datetime
import utils
import pre_proc

############################Connect to Server############################

# connect_time = time.time() # time to connect the server
# Config variables
host = "192.168.0.42" # set to IP address of target computer
port = 13000
addr = (host, port)
UDPSock = socket(AF_INET, SOCK_DGRAM)

data_size = 30000
num_batches = 30000 // 40

POOL = redis.ConnectionPool(host='localhost', port=6379, db=0)
my_server = redis.Redis(connection_pool=POOL)

# connect_time = time.time() - connect_time

###############Receive and Pre-process inital data#######################

# 1. Receive the partial dataset and client_id from server
# 2. Assign initial topics and compute the counting matrix
# pre_time = time.time()
CLOCK = 0
STALE = 0
# MODEL = None
MODEL_STATE = 2
#MODEL_KEY = "W"

# client id
client_id = get_client_id()

local_X = generators(client_id)
print(local_X.shape)
local_D = local_X.shape[0]

def generate_topics(K):
        while True:
            for i in range(K):
                yield i
samples = generate_topics(K)

# WS, DS, local_params = [], [], []
# global_vocab_topics, global_doc_topics = np.zeros((V, K), np.int32), np.zeros((local_D, K), np.int32)
# for m in range(local_D):
#     for n in range(V): # V
#         j = 0
#         while j < local_X[m, n]:
#             new_top = next(samples)
#             DS += [m]
#             WS += [n]
#             local_params += [new_top]
#             global_vocab_topics[n, new_top] += 1
#             global_doc_topics[m, new_top] += 1
#             j += 1
# WS, DS, local_params = np.asarray(WS).astype(np.int32), np.asarray(DS).astype(np.int32), np.asarray(local_params).astype(np.int32)

WS, DS, local_params, global_vocab_topics, global_doc_topics = pre_proc.assign_topics(local_X, K, samples)
WS, DS, local_params = np.asarray(WS).astype(np.int32), np.asarray(DS).astype(np.int32), np.asarray(local_params).astype(np.int32)
global_vocab_topics, global_doc_topics = np.asarray(global_vocab_topics).astype(np.int32), np.asarray(global_doc_topics).astype(np.int32)

vt_dtype = str(global_vocab_topics.dtype)
dt_dtype = str(global_doc_topics.dtype)

scores, times = [], []
avg_sampling_rate_time, avg_sampling_time = 0, 0
j = 0

_rands = np.random.mtrand._rand.rand(1024**2 // 8)
n_rand = np.int32(_rands.shape[0])
N_local = np.sum(global_doc_topics)

# pre_time = time.time() - pre_time

#####################Share the global parameter##########################

# right after each workers have completed counting,
# accumulate all the counting matrix and
# each workers should have the same copied version at the beginning
# share_time = time.time()

key = "dt#" + str(client_id)
my_server.set(key, global_doc_topics.ravel().tostring())
my_server.set("vt#"+str(client_id), global_vocab_topics.ravel().tostring())

UDPSock.sendto(("vt#"+str(client_id)).encode(), addr)
while int(my_server.get("TIME")) == 0:
    pass
start_time = float(my_server.get("TIME")) / 1000.0
start_time_local = time.time()

print("STARTING @:", start_time)
pause.until(start_time)
global_vocab_topics = np.fromstring(my_server.get("vt"), dtype=vt_dtype).reshape((V, K))

# share_time = time.time() - share_time

# Fetches the vocab-topic table from the PS if nessecary,
# otherwise returns the local model
def getModel():
    global global_vocab_topics
    global MODEL_STATE
    # global change_dt, global_doc_topics
    if (not CLOCK == 0) or MODEL_STATE < CLOCK - STALE:
        # waiting_load_time = time.time()
        while CLOCK != MODEL_STATE or MODEL_STATE < CLOCK:
            MODEL_STATE = int(my_server.get("SERVERCLOCK"))
        # waiting_load_time = time.time() - waiting_load_time
            # MODEL = np.fromstring(my_server.get("vt"), dtype=vt_dtype).reshape((V, K))
        global_vocab_topics = np.fromstring(my_server.get("vt"), dtype=vt_dtype).reshape((V, K))
        # return global_vocab_topics, waiting_load_time
        return global_vocab_topics, None
    else:
        # Model is still good, return it
        return global_vocab_topics, None

########################Sampling starts here#############################

# pure_sampling_time, loglikelihood_time, sending_time, waiting_time, load_time = 0, 0, 0, 0, 0
# whole_time = time.time()

for i in range(num_epochs):

    # waiting_start = time.time()
    while my_server.get('open#'+str(client_id)) == 'n':
        pass
    # waiting_time += time.time() - waiting_start

    if i%10 == 0:
        # send_start = time.time()
        my_server.set(key, global_doc_topics.ravel().tostring())
        # sending_time += time.time() - send_start

        # log_start = time.time()
        score = model_calc_loglikelihood()
        # loglikelihood_time += time.time() - log_start

        print("iteration: ", i, " loglikelihood: ", \
                score, " time: ", time.time()-start_time_local)
        scores += [score]
        # print(i, time.time() - start_time_local)

    change_vt_sum = np.zeros_like(global_vocab_topics, np.int32)

    rands = np.copy(_rands)
    np.random.mtrand._rand.shuffle(rands)

    counts_per_topics = np.sum(global_doc_topics, axis=0).astype(np.int32)

    # sample_start = time.time()
    utils.sample_whole_iter(local_params, K, V, n_rand, alpha, eta, global_vocab_topics, global_doc_topics, WS, DS, rands, counts_per_topics, change_vt_sum, N_local)
    # pure_sampling_time += time.time() - sample_start

    # send_start = time.time()
    key_change = '{0}#{1}#{2}'.format(V, K, client_id)
    my_server.set(key_change, change_vt_sum.ravel().tostring())
    UDPSock.sendto(key_change.encode(), addr)
    CLOCK += 1
    # sending_time += time.time() - send_start

    # load_start = time.time()
    global_vocab_topics, waiting_load_time = getModel()
    # load_time += time.time() - load_start - waiting_load_time
    # waiting_time += waiting_load_time

# whole_time = time.time() - whole_time

print("DONE")
print('-'*20)

# print("connect_time", connect_time)
# print("pre_time", pre_time)
# print("share_time", share_time)
# print("pure_sampling_time", pure_sampling_time)
# print("loglikelihood_time", loglikelihood_time)
# print("sending_time", sending_time)
# print("waiting_time", waiting_time)
# print("load_time", load_time)
# print("whole_time", whole_time)

np.save('results/topics/client'+str(client_id), local_params)
np.save('results/dt/client'+str(client_id), global_doc_topics)
np.save('results/logli/client'+str(client_id), scores)


UDPSock.sendto("exit".encode(), addr)
UDPSock.close()
