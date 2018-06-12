import os
from socket import *
import redis
import numpy as np
import io
import json
import time
import pickle
import sys

redis_db = redis.Redis(host="localhost", port=6379, db=0)
redis_db.flushdb()

host = ""
port = 13000
buf = 2048
addr = (host, port)
UDPSock = socket(AF_INET, SOCK_DGRAM)
UDPSock.bind(addr)

print(("Connect to UDP: ", port))
print(("Connect to redis port: ", 6379))
print("Waiting to receive messages...")

assert len(sys.argv) == 2
num_clients = int(sys.argv[1])
redis_db.set('n', str(num_clients))

exit_cnt = 0
redis_db.set("TIME", 0)
redis_db.set("id", str(0).encode())

rec_dict = {}
STALE = 2

start_time = time.time()
K = 20

global_vt = None

# Wait for all clients(workers) to connect
while len(list(rec_dict.keys())) < num_clients:
    (key_encode, addr) = UDPSock.recvfrom(buf)
    key = key_encode.decode()

    client_id = int(key.split('#')[1])
    print(client_id)
    change_str = redis_db.get(key)
    if global_vt is None:
        global_vt = np.fromstring(change_str, dtype=np.int32).reshape(-1, int(K))
    else:
        global_vt += np.fromstring(change_str, dtype=np.int32).reshape(-1, int(K))
    redis_db.set('open#'+str(client_id), 'n')
    rec_dict[addr] = 0
    print(("Recieved %s from %s" % (key_encode.decode(), addr)))
print('-'*20)
redis_db.set("vt", global_vt.ravel().tostring())
cur_time = time.time()
print("Waiting for all clients to connect and pre-proc: ", cur_time - start_time, 's')

p = 1
assert num_clients >= p
for i in range(p):
    redis_db.set('open#'+str(i), 'y')

# Broadcast a start time to all clients
curr_time = int(round(time.time() * 1000)) + 2 * 1000
print("RECIEVED ALL CLIENTS")
print(("BROADCASTING START TIME:", curr_time))
redis_db.set("TIME", curr_time)

redis_db.set("SERVERCLOCK", 0)
print(int(redis_db.get("SERVERCLOCK")))

i, j = 0, 0
server_clock = 0
while True:
    start_time = time.time()
    (key_encode, addr) = UDPSock.recvfrom(buf)
    key = key_encode.decode()

    if key == "exit":
        exit_cnt += 1
        if exit_cnt == num_clients:
            break;
        continue

    V, K, client_id= key.split('#')
    change_str = redis_db.get(key)
    change = np.fromstring(change_str, dtype=np.int32).reshape(int(V), int(K))
    global_vt += change

    redis_db.set('open#'+str(client_id), 'n')
    next_id = int((int(client_id) + p)%num_clients)
    redis_db.set('open#'+str(next_id), 'y')

    redis_db.set("vt", global_vt.ravel().tostring())
    rec_dict[addr] = rec_dict.get(addr) + 1
    redis_db.set("SERVERCLOCK", min(rec_dict.values()))

    i += 1

#file_name = 'results/n=10,p=1,time=200/final_global_vt'
#np.save(file_name, global_vt)
UDPSock.close()

os._exit(0)
