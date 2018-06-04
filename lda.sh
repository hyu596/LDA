#!/bin/sh

# $1: number of workers 

kill $(lsof -t -i:13000)

python3 param.py $1 &
parallel --no-notice python3 client.py ::: $(seq 1 $1) &
wait

echo all processes complete