#!/bin/bash

for(( i=2; i<=16; i*=2 ))
do
    touch ./result/avg_w$i.txt
    touch ./result/grad_w$i.txt
done

for(( i=2; i<=16; i*=2 ))
do
    $(python fedavg.py --workers=$i) 2>&1 ./result/avg_w$i.txt
    $(python fedgrad.py --workers=$i) 2>&1 ./result/grad_w$i.txt
done

touch ./result/benchmark.txt
python benchmark.py 2>&1 ./result/benchmark.txt
