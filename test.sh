#!/bin/bash
<< 'MULTILINE-COMMENT'
for(( i=2; i<=16; i*=2 ))
do
    touch ./result/avg_w$i.txt
    touch ./result/grad_w$i.txt
done

for (( i=5; i<=10; i*=2 ))
do
    touch ./result/avg_niid_w${i}.txt
    touch ./result/grad_niid_w${i}.txt
    touch ./result/avg_w${i}.txt
    touch ./result/grad_w${i}.txt
done

touch ./result/benchmark.txt

for(( i=2; i<=16; i*=2 ))
do
    python fedavg.py "--workers=${i}" > ./result/avg_w${i}.txt
    python fedgrad.py "--workers=${i}" > ./result/grad_w${i}.txt
done
MULTILINE-COMMENT

for (( i=5; i<=10; i*=2 ))
do
    python fedavg.py "--workers=${i}" "--iid=0" > ./result/avg_niid_w${i}.txt
    python fedgrad.py "--workers=${i}" "--iid=0" > ./result/grad_niid_w${i}.txt
    python fedavg.py "--workers=${i}" > ./result/avg_w${i}.txt
    python fedgrad.py "--workers=${i}" > ./result/grad_w${i}.txt
done

python benchmark.py > ./result/benchmark.txt
