#!/bin/bash
<< 'MULTILINE-COMMENT'
for(( i=2; i<=16; i*=2 ))
do
    touch ./result/w${i}.txt
    touch ./result/g${i}.txt
done

for (( i=5; i<=10; i*=2 ))
do
    touch ./result/w_niid_w${i}.txt
    touch ./result/g_niid_w${i}.txt
    touch ./result/w${i}.txt
    touch ./result/g${i}.txt
done

touch ./result/benchmark.txt

touch ./result/w2_ub149.txt
touch ./result/w2_ub545.txt
touch ./result/w2_ub1040.txt
touch ./result/w2_ub2030.txt
touch ./result/g2_ub149.txt
touch ./result/g2_ub545.txt
touch ./result/g2_ub1040.txt
touch ./result/g2_ub2030.txt

touch ./result/w4_ub55535.txt
touch ./result/w4_ub5101520.txt
touch ./result/w4_ub552020.txt
touch ./result/w4_ub12344.txt
touch ./result/g4_ub55535.txt
touch ./result/g4_ub5101520.txt
touch ./result/g4_ub552020.txt
touch ./result/g4_ub212344.txt
MULTILINE-COMMENT

for(( i=2; i<=16; i*=2 ))
do
    python fedweights.py "--workers=${i}" > ./result/w${i}.txt
    #python fedgradients.py "--workers=${i}" > ./result/g${i}.txt
done


for (( i=5; i<=10; i*=2 ))
do
    #python fedweights.py "--workers=${i}" "--iid=0" > ./result/w_niid_w${i}.txt
    #python fedgradients.py "--workers=${i}" "--iid=0" > ./result/g_niid_w${i}.txt
    python fedweights.py "--workers=${i}" > ./result/w${i}.txt
    #python fedgradients.py "--workers=${i}" > ./result/g${i}.txt
done

<< 'MULTILINE-COMMENT'
python benchmark.py > ./result/benchmark.txt
MULTILINE-COMMENT
python fedweights.py "--workers=2" "--proportion=[1000,49000]" > ./result/w2_ub149.txt
python fedweights.py "--workers=2" "--proportion=[5000,45000]" > ./result/w2_ub545.txt
python fedweights.py "--workers=2" "--proportion=[10000,40000]" > ./result/w2_ub1040.txt
python fedweights.py "--workers=2" "--proportion=[20000,30000]" > ./result/w2_ub2030.txt
<< 'MULTILINE-COMMENT'
python fedgradients.py "--workers=2" "--proportion=[1000,49000]" > ./result/g2_ub149.txt
python fedgradients.py "--workers=2" "--proportion=[5000,45000]" > ./result/g2_ub545.txt
python fedgradients.py "--workers=2" "--proportion=[10000,40000]" > ./result/g2_ub1040.txt
python fedgradients.py "--workers=2" "--proportion=[20000,30000]" > ./result/g2_ub2030.txt
MULTILINE-COMMENT

python fedweights.py "--workers=4" "--proportion=[5000,5000,5000,35000]" > ./result/w4_ub55535.txt
python fedweights.py "--workers=4" "--proportion=[1000,2000,3000,44000]" > ./result/w4_ub12344.txt
python fedweights.py "--workers=4" "--proportion=[5000,5000,20000,20000]" > ./result/w4_ub552020.txt
python fedweights.py "--workers=4" "--proportion=[5000,10000,15000,20000]" > ./result/w4_ub5101520.txt
<< 'MULTILINE-COMMENT'
python fedgradients.py "--workers=4" "--proportion=[5000,5000,5000,35000]" > ./result/g4_ub55535.txt
python fedgradients.py "--workers=4" "--proportion=[1000,2000,3000,44000]" > ./result/g4_ub212344.txt
python fedgradients.py "--workers=4" "--proportion=[5000,5000,20000,20000]" > ./result/g4_ub552020.txt
python fedgradients.py "--workers=4" "--proportion=[5000,10000,15000,20000]" > ./result/g4_ub5101520.txt
MULTILINE-COMMENT