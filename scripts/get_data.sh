#!/bin/bash

# Note: if wget doesn't work, try with curl 
wget http://www.robots.ox.ac.uk/~vgg/data/flowers/17/17flowers.tgz
#curl -LO http://www.robots.ox.ac.uk/~vgg/data/flowers/17/17flowers.tgz
tar -xvf 17flowers.tgz
mkdir dataset/flowers/
mkdir dataset/flowers/train
mkdir dataset/flowers/test
mv jpg/* dataset/flowers/
rm -r jpg
rm 17flowers.tgz
for i in $(seq 1 17); 
    do 
    mkdir dataset/flowers/test/$i;
    mkdir dataset/flowers/train/$i;  
done
for f in $(ls dataset/flowers | grep jpg); 
    do
    n=${f:6:4}
    s=train
    b=$(expr $n % 5)
    adjc=$(expr $n - 1)
    c=$(expr $adjc / 80 + 1);
    if [[ $b -eq 0 ]]
    then
        s=test
    fi
    mv dataset/flowers/$f dataset/flowers/$s/$c/$f
done
