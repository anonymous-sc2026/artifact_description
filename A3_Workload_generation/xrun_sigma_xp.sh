#!/bin/sh

SKIP_LINES="0,50000,100000,150000,200000,250000,300000,350000,400000,450000,500000,550000,600000,650000,700000,750000,800000,850000,900000,950000,1000000"
for P in 30 90
do
    ./serving_traces.py --random-deadlines -m out-of-order-discard-most-urgent -a 1 -p $P -r 0.001,0.1 -f 2,3,5 -s 0.1,0.2,0.3,0.6,0.9,1 -n 100 -l "$SKIP_LINES" traces/BurstGPT_new_header.csv
done
