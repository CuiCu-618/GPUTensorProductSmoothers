#!/bin/bash                                                                                                                           

make release

for p in 2
do
    echo "/////////////////////"
    echo "Starting 3D degree $p"
    echo "/////////////////////"
    python3 scripts/ct_parameter.py -DIM 3 -DEG $p -MAXSIZE 500000
    make heat  
    ./apps/heat
done
