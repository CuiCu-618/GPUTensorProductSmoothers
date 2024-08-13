#!/bin/bash                                                                                                                           

make release

for p in 2
do
    echo "/////////////////////"
    echo "Starting 3D degree $p"
    echo "/////////////////////"
    python3 scripts/ct_parameter.py -DIM 3 -DEG $p -N_STAGES 4 -DT 0.1 -ENDT 0.5 -REDUCE 1e-9 -REDUCE_INNER 1e-6 -K FUSED_L -MAXSIZE 5000000 -VNUM double
    make heat  
    ./apps/heat
done
