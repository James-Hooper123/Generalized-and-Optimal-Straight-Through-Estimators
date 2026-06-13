#!/bin/bash

# Runs polynomial programming (integer embedding) for various p and estimators

for p in 1.5 2.0 3.0; do
    for est in ST reinmax STGS GRMC-20 MVE; do
        echo "========================================="
        echo "    Running INT p=$p with estimator=$est "
        echo "========================================="
        python poly_p_int.py --method $est --pnorm $p
    done
done
