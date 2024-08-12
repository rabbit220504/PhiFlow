#!/bin/bash

REY_START=100
REY_END=1000
REY_STEP=100

for ((REY=$REY_START; REY<=$REY_END; REY+=$REY_STEP))
do
    echo "Running simulation with Reynolds number: $REY"
    python PhiFlow/demos/karman2D.py \
        --dataDir 'data/600_varCyl_boundary_varRay' \
        --reynolds_start $REY --reynolds_end $REY
    echo "Simulation completed for Reynolds number: $REY"
done

echo "All simulations completed."

