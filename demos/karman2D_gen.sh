scriptPath="PhiFlow/demos/karman2D.py"

for ((i=1; i<=150; i++))
do
    rey=$(shuf -i 100-1000 -n 1)
    echo "Reynolds number: $rey" >> 'data/400_100_450_cyl_visc/karman2D_gen.log'
    python $scriptPath \
    --dataDir 'data/400_100_450_cyl_visc' \
    --randomParams \
    --res_x 400 \
    --res_y 100 \
    --domain_x 4.0 \
    --domain_y 1.0 \
    --cyl_size 0.3 \
    --steps 450 \
    --reynolds_start $rey  \
    --reynolds_end $rey

done