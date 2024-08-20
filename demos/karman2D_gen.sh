scriptPath="PhiFlow/demos/karman2D.py"

for ((i=1; i<=200; i++))
do
    rey=$(shuf -i 100-1000 -n 1)
    echo "Reynolds number: $rey" >> 'data/600_varCyl_boundary/karman2D_gen.log'
    python $scriptPath \
    --dataDir 'data/600_varCyl_boundary' \
    --randomParams \
    --reynolds_start $rey  \
    --reynolds_end $rey  \
    --steps 600

done