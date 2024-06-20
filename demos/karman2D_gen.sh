scriptPath="demos/karman2D.py"

# python $scriptPath OUTDIR  RES_X RES_Y DT  STEPS WARMUP  CYL_SIZE VEL VISC  REYNOLDS_START REYNOLDS_END
for ((i=1; i<=400; i++))
do
    python $scriptPath
done