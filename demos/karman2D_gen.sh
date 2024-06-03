scriptPath="demos/karman2D.py"

# python $scriptPath OUTDIR  RES_X RES_Y DT  STEPS WARMUP  CYL_SIZE VEL VISC  REYNOLDS_START REYNOLDS_END
reynolds=$(seq 1000 -4 200)
for rey in $reynolds;
do
    python $scriptPath 1300_inc  256 128 0.05  1300 20  0.6 0.5 -  $rey $rey
done