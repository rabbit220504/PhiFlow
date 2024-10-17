""" Karman vortex street
Simulates a viscous fluid flowing around a cylinder.
"""

import random
import os, sys, json, time, datetime
import imageio, matplotlib
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import numpy as np
np.bool = np.bool_
np.object = np.object_
np.int = np.int_
from phi.torch.flow import *
from phi.flow import *
phi.torch.TORCH.set_default_device("GPU")

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Simulation of fluid dynamics around a cylindrical obstacle.")
    parser.add_argument('--dataDir', type=str, default="data/400_100_450_cyl_visc", help='Directory for data storage.')
    parser.add_argument('--write', action='store_true', default=True)
    parser.add_argument('--readOnly', action='store_true', default=False)
    parser.add_argument('--readIdx', type=int, default=0)
    parser.add_argument('--render', action='store_true', default=False)
    parser.add_argument('--writeImageSequence', action='store_true', default=False)
    parser.add_argument('--batch', action='store_true', default=False)
    parser.add_argument('--randomParams', action='store_true', default=False)
    parser.add_argument('--preview', action='store_true', default=True)
    parser.add_argument('--res_x', type=int, default=512, help='Resolution in x-direction.')
    parser.add_argument('--res_y', type=int, default=128, help='Resolution in y-direction.')
    parser.add_argument('--domain_x', type=float, default=8.0, help='Domain size in x-direction.')
    parser.add_argument('--domain_y', type=float, default=2.0, help='Domain size in y-direction.')
    parser.add_argument('--dt', type=float, default=0.05)
    parser.add_argument('--steps', type=int, default=10, help='Number of simulation steps.')
    parser.add_argument('--warmup', type=int, default=0, help='Number of warmup steps.')
    parser.add_argument('--cyl_size', type=float, default=0.5, help='Cylinder size.')
    parser.add_argument('--cyl_num', type=int, default=1, help='Number of cylinders.')
    parser.add_argument('--cyl_x', type=float, default=1/4, help='Cylinder x-coordinate.')
    parser.add_argument('--cyl_y', type=float, default=1/2, help='Cylinder y-coordinate.')
    parser.add_argument('--vel_in', type=float, default=0.5, help='Inlet velocity.')
    parser.add_argument('--visc_start', type=float, default=5e-4, help='Initial viscosity.')
    parser.add_argument('--visc_end', type=float, default=5e-4, help='Final viscosity.')
    parser.add_argument('--reynolds_start', type=int, default=1000, help='Initial Reynolds number.')
    parser.add_argument('--reynolds_end', type=int, default=1000, help='Final Reynolds number.')
    parser.add_argument('--batchSize', type=int, default=3, help='Batch size for simulation.')

    return parser.parse_args()

args = parse_args()
dataDir = args.dataDir
write = args.write
readOnly, readIdx = args.readOnly, args.readIdx
render = args.render
writeImageSequence = args.writeImageSequence
print(args)

BATCH = args.batch
RANDOM_PARAMS = args.randomParams
PREVIEW = args.preview
RES_X, RES_Y = args.res_x, args.res_y
DOMAIN_X = args.domain_x
DOMAIN_Y = args.domain_y
DT = args.dt
STEPS, WARMUP = args.steps, args.warmup
CYL_SIZE = args.cyl_size
CYL_NUM = args.cyl_num
WALL_TOP, WALL_BOTTOM = (1/2)*(DOMAIN_Y - CYL_SIZE), (1/2)*(DOMAIN_Y - CYL_SIZE)
WALL_LEFT, WALL_RIGHT = (1/8)*(DOMAIN_X - CYL_SIZE), (7/8)*(DOMAIN_X - CYL_SIZE)
VEL_IN = args.vel_in
VEL = VEL_IN
batchSize = args.batchSize
REYNOLDS_START = args.reynolds_start
REYNOLDS_END = args.reynolds_end

if REYNOLDS_START == 0 or REYNOLDS_END == 0: # if don't specify the Reynolds number, calculate it
    VISC_START = args.visc_start
    VISC_END = args.visc_end
    REYNOLDS_START = (VEL * CYL_SIZE) / VISC_START
    REYNOLDS_END = (VEL * CYL_SIZE) / VISC_END

gui = "console"

if BATCH:
    NP_NAMES = "batch,vector,x,y"
    TRANSPOSE = [0,3,2,1]
else:
    NP_NAMES = "vector,x,y"
    TRANSPOSE = [2,1,0]
#

### PARAMETER SAMPLING
if RANDOM_PARAMS:
    # uncomment this when more than one cylinder is desired
    # CYL_NUM = torch.randint(1, 4, (1,)).item()
    # CYL_SIZE = random.uniform(0.3, 0.6)

    WALL_TOP, WALL_BOTTOM = (1/2)*(DOMAIN_Y - CYL_SIZE), (1/2)*(DOMAIN_Y - CYL_SIZE)
    WALL_LEFT, WALL_RIGHT = (1/8)*(DOMAIN_X - CYL_SIZE), (7/8)*(DOMAIN_X - CYL_SIZE)
    
    # locations
    BUFFER_V = DOMAIN_Y * 1/8  # buffer on the top and bottom
    BUFFER_HL = DOMAIN_X * 1/16  # buffer on the left
    BUFFER_HR = DOMAIN_X * 9/16  # buffer on the right
    MAX_ITERATIONS = 1000
    cyl_locations = []
    for _ in range(CYL_NUM):
        iterations = 0
        while iterations < MAX_ITERATIONS:
            x = random.uniform(BUFFER_HL + (CYL_SIZE/2), DOMAIN_X - BUFFER_HR - (CYL_SIZE/2))
            y = random.uniform(BUFFER_V + (CYL_SIZE/2), DOMAIN_Y - BUFFER_V - (CYL_SIZE/2))
            
            overlap = False
            for loc in cyl_locations:
                if (x - loc[0])**2 + (y - loc[1])**2 < CYL_SIZE**2:
                    overlap = True
                    break
            
            if not overlap:
                cyl_locations.append((x, y))
                break

            iterations += 1
        
        if iterations == MAX_ITERATIONS:
            print("Failed to find non-overlapping cylinder location")
            sys.exit(1)

if not RANDOM_PARAMS:
    cyl_locations = [(args.cyl_x * DOMAIN_X, args.cyl_y * DOMAIN_Y + 1e-2)]
print("cylinder locations determined")

# viscosity
VISC_START = CYL_SIZE * VEL / REYNOLDS_END
VISC_END = VISC_START
#

print("--------------------------------------------")
print("| Resolution: (%d, %d)" % (RES_X, RES_Y))
print("| Dt: %1.3f" % (DT))
print("| Steps (Warmup): %d (%d)" % (STEPS, WARMUP))
print("| Cylinder Size: %1.3f" % (CYL_SIZE))
print("| Inflow Velocity: %1.3f" % (VEL))
print("| Fluid Viscosity: (%1.8f, %1.8f)" % (VISC_START, VISC_END))
print("| REYNOLDS NUMBER: (%d, %d)" % (REYNOLDS_START, REYNOLDS_END))
print("| Cylinder Number: %d" % (CYL_NUM))
print("| Cylinder Locations: %s" % (cyl_locations))
print("--------------------------------------------\n")



### SCENE SETUP
scene = Scene.create(dataDir) if not readOnly else Scene.at(dataDir, readIdx)

DOMAIN = dict(x=RES_X, y=RES_Y, bounds=Box[0:WALL_LEFT + CYL_SIZE + WALL_RIGHT, 0:WALL_BOTTOM + CYL_SIZE + WALL_TOP])
extr = extrapolation.combine_sides(x=extrapolation.BOUNDARY, y=extrapolation.BOUNDARY)
if BATCH:
    # velocity = StaggeredGrid(math.zeros(batch(batch=batchSize)), extrapolation=extr, **DOMAIN)
    
    # velocity_tensor = math.random_uniform(batch(batch=batchSize))
    # velocity_tensor.device = "cuda"
    # velocity_np = velocity_tensor.numpy()
    # velocity = StaggeredGrid((velocity_np[0],0), extrapolation=extr, **DOMAIN)

    velocity = StaggeredGrid(math.random_normal(batch(batch=batchSize)), extrapolation=extr, **DOMAIN)
else:
    # velocity = StaggeredGrid((VEL,0), extrapolation=extr, **DOMAIN)
    velocity = StaggeredGrid((0,0), extrapolation=extr, **DOMAIN)
pressure = None
BOUNDARY_MASK = StaggeredGrid(HardGeometryMask(Box[:0.2*CYL_SIZE, :]), extrapolation=extr, **DOMAIN)
RESAMPLING_CENTERED = CenteredGrid(0, extrapolation=extr, **DOMAIN)
RESAMPLING_STAGGERED = StaggeredGrid(math.zeros(channel(vector=2)), extrapolation=extr, **DOMAIN)

# print("velocity shape", (velocity @ RESAMPLING_CENTERED).values.numpy(NP_NAMES).shape)
# print("velocity[0] values", (velocity @ RESAMPLING_CENTERED).values.numpy(NP_NAMES)[0,...])
# print("velocity[1] values", (velocity @ RESAMPLING_CENTERED).values.numpy(NP_NAMES)[1,...])
# print("velocity[2] values", (velocity @ RESAMPLING_CENTERED).values.numpy(NP_NAMES)[2,...])

# print("shape of boundary mask", (BOUNDARY_MASK @ RESAMPLING_CENTERED).values.numpy(NP_NAMES).shape)
# print("inside the boundary", (BOUNDARY_MASK @ RESAMPLING_CENTERED).values.numpy(NP_NAMES)[...,0,0])
# print("outside the boundary", (BOUNDARY_MASK @ RESAMPLING_CENTERED).values.numpy(NP_NAMES)[...,100,20])

# single cylinder
# OBSTACLE = Obstacle(Sphere(center=(WALL_LEFT + 0.5*CYL_SIZE, WALL_BOTTOM + 0.5*CYL_SIZE), radius=0.5*CYL_SIZE))
# multiple cylinders
OBSTACLE_GEOMETRIES = [Sphere(center=(cyl_locations[i][0], cyl_locations[i][1]), radius=0.5*CYL_SIZE) for i in range(len(cyl_locations))]
OBSTACLE = Obstacle(union(OBSTACLE_GEOMETRIES))

OBS_MASK = StaggeredGrid(OBSTACLE.geometry, extrapolation=extrapolation.ZERO, **DOMAIN)

# Initialize velocity, only use for warm up!
if BATCH:
    VEL_INIT = np.ones((batchSize, 2, RES_X, RES_Y))
    VEL_INIT = np.random.uniform(0.5, 2, (batchSize, 2, RES_X, RES_Y))
    # for _ in range(0, batchSize):
    #     VEL_INIT[_] = VEL_INIT[_] + (_ * 0.05)

    VEL_INIT = CenteredGrid(tensor(VEL_INIT, batch("batch"), channel("vector"), spatial("x", "y")),extrapolation=extr, **DOMAIN)

else:
    # different initial velocity options
    # VEL_INIT = 0.5 * (np.cos( math.PI * np.arange(0,1,1/RES_Y)[None,...]) + 1) # cosine
    VEL_INIT = 1 * np.ones((1,RES_Y))   # constant
    VEL_INIT = np.repeat(VEL_INIT, RES_X, axis=0)   
    VEL_INIT = np.stack([VEL_INIT, 2*np.ones_like(VEL_INIT)], axis=0)
    VEL_INIT = CenteredGrid(tensor(VEL_INIT, channel("vector"), spatial("x", "y")),extrapolation=extr, **DOMAIN)
VEL_INIT = StaggeredGrid(VEL_INIT @ RESAMPLING_STAGGERED, extrapolation=extr, **DOMAIN)
#

### MAIN LOOP
viewer = view("velocity,pressure,VEL_INIT", namespace=globals(), select="batch", gui=gui, keep_alive=(gui is "dash"))
time.sleep(0.01)
print()
print(scene.path)

recVel = []
recPres = []
recVisc = []
recRey = []

for step in viewer.range(STEPS):
    if step%10 == 0:
        print("\t%s Frame %04d" % ("Reading" if readOnly else "Simulating", step))

    startReyChange = 300
    if REYNOLDS_START == REYNOLDS_END or step < startReyChange:
        visc = VISC_START
        rey = REYNOLDS_START
    else:
        visc = VISC_START + (float(step-startReyChange) / float(STEPS-startReyChange-1)) * (VISC_END - VISC_START)
        rey = REYNOLDS_START + (float(step-startReyChange) / float(STEPS-startReyChange-1)) * (REYNOLDS_END - REYNOLDS_START)
    recVisc += [visc]
    recRey += [rey]

    if not readOnly:
        # simulate
        velocity = advect.mac_cormack(velocity, velocity, DT)
        #velocity = advect.semi_lagrangian(velocity, velocity, DT)
        if step < WARMUP:
            velocity = velocity * (1 - BOUNDARY_MASK) + BOUNDARY_MASK * VEL * VEL_INIT
        else:
            velocity = velocity * (1 - BOUNDARY_MASK) + BOUNDARY_MASK * (VEL, 0)

        velocity, pressure = fluid.make_incompressible(velocity, (OBSTACLE,), Solve('auto', relative_tolerance=1e-5, absolute_tolerance=0, max_iterations=5000, x0=pressure))
        velocity = diffuse.explicit(velocity, visc, DT, substeps=int(max(2000*visc,1)))

        if PREVIEW:
            # preview image
            velNp = (velocity @ RESAMPLING_CENTERED).values.numpy(NP_NAMES)
            if step == 0:
                velNp = np.transpose(velNp, axes=TRANSPOSE)
                if BATCH:
                    for batch_idx in range(velNp.shape[0]):  # Batch dimension
                        for i in range(velNp.shape[-1]):
                            velPart = velNp[batch_idx, ..., i]
                            vMax = max(abs(np.min(velPart)), abs(np.max(velPart)))
                            vMin = -vMax
                            velPart = 255*((velPart - vMin) / (vMax - vMin))
                            imageio.imwrite(f"{scene.path}/preview_batch{batch_idx:02d}_step_{step}_{'X' if i == 0 else 'Y'}.png", velPart.astype(np.uint8))   
                else:
                    for i in range(velNp.shape[-1]):
                        velPart = velNp[..., i]
                        vMax = max(abs(np.min(velPart)), abs(np.max(velPart)))
                        vMin = -vMax
                        velPart = 255*((velPart - vMin) / (vMax - vMin))
                        imageio.imwrite(f"{scene.path}/preview_step_{step}_{'X' if i == 0 else 'Y'}.png", velPart.astype(np.uint8))
            #

        if write:
            # write simulation data
            velNp = (velocity @ RESAMPLING_CENTERED).values.numpy(NP_NAMES)
            presNp = pressure.values.numpy(NP_NAMES)

            # write velocity and pressure
            if BATCH:
                for batch_idx in range(velNp.shape[0]):
                    velNPBatch = velNp[batch_idx].astype(np.float32)
                    presNpBatch = presNp[batch_idx].astype(np.float32)
                    np.savez_compressed( os.path.join(scene.path, f"{batch_idx}_velocity_%06d.npz" % step), velNPBatch)
                    # np.savez_compressed( os.path.join(scene.path, f"{batch_idx}_pressure_%06d.npz" % step), presNpBatch)
                    # utils.np2vtk(os.path.join(scene.path,f'vtk_{batch_idx}'), velNPBatch, presNpBatch, step)
            else:
                velNp = velNp.astype(np.float32)
                presNp = presNp.astype(np.float32)
                np.savez_compressed( os.path.join(scene.path, "velocity_%06d.npz" % step), velNp)
                # np.savez_compressed( os.path.join(scene.path, "pressure_%06d.npz" % step), presNp)
                # utils.np2vtk(os.path.join(scene.pasth,f'vtk'), velNp, presNp, step)
            #

            # obstacle mask
            if not os.path.isfile(os.path.join(scene.path, "obstacle_mask.npz")):
                obsNp = (OBS_MASK @ RESAMPLING_CENTERED).values.numpy(NP_NAMES)
                obsNp = obsNp[0] <= 0
                np.savez_compressed(os.path.join(scene.path, "obstacle_mask.npz"), obsNp.astype(np.int))
                
    else:
        # read existing simulation data
        velNp = np.load(os.path.join(scene.path, "velocity_%06d.npz" % step))["arr_0"]
        presNp = np.load(os.path.join(scene.path, "pressure_%06d.npz" % step))["arr_0"]
        velGrid = CenteredGrid(tensor(velNp, channel("vector"), spatial("x", "y")), extrapolation=extr, **DOMAIN)
        velocity = StaggeredGrid(velGrid @ RESAMPLING_STAGGERED, extrapolation=extr, **DOMAIN)
        pressure = CenteredGrid(tensor(presNp, channel("vector"), spatial("x", "y")), extrapolation=extr, **DOMAIN)

    # recVel += [velNp]
    # recPres += [presNp]

# recVel = np.transpose(np.stack(recVel, axis=0), axes=[0,2,3,1])
# recPres = np.transpose(np.stack(recPres, axis=0), axes=[0,2,3,1])


### RENDERING
if render:
    print("\nRendering...")

    renderpath = os.path.join(scene.path, "render")
    if not os.path.exists(renderpath):
        os.makedirs(renderpath)
    if REYNOLDS_START == REYNOLDS_END:
        renderfile = "cyl%1.2f_vel%1.2f_visc%1.8f_rey%06d" % (CYL_SIZE, VEL, VISC_START, REYNOLDS_START)
    else:
        renderfile = "cyl%1.2f_vel%1.2f_rey%06d-rey%06d" % (CYL_SIZE, VEL, REYNOLDS_START, REYNOLDS_END)

    vx_dx, vx_dy = np.gradient(recVel[...,0][...,None], axis=(1,2))
    vy_dx, vy_dy = np.gradient(recVel[...,1][...,None], axis=(1,2))
    curl = vy_dx - vx_dy
    divergence = vx_dx + vy_dy

    renderdata = [[recVel[...,0][...,None],curl], [recVel[...,1][...,None],divergence], [recVel,recPres]]
    rendercmap = [["seismic","seismic"], ["seismic","coolwarm"], [None,"PuOr"]]

    pad = 8
    result = []
    for i in range(len(renderdata)):
        rows = []
        for j in range(len(renderdata[i])):
            part = np.copy(renderdata[i][j])
            part = np.rot90(part, axes=(1,2))
            cmap = rendercmap[i][j]
            if cmap:
                cmap = matplotlib.cm.get_cmap(cmap)

            for k in range(part.shape[-1]):
                pMax = max(abs(np.min(part[...,k])), abs(np.max(part[...,k])))
                pMin = -pMax
                #pMax = np.max(part[...,k])
                #pMin = np.min(part[...,k])
                part[...,k] = (part[...,k] - pMin) / (pMax - pMin)

            if part.shape[-1] == 1 and cmap:
                part = cmap(np.squeeze(part))

            if part.shape[-1] == 2:
                blue = np.zeros((part.shape[0], part.shape[1], part.shape[2], 1))
                alpha = np.ones_like(blue)
                part = np.concatenate([part, blue, alpha], axis=3)

            if part.shape[-1] == 3:
                alpha = np.ones((part.shape[0], part.shape[1], part.shape[2], 1))
                part = np.concatenate([part, alpha], axis=3)

            part = 255 * np.pad(part, ((0,0), (pad,pad), (pad,pad), (0,0)) )
            rows += [part.astype(np.uint8)]
        result += [np.concatenate(rows, axis=1)]
    result = np.concatenate(result, axis=2)

    vidfile = renderfile + ".mp4"
    imageio.mimwrite(os.path.join(renderpath, vidfile), result, quality=10, fps=int(1/DT), ffmpeg_log_level="error")
    if writeImageSequence:
        for i in range(0,result.shape[0],10):
            imgfile = "img_%06d_%s.png" % (i, renderfile)
            imageio.imwrite(os.path.join(renderpath, imgfile), result[i])
#

## LOGGING
if not readOnly:
    log = {}

    log["Timestamp"] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log["Resolution"] = [RES_X, RES_Y]
    log["Dt"] = DT
    log["Steps, Warmup"] = [STEPS, WARMUP]
    log["Cylinder Size"] = CYL_SIZE
    log["Cylinder Number"] = CYL_NUM
    log["Cylinder Locations"] = cyl_locations
    log["Walls (lrtb)"] = [WALL_LEFT, WALL_RIGHT, WALL_TOP, WALL_BOTTOM]
    log["Inflow Velocity"] = VEL
    log["Fluid Viscosity"] = VISC_START if REYNOLDS_START == REYNOLDS_END else recVisc
    log["Reynolds Number"] = REYNOLDS_START if REYNOLDS_START == REYNOLDS_END else recRey
    log["Stats"] = {"Velocity" : [], "Velocity Magnitude" : [], "Pressure" : []}

    recVelMag = np.linalg.norm(recVel, axis=-1)
    # for i in range(recVel.shape[0]):
    #     log["Stats"]["Velocity"].append( "Min:%2.8f Max:%2.8f Avg: %2.8f" % (np.min(recVel[i]), np.max(recVel[i]), np.mean(recVel[i])) )
    #     log["Stats"]["Velocity Magnitude"].append( "Min:%2.8f Max:%2.8f Avg: %2.8f" % (np.min(recVelMag[i]), np.max(recVelMag[i]), np.mean(recVelMag[i])) )
    #     log["Stats"]["Pressure"].append( "Min:%2.8f Max:%2.8f Avg: %2.8f" % (np.min(recPres[i]), np.max(recPres[i]), np.mean(recPres[i])) )

    if not readOnly:
        logFile = os.path.join(scene.path, "src", "description.json")
        with open(logFile, 'w') as f:
            json.dump(log, f, indent=4)
            f.close()
#

print("Simulation complete\n\n")

if gui == "console":
    os._exit(0)