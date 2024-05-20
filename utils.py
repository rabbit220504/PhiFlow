import argparse
import os
import numpy as np
from pyevtk.hl import imageToVTK


def vecNP2vtk(experiment_dir, fieldNP, time_step):
    if not os.path.exists(experiment_dir):
        os.mkdir(experiment_dir)
    field = fieldNP
    u_component = np.expand_dims(field[0], axis=-1)
    v_component = np.expand_dims(field[1], axis=2)
    imageToVTK(
        f'{experiment_dir}/timestep_{time_step}',
        pointData = {
            "u": np.ascontiguousarray(u_component),
            "v": np.ascontiguousarray(v_component)
        }
    )

def scalarNP2vtk(experiment_dir, fieldNP, time_step):
    if not os.path.exists(experiment_dir):
        os.mkdir(experiment_dir)
    field = fieldNP
    imageToVTK(
        f'{experiment_dir}/{time_step}',
        pointData = {
            "p": np.ascontiguousarray(field)
        }
    )    





   
