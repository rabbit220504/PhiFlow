import argparse
import os
import numpy as np
from pyevtk.hl import imageToVTK


def np2vtk(experiment_dir, velNp, presNp, time_step):
    if not os.path.exists(experiment_dir):
        os.mkdir(experiment_dir)
    u_component = np.expand_dims(velNp[0], axis=-1)
    v_component = np.expand_dims(velNp[1], axis=-1)
    p_component = np.expand_dims(presNp[0], axis=-1)
    imageToVTK(
        f'{experiment_dir}/timestep_{time_step}',
        pointData = {
            "u": np.ascontiguousarray(u_component),
            "v": np.ascontiguousarray(v_component),
            "p": np.ascontiguousarray(p_component)
        }
    )   





   
