import os
import sys
import torch
import numpy as np
import torch
import matplotlib.pyplot as plt

from pytorch3d.datasets import (
    R2N2,
    ShapeNetCore,
    collate_batched_meshes,
    render_cubified_voxels,
)
from pytorch3d.renderer import (
    OpenGLPerspectiveCameras,
    PointLights,
    RasterizationSettings,
    TexturesVertex,
    look_at_view_transform,
)

from pytorch3d.structures import Meshes
from torch.utils.data import DataLoader

# add path for demo utils functions
sys.path.append(os.path.abspath(''))
from plot_image_grid import image_grid

# Setup
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

SHAPENET_PATH = "/Data/leo/Pixel2Mesh_3d/dataset/ShapeNetCore.v2"
shapenet_dataset = ShapeNetCore(SHAPENET_PATH, version=2, synsets=["airplane"])

#shapenet_model = shapenet_dataset[6]
#print("This model belongs to the category " + shapenet_model["synset_id"] + ".")
#print("This model has model id " + shapenet_model["model_id"] + ".")
#model_verts, model_faces = shapenet_model["verts"], shapenet_model["faces"]

#DataLoader

shapenet_loader = DataLoader(shapenet_dataset, batch_size=12, collate_fn=collate_batched_meshes)


#it = iter(shapenet_loader)
#shapenet_batch = next(it)
#print(shapenet_batch.keys())
#batch_renderings = shapenet_batch["images"] # (N, V, H, W, 3), and in this case V is 1.
#image_grid(batch_renderings.squeeze().numpy(), rows=3, cols=4, rgb=True)




#renderer

# Rendering settings.
R, T = look_at_view_transform(1.0, 1.0, 90)
cameras = OpenGLPerspectiveCameras(R=R, T=T, device=device)
raster_settings = RasterizationSettings(image_size=512)
lights = PointLights(location=torch.tensor([0.0, 1.0, -2.0], device=device)[None],device=device)

images_by_idxs = shapenet_dataset.render(
    idxs=list(range(7, 8)),
    device=device,
    cameras=cameras,
    raster_settings=raster_settings,
    lights=lights,
)
image_grid(images_by_idxs.cpu().numpy(), rows=1, cols=3, rgb=True)

#plt.show()