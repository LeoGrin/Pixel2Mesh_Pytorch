import os
import sys
import torch

from pytorch3d.datasets import (
    R2N2,
    ShapeNetCore,
    collate_batched_meshes,
    render_cubified_voxels,
)

from pytorch3d.structures import Meshes
from torch.utils.data import DataLoader

# add path for demo utils functions
sys.path.append(os.path.abspath(''))

# Setup
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

SHAPENET_PATH = "/Data/leo/Pixel2Mesh_3d/dataset/ShapeNetCore.v2"
shapenet_dataset = ShapeNetCore(SHAPENET_PATH, version=2, synsets=["airplane"])

shapenet_loader = DataLoader(shapenet_dataset, batch_size=12, collate_fn=collate_batched_meshes)

shapenet_batch = next(iter(shapenet_loader))

print(shapenet_batch["mesh"].verts_packed().shape)
print(shapenet_batch["mesh"].edges_packed().shape)