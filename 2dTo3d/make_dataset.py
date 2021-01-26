import os
import torch
from pytorch3d.datasets import (
    ShapeNetCore,
    collate_batched_meshes,
)
from pytorch3d.structures import Meshes
from torch.utils.data import DataLoader
import torchvision
import warnings
warnings.filterwarnings('ignore', message = 'Texture file does not exist')
from render import render
from feature_extraction import FeatureExtractor, GraphProjection
from pytorch3d.io import load_obj, save_obj
from pytorch3d.structures import Meshes
from pytorch3d.utils import ico_sphere
from torch_geometric.data import Data
from torch_geometric.transforms import SamplePoints
from pytorch3d.ops import SubdivideMeshes
from pytorch3d.ops import sample_points_from_meshes, graph_conv
from pytorch3d.loss import (
    chamfer_distance,
    mesh_edge_loss,
    mesh_laplacian_smoothing,
    mesh_normal_consistency,
)
from tqdm import trange

from pytorch3d.io import save_obj
from model import GCN, Net, DeeperGCN
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from utils import plot_pointcloud, get_loss, normalize_mesh
mpl.rcParams['savefig.dpi'] = 80
mpl.rcParams['figure.dpi'] = 80

# Set the device
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
    print("WARNING: CPU only, this will be slow!")


SHAPENET_PATH = "/Data/leo/Pixel2Mesh_3d/dataset/ShapeNetCore.v2"
shapenet_dataset = ShapeNetCore(SHAPENET_PATH, version=2, synsets=["airplane"])


shapenet_loader = DataLoader(shapenet_dataset, batch_size=1, collate_fn=collate_batched_meshes, shuffle = False)


print("starting training...")
for i, shape in enumerate(shapenet_loader):
    print("ITERATION {}".format(i))
    #Normalize mesh
    trg_verts, trg_faces = shape["verts"][0], shape["faces"][0] #TODO batch
    trg_mesh = Meshes(
        verts=[trg_verts.to(device)],
        faces=[trg_faces.to(device)],
    )

    trg_mesh = normalize_mesh(trg_mesh)
    #Render the target mesh to make an image
    image, camera = render(trg_mesh, shape["model_id"][0], shapenet_dataset, device)

