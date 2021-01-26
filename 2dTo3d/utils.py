from pytorch3d.ops import sample_points_from_meshes, graph_conv
from pytorch3d.loss import (
    chamfer_distance,
    mesh_edge_loss,
    mesh_laplacian_smoothing,
    mesh_normal_consistency,
)
import torch
from pytorch3d.io import save_obj
from model import GCN, Net
import numpy as np
from tqdm.notebook import tqdm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl

def plot_pointcloud(mesh, title=""):
    # Sample points uniformly from the surface of the mesh.
    points = sample_points_from_meshes(mesh, 5000)
    x, y, z = points.clone().detach().cpu().squeeze().unbind(1)
    fig = plt.figure(figsize=(5, 5))
    ax = Axes3D(fig)
    ax.scatter3D(x, z, -y)
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')
    ax.set_title(title)
    ax.view_init(190, 30)
    plt.show()

def normalize_mesh(mesh):
    verts = mesh.verts_padded()
    center = mesh.verts_padded().mean(1).unsqueeze(1) #0 is the batch index
    verts = verts - center
    scale = torch.max(verts.abs().max(1)[0], 1)[0].reshape(-1, 1, 1)
    verts = verts / scale
    return mesh.update_padded(verts)

def get_loss(mesh, trg_mesh, w_chamfer, w_edge, w_normal, w_laplacian, n_points = 5000):
    # We sample 5k points from the surface of each mesh
    sample_trg = sample_points_from_meshes(trg_mesh, n_points)
    sample_src = sample_points_from_meshes(mesh, n_points)

    # We compare the two sets of pointclouds by computing (a) the chamfer loss

    loss_chamfer, _ = chamfer_distance(sample_trg, sample_src)

    # and (b) the edge length of the predicted mesh
    loss_edge = mesh_edge_loss(mesh)

    # mesh normal consistency
    loss_normal = mesh_normal_consistency(mesh)

    # mesh laplacian smoothing
    loss_laplacian = mesh_laplacian_smoothing(mesh, method="uniform")

    # Weighted sum of the losses
    loss = loss_chamfer * w_chamfer + loss_edge * w_edge + loss_normal * w_normal + loss_laplacian * w_laplacian

    return loss
