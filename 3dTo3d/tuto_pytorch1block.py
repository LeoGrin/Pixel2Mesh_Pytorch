import os
import torch
from pytorch3d.io import load_obj, save_obj
from pytorch3d.structures import Meshes
from pytorch3d.utils import ico_sphere
from torch_geometric.data import Data
from torch_geometric.transforms import SamplePoints
from pytorch3d.ops import sample_points_from_meshes, graph_conv
from pytorch3d.loss import (
    chamfer_distance,
    mesh_edge_loss,
    mesh_laplacian_smoothing,
    mesh_normal_consistency,
)
from tqdm import trange
from model import GCN, Net, DeeperGCN

from pytorch3d.io import save_obj
from model import GCN, Net
import numpy as np
from tqdm.notebook import tqdm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['savefig.dpi'] = 80
mpl.rcParams['figure.dpi'] = 80

# Set the device
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
    print("WARNING: CPU only, this will be slow!")

trg_obj = os.path.join('dolphin.obj')

# We read the target 3D model using load_obj
verts, faces, aux = load_obj(trg_obj)

# verts is a FloatTensor of shape (V, 3) where V is the number of vertices in the mesh
# faces is an object which contains the following LongTensors: verts_idx, normals_idx and textures_idx
# For this tutorial, normals and textures are ignored.
faces_idx = faces.verts_idx.to(device)
verts = verts.to(device)

# We scale normalize and center the target mesh to fit in a sphere of radius 1 centered at (0,0,0).
# (scale, center) will be used to bring the predicted mesh to its original center and scale
# Note that normalizing the target mesh, speeds up the optimization but is not necessary!
center = verts.mean(0)
verts = verts - center
scale = max(verts.abs().max(0)[0])
verts = verts / scale


# We construct a Meshes structure for the target mesh
trg_mesh = Meshes(verts=[verts], faces=[faces_idx])

# We initialize the source shape to be a sphere of radius 1
block1 = ico_sphere(3, device)
#block2 = ico_sphere(3, device)
#block3 = ico_sphere(4, device)
#print(ico_sphere(2, device).verts_packed().shape)
#print(ico_sphere(3, device).verts_packed().shape)
#print(ico_sphere(4, device).verts_packed().shape)
#print(src_mesh.faces_packed().shape)
#print(src_mesh.verts_packed().shape)
#TODO batch size, features

data = Data(x=block1.verts_packed().reshape(-1, 3),
              edge_index=block1.edges_packed().reshape(2, -1),
              face = block1.faces_packed().reshape(3, -1))
data.pos = data.x

#data_2 = Data(x=block2.verts_packed().reshape(-1, 3),
#              edge_index=block2.edges_packed().reshape(2, -1),
#              face = block2.faces_packed().reshape(3, -1))
#block2.pos = block2.x

#data_3 = Data(x=block3.verts_packed().reshape(-1, 3),
#              edge_index=block3.edges_packed().reshape(3, -1),
#              face = block3.faces_packed().reshape(3, -1))
#block3.pos = block3.x

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

#plot_pointcloud(trg_mesh, "Target mesh")
#plot_pointcloud(src_mesh, "Source mesh")
plt.show()

# We will learn to deform the source mesh by offsetting its vertices
# The shape of the deform parameters is equal to the total number of vertices in src_mesh#
#deform_verts = torch.full(block1.verts_packed().shape, 0.0, device=device, requires_grad=True)
#model = GCN()
#print(device)
model = DeeperGCN(3, 128, 3, 5)
model.to(device)

# The optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
#optimizer = torch.optim.SGD([deform_verts], lr=0.2, momentum=0.9)


# Number of optimization steps
Niter = 3001
# Weight for the chamfer loss
w_chamfer = 1.0
# Weight for mesh edge loss
w_edge = 0.1#1.0
# Weight for mesh normal consistency
w_normal = 1.6e-4#0.01
# Weight for mesh laplacian smoothing
w_laplacian = 0.3#0.1
# Plot period for the losses
plot_period = 250

chamfer_losses = []
laplacian_losses = []
edge_losses = []
normal_losses = []

t = trange(Niter)
for i in t:
    # Initialize optimizer
    optimizer.zero_grad()

    # Deform the mesh
    new_features, offsets = model(data)
    new_src_mesh = block1.offset_verts(offsets)
    #data.x = new_features


    # We sample 5k points from the surface of each mesh
    sample_trg = sample_points_from_meshes(trg_mesh, 5000)
    sample_src = sample_points_from_meshes(new_src_mesh, 5000)

    # We compare the two sets of pointclouds by computing (a) the chamfer loss
    loss_chamfer, _ = chamfer_distance(sample_trg, sample_src)

    # and (b) the edge length of the predicted mesh
    loss_edge = mesh_edge_loss(new_src_mesh)

    # mesh normal consistency
    loss_normal = mesh_normal_consistency(new_src_mesh)

    # mesh laplacian smoothing
    loss_laplacian = mesh_laplacian_smoothing(new_src_mesh, method="uniform")

    # Weighted sum of the losses
    loss = loss_chamfer * w_chamfer + loss_edge * w_edge + loss_normal * w_normal + loss_laplacian * w_laplacian

    # Print the losses
    t.set_description('loss={}, c={}, e={}, n={}, l= {}'.format(int(loss * 100) / 100,
                                                                int(loss_chamfer * 100) / 100,
                                                                int(loss_edge * 100) / 100,
                                                                int(loss_normal * 100) / 100,
                                                                int(loss_laplacian * 100) / 100))


    # Save the losses for plotting
    chamfer_losses.append(loss_chamfer)
    edge_losses.append(loss_edge)
    normal_losses.append(loss_normal)
    laplacian_losses.append(loss_laplacian)

    # Plot mesh
    if i % plot_period == 0:
        plot_pointcloud(new_src_mesh, title="iter: %d" % i)

    # Optimization step
    loss.backward()
    optimizer.step()

save_obj("test.obj", new_src_mesh.verts_packed(), new_src_mesh.faces_packed())