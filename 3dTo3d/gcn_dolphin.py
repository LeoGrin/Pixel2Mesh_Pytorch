import os
import torch
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
from model import GCN, Net
import numpy as np
from tqdm.notebook import tqdm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
from utils import plot_pointcloud
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
block1 = ico_sphere(2, device)
uppooling1 = SubdivideMeshes(block1)
block2 = uppooling1(block1)
uppooling2 = SubdivideMeshes(block2)
block3 = uppooling2(block2)

#TODO batch size, features

data1 = Data(x=block1.verts_packed().reshape(-1, 3),
             edge_index=block1.edges_packed().reshape(2, -1),
             face = block1.faces_packed().reshape(3, -1))
data1.pos = data1.x

data2 = Data(x=block2.verts_packed().reshape(-1, 3),
             edge_index=block2.edges_packed().reshape(2, -1),
             face = block2.faces_packed().reshape(3, -1))
data2.pos = data2.x

data3 = Data(x=block3.verts_packed().reshape(-1, 3),
             edge_index=block3.edges_packed().reshape(3, -1),
             face = block3.faces_packed().reshape(3, -1))
data3.pos = data3.x



#plot_pointcloud(trg_mesh, "Target mesh")
#plot_pointcloud(src_mesh, "Source mesh")
plt.show()

# We will learn to deform the source mesh by offsetting its vertices
# The shape of the deform parameters is equal to the total number of vertices in src_mesh#
#deform_verts = torch.full(src_mesh.verts_packed().shape, 0.0, device=device, requires_grad=True)
#model = GCN()
print(device)
#TODO: more features
gcn1 = Net()
gcn1.to(device)
gcn2 = Net()
gcn2.to(device)
gcn3 = Net()
gcn3.to(device)

# The optimizer
optimizer = torch.optim.SGD(list(gcn1.parameters()) +
                            list(gcn2.parameters()) +
                            list(gcn3.parameters()), lr=0.2, momentum=0.9)
#optimizer = torch.optim.SGD([deform_verts], lr=1.0, momentum=0.9)


# Number of optimization steps
Niter = 1000
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

def get_loss(mesh, trg_mesh, n_points = 3000):
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

t = trange(Niter)
for i in t:
    # Initialize optimizer
    optimizer.zero_grad()

    # Deform the mesh
    offset1 = gcn1(data1)
    new_block1 = block1.offset_verts(offset1)
    new_block2, new_features2 = uppooling1(new_block1, data1.x + offset1)
    #data2.x = new_features2 #TODO = ?
    offset2 = gcn2(data2)
    data2.x += offset2
    new_block2 = block2.offset_verts(offset2) #TODO + offset1 ?
    new_block3, new_features3 = uppooling2(new_block2, data2.x)
    data3.x = new_features3
    offset3 = gcn3(data3)
    data3.x += offset3
    new_block3 = block3.offset_verts(offset3)

    #new_block2 = block2.offset_verts(gcn2(data2))
    #new_block3 = block3.offset_verts(gcn3(data3))
    #data.x = new_features


    loss = get_loss(new_block1) + get_loss(new_block2) + get_loss(new_block3)
    # Print the losses
    #t.set_description('loss={}, c={}, e={}, n={}, l= {}'.format(int(loss * 100) / 100,
    #                                                             int(loss_chamfer * 100) / 100,
    #                                                             int(loss_edge * 100) / 100,
    #                                                             int(loss_normal * 100) / 100,
    #                                                             int(loss_laplacian * 100) / 100))
    t.set_description("loss = {}".format(loss))

    # Save the losses for plotting
    #chamfer_losses.append(loss_chamfer)
    #edge_losses.append(loss_edge)
    #normal_losses.append(loss_normal)
    #laplacian_losses.append(loss_laplacian)

    # Plot mesh
    if i % plot_period == 0:
        plot_pointcloud(block1, title="iter: %d" % i)
        plot_pointcloud(block2, title="iter: %d" % i)
        plot_pointcloud(block3, title="iter: %d" % i)

    # Optimization step
    loss.backward()
    optimizer.step()

#save_obj("test.obj", new_src_mesh.verts_packed(), new_src_mesh.faces_packed())