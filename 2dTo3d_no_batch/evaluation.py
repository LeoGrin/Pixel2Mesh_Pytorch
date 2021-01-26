import os
import torch
from pytorch3d.datasets import (
    ShapeNetCore,
    collate_batched_meshes,
)
from pytorch3d.structures import Meshes
from torch.utils.data import DataLoader
import torchvision
from torch.optim.lr_scheduler import MultiStepLR

import warnings
import wandb
from utils import plot_grad_flow
from pytorch3d.renderer import (
    OpenGLPerspectiveCameras,
    PointLights,
    RasterizationSettings,
    TexturesVertex,
    Textures,
    look_at_view_transform,
)
warnings.filterwarnings('ignore', message = 'Texture file does not exist')
from render_no_batch import render
from feature_extraction_no_batch import FeatureExtractor, GraphProjection
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

# Set the device
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
    print("WARNING: CPU only, this will be slow!")
#device = torch.device("cpu")
#DATA:

SHAPENET_PATH = "/Data/leo/Pixel2Mesh_3d/dataset/ShapeNetCore.v2"
shapenet_dataset = ShapeNetCore(SHAPENET_PATH, version=2, synsets=["airplane"])

#shapenet_model = shapenet_dataset[6]
#print("This model belongs to the category " + shapenet_model["synset_id"] + ".")
#print("This model has model id " + shapenet_model["model_id"] + ".")
#model_verts, model_faces = shapenet_model["verts"], shapenet_model["faces"]

#DataLoader

shapenet_loader = DataLoader(shapenet_dataset, batch_size=1, collate_fn=collate_batched_meshes, shuffle = True)


#SETTINGS
# We initialize the source shape to be a sphere of radius 1
block1 = ico_sphere(2, device)
uppooling1 = SubdivideMeshes(block1)
block2 = uppooling1(block1)
uppooling2 = SubdivideMeshes(block2)
block3 = uppooling2(block2)

#TODO batch size, features
data1 = Data(x= block1.verts_packed().reshape(-1, 3),#block1.verts_packed().reshape(-1, 3),
             edge_index=block1.edges_packed().reshape(2, -1),
             face = block1.faces_packed().reshape(3, -1))
data1.pos = data1.x

data2 = Data(x=block2.verts_packed().reshape(-1, 3),
             edge_index=block2.edges_packed().reshape(2, -1),
             face = block2.faces_packed().reshape(3, -1))
data2.pos = data2.x

data3 = Data(x=block3.verts_packed().reshape(-1, 3),
             edge_index=block3.edges_packed().reshape(2, -1),
             face = block3.faces_packed().reshape(3, -1))
data3.pos = data3.x

#Models

gcn1 = DeeperGCN(707, 128, 3, 10)
gcn1.to(device)
gcn2 = DeeperGCN(128 + 3, 128, 3, 10)
gcn2.to(device)
gcn3 = DeeperGCN(128 + 3, 128, 3, 10)
gcn3.to(device)

image_model = torchvision.models.resnet18(pretrained=True)
fe = FeatureExtractor(image_model, ["conv1", "layer2.0.conv1", "layer4.1.conv2"])
#image_model = torchvision.models.vgg16(pretrained=True)
#print(dict([*image_model.named_modules()]).keys())
#fe = FeatureExtractor(image_model, ["conv1", "layer2.0.conv1", "layer4.1.conv2"])
fe.to(device)
image_model.to(device)
#image_model.eval()



#load model
dic = torch.load('/Data/leo/Pixel2Mesh_3d/2dTo3d/wandb/run-20210126_131356-1bkxbzmf/files/dic_checkpoint')
gcn1.load_state_dict(dic['gcn_state_dict'][0])
gcn1.eval()
gcn2.load_state_dict(dic['gcn_state_dict'][1])
gcn2.eval()
gcn3.load_state_dict(dic['gcn_state_dict'][2])
gcn3.eval()
image_model.load_state_dict(dic['cnn_state_dic'])
image_model.eval()
fe.load_state_dict(dic['fe_state_dic'])
fe.eval()
#optimizer.load_state_dict(dic['optimizer_state_dict'])
#epoch = checkpoint['epoch']
#loss = checkpoint['loss']




w_chamfer = 1.0
# Weight for mesh edge loss
w_edge = 0.1#1.0
# Weight for mesh normal consistency
w_normal = 1.6e-4#0.01
# Weight for mesh laplacian smoothing
w_laplacian = 0.3#0.1
# Plot period for the losses
plot_period = 500
save_period = 500
losses = []

EPOCHS = 100


for i, shape in enumerate(shapenet_loader):
    #try:
    print("ITERATION {}".format(i))
    #Initialize optimizer
    #Normalize mesh
    #trg_verts, trg_faces = shape["verts"][0], shape["faces"][0] #TODO batch
    #model_textures = TexturesVertex(verts_features=torch.ones_like(trg_verts, device=device)[None])
    #trg_mesh = Meshes(
    #    verts=[trg_verts.to(device)],
    #    faces=[trg_faces.to(device)],
    #    textures = model_textures
    #)
    trg_mesh = shape["mesh"].to(device)

    trg_mesh = normalize_mesh(trg_mesh)
    #Render the target mesh to make an image
    with torch.no_grad():
        image, camera = render(trg_mesh, shape["model_id"][0], shapenet_dataset, device)
    #to check if it's using the image
    #image = torch.ones(image.shape, device = device)
    #BATCH
    #print("image")
    #print(image.flatten().shape)
    #print(torch.sum(image < 0.99))
    features = fe.add_features(image)
    #print("features")
    #print(features.shape)
    #print(features)
    #feature_to_plot = torch.mean(features, 1)
    #        plt.imshow(feature_to_plot.squeeze().detach().cpu().numpy())
    #        plt.show()
    #        plt.imshow(image.squeeze().permute(1, 2, 0).detach().cpu().numpy())
    #        plt.show()

    #print(features.shape)
    #project
    proj = GraphProjection(camera = camera, device = device)
    proj.to(device)
    #data1.x += torch.cat((data1.x, proj(features, block1.verts_packed()).squeeze()), 1) #TODO batch
    #print(data1.x.shape)
    #print(data1.edge_index)
    #data1.x = proj(features, block1.verts_packed()).squeeze()
    #print("proj")
    #print(proj(features, block2.verts_packed()).shape)
    #plotable_proj = proj(features, block2.verts_packed()).max(2)[0].unsqueeze(2)
    #print(plotable_proj.shape)
    #TEST
    # print("proj")
    # test = proj(features, trg_mesh.verts_packed()).mean(2).unsqueeze(2)
    # print("test")
    # print(test.shape)
    # print(test)
    # test = test - torch.min(test, 1)[0].unsqueeze(2)
    # test =  test * (1 / torch.max(test))
    # test = torch.cat((test,1 - test, torch.zeros(1, trg_mesh.verts_packed().shape[0], 1, device = device)), 2)
    # #test = torch.cat((test, test, test), 2)
    # print("shape")
    # print(test.shape)
    # print(test)
    # #test = test - torch.min(test, 2)[0].unsqueeze(2)
    # model_textures = Textures(verts_rgb=test)
    # ##print(plotable_proj)
    # test_block = Meshes(
    #     verts=[trg_mesh.verts_packed().to(device)],
    #     faces=[trg_mesh.faces_packed().to(device)],
    #     textures = model_textures
    # )
    # #save_obj("texture.obj", test_block.verts_packed(), test_block.faces_packed())
    # plt.imshow(image.squeeze().permute(1, 2, 0).detach().cpu().numpy())
    # plt.show()
    # feature_to_plot = torch.mean(features, 1)
    # plt.imshow(feature_to_plot.squeeze().detach().cpu().numpy())
    # plt.show()
    # new_image, new_camera = render(test_block, shape["model_id"][0], shapenet_dataset, device, camera)
    # plt.imshow(new_image.squeeze().permute(1, 2, 0).detach().cpu().numpy())
    # plt.show()

    data1.x = torch.cat((block1.verts_packed().reshape(-1, 3), proj(features, block1.verts_packed()).squeeze()), 1)
    # Deform the mesh
    out_features1, offset1 = gcn1(data1)
    #out_features1, offset1 = out_features1.squeeze(), offset1.squeeze()
    new_block1 = block1.offset_verts(offset1)
    new_block2, new_features2 = uppooling1(new_block1, torch.cat((new_block1.verts_packed(), out_features1), 1))

    data2.x = new_features2#new_block2.verts_packed().reshape(-1, 3)

    out_features2, offset2 = gcn2(data2)
    out_features2, offset2  = out_features2.squeeze(), offset2.squeeze() #TODO
    new_block2 = new_block2.offset_verts(offset2) #TODO + new_features2?
    new_block3, new_features3 = uppooling2(new_block2, torch.cat((new_block2.verts_packed(), out_features2), 1))
    data3.x = new_features3#new_block3.verts_packed().reshape(-1, 3)


    out_features3, offset3 = gcn3(data3)
    out_features3, offset3 = out_features3.squeeze(), offset3.squeeze()
    new_block3 = new_block3.offset_verts(offset3)


    loss = get_loss(new_block1, trg_mesh, w_chamfer, w_edge, w_normal, w_laplacian) + \
           get_loss(new_block2, trg_mesh, w_chamfer, w_edge, w_normal, w_laplacian) + \
           get_loss(new_block3, trg_mesh, w_chamfer, w_edge, w_normal, w_laplacian)
    print(loss)
    #t.set_description("loss = {}".format(loss))
    losses.append(loss.detach())

    plt.imsave("../results/m{}_image.png".format(shape["model_id"][0]), image.squeeze().permute(1, 2, 0).detach().cpu().numpy())
    save_obj("../results/m{}_model_true.obj".format(shape["model_id"][0]), trg_mesh.verts_packed(), trg_mesh.faces_packed())
    save_obj("../results/m{}_model1.obj".format(shape["model_id"][0]), new_block1.verts_packed(), new_block1.faces_packed())
    save_obj("../results/m{}_model2.obj".format(shape["model_id"][0]), new_block2.verts_packed(), new_block2.faces_packed())
    save_obj("../results/m{}_model3.obj".format(shape["model_id"][0]), new_block3.verts_packed(), new_block3.faces_packed())


    #except:
    #    print("ERROR")
    #    pass
#scheduler.step()
