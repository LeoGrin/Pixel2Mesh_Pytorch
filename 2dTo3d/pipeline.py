import os
import torch
from pytorch3d.datasets import (
    ShapeNetCore,
    collate_batched_meshes,
)
from torch.optim.lr_scheduler import MultiStepLR
from pytorch3d.structures import Meshes
from torch.utils.data import DataLoader
import torchvision
import warnings
import tqdm
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
from pytorch3d.renderer import (
    OpenGLPerspectiveCameras,
    PointLights,
    RasterizationSettings,
    TexturesVertex,
    look_at_view_transform,
)
from tqdm import trange

from pytorch3d.io import save_obj
from model import GCN, Net, DeeperGCN
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import wandb
from utils import plot_pointcloud, get_loss, normalize_mesh

wandb.init(project="pixel2mesh", entity="leogrin")

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

BATCH_SIZE = 2 #number of images rendered for each model (right now we use only one target model per batch)

#SETTINGS
# We initialize the source shape to be a sphere of radius 1
block1 = ico_sphere(2, device)
uppooling1 = SubdivideMeshes(block1)
block2 = uppooling1(block1)
uppooling2 = SubdivideMeshes(block2)
block3 = uppooling2(block2)
block1 = block1.extend(BATCH_SIZE)
block2 = block2.extend(BATCH_SIZE)
block3 = block3.extend(BATCH_SIZE)


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
hidden_dim = 128
features_dim = 960
gcn1 = DeeperGCN(features_dim + 3, hidden_dim, 3, 11)
gcn1.to(device)
#gcn1.half()
gcn2 = DeeperGCN(features_dim + hidden_dim + 3, hidden_dim, 3, 11)
gcn2.to(device)
#gcn2.half()
gcn3 = DeeperGCN(features_dim + hidden_dim + 3, hidden_dim, 3, 11)
gcn3.to(device)
#gcn3.half()

image_model = torchvision.models.resnet18(pretrained=True)
fe = FeatureExtractor(image_model, ["layer1.1.conv1", "layer2.0.conv1", "layer3.0.conv2", "layer4.1.conv2"])
fe.to(device)
image_model.to(device)
#image_model.half()
#fe.half()
wandb.watch(gcn1)
wandb.watch(gcn2)
wandb.watch(gcn3)
wandb.watch(image_model)
wandb.watch(fe)



optimizer = torch.optim.Adam(list(gcn1.parameters()) +
                             list(gcn2.parameters()) +
                             list(gcn3.parameters()) +
                             list(image_model.parameters()), lr=5e-4)
scheduler = MultiStepLR(optimizer, milestones=[5, 10, 30, 80], gamma=0.3)

w_chamfer = 1.0
# Weight for mesh edge loss
w_edge = 0.1#1.0
# Weight for mesh normal consistency
w_normal = 1.6e-4#0.01
# Weight for mesh laplacian smoothing
w_laplacian = 0.3#0.1
# Plot period for the losses
plot_period = 30
save_period = 100
losses = []

EPOCHS = 100

#print("loading")
#dic_saved = wandb.restore('dic_checkpoint', run_path="leogrin/pixel2mesh/104k7dgh")



t = trange(EPOCHS)
print("starting training...")
torch.cuda.empty_cache()
for epoch in t:
    for i, shape in tqdm.tqdm(enumerate(shapenet_loader)):
        #print("ITERATION {}".format(i))
        #Initialize optimizer
        optimizer.zero_grad()
        #Normalize mesh
        #trg_verts, trg_faces = shape["verts"][0], shape["faces"][0] #TODO batch
        #model_textures = TexturesVertex(verts_features=torch.ones_like(trg_verts, device=device)[None])
        ##trg_mesh = Meshes(
        #    verts=[trg_verts.to(device)],
        ##    faces=[trg_faces.to(device)],
         #   textures=model_textures
        #)
        trg_mesh = shape["mesh"].to(device)
        trg_mesh = normalize_mesh(trg_mesh)
        extended_trg_mesh = trg_mesh.extend(BATCH_SIZE)
        #Render the target mesh to make an image
        with torch.no_grad():
            image, camera = render(extended_trg_mesh,
                                   shape["model_id"][0],
                                   shapenet_dataset,
                                   device,
                                   batch_size = BATCH_SIZE)
        #image = image.half()


        #BATCH
        image_features = fe.add_features(image)
        #project
        proj = GraphProjection(camera = camera, device = device)
        proj.to(device)
        #we need packed representation for the GCN, and padded representation for the projection
        indices_padded_to_packed = block1.verts_padded_to_packed_idx()
        features_from_image = proj(image_features, block1.verts_padded())
        features_from_image = features_from_image.reshape(-1, features_from_image.shape[-1])
        features_from_image = features_from_image[indices_padded_to_packed]
        data1.x = torch.cat((block1.verts_packed(), features_from_image), 1)
        # Deform the mesh
        out_features1, offset1 = gcn1(data1)

        #out_features1, offset1 = out_features1.squeeze(), offset1.squeeze()
        new_block1 = block1.offset_verts(offset1)
        new_block2, new_features2 = uppooling1(new_block1, torch.cat((new_block1.verts_packed(), out_features1), 1))
        indices_padded_to_packed = new_block2.verts_padded_to_packed_idx()
        features_from_image = proj(image_features, new_block2.verts_padded())
        features_from_image = features_from_image.reshape(-1, features_from_image.shape[-1])
        features_from_image = features_from_image[indices_padded_to_packed]
        new_features2 = new_features2.reshape(-1, new_features2.shape[-1])
        new_features2 = new_features2[indices_padded_to_packed] #TODO check

        data2.x = torch.cat((new_features2, features_from_image), 1)#new_block2.verts_packed().reshape(-1, 3)

        out_features2, offset2 = gcn2(data2)
        #out_features2, offset2 = out_features2.squeeze(), offset2.squeeze()
        #print(offset2)
        new_block2.offset_verts_(offset2)
        new_block3, new_features3 = uppooling2(new_block2, torch.cat((new_block2.verts_packed(), out_features2), 1))
        #data3.x = new_features3#new_block3.verts_packed().reshape(-1, 3)

        indices_padded_to_packed = new_block3.verts_padded_to_packed_idx()
        features_from_image = proj(image_features, new_block3.verts_padded())
        features_from_image = features_from_image.reshape(-1, features_from_image.shape[-1])
        features_from_image = features_from_image[indices_padded_to_packed]
        new_features3 = new_features3.reshape(-1, new_features3.shape[-1])
        new_features3 = new_features3[indices_padded_to_packed] #TODO check

        data3.x = torch.cat((new_features3, features_from_image), 1)

        out_features3, offset3 = gcn3(data3)
        out_features3, offset3 = out_features3.squeeze(), offset3.squeeze()
        new_block3 = new_block3.offset_verts(offset3)

        loss = get_loss(new_block1, extended_trg_mesh, w_chamfer, w_edge, w_normal, w_laplacian) + \
               get_loss(new_block2, extended_trg_mesh, w_chamfer, w_edge, w_normal, w_laplacian) + \
               get_loss(new_block3, extended_trg_mesh, w_chamfer, w_edge, w_normal, w_laplacian)
        #free memory
        del extended_trg_mesh, trg_mesh, camera, shape
        t.set_description("loss = {}".format(loss))
        wandb.log({"Train Loss": loss})
        losses.append(loss.detach())


        # Plot mesh
        if i % plot_period == 0 and i!=0:
            #plot_pointcloud(new_block1, title="iter: %d" % i)
            #plot_pointcloud(new_block2, title="iter: %d" % i)
            #plot_pointcloud(new_block3, title="iter: %d" % i)
            print("image size")
            print(image[0].shape)
            plt.imshow(image[0].squeeze().permute(1, 2, 0).detach().cpu().numpy())
            #save_obj("model1_{}_{}.obj".format(epoch, i), new_block1.verts_packed(), new_block1.faces_packed())
            #save_obj("model2_{}_{}.obj".format(epoch, i), new_block2.verts_packed(), new_block2.faces_packed())
            #save_obj("model3_{}_{}.obj".format(epoch, i), new_block3.verts_packed(), new_block3.faces_packed())
            plt.plot(range(len(losses)), np.log10(losses))
            plt.show()
        if i % save_period == 0:
            print("saving ...")
            torch.save({
                'epoch': epoch,
                'iteration':i,
                'gcn_state_dict': [gcn1.state_dict(), gcn2.state_dict(), gcn3.state_dict()],
                'fe_state_dic': fe.state_dict(),
                'cnn_state_dic': image_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss}, os.path.join(wandb.run.dir, 'dic_checkpoint'))

        # Optimization step
        loss.backward()

        #torch.nn.utils.clip_grad_norm_(gcn1.parameters(), 1)
        #torch.nn.utils.clip_grad_norm_(gcn2.parameters(), 1)
        #torch.nn.utils.clip_grad_norm_(gcn3.parameters(), 1)
        #torch.nn.utils.clip_grad_norm_(image_model.parameters(), 1)

        optimizer.step()

    scheduler.step()
