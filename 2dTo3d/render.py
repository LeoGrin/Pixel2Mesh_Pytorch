from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    PointLights,
    RasterizationSettings,
    TexturesVertex,
    look_at_view_transform,
)
from pytorch3d.renderer import (
    FoVPerspectiveCameras, look_at_view_transform,
    RasterizationSettings, BlendParams,
    MeshRenderer, MeshRasterizer, HardPhongShader
)
import random
from pytorch3d.io import load_obj, save_obj
import torch
from plot_image_grid import image_grid
import matplotlib.pyplot as plt


def render(meshes, model_id, shapenet_dataset, device, batch_size):
    # Rendering settings.
    #meshes = mesh.extend(batch_size)
    #camera_elevation = [0.5 + 100 * random.random() for _ in range(batch_size)]
    #camera_azimuth = [30 + 90 * random.random() for _ in range(batch_size)]
    camera_elevation = 0 + 180 * torch.rand((batch_size))#torch.linspace(0, 180, batch_size)
    camera_azimuth = -180 + 2 * 180 * torch.rand((batch_size))#torch.linspace(-180, 180, batch_size)
    #R, T = look_at_view_transform(camera_distance, camera_elevation, camera_azimuth)
    R, T = look_at_view_transform(2.0, camera_elevation, camera_azimuth)
    cameras = FoVPerspectiveCameras(R=R, T=T, device=device)
    cameras.eval() #necessary ?
    raster_settings = RasterizationSettings(image_size=512) # TODO ?????
    lights = PointLights(location=torch.tensor([0.0, 1.0, -2.0], device=device)[None],device=device)

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=HardPhongShader(device=device, cameras=cameras)
    )
    renderer.eval()
    #rendering_settings = cameras, raster_settings, lights
    #image = shapenet_dataset.render(
    #    model_ids=[model_id],
    ##    device=device,
     #   cameras=camera,
     #   raster_settings=raster_settings,
     #   lights=lights,
    #)[..., :3]
    image = renderer(meshes)[..., :3]
    #plt.imshow(image[0].squeeze().detach().cpu().numpy())
    #print(image.shape)
    #check images
    #print(image.shape)
    #plt.imshow(image[1].squeeze().detach().cpu().numpy())
    #plt.show()
    image = image.permute(0, 3, 1, 2)

    #plt.show()
    return image, cameras #TODO batch of images
    #image_grid(images_by_idxs.cpu().numpy(), rows=1, cols=3, rgb=True)
    #plt.show()
    #save_obj("model.obj", mesh.verts_packed(), mesh.faces_packed())

