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


def render(mesh, model_id, shapenet_dataset, device, camera = None):
    # Rendering settings.
    # camera_distance = 1
    # camera_elevation = 0.5 + 100 * random.random()
    # camera_azimuth = 30 + 90 * random.random()
    # R, T = look_at_view_transform(camera_distance, camera_elevation, camera_azimuth)
    # camera = FoVPerspectiveCameras(R=R, T=T, device=device)
    # raster_settings = RasterizationSettings(image_size=512)
    # lights = PointLights(location=torch.tensor([0.0, 1.0, -2.0], device=device)[None],device=device)
    # #rendering_settings = cameras, raster_settings, lights
    # image = shapenet_dataset.render(
    #     model_ids=[model_id],
    #     device=device,
    #     cameras=camera,
    #     raster_settings=raster_settings,
    #     lights=lights,
    # )[..., :3]
    if not camera:
        camera_elevation = 0 + 180 * torch.rand((1))#torch.linspace(0, 180, batch_size)
        camera_azimuth = -180 + 2 * 180 * torch.rand((1))#torch.linspace(-180, 180, batch_size)
        #R, T = look_at_view_transform(camera_distance, camera_elevation, camera_azimuth)
        R, T = look_at_view_transform(1.9, camera_elevation, camera_azimuth)
        camera = FoVPerspectiveCameras(R=R, T=T, device=device)
        camera.eval() #necessary ?
    raster_settings = RasterizationSettings(image_size=224) # TODO ?????
    lights = PointLights(location=torch.tensor([0.0, 1.0, -2.0], device=device)[None],device=device)

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=camera, raster_settings=raster_settings),
        shader=HardPhongShader(device=device, cameras=camera)
    )
    renderer.eval()
    #rendering_settings = cameras, raster_settings, lights
    #image = shapenet_dataset.render(
    #   model_ids=[model_id],
    #    device=device,
    #  cameras=camera,
    #  raster_settings=raster_settings,
    #  lights=lights,
    #)[..., :3]
    image = renderer(mesh)[..., :3]
    #plt.imshow(image.squeeze().detach().cpu().numpy())
    #plt.show()
    image = image.permute(0, 3, 1, 2)
    return image, camera #TODO batch of images
    #image_grid(images_by_idxs.cpu().numpy(), rows=1, cols=3, rgb=True)
    #plt.show()
    #save_obj("model.obj", mesh.verts_packed(), mesh.faces_packed())

