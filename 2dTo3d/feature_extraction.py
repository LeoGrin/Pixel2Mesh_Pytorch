import torchvision.models as models
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torch





def image_loader(image_name, imsize):
    """load image, returns cuda tensor"""
    loader = transforms.Compose([transforms.Resize((imsize, imsize)), transforms.ToTensor()])
    image = Image.open(image_name)
    image = loader(image).float()
    image = image[:3] #remove transparency
    image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
    return image.cuda()  #assumes that you're using GPU


from typing import Dict, Iterable, Callable

class FeatureExtractor(nn.Module):
    def __init__(self, model: nn.Module, layers: Iterable[str]):
        super().__init__()
        self.model = model
        self.layers = layers
        self._features = {layer: torch.empty(0) for layer in layers}
        for layer_id in layers:
            layer = dict([*self.model.named_modules()])[layer_id]
            layer.register_forward_hook(self.save_outputs_hook(layer_id))

    def save_outputs_hook(self, layer_id: str) -> Callable:
        def fn(_, __, output):
            self._features[layer_id] = output
        return fn

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        _ = self.model(x)
        return self._features

    def add_features(self, x: torch.Tensor) -> torch.Tensor:
        self.forward(x)
        #WE SUPPOSE THAT THE FEATURES ARE IN ORDER COMPLEX --> SIMPLE
        max_size = self._features[self.layers[0]].shape[2]
        #TODO can we just keep all the features and use them later ?

        for key in self._features.keys():
            self._features[key] = torch.nn.functional.interpolate(self._features[key],
                                                                  (max_size, max_size))
        #TODO check if it's always the same order
        return torch.cat(list(self._features.values()), 1)

class GraphProjection(nn.Module):
    """Graph Projection layer, which pool 2D features to mesh
    The layer projects a vertex of the mesh to the 2D image and use
    bilinear interpolation to get the corresponding feature.
    """

    def __init__(self, camera, device):
        super(GraphProjection, self).__init__()
        self.camera = camera
        self.device = device

    def forward(self, img_features, input):

        t = self.camera.get_world_to_view_transform()
        points2d = t.transform_points(input)
        #TODO check if it's using the different cameras

        #points_screen = self.camera.transform_points(input.unsqueeze(0))

        #TODO check
        self.img_feats = img_features

        #projection ??
        #h = 248 * torch.div(input[:, 1], input[:, 2]) + 111.5
        #w = 248 * torch.div(input[:, 0], -input[:, 2]) + 111.5

        #h = torch.clamp(h, min = 0, max = self.imsize - 1)
        #w = torch.clamp(w, min = 0, max = self.imsize - 1)
        #img_sizes = [56, 28, 14, 7]
        #out_dims = [64, 128, 256, 512]
        #feats = [input]
        #for i in range(4):
        #TODO vectorize
        outputs = []
        for batch in range(img_features.shape[0]):
            output = self.get_image_features(self.img_feats[batch], points2d[batch, :, 0], points2d[batch, :, 1])
            output = output.permute(1, 0)
            outputs.append(output.unsqueeze(0))

        #feats.append(out)

        #output = torch.cat(feats, 1)

        return torch.cat(outputs, 0)

    def get_image_features(self, img_feats, x, y):
        #TODO here we project on the sum of the features
        # which makes a difference for interpolation
        #

        size_features = img_feats.shape[2]

        #interpolate
        x.add_(-min(x))
        x.mul_((size_features - 1) / max(x))
        y.add_(-min(y))
        y.mul_((size_features - 1) / max(y))

        x1, x2 = torch.floor(x).long(), torch.ceil(x).long()
        y1, y2 = torch.floor(y).long(), torch.ceil(y).long()
        x2 = torch.clamp(x2, max = size_features - 1) #TODO unnecessary ?
        y2 = torch.clamp(y2, max = size_features - 1)
        x1, x2, y1, y2 = x1.squeeze(), x2.squeeze(), y1.squeeze(), y2.squeeze()
        Q11 = img_feats[:, x1, y1].clone()
        Q12 = img_feats[:, x1, y2].clone()
        Q21 = img_feats[:, x2, y1].clone()
        Q22 = img_feats[:, x2, y2].clone()

        #x, y = x.long(), y.long() #TODO useful ?
        # interpolation
        weights = torch.mul(x2 - x, y2 - y)
        Q11 = torch.mul(Q11, weights.float())

        weights = torch.mul(x2 - x, y - y1)
        Q12 = torch.mul(weights.float(), Q12)

        weights = torch.mul(x - x1, y2 - y)
        Q21 = torch.mul(weights.float(), Q21)

        weights = torch.mul(x - x1, y - y1)
        Q22 = torch.mul(weights.float(), Q22)

        output = Q11 + Q21 + Q12 + Q22

        return output

def coord_to_features(x, y, features):
    size_features = features.shape[2]
    x1, x2 = torch.floor(x).long(), torch.ceil(x).long()
    y1, y2 = torch.floor(y).long(), torch.ceil(y).long()
    x2 = torch.clamp(x2, max = size_features - 1)
    y2 = torch.clamp(y2, max = size_features - 1)
    Q11 = features[:, :, x1, y1].clone()
    Q12 = features[:, :, x1, y2].clone()
    Q21 = features[:, :, x2, y1].clone()
    Q22 = features[:, :, x2, y2].clone()

    #x, y = x.long(), y.long() #TODO useful ?
    # interpolation
    weights = torch.mul(x2 - x, y2 - y)
    Q11 = torch.mul(Q11, weights.float())

    weights = torch.mul(x2 - x, y - y1)
    Q12 = torch.mul(weights.float(), Q12)

    weights = torch.mul(x - x1, y2 - y)
    Q21 = torch.mul(weights.float(), Q21)

    weights = torch.mul(x - x1, y - y1)
    Q22 = torch.mul(weights.float(), Q22)

    output = Q11 + Q21 + Q12 + Q22

    return output

if __name__ == '__main__':
    # Set the device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
        print("WARNING: CPU only, this will be slow!")

    image = image_loader("umbrella.jpg", 228)


    model = torchvision.models.resnet18(pretrained=True)
    model.to(device)

    outputs = model(image)


    fe = FeatureExtractor(model, ["conv1", "layer2.0.conv1", "layer4.1.conv2"])
    features = fe.add_features(image)
    #print(fe(image)["conv1", "layer4.1.conv2"].shape)
    plt.imshow(features.squeeze()[675].detach().cpu().numpy())
    plt.show()

# #TODO vectorized version
# Q11 = torch.zeros((x1.shape[0], self.img_feats.shape[1], x1.shape[1]), device=self.device)
# Q12 = torch.zeros((x1.shape[0], self.img_feats.shape[1], x1.shape[1]), device=self.device)
# Q21 = torch.zeros((x1.shape[0], self.img_feats.shape[1], x1.shape[1]), device=self.device)
# Q22 = torch.zeros((x1.shape[0], self.img_feats.shape[1], x1.shape[1]), device=self.device)
#
# for batch in range(x1.shape[0]):
#     Q11[batch, :, :,] = self.img_feats[batch, :, x1[batch], y1[batch]].clone()
#     Q12[batch, :, :,] = self.img_feats[batch, :, x1[batch], y2[batch]].clone()
#     Q21[batch, :, :,] = self.img_feats[batch, :, x2[batch], y1[batch]].clone()
#     Q22[batch, :, :,] = self.img_feats[batch, :, x2[batch], y2[batch]].clone()