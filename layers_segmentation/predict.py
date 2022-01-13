import logging

import torch
import torch.nn.functional as F
from PIL import Image

from utils.data_loading import BasicDataset
from unet.unet_model import UNet
from utils.utils import plot_img_and_mask
import yaml

def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()

    img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor))

    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)
        probs = F.softmax(output, dim=1)
        probs = probs.squeeze(0)
        masks = probs.cpu().numpy()

    return masks > out_threshold


def load_model(weight_path, device):
    net = UNet(n_channels=1, n_classes=6)
    net.to(device=device)
    net.load_state_dict(torch.load(weight_path, map_location=device))
    return net


def get_device_auto():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def predict(img, configs):
    device = get_device_auto() if configs["device"] == "auto" else configs["device"]
    net = load_model(configs["weight_path"], device=device)
    mask = predict_img(net=net,
                       full_img=img,
                       scale_factor=configs["scale"],
                       out_threshold=configs["mask_threshold"],
                       device=device)
    return mask


def load_configs(pth="configs\configs.yml"):
    with open(pth, "r") as stream:
        configs = yaml.safe_load(stream)
    return configs


if __name__ == '__main__':

    configs = load_configs()

    if configs["debug_mode"]:
        filename = "raw_1.tif"
        img = Image.open(filename)

    
    masks = predict(img, configs)

    import matplotlib.pyplot as plt
    if configs["debug_mode"]:
        for mask in masks:
            print(mask.shape, mask.sum())
            plt.imshow(mask)
            plt.show()