import logging

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from utils.data_loading import BasicDataset
from unet.unet_model import UNet
from utils.utils import plot_img_and_mask
import yaml


N_CHANNELS = 1
N_CLASSES = 6


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

        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)
        else:
            probs = torch.sigmoid(output)

        probs = probs.squeeze(0)

        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(full_img.size[1]),
                transforms.ToTensor()
            ]
        )

        probs = tf(probs.cpu())
        full_mask = probs.squeeze().cpu().numpy()

    return full_mask > out_threshold


def load_model(weight_path, device):
    net = UNet(n_channels=N_CHANNELS, n_classes=N_CLASSES)
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

    
    mask = predict(img, configs)

    if configs["debug_mode"]:
        plot_img_and_mask(img, mask)