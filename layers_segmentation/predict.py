import logging

import torch
from torchvision import transforms
import matplotlib.pyplot as plt

from cnn_model.unet_model import UNet
import yaml
import cv2
from os.path import dirname, realpath, join

ROOT_REPOSITORY_PATH = dirname(dirname(realpath(__file__)))

N_CHANNELS = 1
N_CLASSES = 6

def preprocess(img, scale):
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    dim = (width, height)

    img = cv2.resize(img, dim)

    tfms = transforms.Compose([
        transforms.ToTensor()   
    ])

    img = tfms(img.copy())[None]  

    return img

def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()

    # img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor))
    
    img = preprocess(full_img, scale_factor)

    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)
        _, _mask = torch.max(output, dim=1)
        masks = _mask.permute(1,2,0).detach().cpu()[:,:,0]

    return masks.numpy()


def load_model(weight_path, device):
    net = UNet(n_channels=N_CHANNELS, n_classes=N_CLASSES)
    net.to(device=device)
    net.load_state_dict(torch.load(weight_path, map_location=device))
    return net


def get_device_auto():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def predict(img, configs):
    device = get_device_auto() if configs["device"] == "auto" else configs["device"]
    weight_path = join(ROOT_REPOSITORY_PATH, configs["weight_path"])
    net = load_model(weight_path, device=device)
    mask = predict_img(net=net,
                       full_img=img,
                       scale_factor=configs["scale"],
                       out_threshold=configs["mask_threshold"],
                       device=device)
    return mask


def load_configs(pth=join(ROOT_REPOSITORY_PATH,
                          "layers_segmentation/configs/configs.yml")):
    with open(pth, "r") as stream:
        configs = yaml.safe_load(stream)
    return configs


if __name__ == '__main__':

    configs = load_configs()

    if configs["debug_mode"]:
        filename = join(ROOT_REPOSITORY_PATH,
                        "layers_segmentation/test/raw_1.tif")
        # img = Image.open(filename)
        img = cv2.imread(filename, -1)

    
    masks = predict(img, configs)
    import matplotlib.pyplot as plt
    if configs["debug_mode"]:
        plt.imshow(masks, cmap='gray')
        # plt.imsave("mask.png", masks)
        plt.show()
