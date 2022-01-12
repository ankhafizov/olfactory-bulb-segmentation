import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from utils.data_loading import BasicDataset
from unet.unet_model import UNet
from utils.utils import plot_img_and_mask

def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)[0]
        else:
            probs = torch.sigmoid(output)[0]

        tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((full_img.size[1], full_img.size[0])),
            transforms.ToTensor()
        ])

        full_mask = tf(probs.cpu()).squeeze()

    if net.n_classes == 1:
        return (full_mask > out_threshold).numpy()
    else:
        return F.one_hot(full_mask.argmax(dim=0), net.n_classes).permute(2, 0, 1).numpy()


def load_model(weight_path, device):
    net = UNet(n_channels=1, n_classes=2)
    net.to(device=device)
    net.load_state_dict(torch.load(weight_path, map_location=device))
    return net


if __name__ == '__main__':
    weight_path = "model_binarization.pth"
    filename = "raw_1.tif"
    scale = 0.5
    mask_threshold = 0.5
    debug = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    logging.info(f'Loading model weights {weight_path}')
    net = load_model(weight_path, device=device)
    logging.info('Model loaded!')

    # logging.info(f'\nPredicting image {filename} ...')
    # img = Image.open(filename)
    # mask = predict_img(net=net,
    #                    full_img=img,
    #                    scale_factor=scale,
    #                    out_threshold=mask_threshold,
    #                    device=device)

    # if debug:
    #     plot_img_and_mask(img, mask)