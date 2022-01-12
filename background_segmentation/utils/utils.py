import matplotlib.pyplot as plt
from skimage import exposure
import numpy as np


def plot_img_and_mask(img, mask):
    img = np.array(img)
    img = (img - img.min()) / (img.max() - img.min())
    img = exposure.equalize_adapthist(img)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.set_title('Input image')
    ax.imshow(img, cmap="gray")
    ax.contour(mask, colors="red")
    plt.xticks([]), plt.yticks([])
    plt.show()
