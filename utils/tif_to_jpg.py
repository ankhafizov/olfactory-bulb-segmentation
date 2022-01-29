import cv2
import matplotlib.pyplot as plt
import matplotlib
from skimage import exposure

filename = "raw_256"
img = cv2.imread(f"{filename}.tif", -1)
img = (img - img.min()) / (img.max() - img.min())
img = exposure.equalize_adapthist(img, clip_limit=0.03)
print(img.shape)

matplotlib.image.imsave(f'{filename}.jpg', img, cmap="gray")

plt.imshow(img)
plt.show()