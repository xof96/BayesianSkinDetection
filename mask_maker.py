import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imshow

n_images = 10

if __name__ == '__main__':
    for n in range(n_images):
        img_path = f'assets/person{n}_mask.jpg'

        # open image
        img = imread(fname=img_path, as_gray=True)
        mask = np.zeros(img.shape)
        mask[img >= 0.7] = 0
        mask[img < 0.7] = 1
        plt.imsave(fname=f'assets/mask{n}.jpg', arr=mask)
