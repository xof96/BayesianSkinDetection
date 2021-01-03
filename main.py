import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
from utils import is_skin

n_images = 10

if __name__ == '__main__':

    for n in range(n_images):
        print('Image:', n, '#####################')
        img_path = f'assets/person{n}.jpg'
        true_mask_path = f'assets/mask{n}.jpg'

        thetas = np.linspace(0.1, 2, 20)

        # open image
        img = imread(fname=img_path)
        height, width, channels = img.shape

        mask = imread(fname=true_mask_path, as_gray=True)
        true_mask = np.zeros(shape=(height, width))
        true_mask[mask >= 0.5] = 1

        tpr = []
        fpr = []

        for th in thetas:
            print('Theta value:', th)
            curr_mask = is_skin(x=img, th=th)

            curr_tp = np.zeros(shape=(height, width))
            curr_tp[curr_mask == 1] = 1
            curr_tp[true_mask == 0] = 0

            curr_fn = np.zeros(shape=(height, width))
            curr_fn[true_mask == 1] = 1
            curr_fn[curr_mask == 1] = 0

            curr_fp = np.zeros(shape=(height, width))
            curr_fp[curr_mask == 1] = 1
            curr_fp[true_mask == 1] = 0

            curr_tn = np.zeros(shape=(height, width))
            curr_tn[true_mask == 0] = 1
            curr_tn[curr_mask == 1] = 0

            curr_tpr = curr_tp.sum() / (curr_tp.sum() + curr_fn.sum())
            curr_fpr = curr_fp.sum() / (curr_fp.sum() + curr_tn.sum())
            print('TPR', curr_tpr)
            print('FPR', curr_fpr)

            tpr.append(curr_tpr)
            fpr.append(curr_fpr)
        print(tpr)
        print(fpr)
        tpr.insert(0, 1)
        fpr.insert(0, 1)
        tpr.append(0)
        fpr.append(0)
        plt.plot(fpr, tpr, label=f'Image{n}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('Talse Positive Rate')
    plt.title('ROC-Curves')
    plt.legend()
    plt.show()
