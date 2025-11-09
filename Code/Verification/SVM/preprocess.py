import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import ndimage
from skimage.filters import threshold_otsu


def rgbgrey(img):
    if img.ndim == 3:
        return np.mean(img, axis=2)
    return img


def greybin(img):
    blur_radius = 0.8
    img = ndimage.gaussian_filter(img, blur_radius)
    thres = threshold_otsu(img)
    binimg = img > thres
    return np.logical_not(binimg)


def preproc(path=None, img=None, display=True):
    if img is None:
        import matplotlib.image as mpimg
        img = mpimg.imread(path)

    if display:
        plt.imshow(img)
        plt.show()

    grey = rgbgrey(img)
    if display:
        plt.imshow(grey, cmap=cm.Greys_r)
        plt.show()

    binimg = greybin(grey)
    if display:
        plt.imshow(binimg, cmap=cm.Greys_r)
        plt.show()

    r, c = np.where(binimg == 1)
    if r.size == 0 or c.size == 0:
        print("WARNING: No signature detected, returning full image")
        signimg = binimg
    else:
        signimg = binimg[r.min(): r.max(), c.min(): c.max()]

    if display:
        plt.imshow(signimg, cmap=cm.Greys_r)
        plt.show()

    signimg = (255 * signimg).astype("uint8")
    return signimg
