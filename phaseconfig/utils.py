import math
import numpy as np
from scipy import ndimage


def next_power_of_two(number):
    """Compute the next power of two of the *number*."""
    return 2 ** int(math.ceil(math.log(number, 2)))


def get_coord_range(size, inverse=True):
    if inverse:
        return (np.arange(size) + 1 - size * 0.5)[::-1]
    else:
        return (np.arange(size) + 1 - size * 0.5)


def swap(v, pos=None):
    if pos == None:
        pos = np.argwhere(v == np.min(np.abs(v)))[0][0]
    return np.concatenate((v[pos:], v[:pos]))


def smooth_1d(f, window_len=11):
    if f.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    w = np.hamming(window_len)
    s = np.r_[f[window_len-1:0:-1], f, f[-2:-window_len-1:-1]]
    cf = np.convolve(w / w.sum(), s, mode='valid')
    return cf[(window_len-1)/2:-(window_len-1)/2]


def get_profile(f, angle=None):
    if f.ndim != 2:
        raise ValueError, "get_profile only accepts 2 dimension arrays."

    H, W = f.shape
    if angle == None:
        # Estimate the angle for a diagonal line
        angle = np.arctan(float(H)/W)

    L = np.ceil(W / np.cos(angle))
    r = np.arange(L) - L * 0.5
    x = r * np.cos(angle) + W * 0.5
    y = r * np.sin(angle) + H * 0.5

    profile = ndimage.map_coordinates(f, np.vstack((y, x)))
    return r, profile


def write_image(image, filename):
    """Write image to a file *filename*."""
    if filename.lower().endswith('.tif') or filename.lower().endswith('.tiff'):
        import tifffile
        tifffile.imsave(filename, image)
    elif '.edf' in filename.lower():
        import fabio
        edf = fabio.edfimage.edfimage(image)
        edf.write(filename)
    else:
        raise ValueError('Unsupported image format')
