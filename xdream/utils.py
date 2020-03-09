import os
import re
from cv2 import imread, imwrite, resize, INTER_CUBIC, INTER_AREA
from time import time, sleep
import h5py
import numpy as np


def resize_image(im_arr, size):
    """
    Resize an image to a given (square) size
    :param im_arr: image array of size (w, h, c)
    :param size: int, size in pixels to resize to
    :return: resized image
    """
    if size is not None and im_arr.shape[1] != size:
        if im_arr.shape[1] < size:    # upsampling
            im_arr = resize(im_arr, (size, size), interpolation=INTER_CUBIC)
        else:                         # downsampling
            im_arr = resize(im_arr, (size, size), interpolation=INTER_AREA)
    return im_arr


def read_image(image_fpath):
    """
    :param image_fpath: path to image
    :return: image (size: (w, h, c), color order: RGB)
    """
    # BGR is flipped to RGB.
    #     Note In the case of color images, the decoded images will have the channels stored in B G R order.
    #     https://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html
    # why BGR?:
    #     https://www.learnopencv.com/why-does-opencv-use-bgr-color-format/
    imarr = imread(image_fpath)[:, :, ::-1]
    return imarr


def write_images(imgs, names, path, size=None, timeout=0.5):
    """
    Save images as 24-bit bmp files to given path with given names
    :param imgs: list of images as numpy arrays with shape (w, h, c) and dtype uint8
    :param names: filenames of images including or excluding '.bmp'
    :param path: path to save to
    :param size: size (pixels) to resize image to; default is unchanged
    :param timeout: timeout for trying to write each image
    :return: None
    """
    for im_arr, name in zip(imgs, names):
        im_arr = resize_image(im_arr, size)
        trying = True
        t0 = time()
        if name.rfind('.bmp') != len(name) - 4:
            name += '.bmp'
        while trying and time() - t0 < timeout:
            try:
                imwrite(os.path.join(path, name), im_arr[:, :, ::-1])
                trying = False
            except IOError as e:
                if e.errno != 35:
                    raise
                sleep(0.01)


def write_codes(codes, names, path, timeout=0.5):
    """
    Save codes as npy files (1 in each file) to given path with given names
    :param codes: list of images as numpy arrays with shape (w, h, c) and dtype uint8
    :param names: filenames of images, excluding extension
    :param path: path to save to
    :param timeout: timeout for trying to write each code
    :return: None
    """
    for name, code in zip(names, codes):
        trying = True
        t0 = time()
        while trying and time() - t0 < timeout:
            try:
                np.save(os.path.join(path, name), code, allow_pickle=False)
                trying = False
            except (OSError, IOError) as e:
                if e.errno != 35 and e.errno != 89:
                    raise
                sleep(0.01)


def savez(fpath, save_kwargs, timeout=1):
    """ wraps numpy.savez, implementing OSError tolerance within timeout """
    trying = True
    t0 = time()
    while trying and time() - t0 < timeout:
        try:
            np.savez(fpath, **save_kwargs)
            trying = False
        except IOError as e:
            if e.errno != 35:
                raise
            sleep(0.01)


save_scores = savez    # a synonym for backward compatibility


def load_codes(codedir, size):
    """
    Load codes, stored in .npy files, from directory
    :param codedir: directory containing .npy files
    :param size: number of codes to load
    :return: list of codes (1-D np arrays)
    """
    # make sure enough codes for requested size
    codefns = sorted([fn for fn in os.listdir(codedir) if '.npy' in fn])
    assert size <= len(codefns), 'not enough codes (%d) to satisfy size (%d)' % (len(codefns), size)
    # load codes
    codes = []
    for codefn in np.random.choice(codefns, size=min(len(codefns), size), replace=False):
        code = np.load(os.path.join(codedir, codefn), allow_pickle=False).flatten()
        codes.append(code)
    codes = np.array(codes)
    return codes


def load_codes2(codedir, size, rand_gen=None):
    """ Same as load_codes, but also returns filename of each loaded .npy file """
    # make sure enough codes for requested size
    codefns = sorted([fn for fn in os.listdir(codedir) if '.npy' in fn])
    assert size <= len(codefns), 'not enough codes (%d) to satisfy size (%d)' % (len(codefns), size)
    # load codes
    if rand_gen is None:
        codefns = list(np.random.choice(codefns, size=min(len(codefns), size), replace=False))
    else:
        codefns = list(rand_gen.choice(codefns, size=min(len(codefns), size), replace=False))
    codes = []
    for codefn in codefns:
        code = np.load(os.path.join(codedir, codefn), allow_pickle=False).flatten()
        codes.append(code)
    codes = np.array(codes)
    return codes, codefns


def load_block_mat(matfpath, image_id_key='stimulusID', response_key='tEvokedResp'):
    """
    Load a .MAT file (hdf5 format) storing image IDs and responses to each image
    :param matfpath: path to MAT file
    :param image_id_key: variable name in .mat file storing image IDs
    :param response_key: variable name in .mat file storing responses to each image
    :return: image IDs, scores
    """
    attempts = 0
    while True:
        try:
            with h5py.File(matfpath, 'r') as f:
                imgids_refs = np.array(f[image_id_key])[0]
                imgids = []
                for ref in imgids_refs:
                    imgpath = ''.join(chr(i) for i in f[ref])
                    imgids.append(imgpath.split('\\')[-1])
                imgids = np.array(imgids)
                scores = np.array(f[response_key])    # shape = (imgs, channels)
            return imgids, scores
        except (KeyError, IOError, OSError):    # if broken mat file or unable to access
            attempts += 1
            if attempts % 100 == 0:
                print('%d failed attempts to read .mat file' % attempts)
            sleep(0.001)


# https://nedbatchelder.com/blog/200712/human_sorting.html
def tryint(s):
    try:
        return int(s)
    except ValueError:
        return s


def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [tryint(c) for c in re.split('([0-9]+)', s)]


def sort_nicely(l):
    """ Return the given list sorted in the way that humans expect.
    """
    newl = l[:]
    newl.sort(key=alphanum_key)
    return newl


def make_ids_unique(ids0):
    """
    Given a list of IDs, make them unique by appending serial numbers to repeated occurences
    :param ids0: input list of IDs
    :return: list of unique IDs
    """
    ids = []
    for id0 in ids0:
        i = 1
        id_ = id0
        while id_ in ids:
            id_ = '%s_%d' % (id0, i)
            i += 1
        ids.append(id_)
    return ids
