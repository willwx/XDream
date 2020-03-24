import os
import re
from time import time, sleep

import h5py
import numpy as np
from cv2 import imread, imwrite, resize, cvtColor, INTER_CUBIC, INTER_AREA, COLOR_RGB2GRAY


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
    imarr = imread(image_fpath)
    if imarr is None:
        return None
    else:
        return imarr[:, :, ::-1]


def write_images(imgs, names, path, size=None, grayscale=False, fmt='png', timeout=0.5):
    """
    Save images to given path with given names
    :param imgs: list of images as numpy arrays with shape (w, h, c) and dtype uint8
    :param names: filenames of images including or excluding '.{fmt}'
    :param path: path to save to
    :param size: size (pixels) to resize image to; default is unchanged
    :param grayscale: if True, color images are converted to grayscale
    :param fmt: file format for image (extension without trailing dot)
    :param timeout: timeout for trying to write each image
    :return: None
    """
    for im_arr, name in zip(imgs, names):
        im_arr = resize_image(im_arr, size)
        if grayscale:
            im_arr = cvtColor(im_arr, COLOR_RGB2GRAY)
        else:
            im_arr = im_arr[:, :, ::-1]    # flip RGB to BGR for cv2 imwrite
        trying = True
        t0 = time()
        if os.path.splitext(name)[-1] != f'.{fmt}':
            name += f'.{fmt}'
        while trying and time() - t0 < timeout:
            try:
                imwrite(os.path.join(path, name), im_arr)
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
    assert size <= len(codefns), f'not enough codes ({len(codefns)}) to satisfy size ({size})'
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
    assert size <= len(codefns), f'not enough codes ({len(codefns)}) to satisfy size ({size})'
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
                imgids = np.array([
                    ''.join(np.array(f[ref]).astype(np.uint32).view('U1').flatten()).split('\\')[-1]
                    for ref in imgids_refs
                ])
                scores = np.array(f[response_key])    # shape = (imgs, channels)
            return imgids, scores
        except (KeyError, IOError, OSError):    # if broken file or unable to access
            attempts += 1
            if attempts % 100 == 0:
                print(f'{attempts} failed attempts to read MAT file: {matfpath}')
            sleep(0.001)


def load_h5_score_file(h5fpath, response_key='scores', indices_key='indices'):
    attempts = 0
    while True:
        try:
            with h5py.File(h5fpath, 'r') as f:
                scores = f[response_key][()]
                if indices_key in f.keys():
                    indices = f[indices_key][()]
                    assert len(indices) == len(scores)
                else:
                    indices = None
            return indices, scores
        except (KeyError, IOError, OSError):    # if broken file or unable to access
            attempts += 1
            if attempts % 100 == 0:
                print(f'{attempts} failed attempts to read hdf5 file: {h5fpath}')
            sleep(0.5)


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


def eq2(v1, v2):
    try:
        return v1 == v2
    except ValueError:    # when comparing multi-dim arrays:
        pass

    try:
        return np.all(v1 == v2)
    except ValueError:    # this catch to be updated when error actually happens
        pass

    return v1 is v2


def make_unique_ids(ids0, existing_ids=(), remove_extension=False, sep_replacer=None, vals=None):
    if vals is not None:
        assert len(vals) == len(ids0)
    assert hasattr(existing_ids, '__iter__')

    ids = []
    id2val = {}
    existing_ids = set(existing_ids)
    for i, id0 in enumerate(ids0):
        id_new = id0
        if remove_extension:
            id_new = id_new[:id_new.rfind('.')]
        if isinstance(sep_replacer, str):
            id_new = id_new.replace(os.sep, sep_replacer)
        j = 1
        while id_new in existing_ids:
            if vals is not None and eq2(id2val[id_new], vals[i]):
                break
            id_new = f'{id0}_{j}'
            j += 1
        ids.append(id_new)
        existing_ids.add(id_new)
        if vals is not None:
            id2val[id_new] = vals[i]
    return ids


def center_crop(im):
    """assumes im.shape = (x, y, colors)"""
    im_dim = min(im.shape[:2])
    im_x0 = int((im.shape[0] - im_dim)/2)
    im_y0 = int((im.shape[1] - im_dim)/2)
    return im[im_x0:im_x0+im_dim, im_y0:im_y0+im_dim]


readable_chrs = ''.join(
    chr(c) for c in np.concatenate((np.arange(48, 58), np.arange(65, 91), np.arange(97, 123)))
)


def n2str(n, chrs=readable_chrs):
    base = len(chrs)
    if n < base:
        return chrs[n]
    else:
        return n2str(n // base, chrs) + chrs[n % base]


def save_unicode_array_to_h5(h5_file, dataset_name, data):
    max_ord = np.max(data.flatten().view(np.uint32))
    if max_ord > 2 ** 16:
        dtype = np.uint32
    else:
        dtype = np.uint16
    i_key = 0
    key = f'#refs#/{n2str(i_key)}'
    refs = np.empty(np.prod(data.shape), dtype=h5py.special_dtype(ref=h5py.Reference))
    for i, d in enumerate(data.flatten()):
        while key in h5_file.keys():
            i_key += 1
            key = f'#refs#/{n2str(i_key)}'
        refs[i] = h5_file.create_dataset(name=key, data=d.flatten().view(dtype)).ref
    refs = refs.reshape(data.shape)
    h5_file.create_dataset(name=dataset_name, data=refs)


def read_unicode_array_from_h5(h5_file, dataset_name):
    oref2str = lambda oref: ''.join(h5_file[oref][:].astype(np.uint32).view('U1').flatten())
    dset = h5_file[dataset_name][:]
    return np.array(tuple(map(oref2str, dset.flatten()))).reshape(dset.shape)
