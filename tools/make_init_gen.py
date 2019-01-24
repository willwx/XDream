"""
Script for making init generation codes from given images
"""


import os
import numpy as np
import Generators
from caffe.io import load_image
from cv2 import imwrite
from shutil import copyfile


# parameters
generator_name = 'deepsim-conv4'
generator = Generators.get_generator(generator_name)
srcdirs = ('imgs',)    # to be changed; list of directories containing source images
dstrootdir = os.path.join('codes', generator_name)


# main
for srcdir in srcdirs:
    leafdir = os.path.basename(srcdir)
    dstdir = os.path.join(dstrootdir, leafdir)
    if not os.path.isdir(dstdir):
        os.makedirs(dstdir)
    imgfns = [fn for fn in os.listdir(srcdir) if len(fn) > 3 and fn[-4:] == '.bmp']
    for imgfn in imgfns:
        im = load_image(os.path.join(srcdir, imgfn))
        if generator.name == 'raw_pixel':
            code = generator.encode(im)
        else:
            code = generator.encode(im, steps=200)
        np.save(os.path.join(dstdir, imgfn[:-4]+'.npy'), code, allow_pickle=False)
        imwrite(os.path.join(dstdir, imgfn[:-4]+'_syn.png'), generator.visualize(code)[:, :, ::-1])
        copyfile(os.path.join(srcdir, imgfn), os.path.join(dstdir, imgfn))
