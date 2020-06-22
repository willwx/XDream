import os
import numpy as np
try:
    import caffe
    caffe_available = True
except ImportError:
    caffe = None
    caffe_available = False
try:
    import torch
    from net_utils import torch_nets
    torch_available = True
except ImportError:
    torch = None
    torch_available = False
from .local_settings import gpu_available
from .net_catalogue import *
from .net_catalogue import net_paths, net_meta
from .transformer import Transformer

if not (caffe_available or torch_available):
    raise ImportError('please install either caffe or pytorch')
if caffe_available and gpu_available:
    caffe.set_mode_gpu()


__all__ = ['supported_engines', 'load_net', 'get_transformer']


supported_engines = ('caffe', 'pytorch')    # only these two libraries are supported here
ilsvrc2012_mean_bgr = np.array((104.0, 117.0, 123.0)) / 255    # ImageNet Mean in BGR order
ilsvrc2012_mean_rgb_1 = np.array((0.485, 0.456, 0.406))        # RGB; for pytorch models
loaded_nets = {}


def get_paths(net_name, engine):
    assert engine in supported_engines, f'engine {engine} is not supported'

    # check that net files are specified in catalogue
    assert net_name in net_paths[engine], \
        f'{net_name} (engine: {engine}) is not defined in net_catalogue'

    # check whether files are available for net; if not, specify what is missing
    assert net_name in available_nets[engine], \
        f'{net_name} (engine: {engine}) not available; ' + \
        '; '.join(f'{path_name} path exists: {exists}'
                  for path_name, exists in net_paths_exist[engine][net_name].items()) + \
        '. Please check paths in net_catalogue.net_paths'

    path_dict = net_paths[engine][net_name]
    return path_dict.get('definition', None), path_dict['weights']


def load_net(net_name, engine=None, fresh_copy=False):
    # in general, do not load the same net multiple times
    if not fresh_copy:
        try:
            return loaded_nets[(net_name, engine)], engine
        except KeyError:
            pass

    if engine is None:
        engine = 'caffe' if caffe_available else 'pytorch'

    if engine == 'caffe':
        assert caffe_available, 'please install caffe to use caffe nets'
        net_def, net_weights = get_paths(net_name, 'caffe')
        if os.name == 'nt':
            # tested to work on windows caffe
            # installed from conda-forge (by willyd)
            net = caffe.Net(net_def, caffe.TEST)
            net.copy_from(net_weights)
        else:
            net = caffe.Net(net_def, caffe.TEST, weights=net_weights)
    elif engine == 'pytorch':
        assert torch_available, 'please install pytorch to use torch nets'
        net = torch_nets.load_net(net_name)
        _, net_weights = get_paths(net_name, 'pytorch')
        if gpu_available:
            net.load_state_dict(torch.load(net_weights, map_location='cuda'))
            net.cuda()
        else:
            net.load_state_dict(torch.load(net_weights, map_location='cpu'))
        net.eval()    # only using pretrained nets for inference here
    else:
        raise ValueError(f'engine {engine} is not supported')

    if not fresh_copy:
        loaded_nets[(net_name, engine)] = net

    return net, engine


def get_transformer(net_name, net_engine='caffe',
                    net=None, scale=None, outputs_image=None):
    if scale is None:
        scale = net_meta[net_name].get('scale', 255)
    if outputs_image is None:
        try:
            outputs_image = net_meta[net_name]['type'] == 'generator'
        except KeyError:
            outputs_image = False
    im_layer_shape = None
    if net is not None and net_engine == 'caffe':    # pytorch has no uniform API for layer shape
        im_layer_name = net.outputs if outputs_image else net.inputs
        im_layer_name = im_layer_name.__iter__().__next__()
        im_layer_shape = tuple(net.blobs[im_layer_name].shape[1:])
    if im_layer_shape is None:    # try static catalogue
        try:
            key = 'output_layer_shape' if outputs_image else 'input_layer_shape'
            im_layer_shape = net_meta[net_name][key]
        except KeyError:
            pass
    transformer = Transformer({'data': im_layer_shape})
    transformer.set_transpose('data', (2, 0, 1))           # move color channels to outermost dimension
    if net_engine == 'caffe' or 'deepsim' in net_name:     # pytorch version of deepsim still uses caffe preprocessing
        transformer.set_channel_swap('data', (2, 1, 0))    # swap channels from RGB to BGR
        transformer.set_mean(                              # subtract the dataset-mean value in each channel
            'data', (ilsvrc2012_mean_bgr.reshape(-1, 1, 1) * scale).flatten())
    else:
        transformer.set_mean(                              # default mean for pretrained pytorch nets
            'data', (ilsvrc2012_mean_rgb_1.reshape(-1, 1, 1) * scale).flatten())
    transformer.set_raw_scale('data', scale)               # should be set if pixel values are in [0, scale], not [0, 1]
    return transformer
