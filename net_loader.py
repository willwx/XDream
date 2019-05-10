import os
import numpy as np
try:
    import caffe
    caffe_available = True
except ImportError:
    caffe_available = False
try:
    import torch
    import torch_nets
    torch_available = True
except ImportError:
    torch_available = False
import net_catalogue
from local_settings import gpu_available
from caffe_transformer import Transformer

if not (caffe_available or torch_available):
    raise ImportError('Both caffe and pytorch are missing! Please install either library')


if caffe_available and gpu_available:
    caffe.set_mode_gpu()

ilsvrc2012_mean_bgr = np.array((104.0, 117.0, 123.0)) / 255    # ImageNet Mean in BGR order
ilsvrc2012_mean_rgb_1 = np.array((0.485, 0.456, 0.406))        # RGB; for pytorch models
loaded_nets = {}


def get_paths(net_name, engine):
    if engine not in net_catalogue.defined_engines:
        raise ValueError(f'engine {engine} is not defined in net_catalogue')
    if net_name not in net_catalogue.net_paths[engine]:
        raise ValueError(f'{net_name} (engine: {engine}) not defined in net_catalogue')
    assert net_name in net_catalogue.available_nets[engine], f'{net_name} (engine: {engine}) not available; ' +\
        '; '.join(
            f'{file_name} file exists: {net_catalogue.net_paths_exist[engine][net_name][file_name]}'
            for file_name in net_catalogue.net_paths_exist[engine][net_name]
        ) + '. Please check paths in net_catalogue.net_paths'
    net_definition = net_catalogue.net_paths[engine][net_name].get('definition', None)
    net_weights = net_catalogue.net_paths[engine][net_name]['weights']
    for p in (net_definition, net_weights):
        if p is not None and not os.path.isfile(p):
            raise ValueError(f'Weights or definition file does not exist for net {net_name} at {p}')
    return net_definition, net_weights


def load(net_name, engine=None, fresh_copy=False):
    if not fresh_copy:
        try:        # in general, do not load the same net multiple times
            return loaded_nets[(net_name, engine)], engine
        except KeyError:
            pass

    if engine is None:
        engine = 'caffe' if caffe_available else 'pytorch'
    else:
        assert engine in net_catalogue.defined_engines, f'engine {engine} not defined in net_catalogue'

    if engine == 'caffe':
        assert caffe_available
        net_def, net_weights = get_paths(net_name, 'caffe')
        if os.name == 'nt':
            net = caffe.Net(net_def, caffe.TEST)
            net.copy_from(net_weights)
        else:
            net = caffe.Net(net_def, caffe.TEST, weights=net_weights)
    elif engine == 'pytorch':
        assert torch_available
        net = torch_nets.load_net(net_name)
        _, net_weights = get_paths(net_name, 'pytorch')
        if gpu_available:
            net.load_state_dict(torch.load(net_weights, map_location='cuda'))
        else:
            net.load_state_dict(torch.load(net_weights, map_location='cpu'))
        net.eval()
        if gpu_available:
            net.cuda()
    else:
        raise NotImplemented(f"net engine '{engine}' is not currently supported")
    if not fresh_copy:
        loaded_nets[(net_name, engine)] = net

    return net, engine


def get_transformer(net_name, net_engine='caffe', scale=None, outputs_image=False):
    if scale is None:
        try:
            scale = net_catalogue.net_scales[net_name]
        except KeyError:
            scale = 255
    if outputs_image:    # True if is a generator
        transformer = Transformer({'data': (1, *net_catalogue.net_io_layers[net_name]['output_layer_shape'])})
    else:
        transformer = Transformer({'data': (1, *net_catalogue.net_io_layers[net_name]['input_layer_shape'])})
    transformer.set_transpose('data', (2, 0, 1))       # move color channels to outermost dimension
    transformer.set_raw_scale('data', scale)           # should be set if pixel values are in [0, scale], not [0, 1]
    if net_engine == 'caffe' or 'deepsim' in net_name:     # pytorch version of deepsim still uses caffe preprocessing
        transformer.set_channel_swap('data', (2, 1, 0))    # swap channels from RGB to BGR
        transformer.set_mean('data',                       # subtract the dataset-mean value in each channel
                             (ilsvrc2012_mean_bgr.reshape(-1, 1, 1) * scale).flatten())
    else:
        transformer.set_mean('data', (ilsvrc2012_mean_rgb_1.reshape(-1, 1, 1) * scale).flatten())
    return transformer
