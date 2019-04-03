import os
try:
    import caffe
    caffe_avail = True
except ImportError:
    caffe_avail = False
try:
    import torch
    import torch_nets
    torch_avail = True
except ImportError:
    torch_avail = False
import numpy as np
import net_catalogue
from local_settings import gpu_available

if not (caffe_avail or torch_avail):
    raise ImportError('Both caffe and pytorch are missing! Please install either library')


if gpu_available:
    caffe.set_mode_gpu()

ilsvrc2012_mean = np.array((104.0, 117.0, 123.0))  # ImageNet Mean in BGR order
loaded_nets = {}


def get_paths(net_name, engine):
    assert engine in net_catalogue.defined_nets and net_name in net_catalogue.defined_nets[engine], \
        f'{net_name} (engine: {engine}) not defined in net_catalogue'
    net_definition = net_catalogue.net_paths[engine][net_name].get('definition', None)
    net_weights = net_catalogue.net_paths[engine][net_name]['weights']
    for p in (net_definition, net_weights):
        if p is not None and not os.path.isfile(p):
            raise ValueError(f'Weights or definition file does not exist for net {net_name} at {p}')
    return net_definition, net_weights


def load(net_name, engine=None, fresh_copy=False):
    if not fresh_copy and net_name in loaded_nets.keys():
        return loaded_nets[net_name]    # in general, do not load the same net multiple times
    else:
        if engine is None:
            engine = 'caffe' if caffe_avail else 'pytorch'
        else:
            assert engine in net_catalogue.defined_engines, f'engine {engine} not defined in net_catalogue'
        if engine == 'caffe':
            assert caffe_avail
            net_def, net_weights = get_paths(net_name, 'caffe')
            if os.name == 'nt':
                net = caffe.Net(net_def, caffe.TEST)
                net.copy_from(net_weights)
            else:
                net = caffe.Net(net_def, caffe.TEST, weights=net_weights)
        elif engine == 'pytorch':
            assert torch_avail
            net = torch_nets.load_net(net_name)
            _, net_weights = get_paths(net_name, 'pytorch')
            net.load_state_dict(torch.load(net_weights))
            if gpu_available:
                net.to('cuda')
        else:
            raise NotImplemented
        if not fresh_copy:
            loaded_nets[net_name] = net
        return net, engine


def get_transformer(net_name, scale=None, outputs_image=False):
    if scale is None:
        try:
            scale = net_catalogue.net_scales[net_name]
        except KeyError:
            scale = 255
    if outputs_image:    # True if is a generator
        transformer = caffe.io.Transformer({'data': (1, *net_catalogue.net_io_layers[net_name]['output_layer_shape'])})
    else:
        transformer = caffe.io.Transformer({'data': (1, *net_catalogue.net_io_layers[net_name]['input_layer_shape'])})
    transformer.set_transpose('data', (2, 0, 1))       # move color channels to outermost dimension
    transformer.set_raw_scale('data', scale)           # should be set if pixel values are in [0, scale], not [0, 1]
    transformer.set_mean('data', ilsvrc2012_mean / (255 / scale))    # subtract the dataset-mean value in each channel
    transformer.set_channel_swap('data', (2, 1, 0))    # swap channels from RGB to BGR
    return transformer
