import os
import caffe
import numpy as np
import net_catalogue
from local_settings import gpu_available

if gpu_available:
    caffe.set_mode_gpu()

ilsvrc2012_mean = np.array((104.0, 117.0, 123.0))  # ImageNet Mean in BGR order
loaded_nets = {}


def get_paths(net_name):
    assert net_name in net_catalogue.defined_nets, '%s not defined in net_catalogue' % net_name
    net_definition = net_catalogue.net_paths[net_name]['definition']
    net_weights = net_catalogue.net_paths[net_name]['weights']
    for p in (net_definition, net_weights):
        if not os.path.isfile(p):
            raise ValueError('Weights or definition does not exist for net %s (%s)' % (net_name, p))
    return net_definition, net_weights


def load(net_name, fresh_copy=False):
    if not fresh_copy and net_name in loaded_nets.keys():
        return loaded_nets[net_name]    # in general, do not load the same net multiple times
    else:
        net_def, net_weights = get_paths(net_name)
        if os.name == 'nt':
            net = caffe.Net(net_def, caffe.TEST)
            net.copy_from(net_weights)
        else:
            net = caffe.Net(net_def, caffe.TEST, weights=net_weights)
        if not fresh_copy:
            loaded_nets[net_name] = net
        return net


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
