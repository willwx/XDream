"""
Database of neural networks to support easy API
To add a new network, edit `net_paths`, `net_io_layers`,
    `all_classifiers`/`all_generators`, and `net_scales`
"""
from os.path import join, exists
from numpy import array
from .local_settings import nets_dir


__all__ = ['refresh_available_nets',
           'net_paths_exist', 'available_nets',
           'available_classifiers', 'available_generators']


# paths for nets
#   - manual entries for two classifiers
net_paths = {
    'caffe': {
        'caffenet': {'definition':  join(nets_dir, 'caffenet', 'caffenet.prototxt'),
                     'weights': join(nets_dir, 'caffenet', 'bvlc_reference_caffenet.caffemodel')},
    },
    'pytorch': {
        'alexnet':  {'weights': join(nets_dir, 'pytorch', 'alexnet', 'alexnet-owt-4df8aa71.pt')}
                     # 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth'
    }
}
#   - DeePSiM generators
_deepsim_layers = ('norm1', 'norm2', 'conv3', 'conv4', 'pool5', 'fc6', 'fc7', 'fc8')
for n in _deepsim_layers:
    net_paths['caffe'][f'deepsim-{n}'] = {
        'definition': join(nets_dir, 'caffe', 'deepsim', n, 'generator_no_batch.prototxt'),
        'weights': join(nets_dir, 'caffe', 'deepsim', n, 'generator.caffemodel')
    }
    net_paths['pytorch'][f'deepsim-{n}'] = {
        'weights': join(nets_dir, 'pytorch', 'deepsim', f'{n}.pt')
    }


# #####################################
# Metadata
# #####################################
# 1. Type: generator or classifier
# 2. Layer info:
#    Layer info is various used in modules that use a neural net.
#        It is usually inferred from a loaded net when availble. However, in
#        multiprocessing cases involving caffe, nets have to be loaded in each
#        subprocess separately, while dependent modules like Generator need to
#        be constructed beforehand in the main process. This necessiates a
#        static catelog of layer info.
#    For caffe classifiers only,
#        input layer shape should be defined (name assumed to be 'data' by
#        convention); this is used to automatically construct a data transformer
#        when loading net
#    For generators,
#        if caffe: input and output layer shapes and names should be defined
#            (see prototxt for names);
#        if pytorch: input layer shape should be defined.
#        Input shapes allow Generator to know the shape of the image code.
#        Output shape is only used when Generator.reencode() is called
#    Shape is without the first (batch) dimension.
#    This is not per-engine; i.e., nets with the same name are
#        assumed to have the same layer shapes & names regardless of engine.
# 3. Scale for preprocessing:
#    The range of image data input/output for net (without subtracting mean)
#    E.g., scale of 255 means image is on the scale of 0-255.
#    Notably, inception networks use scale 0-1, and pretrained pytorch networks
#    use something else (see 'alexnet').
net_meta = {'caffenet':      {'type': 'classifier', 'input_layer_shape': (3, 227, 227,)},
            'alexnet':       {'type': 'classifier', 'input_layer_shape': (3, 224, 224,),
                              'scale': 1 / array([0.229, 0.224, 0.225])[:, None, None]},
            'deepsim-norm1': {'input_layer_shape':  (96, 27, 27,)},
            'deepsim-norm2': {'input_layer_shape':  (256, 13, 13,)},
            'deepsim-conv3': {'input_layer_shape':  (384, 13, 13,)},
            'deepsim-conv4': {'input_layer_shape':  (384, 13, 13,)},
            'deepsim-pool5': {'input_layer_shape':  (256, 6, 6,)},
            'deepsim-fc6':   {'input_layer_shape':  (4096,)},
            'deepsim-fc7':   {'input_layer_shape':  (4096,)},
            'deepsim-fc8':   {'input_layer_shape':  (1000,)}}
for d in net_meta.values():
    d.setdefault('scale', 255)
for n in _deepsim_layers:
    net_meta[f'deepsim-{n}'].update({
        'type': 'generator',
        'input_layer_name': 'feat',
        'output_layer_name': 'generated',
        'output_layer_shape': (3, 240, 240,) if 'norm' in n else (3, 256, 256)
    })


net_paths_exist = available_nets = available_classifiers \
    = available_generators = None


def refresh_available_nets():
    global net_paths_exist, available_nets, available_classifiers, available_generators
    net_paths_exist = {
        engine: {net_name: {path_name: bool(exists(path))
                            for path_name, path in paths.items()}
                 for net_name, paths in nets_paths.items()}
        for engine, nets_paths in net_paths.items()
    }
    available_nets = {
        engine: tuple(net_name for (net_name, paths_exist) in nets_paths_exist.items()
                      if all(paths_exist.values()))
        for engine, nets_paths_exist in net_paths_exist.items()
    }
    available_classifiers = {
        engine: tuple(net_name for net_name in net_names
                      if net_meta[net_name]['type'] == 'classifier')
        for engine, net_names in available_nets.items()
    }
    available_generators = {
        engine: tuple(net_name for net_name in net_names
                      if net_meta[net_name]['type'] == 'generator')
        for engine, net_names in available_nets.items()
    }


refresh_available_nets()
