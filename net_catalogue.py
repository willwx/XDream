"""
Database of caffe networks to support easy API
To add a new network, edit `net_paths`, `net_io_layers`, `all_classifiers`/`all_generators`, and `net_scales`
"""


import os
from local_settings import nets_dir


net_paths = {'caffenet':      {'definition': os.path.join(nets_dir, 'caffenet', 'caffenet.prototxt'),
                               'weights':    os.path.join(nets_dir, 'caffenet', 'bvlc_reference_caffenet.caffemodel')},
             'deepsim-norm1': {'definition': os.path.join(nets_dir, 'deepsim', 'norm1', 'generator_no_batch.prototxt'),
                               'weights':    os.path.join(nets_dir, 'deepsim', 'norm1', 'generator.caffemodel')},
             'deepsim-norm2': {'definition': os.path.join(nets_dir, 'deepsim', 'norm2', 'generator_no_batch.prototxt'),
                               'weights':    os.path.join(nets_dir, 'deepsim', 'norm2', 'generator.caffemodel')},
             'deepsim-conv3': {'definition': os.path.join(nets_dir, 'deepsim', 'conv3', 'generator_no_batch.prototxt'),
                               'weights':    os.path.join(nets_dir, 'deepsim', 'conv3', 'generator.caffemodel')},
             'deepsim-conv4': {'definition': os.path.join(nets_dir, 'deepsim', 'conv4', 'generator_no_batch.prototxt'),
                               'weights':    os.path.join(nets_dir, 'deepsim', 'conv4', 'generator.caffemodel')},
             'deepsim-pool5': {'definition': os.path.join(nets_dir, 'deepsim', 'pool5', 'generator_no_batch.prototxt'),
                               'weights':    os.path.join(nets_dir, 'deepsim', 'pool5', 'generator.caffemodel')},
             'deepsim-fc6':   {'definition': os.path.join(nets_dir, 'deepsim', 'fc6', 'generator_no_batch.prototxt'),
                               'weights':    os.path.join(nets_dir, 'deepsim', 'fc6', 'generator.caffemodel')},
             'deepsim-fc7':   {'definition': os.path.join(nets_dir, 'deepsim', 'fc7', 'generator_no_batch.prototxt'),
                               'weights':    os.path.join(nets_dir, 'deepsim', 'fc7', 'generator.caffemodel')},
             'deepsim-fc8':   {'definition': os.path.join(nets_dir, 'deepsim', 'fc8', 'generator_no_batch.prototxt'),
                               'weights':    os.path.join(nets_dir, 'deepsim', 'fc8', 'generator.caffemodel')}}

net_io_layers = {'caffenet':      {'input_layer_name':   'data',
                                   'input_layer_shape':  (3, 227, 227,),   # shape is without the first, batch dimension
                                   'output_layer_name':  'fc8',            # for classifier, output is layer before prob
                                   'output_layer_shape': (1000,)},
                 'deepsim-norm1': {'input_layer_name':   'feat',           # modify these names to match prototxt
                                   'input_layer_shape':  (96, 27, 27,),
                                   'output_layer_name':  'generated',
                                   'output_layer_shape': (3, 240, 240,)},
                 'deepsim-norm2': {'input_layer_name':   'feat',
                                   'input_layer_shape':  (256, 13, 13,),
                                   'output_layer_name':  'generated',
                                   'output_layer_shape': (3, 240, 240,)},
                 'deepsim-conv3': {'input_layer_name':   'feat',
                                   'input_layer_shape':  (384, 13, 13,),
                                   'output_layer_name':  'generated',
                                   'output_layer_shape': (3, 256, 256,)},
                 'deepsim-conv4': {'input_layer_name':   'feat',
                                   'input_layer_shape':  (384, 13, 13,),
                                   'output_layer_name':  'generated',
                                   'output_layer_shape': (3, 256, 256,)},
                 'deepsim-pool5': {'input_layer_name':   'feat',
                                   'input_layer_shape':  (256, 6, 6,),
                                   'output_layer_name':  'generated',
                                   'output_layer_shape': (3, 256, 256,)},
                 'deepsim-fc6':   {'input_layer_name':   'feat',
                                   'input_layer_shape':  (4096,),
                                   'output_layer_name':  'generated',
                                   'output_layer_shape': (3, 256, 256,)},
                 'deepsim-fc7':   {'input_layer_name':   'feat',
                                   'input_layer_shape':  (4096,),
                                   'output_layer_name':  'generated',
                                   'output_layer_shape': (3, 256, 256,)},
                 'deepsim-fc8':   {'input_layer_name':   'feat',
                                   'input_layer_shape':  (1000,),
                                   'output_layer_name':  'generated',
                                   'output_layer_shape': (3, 256, 256,)}}


all_classifiers = ('caffenet',)
all_generators = ('deepsim-norm1', 'deepsim-norm2', 'deepsim-conv3', 'deepsim-conv4',
                  'deepsim-pool5', 'deepsim-fc6', 'deepsim-fc7', 'deepsim-fc8')


defined_classifiers = \
    tuple([n for n, p in net_paths.items()
           if os.path.isfile(p['definition']) and os.path.isfile(p['weights']) and n in all_classifiers])
defined_generators = \
    tuple([n for n, p in net_paths.items()
           if os.path.isfile(p['definition']) and os.path.isfile(p['weights']) and n in all_generators])
defined_nets = tuple(list(defined_classifiers) + list(defined_generators))


# whether raw input to net is on the scale of 0-255 (before subtracting mean)
# or something else (e.g., inception networks use scale 0-1)
net_scales = {n: 255 for n in defined_nets}
