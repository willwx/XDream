"""
Example script for running an experiment to maximize activity of CNN units (without backprop)
"""

import os
from Experiments import CNNExperiment


# some arbitrary targets
target_neurons = (('caffenet', 'fc8', 1), ('caffenet', 'fc8', 407), ('caffenet', 'fc8', 632),
                  # ('placesCNN', 'fc8', 55), ('placesCNN', 'fc8', 74), ('placesCNN', 'fc8', 162),
                  # ('googlenet', 'loss3/classifier', 1), ('googlenet', 'loss3/classifier', 407),
                  # ('resnet-152', 'fc1000', 1), ('resnet-152', 'fc1000', 407),
                  )
project_root_dir = 'temp_exp'    # to be changed; dir for writing experiment data & logs
generator_name = 'deepsim-fc6'
exp_settings = {
    'optimizer_name': 'genetic',
    'optimizer_parameters': {
        'generator_name': generator_name,
        'population_size': 20,
        'mutation_rate': 0.5,
        'mutation_size': 0.5,
        'selectivity': 2,
        'heritability': 0.5,
        'n_conserve': 0,
    },
    'with_write': False,
    'image_size': 64,
    'max_images': 1000,
    'random_seed': 0,
    'stochastic': False,
    'config_file_path': __file__,
}


if __name__ == '__main__':
    for target_neuron in target_neurons:
        neuron = target_neuron
        subdir = '_'.join([str(i).replace('/', '_') for i in neuron])
        project_dir = os.path.join(project_root_dir, subdir)
        if not os.path.isdir(project_dir):
            os.makedirs(project_dir)
        else:
            continue
        exp_settings['project_dir'] = project_dir
        exp_settings['target_neuron'] = target_neuron

        experiment = CNNExperiment(**exp_settings)
        experiment.run()
