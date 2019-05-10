"""
Example script for running an experiment to maximize activity of CNN units (without backprop)
"""

import os
from Experiments import CNNExperiment


# some arbitrary targets
target_neuron = ('alexnet', 'classifier.6', 1)    # for caffenet, use ('caffenet', 'fc8', 1)
project_dir = os.path.expanduser('~/Documents/data/with_CNN2')
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
        'generator_parameters': {'engine': 'pytorch'}
    },
    'with_write': True,
    'image_size': 64,
    'max_images': 1000,
    'random_seed': 0,
    'stochastic': False,
    'config_file_path': __file__,
    # 'wait_each_step': 0
}
scorer_parameters = {'engine': 'pytorch'}    # comment out to use CaffeNet in caffe


if __name__ == '__main__':
    os.makedirs(project_dir, exist_ok=True)
    exp_settings['scorer_parameters'] = scorer_parameters    # comment out to use CaffeNet in caffe
    exp_settings['project_dir'] = project_dir
    exp_settings['target_neuron'] = target_neuron
    experiment = CNNExperiment(**exp_settings)
    experiment.run()
