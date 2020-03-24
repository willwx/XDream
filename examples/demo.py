"""
Runs an experiment to maximize activity of a 'neuron,' simulated
    by a unit in a CNN.
Demonstrates a high-level API: `CNNExperiment`, which is designed to simulate an
    electrophysiology experiment and behaves similarly to `EphysExperiment`
"""

from pathlib import Path
import sys
sys.path.append('../xdream')    # or, add to path in your environment
from Experiments import CNNExperiment


# parameters
exp_settings = {
    'project_dir': 'demo',    # change as needed
    'target_neuron': ('alexnet', 'classifier.6', 1),
    # 'target_neuron': ('caffenet', 'fc8', 1),    # toggle with above to use CaffeNet
    'scorer_parameters': {'engine': 'pytorch'},    # comment out to use CaffeNet
    'optimizer_name': 'genetic',
    'optimizer_parameters': {
        'generator_parameters': {'engine': 'pytorch'},    # comment out to use caffe
        'generator_name': 'deepsim-fc6',
        'population_size': 20,
        'mutation_rate': 0.5,
        'mutation_size': 0.5,
        'selectivity': 2,
        'heritability': 0.5,
        'n_conserve': 0},
    'with_write': True,    # simulates file-writing behavior of EphysExperiment
    'image_size': 128,     # comment out to use default generator output size
    'max_optimize_images': 1000,     # query at most this many images
    'random_seed': 0,
    'config_file_path': __file__,    # makes a copy of this file in exp_dir, for debugging/recordkeeping
    # 'stochastic': False,    # simulates neuronal noise
    # 'wait_each_step': 0     # simulates a delay of image presentation
}


Path(exp_settings['project_dir']).mkdir(exist_ok=True)
experiment = CNNExperiment(**exp_settings)
experiment.run()
