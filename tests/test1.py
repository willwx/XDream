"""
Tests optimizing a few layers in AlexNet/CaffeNet
    with DeePSiM-fc6 and genetic optimizer
Running time (approx.; steps == 200)
- n_units == 1 (default): 3 mins (GTX 2080); 6 mins (GTX 1060)
- n_units == None (uses all 25): 2 hrs (GTX 2080); 5 hrs (GTX 1060)
"""

from time import time
from pathlib import Path
import h5py as h5
import numpy as np
from Experiments import CNNExperiment


save_root = Path('temp')
engine = 'pytorch'    # or caffe; will switch engine in generator and target
optimizer_name = 'genetic'
generator_name = 'deepsim-fc6'
steps = 200
n_units = 1    # num units to test out of all 25 in target_neurons.h5
init_rand_seed = 0
exp_settings = {
    'optimizer_name': optimizer_name,
    'optimizer_parameters': {
        'generator_name': generator_name,
        'generator_parameters': {'engine': engine},
        'population_size': 20,
        'mutation_rate': 0.5,
        'mutation_size': 0.5,
        'selectivity': 2,
        'heritability': 0.5,
        'n_conserve': 0},
    'scorer_parameters': {'engine': engine},
    'image_size': 85,
    'with_write': False,
    'max_optimize_steps': steps,
    'random_seed': 0,
    'stochastic': False,
    'config_file_path': __file__}
t0 = time()


# load target neurons
with h5.File('test_data/target_neurons.h5', 'r') as f:
    target_net = f[f'{engine}/target_net'][()]
    target_layers = f[f'{engine}/target_layers'][()].astype(np.str_)
    target_units = [f[f'{engine}/target_units'][layer][()]
                    for layer in target_layers]


# specify initialization (for reproducibility)
with h5.File('test_data/stats_caffenet.h5', 'r') as f:
    mu = f['fc6/mu'][()]
    sig = f['fc6/sig'][()]
init_randgen = np.random.RandomState(init_rand_seed)
pop_size = exp_settings['optimizer_parameters']['population_size']
init_codes = init_randgen.normal(mu, sig, size=(pop_size, *mu.shape))
init_codes = np.clip(init_codes, 0, None)    # generator was trained on post-ReLU input


# main loop
save_root2 = (save_root / engine / optimizer_name / generator_name
              / 'random_dist'    # init name
              / target_net)
for layer, units in zip(target_layers, target_units):
    n_units = n_units if n_units else len(units)
    for unit in units[:n_units]:
        unit = np.atleast_1d(unit)
        target_s = '_'.join(map(str, unit))
        project_dir = save_root2 / layer / target_s
        try:
            project_dir.mkdir(parents=True)
        except FileExistsError:
            # skips finished, running, or aborted experiments
            # this allows running test1.py in parallel with (almost) no
            # interference (unless timing is so precise---for example in batch
            # jobs---that two processes attempted to create the same folder
            # at the same time. Downside is, for aborted experiments,
            # project_dir needs to be manually removed
            continue
        exp_settings['project_dir'] = project_dir
        exp_settings['target_neuron'] = (target_net, layer) + tuple(unit)
        experiment = CNNExperiment(**exp_settings)
        experiment.optimizer.set_initialization(init_codes)    # manual init
        experiment.run()


# load and print results
for layer, units in zip(target_layers, target_units):
    init_acts = []
    evo_acts = []
    for unit in units[:n_units]:
        target_s = '_'.join(map(str, np.atleast_1d(unit)))
        for l, i in zip((init_acts, evo_acts), (0, steps-1)):
            score_f = save_root2 / layer / target_s / f'scores_step{i:03d}.npz'
            try:
                l.append(np.load(score_f)['scores'].max())
            except FileNotFoundError:
                continue
    init_acts = np.array(init_acts)
    evo_acts = np.array(evo_acts)
    if n_units > 1:
        print(f'{target_net+" "+layer+":":<20}\t'
              f'{init_acts.mean():>4.1f} +/- {init_acts.std():>4.1f} (init) -> '
              f'{evo_acts.mean():>5.1f} +/- {evo_acts.std():>5.1f} (evolved)')
    else:
        print(f'{target_net+" "+layer+":":<20}\t'
              f'{init_acts.mean():>4.1f} -> {evo_acts.mean():>5.1f}')
print(f'took time: {time() - t0:.0f} s')


"""
results (n_units == 1):
>>> cuda 10.1.243, python 3.7.5, Ubuntu 19.10
    >>> torch 1.3.1
        alexnet features.3: 	 2.3 ->  94.2
        alexnet features.8: 	 0.1 ->  39.7
        alexnet classifier.1:	 0.0 ->  23.9
        alexnet classifier.6:	 2.7 ->  36.6
    >>> caffe
        caffenet conv2:     	38.2 -> 246.9
        caffenet conv4:     	36.0 -> 213.1
        caffenet fc6:       	 0.2 ->  74.9
        caffenet fc8:       	 5.5 ->  43.3
>>> cuda 9.0, python 3.7.1 (anaconda), Windows 10
    >>> torch 1.0.1
        alexnet features.3: 	 2.3 -> 111.7
        alexnet features.8: 	 0.1 ->  40.2
        alexnet classifier.1:	 0.0 ->  19.0
        alexnet classifier.6:	 2.7 ->  44.1
    >>> caffe
        caffenet conv2:     	38.2 -> 224.3
        caffenet conv4:     	36.0 -> 226.1
        caffenet fc6:       	 0.2 ->  86.8
        caffenet fc8:       	 5.5 ->  44.6
"""
