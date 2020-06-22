"""
Tests optimizing a few layers in AlexNet/CaffeNet
    with DeePSiM-fc6 and three optimizers: genetic, FDGD, NES
Running time (approx.; steps == 200)
- n_units == 1 (default): 8 mins (GTX 2080); 15 mins (GTX 1060)
"""

from time import time
from pathlib import Path
import h5py as h5
import numpy as np
from Experiments import CNNExperiment
from copy import deepcopy

save_root = Path('temp')
engine = 'pytorch'    # or caffe; will switch engine in generator and target
optimizer_names = ('genetic', 'FDGD', 'NES')
generator_name = 'deepsim-fc6'
steps = 200
n_units = 1    # num units to test out of all 25 in target_neurons.h5
init_rand_seed = 0
exp_settings = {
    'optimizer_parameters': {
        'generator_name': generator_name,
        'generator_parameters': {'engine': engine}},
    'scorer_parameters': {'engine': engine},
    'image_size': 85,
    'with_write': False,
    'max_optimize_steps': steps,
    'random_seed': 0,
    'stochastic': False,
    'config_file_path': __file__}
by_optimizer_params = {    # for deepsim-fc6 generator
    'genetic': {'population_size': 20, 'mutation_rate': 0.5,
                'mutation_size': 0.5, 'selectivity': 2, 'heritability': 0.5},
    'FDGD': {'n_samples': 20, 'search_radius': 1.25, 'learning_rate': 1.25,
             'antithetic': True},
    'NES': {'n_samples': 20, 'search_radius': 1.25, 'learning_rate': 1.75,
            'antithetic': True, 'search_radius_learning_rate': 0.05}}
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


# main loop
opt_params0 = deepcopy(exp_settings['optimizer_parameters'])
for optimizer_name in optimizer_names:
    # set optimizer-specific parameters
    save_root2 = (save_root / engine / optimizer_name / generator_name
                  / 'random_dist'    # init name
                  / target_net)
    opt_params = deepcopy(opt_params0)
    opt_params.update(by_optimizer_params[optimizer_name])
    exp_settings['optimizer_name'] = optimizer_name
    exp_settings['optimizer_parameters'] = opt_params

    # get appropriate-sized init
    if optimizer_name == 'genetic':
        pop_size = opt_params['population_size']
        init_codes = init_randgen.normal(mu, sig, size=(pop_size, *mu.shape))
    else:
        # init_codes = init_randgen.normal(mu, sig, size=mu.shape)
        init_codes = np.zeros(mu.shape)
    init_codes = np.clip(init_codes, 0, None)    # generator was trained on post-ReLU input

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
                # interference (unless timing is so precise---for ex in batch
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
for optimizer_name in optimizer_names:
    # set optimizer-specific parameters
    save_root2 = (save_root / engine / optimizer_name / generator_name
                  / 'random_dist'    # init name
                  / target_net)
    print('optimizer:', optimizer_name)
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
            print(f'\t{target_net+" "+layer+":":<20}\t'
                  f'{init_acts.mean():>4.1f} +/- {init_acts.std():>4.1f} (init) -> '
                  f'{evo_acts.mean():>5.1f} +/- {evo_acts.std():>5.1f} (evolved)')
        else:
            print(f'\t{target_net+" "+layer+":":<20}\t'
                  f'{init_acts.mean():>4.1f} -> {evo_acts.mean():>5.1f}')
print(f'took time: {time() - t0:.0f} s')


"""
Note: for unclear reasons, torch results are slightly different
  on different runs; caffe results are consistent.
results (n_units == 1):
>>> cuda 10.1.243, python 3.7.5, Ubuntu 19.10
    >>> torch 1.3.1
        optimizer: genetic
            alexnet features.3: 	 2.3 -> 100.6
            alexnet features.8: 	 0.1 ->  42.7
            alexnet classifier.1:	 0.0 ->  24.8
            alexnet classifier.6:	 2.7 ->  43.5
        optimizer: FDGD
            alexnet features.3: 	 1.2 ->  82.2
            alexnet features.8: 	 0.0 -> 105.2
            alexnet classifier.1:	 0.0 ->  32.6
            alexnet classifier.6:	-0.1 ->  54.5
        optimizer: NES
            alexnet features.3: 	 1.2 ->  84.4
            alexnet features.8: 	 0.0 -> 105.7
            alexnet classifier.1:	 0.0 ->  23.2
            alexnet classifier.6:	-0.1 ->  57.7
    >>> caffe
        optimizer: genetic
            caffenet conv2:     	38.2 -> 246.9
            caffenet conv4:     	36.0 -> 213.1
            caffenet fc6:       	 0.2 ->  74.9
            caffenet fc8:       	 5.5 ->  43.3
        optimizer: FDGD
            caffenet conv2:     	14.3 -> 285.5
            caffenet conv4:     	40.4 -> 230.6
            caffenet fc6:       	-4.7 -> 107.0
            caffenet fc8:       	 0.9 ->  71.7
        optimizer: NES
            caffenet conv2:     	14.3 -> 253.0
            caffenet conv4:     	40.4 -> 242.4
            caffenet fc6:       	-4.7 -> 118.5
            caffenet fc8:       	 0.9 ->  70.8
>>> cuda 9.0, python 3.7.1 (anaconda), Windows 10
    >>> torch 1.0.1
        optimizer: genetic
            alexnet features.3: 	 2.3 -> 119.4
            alexnet features.8: 	 0.1 ->  41.2
            alexnet classifier.1:	 0.0 ->  25.7
            alexnet classifier.6:	 2.7 ->  37.2
        optimizer: FDGD
            alexnet features.3: 	 1.2 ->  81.6
            alexnet features.8: 	 0.0 ->  90.2
            alexnet classifier.1:	 0.0 ->  34.7
            alexnet classifier.6:	-0.1 ->  61.4
        optimizer: NES
            alexnet features.3: 	 1.2 ->  80.1
            alexnet features.8: 	 0.0 ->  98.0
            alexnet classifier.1:	 0.0 ->  30.0
            alexnet classifier.6:	-0.1 ->  61.7
    >>> caffe
        optimizer: genetic
            caffenet conv2:     	38.2 -> 224.3
            caffenet conv4:     	36.0 -> 226.1
            caffenet fc6:       	 0.2 ->  86.8
            caffenet fc8:       	 5.5 ->  44.6
        optimizer: FDGD
            caffenet conv2:     	14.3 -> 233.0
            caffenet conv4:     	40.4 -> 245.0
            caffenet fc6:       	-4.7 -> 118.3
            caffenet fc8:       	 0.9 ->  71.7
        optimizer: NES
            caffenet conv2:     	14.3 -> 302.4
            caffenet conv4:     	40.4 -> 254.9
            caffenet fc6:       	-4.7 -> 128.3
            caffenet fc8:       	 0.9 ->  68.7
"""
