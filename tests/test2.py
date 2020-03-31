"""
Tests optimizing a unit in AlexNet/CaffeNet with DeePSiM generators 1-8
Running time (approx.; steps == 200; n_units == 1):
    5 mins (GTX 2080); 10 mins (GTX 1060)
"""

from time import time
from pathlib import Path
import h5py as h5
import numpy as np
from Experiments import CNNExperiment


save_root = Path('temp')
engine = 'pytorch'    # or caffe; will switch engine in generator and target
optimizer_name = 'genetic'
generator_names = (
    'deepsim-norm1', 'deepsim-norm2', 'deepsim-conv3', 'deepsim-conv4',
    'deepsim-pool5', 'deepsim-fc6', 'deepsim-fc7', 'deepsim-fc8')
steps = 200
target_unit = {    # goldfish neuron
    'caffe': ('caffenet', 'fc8', 1),
    'pytorch': ('alexnet', 'classifier.6', 1)}[engine]
init_rand_seed = 0
exp_settings = {
    'optimizer_name': optimizer_name,
    'optimizer_parameters': {'generator_parameters': {'engine': engine}},
    'scorer_parameters': {'engine': engine},
    'image_size': 85,
    'with_write': False,
    'max_optimize_steps': steps,
    'random_seed': 0,
    'stochastic': False,
    'config_file_path': __file__}
optimizer_setting_names = (
    'population_size', 'mutation_rate', 'mutation_size', 'selectivity',
    'heritability', 'n_conserve')
by_generator_optimizer_settings = {
    'deepsim-norm1': (15, 1, 1.5, 2, 0.5, 0),
    'deepsim-norm2': (10, 0.5, 0.7, 4, 0.5, 0),
    'deepsim-conv3': (12, 0.65, 0.75, 2.25, 0.5, 0),
    'deepsim-conv4': (10, 0.9, 0.75, 2.5, 0.5, 0),
    'deepsim-pool5': (10, 0.6, 1, 2.5, 0.75, 0),
    'deepsim-fc6': (20, 0.5, 0.5, 2, 0.5, 0),
    'deepsim-fc7': (45, 0.6, 0.3, 1.25, 0.5, 0),
    'deepsim-fc8': (20, 0.2, 0.6, 2, 0.5, 0)}
t0 = time()


# main loop
unit = np.atleast_1d(target_unit[2:])
target_s = '_'.join(map(str, unit))
init_randgen = np.random.RandomState(init_rand_seed)
for generator_name in generator_names:
    # set generator-specific parameters
    optimizer_parameters = exp_settings['optimizer_parameters']
    optimizer_parameters['generator_name'] = generator_name
    for p, v in zip(
            optimizer_setting_names,
            by_generator_optimizer_settings[generator_name]):
        optimizer_parameters[p] = v
    exp_settings['optimizer_parameters'] = optimizer_parameters

    # specify initialization (for reproducibility)
    gen_layer_name = generator_name.split('-')[1]
    with h5.File('test_data/stats_caffenet.h5', 'r') as f:
        mu = f[gen_layer_name]['mu'][()]
        sig = f[gen_layer_name]['sig'][()]
    pop_size = exp_settings['optimizer_parameters']['population_size']
    init_codes = init_randgen.normal(mu, sig, size=(pop_size, *mu.shape))
    if generator_name != 'deepsim-fc8':
        # generator was trained on post-ReLU input
        init_codes = np.clip(init_codes, 0, None)

    project_dir = (
        save_root / engine / optimizer_name / generator_name
        / 'random_dist'    # init name
        / target_unit[0] / target_unit[1] / target_s)
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
    exp_settings['target_neuron'] = target_unit
    experiment = CNNExperiment(**exp_settings)
    experiment.optimizer.set_initialization(init_codes)    # manual init
    experiment.run()


# load and print results
print('target unit:\t', target_unit)
for generator_name in generator_names:
    project_dir = (
            save_root / engine / optimizer_name / generator_name
            / 'random_dist'  # init name
            / target_unit[0] / target_unit[1] / target_s)
    acts = []
    for i in (0, steps-1):
        score_f = project_dir / f'scores_step{i:03d}.npz'
        try:
            acts.append(np.load(score_f)['scores'].max())
        except FileNotFoundError:
            acts.append(np.nan)
    print(f'{generator_name}\t{acts[0]:>4.1f} -> {acts[1]:>5.1f}')
print(f'took time: {time() - t0:.0f} s')


"""
Note: for unclear reasons, torch results are slightly different
  on different runs; caffe results are consistent.
results:
>>> cuda 10.1.243, python 3.7.5, Ubuntu 19.10
    >>> torch 1.3.1
        deepsim-norm1	-1.6 ->   8.5
        deepsim-norm2	 0.8 ->   6.7
        deepsim-conv3	 1.6 ->  10.1
        deepsim-conv4	 1.8 ->  12.8
        deepsim-pool5	 2.0 ->  18.2
        deepsim-fc6	 5.4 ->  31.6
        deepsim-fc7	 5.8 ->  58.5
        deepsim-fc8	 6.2 ->  38.7
    >>> caffe
        deepsim-norm1	 0.0 ->  16.4
        deepsim-norm2	 3.5 ->  10.8
        deepsim-conv3	 4.8 ->  15.3
        deepsim-conv4	 5.4 ->  21.8
        deepsim-pool5	 3.5 ->  27.3
        deepsim-fc6	 6.3 ->  51.1
        deepsim-fc7	 6.9 ->  74.1
        deepsim-fc8	 9.1 ->  52.7
>>> cuda 9.0, python 3.7.1 (anaconda), Windows 10
    >>> torch 1.0.1
        deepsim-norm1	-1.6 ->   8.5
        deepsim-norm2	 0.8 ->   6.3
        deepsim-conv3	 1.6 ->   9.5
        deepsim-conv4	 1.8 ->  13.7
        deepsim-pool5	 2.0 ->  19.2
        deepsim-fc6	 5.4 ->  26.6
        deepsim-fc7	 5.8 ->  45.7
        deepsim-fc8	 6.2 ->  35.9
    >>> caffe
        deepsim-norm1	 0.0 ->  15.2
        deepsim-norm2	 3.5 ->  11.4
        deepsim-conv3	 4.8 ->  15.7
        deepsim-conv4	 5.4 ->  21.1
        deepsim-pool5	 3.5 ->  27.6
        deepsim-fc6	 6.3 ->  49.8
        deepsim-fc7	 6.9 ->  64.3
        deepsim-fc8	 9.1 ->  49.1
"""
