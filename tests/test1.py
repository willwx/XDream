from pathlib import Path
import h5py as h5
import numpy as np
from Experiments import CNNExperiment


save_root = Path('test1')
optimizer_name = 'genetic'
generator_name = 'deepsim-fc6'
generator_engine = 'pytorch'
init_rand_seed = 0

exp_settings = {
    'optimizer_name': optimizer_name,
    'optimizer_parameters': {
        'generator_name': generator_name,
        'generator_parameters': {'engine': generator_engine},
        'population_size': 20,
        'mutation_rate': 0.5,
        'mutation_size': 0.5,
        'selectivity': 2,
        'heritability': 0.5,
        'n_conserve': 0
    },
    'image_size': 85,
    'with_write': False,
    'max_optimize_images': 10000,
    'random_seed': 0,
    'stochastic': False,
    'config_file_path': __file__,
}

with h5.File('test_data/target_neurons.h5', 'r') as f:
    target_net = f['target_net'][()]
    target_layers = f['target_layers'][()].astype(np.str_)
    target_units = [f['target_units'][layer][()] for layer in target_layers]

# specify initialization (for reproducibility)
with h5.File('test_data/stats_caffenet_fc6.h5', 'r') as f:
    mu = f['mu'][()]
    sig = f['sig'][()]
    # cov = f['cov'][()]
init_randgen = np.random.RandomState(init_rand_seed)
pop_size = exp_settings['optimizer_parameters']['population_size']
init_codes = init_randgen.normal(mu, sig, size=(pop_size, *mu.shape))
# init_codes = init_randgen.multivariate_normal(mu, cov, size=pop_size)
init_codes = np.clip(init_codes, 0, None)    # generator was trained on post-ReLU input

save_root2 = (save_root / optimizer_name / generator_name
              / 'random_dist'    # init name
              / target_net)
for layer, units in zip(target_layers, target_units):
    for unit in units:
        unit = np.atleast_1d(unit)
        target_s = '_'.join(map(str, unit))
        project_dir = save_root2 / layer / target_s
        try:
            project_dir.mkdir(parents=True)
        except FileExistsError:
            # skips finished, running, or aborted experiments
            # this allows running test1.py in parallel with (almost) no
            # interference (unless timing is so precise, for example in batch jobs,
            # that two processes attempted to create the same folder at the same time
            # downside is, for aborted experiments, project_dir needs to be removed
            continue
        exp_settings['project_dir'] = project_dir
        exp_settings['target_neuron'] = (target_net, layer) + tuple(unit)
        experiment = CNNExperiment(**exp_settings)
        experiment.optimizer.set_initialization(init_codes)    # manual init
        experiment.run()

for layer, units in zip(target_layers, target_units):
    evo_acts = []
    for unit in units:
        target_s = '_'.join(map(str, np.atleast_1d(unit)))
        score_f = save_root2 / layer / target_s / 'scores_step499.npz'
        try:
            evo_acts.append(np.load(score_f)['scores'].max())
        except FileNotFoundError:
            continue
    evo_acts = np.array(evo_acts)
    print(f'{target_net} {layer:<10} evolved activation:\t'
          f'{evo_acts.mean():.1f} +/- {evo_acts.std():.1f}')
