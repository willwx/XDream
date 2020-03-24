"""
Runs an experiment to maximize activity of a 'neuron,' simulated
    by a unit in a CNN
Demonstrates low-level APIs:
    - optimizer (`Genetic`)
    - scorer (`EphysScorer`)
    - `ReferenceImagesLoader`
These are the components that make up high-level APIs like `EphysExperiment`
"""

from pathlib import Path
from time import time
import numpy as np
import sys
sys.path.append('../xdream')    # or, add to path in your environment
from Experiments import ReferenceImagesLoader
from Optimizers import Genetic
from Scorers import EPhysScorer


# ==============================================================================
# parameters
steps = 50
exp_dir = Path('demo')
image_dir = exp_dir / 'stimuli'
score_dir = exp_dir / 'scores'
backup_dir = exp_dir / 'backup'    # saves images and codes during opt process
n_threads = 2    # how many optimizers to run concurrently
optimizer_parameters = {
    'generator_name': 'deepsim-fc6',                  # see net_catalogue for options
    'generator_parameters': {'engine': 'pytorch'},    # comment out to use caffe
    'initial_codes_dir': None,
    'population_size': 20,       # size of population each generation
    'mutation_rate': 0.5,        # fraction of code elements to mutate(on average); range 0 - 1
    'mutation_size': 0.5,        # magnitude of mutation (on average); meaningful range 0 - ~1.5
    'selectivity': 2,            # selective pressure, with lower being more selective; range 0 - inf
    'heritability': 0.5,         # how much one parent (of 2) contributes to each progeny; meaningful range 0.5 - 1
    'n_conserve': 1,             # number of best images to keep untouched per generation; range 0 - populationsize
    'log_dir': backup_dir}

# the channel each optimizer targets;
# relative to what's defined in `demo2_score.py`
channel_idc = tuple(range(n_threads))

# ==============================================================================
# initialize scorer, ref_im_loader, and optimizers
for d in (exp_dir, image_dir, score_dir, backup_dir):
    d.mkdir(exist_ok=True)    # first, create required dirs

# scorer:
# It doesn't literally 'score'; rather, it writes images to image_dir,
# and waits for a 'score file', to be provided externally, that contains the
# corresponding scores. In this case, the score file is provided by
# `demo2_score.py`. In an actual experiment, the score file will be provided by
# the recording software, and `EPhysScorer` can be modified to interface with
# the specific file format (in the class method `_with_io_get_scores`)
scorer = EPhysScorer(
    image_dir, log_dir=backup_dir, score_dir=score_dir,
    image_size=128,        # None to use native generator output size
    backup_images=True,    # if False, evo images will not be saved (codes will)
    channel=channel_idc,
    score_format='h5')

# optimizer: It
# - integrates a 'generator', such as a generative neural network
# - defines a `current_images` property that contains the current images
# - defines a `step()` method that uses scores corresponding to the current
#   images to update the images (by updating their underlying image codes)
optimizers = [Genetic(thread=i, **optimizer_parameters) for i in range(n_threads)]

# ref_im_loader: a utility to load images to show alongside optimized images
ref_im_loader = ReferenceImagesLoader(
    reference_images_dir='ref_images',
    n_reference_images=2)


# ==============================================================================
# preamble
image_loaders = [ref_im_loader] + optimizers
assert len(channel_idc) == len(optimizers), \
    'channel indices and optimizers must be 1-to-1 corresponding'
for optim in optimizers:
    optim.save_current_codes()


# ==============================================================================
# main loop
for i_step in range(steps):
    print('\n'+'='*80)
    print('step', i_step)

    # collect current images
    curr_ims = []
    curr_imids = []
    curr_split_idc = []
    for im_loader in image_loaders:
        curr_ims += im_loader.current_images
        curr_imids += im_loader.current_image_ids
        curr_split_idc.append(len(curr_ims))

    # show them and wait for scores
    print('writing new images and waiting for scores...')
    t0 = time()
    scores = scorer.score(curr_ims, curr_imids)
    print(f'done ({time() - t0:.1f} s)\n')

    # update optimizers
    print('updating optimizers...')
    t0 = time()
    byloader_scores = np.split(scores, curr_split_idc[:-1])
    i_optim = 0
    for loader, scores in zip(image_loaders, byloader_scores):
        if hasattr(loader, 'step'):
            loader.step(scores[:, i_optim])
            loader.save_current_codes()
            i_optim += 1
        else:
            loader.refresh_images()
    print(f'done ({time() - t0:.1f} s)\n')
