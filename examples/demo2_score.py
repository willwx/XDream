"""
Runs alongside `demo2_main.py` to evaluate proposed images
Demonstrates:
    - the format of score files expected by `scorer` in main script
    - `NoIOCNNScorer`: a wrapper around a CNN; specifies target unit(s),
        takes images, and returns activation of the target units
"""

from pathlib import Path
from time import time, sleep
import h5py as h5
import numpy as np
import sys
sys.path.append('../xdream')    # or, add to path in your environment
import utils
from CNNScorers import NoIOCNNScorer


# parameters
exp_dir = Path('demo')
score_dir = exp_dir / 'scores'
evo_log_dir = exp_dir / 'backup'
image_dir = exp_dir / 'stimuli'
steps = 50    # match main script
wait_each_step = 3

# format: (net_name, layer_name, unit index(/indices))
#   (add [, x, y] for conv layers)
# default setting is the goldfish and ambulance units in alexnet (pre-softmax)
# `channel_idc` in the main script is relative to indices here
target_neuron = ('alexnet', 'classifier.6', [1, 407])
scorer = NoIOCNNScorer(
    log_dir=evo_log_dir, target_neuron=target_neuron, engine='pytorch')


# main loop
for i_block in range(steps):

    # this loop checks image_dir for this block's images (blockxxx*.png)
    # every second  until there is no change
    n_ims_old = 0
    while True:
        imfps = sorted(fp for fp in image_dir.iterdir()
                       if fp.suffix == '.png'
                       and f'block{i_block:03d}' in fp.name)
        if len(imfps) > 0 and len(imfps) == n_ims_old:
            break
        n_ims_old = len(imfps)
        sleep(1)

    print(f'scoring block{i_block:03d}...', end=' ')
    t0 = time()

    # read images and obtain unit activations
    scores = [scorer._score_image(utils.read_image(str(fp))) for fp in imfps]

    # save scores to score file
    scores = np.array(scores).reshape(len(scores), -1)
    with h5.File(score_dir / f'block{i_block:03d}.h5', 'w') as f:
        f.create_dataset('scores', data=scores)
    print(f'done ({time() - t0:.1f} s)')

    if wait_each_step > 0:
        sleep(wait_each_step)
