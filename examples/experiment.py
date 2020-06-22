"""
Example script for running an experiment to maximize activity of a unit
- images are written to disk
- the program waits for a .mat file that contains responses to the given images;
    in the meantime, some separate program(s) should present the images to a subject,
    record responses from a unit online, and write responses to a .mat file
- the responses are loaded and used to optimize the images
- the process is repeated iteratively until manually stopped (by a keyboard interrupt)
"""

from Experiments import EphysExperiment


# set file directories for I/O
initcode_dir = None
natural_stimuli_dir = None    # to be changed; dir containing control images to be shown along with generated ims
project_dir = 'demo'          # to be changed; dir for writing experiment data & logs


# set parameters
nthreads = 2
nchannels_per_thread = 1          # unused if nthreads == 1
ichannels = None                  # 0-based index or list of indices for channels used in score; None means average all

n_natural_stimuli = 20            # None means default to however many is in natstimdir
cycle_natural_stimuli = True      # True: cycle through images in natstimdir

image_size = 83                   # size (height/width) to which to resize synthesized & natural images
random_seed = 0                   # seed for all random generators used in the experiment (to ensure reproducibility)

optimizer_name = 'genetic'
optimizer_parameters =\
    {'generator_name': 'deepsim-fc6',                        # see net_catalogue for available options
     # 'generator_parameters': {'engine': 'pytorch'},        # see Generators.py for available options
     'initial_codes_dir': initcode_dir,
     'population_size': 20,       # size of population each generation
     'mutation_rate': 0.5,        # fraction of code elements to mutate(on average); range 0 - 1
     'mutation_size': 0.5,        # magnitude of mutation (on average); meaningful range 0 - ~1.5
     'selectivity': 2,            # selective pressure, with higher being more selective; range 0 - inf
     'heritability': 0.5,         # how much one parent (of 2) contributes to each progeny; meaningful range 0.5 - 1
     'n_conserve': 1}             # number of best images to keep untouched per generation; range 0 - populationsize


if __name__ == '__main__':
    experiment = EphysExperiment(
        project_dir, optimizer_name, optimizer_parameters,
        ichannels, nthreads=nthreads, nchannels_per_thread=nchannels_per_thread, image_size=image_size,
        natural_stimuli_dir=natural_stimuli_dir, n_natural_stimuli=n_natural_stimuli,
        cycle_natural_stimuli=cycle_natural_stimuli, random_seed=random_seed, config_file_path=__file__)
    experiment.run()
