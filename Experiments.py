import os
from shutil import copyfile
from time import time
import numpy as np
import CNNScorers
import Optimizers
import Scorers
import utils
from Logger import Tee

np.set_printoptions(precision=4, suppress=True)


class ExperimentBase:
    """ Base class for Experiment implementing commonly used routines """
    def __init__(self, log_dir, optimizer_name, optimizer_parameters, scorer_name, scorer_parameters, nthreads=1,
                 natural_stimuli_dir=None, n_natural_stimuli=None, save_natural_stimuli_copy=False,
                 random_seed=None, config_file_path=None):
        """
        :param log_dir: str (path), directory to which to write experiment log files
        :param optimizer_name: str or iter of strs, see `defined_optimizers` in `Optimizers.py`
        :param optimizer_parameters: dict, iter of dicts, or dicts with certain iterables values
            interpreted into kwargs passed when initializing the optimizer
            when iterables are interpreted over threads, their len must be 1 or `nthreads`
        :param scorer_name: str, see `defined_scorers` in `Scorers.py`
        :param scorer_parameters: dict, kwargs passed when initializing the scorer
        :param nthreads: int, number of concurrent threads to run; default is 1
        :param natural_stimuli_dir: str (path)
            directory containing "natural"/reference images to show
            interleaved with synthetic stimuli during the experiment
            default is to not show natural images
        :param n_natural_stimuli: int, number of natural images to show per step; default is not to show
        :param save_natural_stimuli_copy: bool
            make a copy in log_dir of natural images shown; default is to not use
            do not use; will not save new images if natural stimuli are changed
        :param random_seed: int
            when set, the experiment will have deterministic behavior; default is an arbitrary integer
        :param config_file_path: str (path),
            path to the .py file defining the experiment,
            intended for saving a copy of it as part of the log
            default is to not save any file
        """
        assert isinstance(nthreads, int) and nthreads > 0, 'nthreads must be a positive integer'
        if not os.path.isdir(log_dir):
            os.mkdir(log_dir)
        if isinstance(optimizer_name, str):
            assert optimizer_name in Optimizers.defined_optimizers
            optimizer_names = tuple([optimizer_name for _ in range(nthreads)])
        else:
            assert hasattr(optimizer_name, '__iter__')
            optimizer_names = tuple(optimizer_name)
            assert len(optimizer_names) == 1 or len(optimizer_names) >= nthreads
            if len(optimizer_names) > nthreads:
                print('note: more values than nthreads passed for optimizer_name; ignoring extra values')
            optimizer_names = tuple([optimizer_name[t % len(optimizer_name)] for t in range(nthreads)])
        optimizer_parameters = optimizer_parameters.copy()
        if hasattr(optimizer_parameters, 'keys'):
            if 'log_dir' not in optimizer_parameters.keys():
                optimizer_parameters['log_dir'] = log_dir
            if 'random_seed' not in optimizer_parameters.keys():
                optimizer_parameters['random_seed'] = random_seed
            optimizer_parameterss = [{} for _ in range(nthreads)]
            for k, param in optimizer_parameters.items():
                if isinstance(param, str) or not hasattr(param, '__len__'):
                    for t in range(nthreads):
                        optimizer_parameterss[t][k] = param
                else:
                    assert len(param) == 1 or len(param) >= nthreads
                    if len(param) > nthreads:
                        print('note: more values than nthreads passed for %s; ignoring extra values' % k)
                    for t in range(nthreads):
                        optimizer_parameterss[t][k] = param[t % len(param)]
        else:
            assert len(optimizer_parameters) == 1 or len(optimizer_parameters) >= nthreads
            if len(optimizer_parameters) > nthreads:
                print('note: more values than nthreads passed for optimizer_parameters; ignoring extra values')
            optimizer_parameterss = []
            for t in range(nthreads):
                params = optimizer_parameters[t % len(optimizer_parameters)]
                assert hasattr(params, 'keys')
                if 'log_dir' not in optimizer_parameters.keys():
                    params['log_dir'] = log_dir
                if 'random_seed' not in params:
                    params['random_seed'] = random_seed
                optimizer_parameterss.append(params.copy())
        assert scorer_name in Scorers.defined_scorers or scorer_name in CNNScorers.defined_scorers
        if 'log_dir' not in scorer_parameters.keys():
            scorer_parameters['log_dir'] = log_dir
        if 'random_seed' not in scorer_parameters.keys():
            scorer_parameters['random_seed'] = random_seed
        assert natural_stimuli_dir is None or os.path.isdir(natural_stimuli_dir)
        # an experiment config file is the file defining experiment parameters, e.g., experiment.py, experiment_CNN.py
        # used for record-keeping
        assert config_file_path is None or os.path.isfile(config_file_path),\
            'experiment config file not found: %s' % config_file_path

        self._istep = -1  # score/optimize cycle we are currently at; will count from 0 once experiment starts
        self._nthreads = max(1, int(nthreads))
        self._optimizers = None
        self._optimizer_parameterss = optimizer_parameterss
        self._scorer = None

        # logging utilities
        self._logdir = log_dir
        self._logger = Tee(os.path.join(log_dir, 'log.txt'))  # helps print to stdout & save a copy to logfpath
        self._config_fpath = config_file_path
        self._copy_config_file()

        # only used if natural stimuli are attached
        self._natstimdir = natural_stimuli_dir  # str
        self._natstimuli = None  # list of images: (w, h, c) arrays, uint8
        self._natstimids = None  # list of strs
        self._natstimfns = None  # list of strs
        self._n_natstim_to_show = 0  # int
        self._all_natstimids = None  # array of strs
        self._all_natstimfns = None  # array of strs
        self._all_natstim_times_shown = None
        if natural_stimuli_dir is not None and \
                (n_natural_stimuli is None or
                 (n_natural_stimuli > 0 and isinstance(n_natural_stimuli, int))):
            self._load_natural_stimuli(natural_stimuli_dir, size=n_natural_stimuli)
            if save_natural_stimuli_copy:
                self._save_natural_stimuli_copy()

        # attach optimizer(s)
        for thread in range(self._nthreads):
            if self._nthreads == 1:
                thread_code = None  # this supresses prefix of 'threadxx' on saved files when running only one thread
            else:
                thread_code = thread
            optimizer = Optimizers.load_optimizer(optimizer_names[thread], thread_code, optimizer_parameterss[thread])
            self._attach_optimizer(optimizer)

        # attach scorer
        if scorer_name in Scorers.defined_scorers:
            scorer = Scorers.load_scorer(scorer_name, scorer_parameters)
        else:
            scorer = CNNScorers.load_scorer(scorer_name, scorer_parameters)
        self._attach_scorer(scorer)

        # apply random seed
        if random_seed is None:
            random_seed = np.random.randint(100000)
            print('%s: random seed not provided, using %d for reproducibility' %
                  (self.__class__.__name__, random_seed))
        else:
            print('%s: random seed set to %d' % (self.__class__.__name__, random_seed))
        self._random_generator = np.random.RandomState(seed=random_seed)
        self._random_seed = random_seed

    def _load_natural_stimuli(self, natstimdir, size=None, natstim_catalogue_fpath=None, shuffle=False):
        all_natstimfns = utils.sort_nicely(
            [fn for fn in os.listdir(natstimdir) if '.bmp' in fn or '.jpg' in fn or '.png' in fn
             or '.tif' in fn or '.tiff' in fn or '.jpeg' in fn or '.JPEG' in fn]
        )
        all_natstimnames = [fn[:fn.rfind('.')] for fn in all_natstimfns]
        nstimuli = len(all_natstimnames)
        all_natstim_times_shown = np.zeros(nstimuli, dtype=int)
        if nstimuli == 0:
            raise Exception('no images found in natual stimulus directory %s' % natstimdir)
        if size is None:
            size = nstimuli
        else:
            size = int(size)

        # make ids (names mapped to catalogue if any) from names (fn without extension)
        all_natstimids = all_natstimnames[:]
        #   try to map filename (no extension) to short id, if catalogue given
        if natstim_catalogue_fpath is not None:
            catalogue = np.load(natstim_catalogue_fpath)
            name_2_id = {name: id_ for name, id_ in zip(catalogue['stimnames'], catalogue['stimids'])}
            for i, name in enumerate(all_natstimids):
                try:
                    all_natstimids[i] = name_2_id[name]
                except KeyError:
                    all_natstimids[i] = name
        #   resolve nonunique ids, if any
        if nstimuli < size:
            for i, id_ in enumerate(all_natstimids):
                icopy = 1
                new_id = id_
                while new_id in all_natstimids[:i]:
                    new_id = '%s_copy%02d' % (id_, icopy)
                all_natstimids[i] = new_id

        # choose images to load (if nstimuli != size)
        toshow_args = np.arange(nstimuli)
        if nstimuli < size:
            if shuffle:
                print('note: number of natural images (%d) < requested (%d); sampling with replacement'
                      % (nstimuli, size))
                toshow_args = self._random_generator.choice(toshow_args, size=size, replace=True)
            else:
                print('note: number of natural images (%d) < requested (%d); repeating images'
                      % (nstimuli, size))
                toshow_args = np.repeat(toshow_args, int(np.ceil(size / float(nstimuli))))[:size]
        elif nstimuli > size:
            if shuffle:
                print('note: number of natural images (%d) > requested (%d); sampling (no replacement)'
                      % (nstimuli, size))
                toshow_args = self._random_generator.choice(toshow_args, size=size, replace=False)
            else:
                print('note: number of natural images (%d) > requested (%d); taking first %d images'
                      % (nstimuli, size, size))
                toshow_args = toshow_args[:size]
        natstimids = [all_natstimids[arg] for arg in toshow_args]
        natstimfns = [all_natstimfns[arg] for arg in toshow_args]
        all_natstim_times_shown[toshow_args] += 1

        # load images
        natstimuli = []
        for natstimfn in natstimfns:
            natstimuli.append(utils.read_image(os.path.join(natstimdir, natstimfn)))

        print('showing the following %d natural stimuli loaded from %s:' % (size, natstimdir))
        print(natstimids)

        # save results
        self._natstimdir = natstimdir
        self._natstimuli = natstimuli
        self._natstimids = natstimids
        self._natstimfns = natstimfns
        self._n_natstim_to_show = size
        self._all_natstimids = np.array(all_natstimids)
        self._all_natstimfns = np.array(all_natstimfns)
        self._all_natstim_times_shown = all_natstim_times_shown

    def _save_natural_stimuli_copy(self):
        assert (self._natstimfns is not None) and (self._natstimdir is not None), \
            'please load natural stimuli first by calling _load_natural_stimuli()'
        savedir = os.path.join(self._logdir, 'natural_stimuli')
        try:
            os.mkdir(savedir)
        except OSError as e:
            if e.errno == 17:
                raise OSError('trying to save natural stimuli but directory already exists: %s' % savedir)
            else:
                raise
        for fn in self._natstimfns:
            copyfile(os.path.join(self._natstimdir, fn), os.path.join(savedir, fn))

    def refresh_natural_stimuli(self):
        """ Refresh natural stimuli with a new set of images from natural_stimuli_dir """
        view = self._random_generator.permutation(len(self._all_natstimfns))
        prioritized_view = np.argsort(self._all_natstim_times_shown[view])[:self._n_natstim_to_show]
        natstimids = list(self._all_natstimids[view[prioritized_view]])
        natstimfns = list(self._all_natstimfns[view[prioritized_view]])
        natstimuli = []
        for natstimfn in natstimfns:
            natstimuli.append(utils.read_image(os.path.join(self._natstimdir, natstimfn)))
        print('changing natural stimuli to the following:')
        print(natstimids)
        self._natstimuli = natstimuli
        self._natstimids = natstimids
        self._natstimfns = natstimfns
        self._all_natstim_times_shown[view[prioritized_view]] += 1

    def _attach_optimizer(self, optimizer):
        if self._optimizers is None:
            self._optimizers = [optimizer]
        else:
            self._optimizers.append(optimizer)

    def _attach_scorer(self, scorer):
        self._scorer = scorer

    def _copy_config_file(self):
        if self._config_fpath is None:
            return
        copyfile(self._config_fpath, os.path.join(self._logdir, 'config.py'))

    def _save_parameters(self):
        paramfpath = os.path.join(self._logdir, 'parameters.txt')
        with open(paramfpath, 'w') as f:
            f.write(str(self.parameters))

    def _load_nets(self):
        if self._optimizers is None:
            raise RuntimeError('cannot load nets before optimizers are attached')
        for optimizer in self._optimizers:
            optimizer.load_generator()

    def run(self):
        """ For implementation of experiment loop """
        raise NotImplementedError

    @property
    def istep(self):
        """ Index of current loop in the experiment """
        return self._istep

    @istep.setter
    def istep(self, istep):
        self._istep = max(0, int(istep))

    @property
    def optimizer(self):

        if self._optimizers is None:
            return None
        elif len(self._optimizers) == 1:
            return self._optimizers[0]
        else:
            raise RuntimeError('more than 1 (%d) optimizers have been loaded; asking for "optimizer" is ambiguous'
                               % len(self._optimizers))

    @property
    def optimizers(self):
        return self._optimizers

    @property
    def scorer(self):
        return self._scorer

    @property
    def natural_stimuli(self):
        return self._natstimuli

    @property
    def natural_stimuli_ids(self):
        return self._natstimids

    @property
    def logger(self):
        return self._logger

    @property
    def logdir(self):
        return self._logdir

    @property
    def parameters(self):
        """ Returns dict storing parameters defining the experiment """
        params = {'class': self.__class__.__name__, 'log_dir': self._logdir,
                  'nthreads': self._nthreads, 'random_seed': self._random_seed}
        if self._nthreads == 1:
            params['optimizer'] = self.optimizer.parameters
        else:
            for thread in range(self._nthreads):
                params['optimizer_thread%02d' % thread] = self.optimizers[thread].parameters
        params['scorer'] = self.scorer.parameters
        params.update({'natural_stimuli_dir': self._natstimdir, 'n_natural_stimuli': self._n_natstim_to_show})
        return params


class EphysExperiment(ExperimentBase):
    """
    Implements an experiment that writes images-to-show to disk,
    waits for and loads responses from .mat file from disk, and iterates automatically
    """
    def __init__(self, project_dir, optimizer_name, optimizer_parameters, ichannels,
                 nthreads=1, nchannels_per_thread=None,
                 mat_dir=None, log_dir=None, image_size=None, reps=None, block_size=None, scorer_parameters=None,
                 natural_stimuli_dir=None, n_natural_stimuli=None, save_natural_stimuli_copy=False,
                 cycle_natural_stimuli=True, random_seed=None, config_file_path=None):
        """
        :param project_dir: str (path), directory for experiment files I/O (write: images and log, read: responses)
        :param optimizer_name: str or iter of strs, see `defined_optimizers` in `Optimizers.py`
        :param optimizer_parameters: dict, iter of dicts, or dicts with certain iterables values
            interpreted into kwargs passed when initializing the optimizer
            when iterables are interpreted over threads, their len must be 1 or `nthreads`
        :param ichannels: int, iterable of [iterable of] ints, None, or Ellipsis
            index of channels on which to listen for responses given response matrix of dimension n_images x n_channels
            int means to return the score at the index
            iterable of [iterable of] ints means, for each index/iterable of indices:
                if index, to return the score at the index
                if iterable of indices, to return the score averaged over the indices
            None means to average over all channels
            Ellipsis means to return all channels
        :param nthreads: int, number of concurrent threads to run; default is 1
        :param nchannels_per_thread: int, offset added to ichannels for successive threads; not needed if nthreads == 1
        :param mat_dir: str (path), directory for reading .mat file responses; default is project_dir
        :param log_dir: str (path), directory for saving experiment logs; default is project_dir/backup
        :param image_size: int, size in pixels to which to resize all images; default is to leave unchanged
        :param reps: int, number of times to show each image
        :param block_size: int
            number of images to write each time before waiting for responses
            default is however many images there are to show each step
        :param scorer_parameters: dict, kwargs passed when initializing the scorer; default is empty
        :param natural_stimuli_dir: str (path)
            directory containing "natural"/reference images to show
            interleaved with synthetic stimuli during the experiment
            default is to not show natural images
        :param n_natural_stimuli: int, number of natural images to show per step; default is not to show
        :param save_natural_stimuli_copy: bool
            make a copy in log_dir of natural images shown; default is to not use
            do not use; will not save new images if natural stimuli are changed
        :param cycle_natural_stimuli: bool
            cycle through images in natural_stimuli_dir, refreshing each step
        :param random_seed: int
            when set, the experiment will have deterministic behavior; default is an arbitrary integer
        :param config_file_path: str (path),
            path to the .py file defining the experiment,
            intended for saving a copy of it as part of the log
            default is to not save any file
        """
        assert os.path.isdir(project_dir), 'project directory %s is not a valid directory' % project_dir
        assert mat_dir is None or os.path.isdir(mat_dir), 'mat file directory %s is not a valid directory' % mat_dir
        if log_dir is None:
            log_dir = os.path.join(project_dir, 'backup')
        if scorer_parameters is None:
            scorer_parameters = {}
        assert isinstance(scorer_parameters, dict)
        scorer_parameters = scorer_parameters.copy()
        if nthreads > 1:
            assert isinstance(nchannels_per_thread, int) and nchannels_per_thread > 0, \
                'must define nchannels_per_thread as a positive integer for multithreaded experiment'
        self._ichannels_input = ichannels
        self._nchannels_per_thread = nchannels_per_thread
        if hasattr(ichannels, '__iter__'):
            for ic in ichannels:
                assert isinstance(ic, int)
        elif ichannels is None:
            ichannels = ...
        else:
            assert isinstance(ichannels, int)
        if nthreads > 1:
            ichannels_new = []
            for thread in range(nthreads):
                ichannels_new.append(thread * nchannels_per_thread + (np.arange(nchannels_per_thread))[ichannels])
            ichannels = ichannels_new
            print('%s: listening on the following channel(s) for each thread' % self.__class__.__name__)
            for thread in range(nthreads):
                print('\tthread %d: %s' % (thread, str(ichannels[thread])))
        else:
            print('%s: listening on the following channel(s): %s' % (self.__class__.__name__, str(ichannels)))

        scorer_name = 'ephys'
        for param, val in zip(('write_dir', 'log_dir', 'channel'), (project_dir, log_dir, ichannels)):
            if param not in scorer_parameters.keys():
                scorer_parameters[param] = val
        for param, val in zip(('mat_dir', 'image_size', 'reps', 'block_size'), (mat_dir, image_size, reps, block_size)):
            if val is not None:
                scorer_parameters[param] = val
        super(EphysExperiment, self).__init__(
            log_dir=log_dir, optimizer_name=optimizer_name, optimizer_parameters=optimizer_parameters,
            scorer_name=scorer_name, scorer_parameters=scorer_parameters,
            natural_stimuli_dir=natural_stimuli_dir, n_natural_stimuli=n_natural_stimuli,
            save_natural_stimuli_copy=save_natural_stimuli_copy,
            nthreads=nthreads, random_seed=random_seed, config_file_path=config_file_path
        )

        self._imsize = self._scorer.parameters['image_size']
        self._cycle_natstim = bool(cycle_natural_stimuli)

    def run(self):
        """ Main experiment loop """
        self._load_nets()  # nets are not loaded in __init__
        self._save_parameters()
        self.istep = 0

        try:
            while True:
                print('\n>>> step %d' % self.istep)
                t00 = time()

                # before scoring, backup codes (optimizer)
                for optimizer in self.optimizers:
                    optimizer.save_current_codes()
                    if hasattr(optimizer, 'save_current_genealogy'):
                        optimizer.save_current_genealogy()
                t01 = time()

                # get scores of images:
                #    1) combine synthesized & natural images
                #    2) write images to disk for evaluation; also, copy them to backup
                #    3) wait for & read results
                syn_nimgs = 0
                syn_sections = [0]
                syn_images = []
                syn_image_ids = []
                for optimizer in self.optimizers:
                    syn_nimgs += optimizer.nsamples
                    syn_sections.append(syn_nimgs)
                    syn_images += optimizer.current_images
                    syn_image_ids += optimizer.current_image_ids
                combined_scores = self.scorer.score(syn_images + self.natural_stimuli,
                                                    syn_image_ids + self.natural_stimuli_ids)
                t1 = time()

                # after scoring, backup scores (scorer)
                self.scorer.save_current_scores()
                t2 = time()

                # use results to update optimizer
                threads_synscores = []
                threads_natscores = []
                for i, optimizer in enumerate(self.optimizers):
                    thread_synscores = combined_scores[syn_sections[i]:syn_sections[i + 1], i]
                    thread_natscores = combined_scores[syn_nimgs:, i]
                    if len(thread_synscores.shape) > 1:  # if scores for list of channels returned, pool channels
                        thread_synscores = np.mean(thread_synscores, axis=-1)
                        thread_natscores = np.mean(thread_natscores, axis=-1)  # unused by optimizer but used in summary
                    threads_synscores.append(thread_synscores)
                    threads_natscores.append(thread_natscores)
                    optimizer.step(thread_synscores)  # update optimizer
                t3 = time()

                # summarize scores & delays, & save log
                for thread in range(self._nthreads):
                    if not self._nthreads == 1:
                        print('thread %d: ' % thread)
                    print('synthetic img scores: mean {}, all {}'.
                          format(np.nanmean(threads_synscores[thread]), threads_synscores[thread]))
                    print('natural image scores: mean {}, all {}'.
                          format(np.nanmean(threads_natscores[thread]), threads_natscores[thread]))
                print(('step %d time: total %.2fs | ' +
                       'wait for results %.2fs  optimizer update %.2fs  write records %.2fs')
                      % (self.istep, t3 - t00, t1 - t01, t3 - t2, t2 - t1 + t01 - t00))
                self.logger.flush()

                # refresh natural stimuli being shown
                if self._cycle_natstim:
                    self.refresh_natural_stimuli()

                self.istep += 1

        # gracefully exit
        except KeyboardInterrupt:
            print()
            print('... keyboard interrupt')
            print('stopped at step %d <<<\n\n' % self.istep)
            self.logger.stop()

    @property
    def parameters(self):
        """ Returns dict storing parameters defining the experiment """
        params = super(EphysExperiment, self).parameters
        params.update({'ichannels': self._ichannels_input, 'nchannels_per_thread': self._nchannels_per_thread})
        return params


class CNNExperiment(ExperimentBase):
    """
    Implements a simulated experiment using a CNN to score images
    """
    def __init__(self, project_dir, optimizer_name, optimizer_parameters, target_neuron, with_write,
                 image_size=None, stochastic=None, stochastic_random_seed=None, reps=None, scorer_parameters=None,
                 natural_stimuli_dir=None, n_natural_stimuli=None, save_natural_stimuli_copy=False,
                 random_seed=None, config_file_path=None, max_images=None, max_steps=None,
                 write_codes=False, write_last_codes=False, write_best_last_code=True,
                 write_last_images=False, write_best_last_image=True):
        """
        :param project_dir: str (path), directory for experiment files [I/]O
        :param optimizer_name: str or iter of strs, see `defined_optimizers` in `Optimizers.py`
        :param optimizer_parameters: dict, iter of dicts, or dicts with certain iterables values
            interpreted into kwargs passed when initializing the optimizer
            when iterables are interpreted over threads, their len must be 1 or `nthreads`
        :param target_neuron: 5-tuple
            (network_name (str), layer_name (str), unit_index (int) [, x_index (int), y_index (int)])
        :param with_write: bool
            if False, images are passed to CNN for evaluation in memory
            if True, images are written to disk, loaded from disk, then passed to CNN for evaluation,
            to fully simulate an EphysExperiment
        :param image_size: int, size in pixels to which to resize all images; default is to leave unchanged
        :param stochastic: bool, whether to inject random Poisson noise to CNN responses; default is False
        :param stochastic_random_seed: int
            when set, the pseudo-stochastic noice will be deterministic; default is to set to `random_seed`
        :param reps: int
            number of stochastic scores to produce for the same image
            only meaningful if stochastic == True
        :param scorer_parameters: dict, kwargs passed when initializing the scorer; default is empty
        :param natural_stimuli_dir: str (path)
            directory containing "natural"/reference images to show
            interleaved with synthetic stimuli during the experiment
            default is to not show natural images
        :param n_natural_stimuli: int, number of natural images to show per step; default is not to show
        :param save_natural_stimuli_copy: bool
            make a copy in log_dir of natural images shown; default is to not use
            do not use; will not save new images if natural stimuli are changed
        :param random_seed: int
            when set, the experiment will have deterministic behavior; default is an arbitrary integer
        :param config_file_path: str (path),
            path to the .py file defining the experiment,
            intended for saving a copy of it as part of the log
            default is to not save any file
        :param max_images: int
            max number of images to show; default is not set (experiment must be interrupted manually)
        :param max_steps: int
            max number of steps to run; superseded by `max_images`; default is not set
        :param write_codes: bool, whether to save codes each step
        :param write_last_codes: bool, whether to save codes at the last step
        :param write_best_last_code: bool, whether to save the best code at the last step
        :param write_last_images: bool, whether to save the images at the last step
        :param write_best_last_image: bool, whether to save the best image at the last step
        """
        assert os.path.isdir(project_dir), 'project directory %s is not a valid directory' % project_dir
        if scorer_parameters is None:
            scorer_parameters = {}
        else:
            assert hasattr(scorer_parameters, 'keys')
            scorer_parameters = scorer_parameters.copy()
        if with_write:
            log_dir = os.path.join(project_dir, 'backup')
            self._write_codes = True
            self._write_last_codes = False
            self._write_last_images = False
            self._write_best_last_code = False
            self._write_best_last_image = False
            scorer_name = 'cnn_with_io'
        else:
            log_dir = project_dir
            self._write_codes = bool(write_codes)
            self._write_last_codes = bool(write_last_codes)
            self._write_last_images = bool(write_last_images)
            self._write_best_last_code = bool(write_best_last_code)
            self._write_best_last_image = bool(write_best_last_image)
            scorer_name = 'cnn_no_io'
        if stochastic_random_seed is None:
            stochastic_random_seed = random_seed
        for param, val in zip(
                ('target_neuron', 'write_dir', 'target_neuron', 'image_size', 'stochastic',
                 'stochastic_random_seed', 'reps'),
                (target_neuron, project_dir, target_neuron, image_size, stochastic, stochastic_random_seed, reps)
        ):
            if param not in scorer_parameters.keys() and val is not None:
                scorer_parameters[param] = val

        super(CNNExperiment, self).__init__(
            log_dir=log_dir, optimizer_name=optimizer_name, optimizer_parameters=optimizer_parameters,
            scorer_name=scorer_name, scorer_parameters=scorer_parameters,
            natural_stimuli_dir=natural_stimuli_dir, n_natural_stimuli=n_natural_stimuli,
            save_natural_stimuli_copy=save_natural_stimuli_copy,
            nthreads=1, random_seed=random_seed, config_file_path=config_file_path
        )

        if max_images is not None:
            if stochastic:
                reps = self.scorer.parameters['reps']
                self._max_steps = int(max_images / self.optimizer.n_samples / reps)
            else:
                self._max_steps = int(max_images / self.optimizer.n_samples)
            self._max_steps = max(1, self._max_steps)
        elif max_steps is not None:
            self._max_steps = max(1, int(max_steps))
        else:
            self._max_steps = None

    def _load_nets(self):
        super(CNNExperiment, self)._load_nets()
        self.scorer.load_classifier()

    def run(self):
        """ Main experiment loop """
        self._load_nets()
        self._save_parameters()
        self.istep = 0

        try:
            while self._max_steps is None or self.istep < self._max_steps:
                print('\n>>> step %d' % self.istep)
                last_codes = self.optimizer.current_samples_copy
                last_images = self.optimizer.current_images
                last_imgids = self.optimizer.current_image_ids
                last_scores = None
                t0 = time()

                # if any natural images to show, show all at the first step
                if self.istep == 0 and self._n_natstim_to_show > 0:
                    # score images
                    natscores = self.scorer.score(self.natural_stimuli, self.natural_stimuli_ids)
                    t1 = time()
                    # backup scores
                    self.scorer.save_current_scores()
                    t2 = time()
                    # summarize scores & delays
                    print('natural image scores: mean {}, all {}'.format(np.nanmean(natscores), natscores))
                    print('step %d time: total %.2fs | wait for results %.2fs  write records %.2fs'
                          % (self.istep, t2 - t0, t1 - t0, t2 - t1))

                else:
                    # score images
                    synscores = self.scorer.score(self.optimizer.current_images, self.optimizer.current_image_ids)
                    t1 = time()
                    # before update, backup codes and scores
                    if self._write_codes:
                        self.optimizer.save_current_codes()
                        if hasattr(self.optimizer, 'save_current_genealogy'):
                            self.optimizer.save_current_genealogy()
                    last_scores = synscores
                    self.scorer.save_current_scores()
                    t2 = time()
                    # use results to update optimizer
                    self.optimizer.step(synscores)
                    t3 = time()
                    # summarize scores & delays
                    print('synthetic img scores: mean {}, all {}'.format(np.nanmean(synscores), synscores))
                    print(('step %d time: total %.2fs | ' +
                           'wait for results %.2fs  write records %.2fs  optimizer update %.2fs')
                          % (self.istep, t3 - t0, t1 - t0, t2 - t1, t3 - t2))

                self.logger.flush()
                self.istep += 1
            print('\nfinished <<<\n\n')

        # gracefully exit
        except KeyboardInterrupt:
            print()
            print('... keyboard interrupt')
            print('stopped at step %d <<<\n\n' % self.istep)

        # save final results when stopped
        try:
            ibest = None
            if last_scores is not None:
                ibest = np.argmax(last_scores)
            if not self._write_codes:
                if self._write_last_codes or (self._write_best_last_code and ibest is None):
                    utils.write_codes(last_codes, last_imgids, self.logdir)
                elif self._write_best_last_code and ibest is not None:
                    utils.write_codes([last_codes[ibest]], [last_imgids[ibest]], self.logdir)
            if self._write_last_images or (self._write_best_last_image and ibest is None):
                utils.write_images(last_images, last_imgids, self.logdir)
            elif self._write_best_last_image and ibest is not None:
                utils.write_images([last_images[ibest]], [last_imgids[ibest]], self.logdir)
        except NameError:
            pass

        self.logger.stop()

    @property
    def parameters(self):
        """ Returns dict storing parameters defining the experiment """
        params = super(CNNExperiment, self).parameters
        params.update({'max_steps': self._max_steps})
        return params
