import os
from shutil import copyfile
from time import time, sleep

import h5py as h5
import numpy as np

import CNNScorers
import Optimizers
import Scorers
import utils
from Logger import Tee

np.set_printoptions(precision=4, suppress=True)


class ReferenceImagesLoader:
    image_extensions = ('.bmp', '.jpg', '.png', '.tif', '.tiff', '.jpeg')

    def __init__(self, reference_images_dir, dynamic_reference_images_dir=None, n_reference_images=None,
                 random_seed=None, shuffle=True, max_recursion_depth=None,
                 sep_replacer='-', crop_center_square=True, max_n_loaded=500, verbose=False,
                 max_n_dynref=None, dynref_ref_ratio=1,
                 cache_dir='cache', use_cache=True, save_cache=False, idx=None):
        """
        # TODO
        :param reference_images_dir:  str (path)
            directory containing "natural"/reference images to show
            interleaved with synthetic stimuli during the experiment
            default is to not show natural images
        :param dynamic_reference_images_dir:
        :param n_reference_images: int
            number of natural images to show per step; default is not to show
        :param random_seed:
        :param shuffle:
        :param max_recursion_depth:
        :param sep_replacer:
        :param crop_center_square:
        :param max_n_loaded:
        :param verbose:
        :param max_n_dynref:
        :param dynref_ref_ratio:
        :param cache_dir:
        :param use_cache:
        :param idx:
        """
        assert os.path.isdir(reference_images_dir)
        if dynamic_reference_images_dir is not None:
            dynamic_reference_images_dir = str(dynamic_reference_images_dir)
        self._refimdir = str(reference_images_dir)
        self._dynrefimdir = dynamic_reference_images_dir
        self._max_n_dynref = None if max_n_dynref is None else int(max_n_dynref)
        self._dynref_rps_seen = set()
        self._dynref_ref_ratio = float(dynref_ref_ratio)
        if n_reference_images is None:
            self._nrefims = None
        else:
            assert n_reference_images >= 1, f'invalid number of requested reference images: {n_reference_images}'
            assert n_reference_images <= max_n_loaded, \
                f'requested more reference images {n_reference_images} than max number of loaded images {max_n_loaded}'
            self._nrefims = int(n_reference_images)
        if max_recursion_depth is None:
            self._maxrecur = None
        else:
            self._maxrecur = int(max_recursion_depth)
        self._crop_center = bool(crop_center_square)
        self._max_n_loaded = max(0, int(max_n_loaded))
        self._verbose = bool(verbose)
        self._seprep = str(sep_replacer)
        self._idx = idx if idx is None else int(idx)
        self._idx_str = '' if self._idx is None else f' {self._idx:d}'

        self._all_refimrpaths = None
        self._all_refimids = None
        self._prepare_refim_catalogue(use_cache, cache_dir, save_cache)
        self._n_all_refims = len(self._all_refimids)
        if self._n_all_refims == 0:
            raise RuntimeError(f'no images found in reference images directory {reference_images_dir}')
        if self._nrefims is None:
            self._nrefims = min(self._n_all_refims, self._max_n_loaded)
        self._all_refims_is_valid = np.ones(self._n_all_refims, dtype=bool)

        self._shuffle = bool(shuffle)
        if self._shuffle:
            if random_seed is None:
                random_seed = np.random.randint(100000)
                print(f'{self.__class__.__name__}{self._idx_str}: random seed not provided, '
                      f'using {random_seed} for reproducibility')
            else:
                print(f'{self.__class__.__name__}{self._idx_str}: random seed set to {random_seed}')
            self._random_seed = random_seed
            self._random_generator = np.random.RandomState(seed=random_seed)
        else:
            self._random_seed = None
            self._random_generator = None

        self._curr_refims = None
        self._curr_refimids = None
        self._all_view = np.arange(self._n_all_refims)
        if self._shuffle:
            self._random_generator.shuffle(self._all_view)
        self._i_toshow = 0
        self._epoch = 0
        self._dynref_epoch = 0
        self._i_loaded = -1
        self._loaded_refims = {'stat': {}, 'dyn': {}}
        self._loaded_refimids = []
        self.refresh_images()

    def _find_refims_recursively(self, curr_rpath='', curr_depth=0):
        if self._maxrecur is not None and curr_depth > self._maxrecur:
            return

        curr_wd = os.path.join(self._refimdir, curr_rpath)
        refstimrpaths = []
        for fn in os.listdir(curr_wd):
            if os.path.isdir(os.path.join(curr_wd, fn)):
                refstimrpaths += self._find_refims_recursively(curr_rpath=curr_rpath+fn+os.sep,
                                                               curr_depth=curr_depth+1)
            elif fn[fn.rfind('.'):].lower() in self.image_extensions:
                refstimrpaths.append(curr_rpath + fn)
        return refstimrpaths

    def _prepare_refim_catalogue(self, use_cache, cache_dir, save_cache=True):
        loaded_cache = False
        cache_fpath = os.path.join(cache_dir, 'reference_images.hdf5')
        refimdir_str = self._refimdir.replace(os.sep, '|') + '|' + \
            ('' if not self._maxrecur else f'maxrecur{self._maxrecur:d}_') + f'sep{self._seprep}'
        if not os.path.isdir(cache_dir):
            try:
                os.mkdir(cache_dir)
            except OSError:
                print(f'{self.__class__.__name__}{self._idx_str}: cannot create cache directory; not saving cache file')
                save_cache = False

        refimdir_mtime = os.path.getmtime(self._refimdir)
        if use_cache and os.path.isfile(cache_fpath):
            with h5.File(cache_fpath, 'r') as f:
                try:
                    cached_mtime = np.array(f[f'{refimdir_str}/mtime'])
                    if cached_mtime != refimdir_mtime:
                        use_cache = False
                except KeyError:
                    use_cache = False

            if use_cache:
                with h5.File(cache_fpath, 'r') as f:
                    try:
                        self._all_refimrpaths = np.array(f[f'{refimdir_str}/rpaths']).astype(str)
                        self._all_refimids = np.array(f[f'{refimdir_str}/ids']).astype(str)
                        loaded_cache = True
                    except KeyError:  # no cached result
                        pass

        if not loaded_cache:
            self._all_refimrpaths = utils.sort_nicely(self._find_refims_recursively())
            self._all_refimids = utils.make_unique_ids(self._all_refimrpaths,
                                                       remove_extension=True, sep_replacer=self._seprep)
            if save_cache:
                with h5.File(cache_fpath, 'a') as f:
                    if refimdir_str in f.keys():
                        del f[refimdir_str]
                    f.create_dataset(f'{refimdir_str}/rpaths', data=np.array(self._all_refimrpaths).astype('S'))
                    f.create_dataset(f'{refimdir_str}/ids', data=np.array(self._all_refimids).astype('S'))
                    f.create_dataset(f'{refimdir_str}/mtime', data=refimdir_mtime)

    def _check_add_to_buffer(self, im, imid, is_dyn=False):
        if self._max_n_loaded > 0:
            dyn_key = ('stat', 'dyn')[is_dyn]
            self._loaded_refims[dyn_key][imid] = im
            self._i_loaded += 1
            if self._i_loaded >= self._max_n_loaded:
                i_rolled = self._i_loaded % len(self._loaded_refimids)
                dyn_key2, imid2 = self._loaded_refimids[i_rolled]
                del self._loaded_refims[dyn_key2][imid2]
                self._loaded_refimids[i_rolled] = (dyn_key, imid)
            else:
                self._loaded_refimids.append((dyn_key, imid))

    def refresh_images(self):
        curr_refims = []
        curr_refimids = []

        # add images until nrefims satisfied
        while len(curr_refims) < self._nrefims:
            n_to_show = self._nrefims - len(curr_refims)

            # dynamic reference image dir takes precedence
            if self._dynrefimdir and os.path.isdir(self._dynrefimdir):
                dynref_rps_added = []
                dynref_rps_unseen = set(os.listdir(self._dynrefimdir)) - self._dynref_rps_seen
                invalid_rps = set(rp for rp in dynref_rps_unseen
                                  if os.path.splitext(rp)[1].lower() not in self.image_extensions)
                self._dynref_rps_seen |= invalid_rps
                dynref_rps_unseen -= invalid_rps
                n_dynref_to_show = n_to_show if self._max_n_dynref is None \
                    else min(n_to_show, self._max_n_dynref - len(curr_refims))
                if dynref_rps_unseen and n_dynref_to_show > 0:
                    for imfn, i in zip(dynref_rps_unseen, range(n_dynref_to_show)):
                        self._dynref_rps_seen.add(imfn)
                        imid = os.path.splitext(imfn)[0]
                        try:    # if already loaded and cached
                            curr_refims.append(self._loaded_refims['dyn'][imid])
                            curr_refimids.append(imid)
                            dynref_rps_added.append(imfn)
                        except KeyError:
                            im = utils.read_image(os.path.join(self._dynrefimdir, imfn))
                            if im is not None:
                                if self._crop_center:
                                    im = utils.center_crop(im)
                                self._check_add_to_buffer(im, imid, is_dyn=True)
                                curr_refims.append(im)
                                curr_refimids.append(imid)
                                dynref_rps_added.append(imfn)
                    if len(dynref_rps_added) > 0:
                        print('added dynref images:', dynref_rps_added)
                    continue    # check dyn ref again; only check stat ref if no dyn imids processed

            # static reference image dir
            for i in range(n_to_show):
                idx_toshow = self._all_view[self._i_toshow % self._n_all_refims]
                if self._all_refims_is_valid[idx_toshow]:
                    rp = self._all_refimrpaths[idx_toshow]
                    imid = self._all_refimids[idx_toshow]
                    try:
                        curr_refims.append(self._loaded_refims['stat'][imid])
                        curr_refimids.append(imid)
                    except KeyError:    # not found in loaded images
                        im = utils.read_image(os.path.join(self._refimdir, rp))
                        if im is None:    # not a valid image
                            self._all_refims_is_valid[idx_toshow] = False
                            print(f'{self.__class__.__name__}{self._idx_str}: invalide reference image: {rp}')
                        else:
                            if self._crop_center:
                                im = utils.center_crop(im)
                            self._check_add_to_buffer(im, imid)
                            curr_refims.append(im)
                            curr_refimids.append(imid)

                # check change of epoch
                self._i_toshow += 1
                epoch = self._i_toshow / self._n_all_refims
                dynref_epoch = epoch * self._dynref_ref_ratio
                if epoch >= self._epoch + 1:
                    self._epoch = int(epoch)
                    if self._shuffle:
                        self._random_generator.shuffle(self._all_view)
                if dynref_epoch >= self._dynref_epoch + 1:
                    self._dynref_epoch = int(dynref_epoch)
                    self._dynref_rps_seen = set()

        self._curr_refims = curr_refims
        self._curr_refimids = curr_refimids
        if self._verbose:
            print(f'{self.__class__.__name__}{self._idx_str}: showing the following {self._nrefims} reference iamges')
            print(curr_refimids)

    @property
    def n_images(self):
        return self._nrefims

    @property
    def current_images(self):
        return self._curr_refims

    @property
    def current_image_ids(self):
        return self._curr_refimids

    @property
    def parameters(self):
        params = {'reference_images_dir': self._refimdir, 'n_reference_images': self._nrefims, 'shuffle': self._shuffle,
                  'crop_center_square': self._crop_center}
        if self._maxrecur is not None:
            params['max_recursion_depth'] = self._maxrecur
        if self._maxrecur is None or self._maxrecur > 0:
            params['sep_replacer'] = self._seprep
        if self._shuffle:
            params['random_seed'] = self._random_seed
        if self._max_n_dynref is not None:
            params['max_n_dynamic_reference_images'] = self._max_n_dynref
        return params


class ExperimentBase:
    """ Base class for Experiment implementing commonly used routines """
    def __init__(self, log_dir, optimizer_name, optimizer_parameters, scorer_name, scorer_parameters,
                 ref_images_loader_parameters=None, cycle_reference_images=True,
                 nthreads=1, random_seed=None, config_file_path=None):
        """
        :param log_dir: str (path), directory to which to write experiment log files
        :param optimizer_name: str or iter of strs, see `defined_optimizers` in `Optimizers.py`
        :param optimizer_parameters: dict, iter of dicts, or dicts with certain iterables values
            interpreted into kwargs passed when initializing the optimizer
            when iterables are interpreted over threads, their len must be 1 or `nthreads`
        :param scorer_name: str, see `defined_scorers` in `Scorers.py`
        :param scorer_parameters: dict, kwargs passed when initializing the scorer
        :param nthreads: int, number of concurrent threads to run; default is 1
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
            assert optimizer_name.lower() in Optimizers.defined_optimizers
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
                if isinstance(param, str) or isinstance(param, dict) or not hasattr(param, '__len__'):
                    for t in range(nthreads):
                        optimizer_parameterss[t][k] = param
                else:
                    assert len(param) == 1 or len(param) >= nthreads
                    if len(param) > nthreads:
                        print(f'note: more values than nthreads passed for {k}; ignoring extra values')
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
                if 'log_dir' not in params.keys():
                    params['log_dir'] = log_dir
                if 'random_seed' not in params:
                    params['random_seed'] = random_seed
                optimizer_parameterss.append(params.copy())
        assert scorer_name in Scorers.defined_scorers or scorer_name in CNNScorers.defined_scorers
        if 'log_dir' not in scorer_parameters.keys():
            scorer_parameters['log_dir'] = log_dir
        if 'random_seed' not in scorer_parameters.keys():
            scorer_parameters['random_seed'] = random_seed
        if ref_images_loader_parameters is not None:
            if not hasattr(ref_images_loader_parameters, 'keys'):
                err_msg = 'ref_images_loader_parameters should be a dict or a list of dicts'
                assert hasattr(ref_images_loader_parameters, '__iter__'), err_msg
                for params in ref_images_loader_parameters:
                    assert hasattr(params, 'keys'), err_msg
                refim_loader_paramss = ref_images_loader_parameters[:]
            else:
                refim_loader_paramss = [ref_images_loader_parameters]
            for iparam, params in enumerate(refim_loader_paramss):
                # if 'random_seed' not in params.keys():
                #     params['random_seed'] = random_seed
                if len(refim_loader_paramss) > 1:
                    params['idx'] = iparam
        if hasattr(cycle_reference_images, '__len__'):
            assert len(cycle_reference_images) == 1 or len(cycle_reference_images) == len(refim_loader_paramss),\
                'cycle_reference_images should be a bool ' \
                'or list of bools with len 1 or same len as number of requested ref_images_loaders'
            cycle_refim_flags = [bool(flag) for flag in cycle_reference_images]
        else:
            cycle_refim_flags = [bool(cycle_reference_images)]
        # an experiment config file is the file defining experiment parameters, e.g., experiment.py, experiment_CNN.py
        # used for record-keeping
        assert config_file_path is None or os.path.isfile(config_file_path), \
            f'experiment config file not found: {config_file_path}'

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

        # attach reference images loader
        self._ref_images_loaders = None
        if ref_images_loader_parameters is not None:
            self._ref_images_loaders = [ReferenceImagesLoader(**params) for params in refim_loader_paramss]
        self._cycle_refim_flags = cycle_refim_flags

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

    def __del__(self):
        try:
            # if many experiments are created (unlikely), logging could exceed max recursion depth
            self._logger.stop()
        except (AttributeError, ValueError):
            # if before full initialization or after log file closed
            pass

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
            raise RuntimeError(f'multiple ({len(self._optimizers)}) optimizers have been loaded; '
                               'asking for "optimizer" is ambiguous')

    @property
    def optimizers(self):
        return self._optimizers

    @property
    def scorer(self):
        return self._scorer

    @property
    def reference_images(self):
        rtn = []
        if self._ref_images_loaders is not None:
            for loader in self._ref_images_loaders:
                rtn += loader.current_images
        return rtn

    @property
    def reference_image_ids(self):
        rtn = []
        if self._ref_images_loaders is not None:
            for loader in self._ref_images_loaders:
                rtn += loader.current_image_ids
        return rtn

    def check_refresh_ref_ims(self):
        if self._ref_images_loaders is not None:
            nflags = len(self._cycle_refim_flags)
            for iloader, loader in enumerate(self._ref_images_loaders):
                if self._cycle_refim_flags[iloader % nflags]:
                    loader.refresh_images()

    @property
    def logger(self):
        return self._logger

    @property
    def logdir(self):
        return self._logdir

    @property
    def parameters(self):
        """ Returns dict storing parameters defining the experiment """
        params = {'class': self.__class__.__name__, 'log_dir': self._logdir, 'nthreads': self._nthreads}
        if self._ref_images_loaders is not None:
            if len(self._ref_images_loaders) > 1:
                for iloader, loader in enumerate(self._ref_images_loaders):
                    params[f'reference_image_loader_{iloader:d}'] = loader.parameters
            else:
                params['reference_image_loader'] = self._ref_images_loaders[0].parameters
            nflags = len(self._cycle_refim_flags)
            if nflags > 1:
                for iloader in range(len(self._ref_images_loaders)):
                    params[f'cycle_reference_images_{iloader:d}'] = self._cycle_refim_flags[iloader % nflags]
            else:
                params['cycle_reference_images'] = self._cycle_refim_flags[0]

        if self._nthreads == 1:
            params['optimizer'] = self.optimizer.parameters
        else:
            for thread in range(self._nthreads):
                params[f'optimizer_thread{thread:02d}'] = self.optimizers[thread].parameters

        params['scorer'] = self.scorer.parameters

        return params


class EphysExperiment(ExperimentBase):
    """
    Implements an experiment that writes images-to-show to disk,
    waits for and loads responses from .mat file from disk, and iterates automatically
    """
    def __init__(self, project_dir, optimizer_name, optimizer_parameters, ichannels,
                 nthreads=1, nchannels_per_thread=None,
                 score_dir=None, log_dir=None, image_size=None, reps=None, block_size=None, scorer_parameters=None,
                 ref_images_loader_parameters=None, cycle_reference_images=True,
                 random_seed=None, config_file_path=None):
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
        :param score_dir: str (path), directory for reading .mat file responses; default is project_dir
        :param log_dir: str (path), directory for saving experiment logs; default is project_dir/backup
        :param image_size: int, size in pixels to which to resize all images; default is to leave unchanged
        :param reps: int, number of times to show each image
        :param block_size: int
            number of images to write each time before waiting for responses
            default is however many images there are to show each step
        :param scorer_parameters: dict, kwargs passed when initializing the scorer; default is empty
        :param random_seed: int
            when set, the experiment will have deterministic behavior; default is an arbitrary integer
        :param config_file_path: str (path),
            path to the .py file defining the experiment,
            intended for saving a copy of it as part of the log
            default is to not save any file
        """
        assert os.path.isdir(project_dir), f'project directory is not a valid directory: {project_dir}'
        assert score_dir is None or os.path.isdir(score_dir), f'mat file directory  is not a valid directory: {score_dir}'
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
            print(f'{self.__class__.__name__}: listening on the following channel(s) for each thread')
            for thread in range(nthreads):
                print(f'\tthread {thread}: {ichannels[thread]}')
        else:
            print(f'{self.__class__.__name__}: listening on the following channel(s): {ichannels}')

        scorer_name = 'ephys'
        for param, val in zip(('write_dir', 'log_dir', 'channel'), (project_dir, log_dir, ichannels)):
            if param not in scorer_parameters.keys():
                scorer_parameters[param] = val
        for param, val in zip(('score_dir', 'image_size', 'reps', 'block_size'), (score_dir, image_size, reps, block_size)):
            if val is not None:
                scorer_parameters[param] = val
        super().__init__(
            log_dir=log_dir, optimizer_name=optimizer_name, optimizer_parameters=optimizer_parameters,
            scorer_name=scorer_name, scorer_parameters=scorer_parameters,
            ref_images_loader_parameters=ref_images_loader_parameters, cycle_reference_images=cycle_reference_images,
            nthreads=nthreads, random_seed=random_seed, config_file_path=config_file_path
        )

        self._imsize = self._scorer.parameters['image_size']

    def run(self):
        """ Main experiment loop """
        self._load_nets()  # nets are not loaded in __init__
        self._save_parameters()
        self.istep = 0

        try:
            while True:
                print(f'\n>>> step {self.istep:d}')
                t00 = time()

                # before scoring, backup codes (optimizer)
                for optimizer in self.optimizers:
                    optimizer.save_current_codes()
                    if hasattr(optimizer, 'save_current_genealogy'):
                        optimizer.save_current_genealogy()
                t01 = time()

                # get scores of images:
                #    1) combine synthesized & reference images
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
                if self.reference_images:
                    combined_scores = self.scorer.score(syn_images + self.reference_images,
                                                        syn_image_ids + self.reference_image_ids)
                else:
                    combined_scores = self.scorer.score(syn_images, syn_image_ids)
                t1 = time()

                # after scoring, backup scores (scorer)
                self.scorer.save_current_scores()
                t2 = time()

                # use results to update optimizer
                threads_synscores = []
                threads_refscores = []
                for i, optimizer in enumerate(self.optimizers):
                    thread_synscores = combined_scores[syn_sections[i]:syn_sections[i + 1], i]
                    thread_refscores = combined_scores[syn_nimgs:, i]
                    if len(thread_synscores.shape) > 1:  # if scores for list of channels returned, pool channels
                        thread_synscores = np.mean(thread_synscores, axis=-1)
                        thread_refscores = np.mean(thread_refscores, axis=-1)  # unused by optimizer but used in summary
                    threads_synscores.append(thread_synscores)
                    threads_refscores.append(thread_refscores)
                    optimizer.step(thread_synscores)  # update optimizer
                t3 = time()

                # summarize scores & delays, & save log
                for thread in range(self._nthreads):
                    if not self._nthreads == 1:
                        print(f'thread {thread:d}: ')
                    print(f'synthetic img scores: mean {np.nanmean(threads_synscores[thread], axis=0)}, '
                          f'all {threads_synscores[thread]}')
                    print('reference image scores: mean {}, all {}'.
                          format(np.nanmean(threads_refscores[thread], axis=0), threads_refscores[thread]))
                print(f'step {self.istep:d} time: total {t3 - t00:.2f}s | ' +
                      f'wait for results {t1 - t01:.2f}s  optimizer update {t3 - t2:.2f}s  '
                      f'write records {t2 - t1 + t01 - t00:.2f}s')
                self.logger.flush()

                # refresh reference images being shown
                self.check_refresh_ref_ims()

                self.istep += 1

        # gracefully exit
        except KeyboardInterrupt:
            print()
            print('... keyboard interrupt')
            print(f'stopped at step {self.istep:d} <<<\n\n')
            self.logger.stop()

    @property
    def parameters(self):
        """ Returns dict storing parameters defining the experiment """
        params = super().parameters
        params.update({'ichannels': self._ichannels_input, 'nchannels_per_thread': self._nchannels_per_thread})
        return params


class CNNExperiment(ExperimentBase):
    """
    Implements a simulated experiment using a CNN to score images
    """
    def __init__(self, project_dir, optimizer_name, optimizer_parameters, target_neuron, with_write,
                 image_size=None, stochastic=None, stochastic_random_seed=None, reps=None, scorer_parameters=None,
                 ref_images_loader_parameters=None, cycle_reference_images=False, random_seed=None, config_file_path=None,
                 max_optimize_images=None, max_optimize_steps=None,
                 write_codes=False, write_last_codes=False, write_best_last_code=True,
                 write_last_images=False, write_best_last_image=True, save_init=False,
                 wait_each_step=None):
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
        :param random_seed: int
            when set, the experiment will have deterministic behavior; default is an arbitrary integer
        :param config_file_path: str (path),
            path to the .py file defining the experiment,
            intended for saving a copy of it as part of the log
            default is to not save any file
        :param max_optimize_images: int
            max number of images to show; default is not set (experiment must be interrupted manually)
        :param max_optimize_steps: int
            max number of steps to run; superseded by `max_images`; default is not set
        :param write_codes: bool, whether to save codes each step
        :param write_last_codes: bool, whether to save codes at the last step
        :param write_best_last_code: bool, whether to save the best code at the last step
        :param write_last_images: bool, whether to save the images at the last step
        :param write_best_last_image: bool, whether to save the best image at the last step
        """
        assert os.path.isdir(project_dir), f'project directory is not a valid directory: {project_dir}'
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
        if 'save_init' not in optimizer_parameters.keys():
            optimizer_parameters['save_init'] = bool(save_init)
        if wait_each_step is not None:
            wait_each_step = float(wait_each_step)
            assert wait_each_step >= 0

        super().__init__(
            log_dir=log_dir, optimizer_name=optimizer_name, optimizer_parameters=optimizer_parameters,
            scorer_name=scorer_name, scorer_parameters=scorer_parameters,
            ref_images_loader_parameters=ref_images_loader_parameters, cycle_reference_images=cycle_reference_images,
            nthreads=1, random_seed=random_seed, config_file_path=config_file_path
        )

        if max_optimize_images is not None:
            if stochastic:
                reps = self.scorer.parameters['reps']
                self._max_steps = int(max_optimize_images / self.optimizer.n_samples / reps)
            else:
                self._max_steps = int(max_optimize_images / self.optimizer.n_samples)
            self._max_steps = max(1, self._max_steps)
        elif max_optimize_steps is not None:
            self._max_steps = max(1, int(max_optimize_steps))
        else:
            self._max_steps = None
        self._wait_each_step = wait_each_step

    def _load_nets(self):
        super()._load_nets()
        self.scorer.load_classifier()

    def run(self):
        """ Main experiment loop """
        self._load_nets()
        self._save_parameters()
        self.istep = 0

        try:
            while self._max_steps is None or self.istep < self._max_steps:
                print(f'\n>>> step {self.istep:d}')
                last_codes = self.optimizer.current_samples_copy
                last_images = self.optimizer.current_images
                last_imgids = self.optimizer.current_image_ids
                last_scores = None
                t0 = time()

                if not self._cycle_refim_flags and self.istep == 0 and self.reference_images:
                    # score images
                    refscores = self.scorer.score(self.reference_images, self.reference_image_ids)
                    t1 = time()
                    # backup scores
                    self.scorer.save_current_scores()
                    t2 = time()
                    # summarize scores & delays
                    print(f'reference image scores: mean {np.nanmean(refscores)}, all {refscores}')
                    print(f'step {self.istep:d} time: total {t2 - t0:.2f}s | wait for results {t1 - t0:.2f}s  '
                          f'write records {t2 - t1:.2f}s')

                else:
                    # score images
                    syn_images = self.optimizer.current_images
                    syn_image_ids = self.optimizer.current_image_ids
                    if self.reference_images:
                        combined_scores = self.scorer.score(syn_images + self.reference_images,
                                                            syn_image_ids + self.reference_image_ids)
                        refscores = combined_scores[self.optimizer.nsamples:]
                    else:
                        combined_scores = self.scorer.score(syn_images, syn_image_ids)
                    synscores = combined_scores[:self.optimizer.nsamples]
                    t1 = time()
                    # before update, backup codes (optimizer) and scores (scorer)
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
                    print(f'synthetic img scores: mean {np.nanmean(synscores, axis=0)}, all {synscores}')
                    if self.reference_images:
                        print(f'reference image scores: mean {np.nanmean(refscores, axis=0)}, all {refscores}')
                    print(f'step {self.istep:d} time: total {t3 - t0:.2f}s | ' +
                          f'wait for results {t1 - t0:.2f}s  write records {t2 - t1:.2f}s  '
                          f'optimizer update {t3 - t2:.2f}s')

                    # refresh reference images being shown
                    self.check_refresh_ref_ims()

                self.logger.flush()
                self.istep += 1
                if self._wait_each_step:
                    sleep(self._wait_each_step)
            print('\nfinished <<<\n\n')

        # gracefully exit
        except KeyboardInterrupt:
            print()
            print('... keyboard interrupt')
            print(f'stopped at step {self.istep:d} <<<\n\n')

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
        params = super().parameters
        params.update({'max_optimize_steps': self._max_steps})
        return params
